"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 设计用于在 OpenWebText 上训练 gpt2 (124M) 的默认配置值
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # 如果为真，脚本在第一次评估后立即退出
always_save_checkpoint = True # 如果为真，每次评估后总是保存一个检查点
init_from = 'scratch' # 'scratch' 或 'resume' 或 'gpt2*'
# wandb日志记录
wandb_log = False # 默认禁用
wandb_project = 'owt'
wandb_run_name = 'gpt2' # '运行' + str(time.time())
#数据
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # 用于模拟更大的批量
batch_size = 12 # 如果 gradient_accumulation_steps > 1，这是微批量大小
block_size = 1024
# 模型
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 #对于预训练 0 很好，对于微调尝试 0.1+
bias = False #我们在 LayerNorm 和 Linear 层中使用偏差吗？
#优化器
learning_rate = 6e-4 # 最大学习率
max_iters = 600000 # 训练迭代总数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # 以此值剪裁渐变，如果 == 0.0 则禁用
# 学习率衰减设置
decay_lr = True # 是否衰减学习率
warmup_iters = 2000 # 要热身多少步
lr_decay_iters = 600000 # 应该是 ~= max_iters per Chinchilla
min_lr = 6e-5 # 最小学习率，应该是 ~= learning_rate/10 per Chinchilla
# DDP 设置
backend = 'nccl' # “nccl”、“gloo”等
# 系统
device = 'cuda' # 示例：“cpu”、“cuda”、“cuda:0”、“cuda:1”等，或在 macbook 上尝试“mps”
dtype = 'bfloat16' # 'float32'、'bfloat16' 或 'float16'，后者将自动实现 GradScaler
compile = True # 使用 PyTorch 2.0 编译模型更快
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # 从命令行或配置文件覆盖
config = {k: globals()[k] for k in config_keys} # 将对日志记录有用
# -----------------------------------------------------------------------------

# 各种初始化、派生属性、I/O 设置
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是ddp运行吗？
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 这个过程将进行日志记录、检查点等。
    seed_offset = ddp_rank # 每个进程得到不同的种子
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # 如果不是 ddp，我们将在单个 gpu 和一个进程上运行
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # 在 matmul 上允许 tf32
torch.backends.cudnn.allow_tf32 = True # 在 cudnn 上允许 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供以后在 torch.autocast 中使用
# 注意：float16 数据类型将自动使用 GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 穷人的数据加载器
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin 数组 x,y，这允许我们将它们异步移动到 GPU (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 在这里初始化这些，如果 init_from='resume'（即从检查点）可以覆盖
iter_num = 0
best_val_loss = 1e9

# 尝试从数据集中导出 vocab_size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# 初始化模式
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # 从命令行使用 model_args 开始
if init_from == 'scratch':
    # 从头开始创建一个新模型
    print("Initializing a new model from scratch")
    # 确定我们将用于从头开始训练的词汇量
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # 从检查点恢复训练。
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制这些配置属性相等，否则我们甚至无法恢复训练
    # 其余属性（例如 dropout）可以根据需要从命令行保留
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复状态字典的键:(
    # 老实说，不知道检查点有时是如何获得这个前缀的，必须进行更多调试
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # 从 OpenAI GPT-2 权重初始化
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 读取创建的配置参数，以便我们可以将它们正确存储到检查点
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 如果需要，使用模型手术缩小模型块大小
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # 这样检查点就会有正确的值
model.to(device)

# 初始化 GradScaler。如果 enabled=False 缩放器是空操作
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 编译模型
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # 需要 PyTorch 2.0

# 将模型包装到 DDP 容器中
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 帮助估计使用许多批次的拆分的任意准确损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 学习率衰减调度器（带预热的余弦）
def get_lr(it):
    # 1) warmup_iters 步骤的线性预热
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) 如果它 > lr_decay_iters，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 3) 在两者之间，使用余弦衰减到最小学习率
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 系数范围 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# 记录
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练循环
X, Y = get_batch('train') # 获取第一批
t0 = time.time()
local_iter_num = 0 # 此过程生命周期中的迭代次数
raw_model = model.module if ddp else model # 如果需要，打开 DDP 容器
running_mfu = -1.0
while True:

    # 确定并设置本次迭代的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 评估训练/验证集的损失并编写检查点
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # 转换为百分比
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 前向后向更新，可选的梯度累积来模拟更大的批量大小
    # 如果数据类型为 float16，则使用 GradScaler
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 在 DDP 训练中，我们只需要在最后一个微步同步梯度。
            # 官方的方法是使用 model.no_sync() 上下文管理器，但是
            # 我真的不喜欢这会使代码膨胀并迫使我们重复代码
            # 查看那个上下文管理器的来源，它只是切换这个变量
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 缩放损失以考虑梯度累积
        # 当模型在 GPU 上进行正向传递时立即异步预取下一批
        X, Y = get_batch('train')
        # 向后传递，如果在 fp16 中训练，则使用梯度缩放
        scaler.scale(loss).backward()
    # 剪裁渐变
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 如果在 fp16 中训练，则执行优化器和缩放器
    scaler.step(optimizer)
    scaler.update()
    # 尽快刷新梯度，不再需要这个内存
    optimizer.zero_grad(set_to_none=True)

    # 计时和记录
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 损失为浮点数。注意：这是一个 CPU-GPU 同步点
        # 放大以撤消上面的除法，近似真实的总损失（准确的是总和）
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 让训练循环稍微稳定一下
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
