"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # “resume”（来自 out_dir）或 gpt2 变体（例如“gpt2-xl”）
out_dir = 'out' # 如果 init_from 不是“resume”则忽略
start = "\n" # 或“<|endoftext|>”等。也可以指定一个文件，用作：“FILE:prompt.txt”
num_samples = 10 # 要绘制的样本数
max_new_tokens = 500 # 每个样本中生成的标记数
temperature = 0.8 # 1.0 = 没有变化，< 1.0 = 随机性较低，> 1.0 = 随机性更高，在预测中
top_k = 200 # 仅保留 top_k 个最有可能的标记，将其他标记限制为 0 概率
seed = 1337
device = 'cuda' # 示例：“cpu”、“cuda”、“cuda:0”、“cuda:1”等。
dtype = 'bfloat16' # 'float32' 或 'bfloat16' 或 'float16'
compile = False # 使用 PyTorch 2.0 编译模型更快
exec(open('configurator.py').read()) # 从命令行或配置文件覆盖
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 在 matmul 上允许 tf32
torch.backends.cudnn.allow_tf32 = True # 在 cudnn 上允许 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供以后在 torch.autocast 中使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型
if init_from == 'resume':
    # 从保存在特定目录中的模型初始化
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # 从给定的 GPT-2 模型初始化
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # 需要 PyTorch 2.0（可选）

# 查找 meta pickle，以防它在数据集文件夹中可用
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # 较旧的检查站可能没有这些......
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO 想让这对任意编码器/解码器方案更通用
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # 好的，让我们默认使用 gpt-2 编码
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 对提示的开头进行编码
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 跑一代
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
