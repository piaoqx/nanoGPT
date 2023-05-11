
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

用于训练/微调中型 GPT 的最简单、最快的存储库。这是一个重写 [minGPT](https://github.com/karpathy/minGPT) 优先考虑使用上的实用性而不是教育。仍在积极开发中，但目前文件 `train.py` 在 OpenWebText 上重现 GPT-2 (124M)，在大约 4 天的训练中在单个 8XA100 40GB 节点上运行。代码本身简单易读： `train.py` 是一个约 300 行的样板训练循环，`model.py` 是一个约 300 行的 GPT 模型定义，它可以选择从 OpenAI 加载 GPT-2 权重。就是这样。

![repro124m](assets/gpt2_124M_loss.png)

因为代码非常简单，所以很容易破解您的需求，从头开始训练新模型，或微调预训练检查点（例如，目前可用的最大起点是来自 OpenAI 的 GPT-2 1.3B 模型）。

## 安装

依赖关系:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm` <3

## 快速开始

如果你不是深度学习专业人士，只是想感受一下魔力，尝试一下，那么最快的入门方式就是在莎士比亚的作品上训练一个角色级别的 GPT。首先，我们将其下载为单个 (1MB) 文件，并将其从原始文本转换为一个大的整数流：

```
$ python data/shakespeare_char/prepare.py
```

这会在该数据目录中创建一个“train.bin”和“val.bin”。现在是时候训练您的 GPT 了。它的大小在很大程度上取决于系统的计算资源：

**我有一个 GPU**。太好了，我们可以使用 [config/train_shakespeare_char.py](config/train_shakespeare_char.py) 配置文件中提供的设置快速训练一个婴儿 GPT：

```
$ python train.py config/train_shakespeare_char.py
```


如果你往里看，你会发现我们正在训练一个 GPT，其上下文大小最多为 256 个字符、384 个特征通道，并且它是一个 6 层的 Transformer，每层有 6 个头。在一个 A100 GPU 上，此训练运行大约需要 3 分钟，最佳验证损失为 1.4697。根据配置，模型检查点被写入“--out_dir”目录“out-shakespeare-char”。因此，一旦训练完成，我们就可以通过将采样脚本指向此目录来从最佳模型中采样：

```
$ python sample.py --out_dir=out-shakespeare-char
```

这会生成一些示例，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

大声笑`¯\_(ツ)_/¯`。在 GPU 上训练 3 分钟后，对于字符级模型来说还不错。通过在此数据集上微调预训练的 GPT-2 模型，很可能会获得更好的结果（请参阅后面的微调部分）。

**我只有一台 macbook**（或其他便宜的电脑）。不用担心，我们仍然可以训练 GPT，但我们想降低一个档次。我建议每晚获取最新的 PyTorch（安装时[在此处选择](https://pytorch.org/get-started/locally/)），因为它目前很可能使您的代码更高效。但即使没有它，简单的火车运行也可能如下所示：

```
$ python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```


在这里，由于我们在 CPU 而不是 GPU 上运行，我们必须同时设置 `--device=cpu` 并使用 `--compile=False` 关闭 PyTorch 2.0 编译。然后，当我们评估时，我们得到了更多的噪音但更快的估计（`--eval_iters=20`，从 200 下降），我们的上下文大小只有 64 个字符而不是 256，并且每次迭代的批量大小只有 12 个示例，而不是 64 . 我们还将使用一个小得多的 Transformer（4 层、4 个头、128 嵌入大小），并将迭代次数减少到 2000（相应地通常将学习率衰减到 max_iters 左右，使用 `--lr_decay_iters`）。因为我们的网络很小，所以我们也简化了正则化 (`--dropout=0.0`)。这仍然会在大约 3 分钟内运行，但我们只损失了 1.88，因此样本也更差，但它仍然很有趣：

```
$ python sample.py --out_dir=out-shakespeare-char --device=cpu
```
生成这样的样本：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

在 CPU 上大约 3 分钟还不错，可以提示正确的字符格式塔。如果您愿意等待更长的时间，请随意调整超参数、增加网络的大小、上下文长度（`--block_size`）、训练长度等。

最后，在 Apple Silicon Macbook 和最新的 PyTorch 版本上确保添加 `--device=mps`（“Metal Performance Shaders”的缩写）； PyTorch 然后使用可以*显着*加速训练（2-3 倍）并允许您使用更大网络的片上 GPU。有关更多信息，请参阅[第 28 期](https://github.com/karpathy/nanoGPT/issues/28)。

## 复制 GPT-2

更认真的深度学习专业人士可能对重现 GPT-2 结果更感兴趣。所以我们开始 -我们首先标记数据集，在本例中是 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)，OpenAI 的（私有）WebText 的开放复制：

```
$ python data/openwebtext/prepare.py
```

这将下载并标记 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集。它将创建一个 `train.bin` 和 `val.bin`，它们以一个序列保存 GPT2 BPE 令牌 ID，存储为原始 uint16 字节。然后我们准备开始训练。要重现 GPT-2 (124M)，您至少需要一个 8X A100 40GB 节点并运行：

```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

这将使用 PyTorch 分布式数据并行 (DDP) 运行大约 4 天，并下降到约 2.85 的损失。现在，刚刚在 OWT 上评估的 GPT-2 模型得到了大约 3.11 的 val 损失，但如果你对其进行微调，它将下降到 ~2.85 区域（由于明显的域差距），使两个模型~匹配。

如果你在一个集群环境中并且你有幸拥有多个 GPU 节点，你可以让 GPU 运行 brrrr 例如。跨越 2 个节点，例如：

```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

对您的互连（例如 iperf3）进行基准测试是个好主意。特别是，如果您没有 Infiniband，则还要在上述启动前加上 `NCCL_IB_DISABLE=1`。您的多节点训练会起作用，但很可能_爬行_。默认情况下，检查点会定期写入“--out_dir”。我们可以通过简单的 `$ python sample.py` 从模型中采样。
最后，要在单个 GPU 上进行训练，只需运行“$ python train.py”脚本即可。看看它的所有参数，脚本试图变得非常可读、可破解和透明。您很可能希望根据需要调整其中一些变量。

## 基线

OpenAI GPT-2 检查点允许我们为 openwebtext 设置一些基线。我们可以得到如下数字：

```
$ python train.py eval_gpt2
$ python train.py eval_gpt2_medium
$ python train.py eval_gpt2_large
$ python train.py eval_gpt2_xl
```

并在 train 和 val 上观察以下损失：

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

然而，我们必须注意到 GPT-2 是在（封闭的，从未发布的）WebText 上训练的，而 OpenWebText 只是该数据集的最大努力开放复制。这意味着存在数据集域差距。事实上，直接在 OWT 上使用 GPT-2 (124M) 检查点和微调一段时间可以将损失降至 ~2.85。然后这成为更合适的基线 w.r.t.再生产。

＃＃ 微调
微调与训练没有什么不同，我们只是确保从预训练模型进行初始化并以较小的学习率进行训练。有关如何在新文本上微调 GPT 的示例，请转到“data/shakespeare”并运行“prepare.py”以下载小型莎士比亚数据集并将其呈现为“train.bin”和“val.bin”，使用来自 GPT-2 的 OpenAI BPE 分词器。与 OpenWebText 不同，这将在几秒钟内运行。微调可能需要很少的时间，例如在单个 GPU 上只需几分钟。运行一个示例微调，如：

```
$ python train.py config/finetune_shakespeare.py
```


这将在 config/finetune_shakespeare.py 中加载配置参数覆盖（虽然我没有对它们进行太多调整）。基本上，我们使用“init_from”从 GPT2 检查点进行初始化并正常训练，只是时间更短且学习率较小。如果内存不足，请尝试减小模型大小（它们是 `{'gpt2'、'gpt2-medium'、'gpt2-large'、'gpt2-xl'}`）或可能减小 `block_size`（上下文长度）。最佳检查点（最低验证损失）将在 out_dir 目录中，例如根据配置文件，默认情况下在 out-shakespeare 中。然后您可以运行 `sample.py --out_dir=out-shakespeare` 中的代码：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

哇哦，GPT，进入了那边的某个黑暗地方。我并没有对配置中的超参数进行太多调整，请随意尝试！

##采样/推理

使用脚本“sample.py”从 OpenAI 发布的预训练 GPT-2 模型或您自己训练的模型中进行采样。例如，这是一种从最大的可用“gpt2-xl”模型中采样的方法：

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

如果您想从您训练的模型中采样，请使用 `--out_dir` 适当地指向代码。您还可以使用文件中的一些文本提示模型，例如`$ python sample.py --start=FILE:prompt.txt`。

## 效率笔记

对于简单的模型基准测试和分析，“bench.py​​”可能会有用。它与 train.py 的训练循环中发生的事情相同，但省略了许多其他复杂性。
请注意，代码默认使用 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。在撰写本文时（2022 年 12 月 29 日），这使得 `torch.compile()` 在夜间发布中可用。一行代码的改进是显而易见的，例如将迭代时间从 ~250ms/iter 减少到 135ms/iter。 PyTorch 团队干得好！

## todos

-调查并添加 FSDP 而不是 DDP
-标准评估的评估零射击困惑（例如 LAMBADA？HELM？等）
-微调微调脚本，我认为超参数不是很好
-训练期间线性批量增加的时间表
-结合其他嵌入（旋转，不在场证明）
-我认为在检查点中将优化缓冲区与模型参数分开
-额外记录网络健康状况（例如梯度剪辑事件、幅度）
-围绕更好的 init 等进行更多调查。

## 故障排除

请注意，默认情况下，此 repo 使用 PyTorch 2.0（即 `torch.compile`）。这是相当新的和实验性的，尚未在所有平台（例如 Windows）上可用。如果您遇到相关的错误消息，请尝试通过添加 `--compile=False` 标志来禁用它。这会减慢代码但至少它会运行。
对于此存储库、GPT 和语言建模的某些上下文，观看我的 [从零到英雄系列](https://karpathy.ai/zero-to-hero.html) 可能会有所帮助。具体来说，如果您之前有一些语言建模背景，[GPT 视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) 很受欢迎。

如需更多问题/讨论，请随时访问 Discord 上的 **#nanoGPT**：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 致谢

所有 nanoGPT 实验均由 [Lambda 实验室](https://lambdalabs.com) 上的 GPU 提供支持，这是我最喜欢的云 GPU 提供商。感谢 Lambda 实验室赞助 nanoGPT！
