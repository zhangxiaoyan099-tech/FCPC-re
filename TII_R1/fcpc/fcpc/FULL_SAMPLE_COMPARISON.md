# CIFAR-10 全样本 GPU 对比实验

## 实验原则

- CIFAR-10 官方训练集先固定划分为 45,000 个训练样本和 5,000 个验证样本；
- 45,000 个训练样本全部分配给客户端，每个样本恰好出现一次；
- dual-skew 同时使用类别 Dirichlet 权重和客户端数量权重，但不再下采样丢弃数据；
- 所有方法使用同一随机种子、数据划分、ResNet-18、客户端参与率、轮数、优化器和学习率日程；
- 每轮参与 6/10 个客户端，既给 FCPC 留出配对选择空间，也让 FBLG 的客户端选择机制有效；
- `device` 固定为 `cuda:0`，CUDA 不可用时直接报错，不允许静默退回 CPU；
- CSV 必须满足 `train_pool_examples=assigned_unique_examples=45000`；
- 先跑核心四组，再跑全部八组，最终使用种子 42、43、44 报告均值和标准差。

## 方法集合

核心四组：

```text
FedAvg
FedProx
原 FCPC（JSDN + partner）
新 FCPC（weighted JS complementarity + pair center）
```

完整八组另外加入：

```text
MOON
FedDyn-DynamicReg
FBLG
FedCFA
```

其中 `FedDyn-DynamicReg` 明确表示标准动态正则化方法，不等同于原稿误引的推荐系统 FedDyn。

## 服务器检查

```bash
cd ~/FCPC-re/TII_R1/fcpc/fcpc
git log -1 --oneline
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

## 先做两轮 GPU 冒烟

```bash
python scripts/run_cifar10_full_comparison.py \
  --methods core \
  --seeds 42 \
  --rounds 2 \
  --force
```

检查控制台文件：

```bash
tail -n 50 outputs/cifar10_full_comparison/console/*.log
```

检查 CSV 中的 GPU 和样本覆盖字段：

```bash
head -n 2 outputs/cifar10_full_comparison/logs/*.csv
```

必须确认：

```text
device = cuda:0
gpu_name 非空
gpu_sample_count > 0
train_pool_examples = 45000
assigned_unique_examples = 45000
```

## 核心正式实验

```bash
mkdir -p outputs/cifar10_full_comparison

nohup python -u scripts/run_cifar10_full_comparison.py \
  --methods core \
  --seeds 42 \
  > outputs/cifar10_full_comparison/core_seed42_runner.log 2>&1 &
```

查看方法切换进度：

```bash
tail -f outputs/cifar10_full_comparison/core_seed42_runner.log
```

查看当前方法的逐轮 CSV：

```bash
tail -n 3 outputs/cifar10_full_comparison/logs/*.csv
```

## 完整多种子实验

核心实验稳定后运行：

```bash
nohup python -u scripts/run_cifar10_full_comparison.py \
  --methods all \
  --seeds 42,43,44 \
  > outputs/cifar10_full_comparison/all_seeds_runner.log 2>&1 &
```

运行器会自动跳过已经完整存在的同名结果；只有显式使用 `--force` 才会覆盖重跑。

## GPU 监控

另开一个 SSH 终端：

```bash
watch -n 1 nvidia-smi
```

训练程序启动时会立即打印实际设备名和 PyTorch CUDA 版本。资源监控优先使用 `torch.cuda.utilization`，不可用时退回 `nvidia-smi`，并在 CSV 中记录监控后端和有效样本数。
