# CIFAR-10 统一基线对比

## 1. 本轮目的

在完全相同的 CIFAR-10 dual-skew 协议下比较：

- FedAvg；
- FedProx；
- MOON；
- FedDyn；
- FBLG；
- FedCFA；
- FCPC-grad。

统一基础配置位于 `configs/cifar10_full_comparison_base.yaml`：10 个客户端、每轮 6 个客户端、ResNet-18、batch size 128、SGD、初始学习率 0.05、余弦学习率和 200 轮训练。不要将 `configs/baselines/*cpr2*` 的结果混入本表，因为这些文件使用另一套训练协议。

FCPC-grad 的固定开发参数来自种子 42 的验证集选择：

\[
\beta_0=0.2,\qquad \xi=1,\qquad s=0.5,
\]

并使用 cosine-to-zero 的 \(\beta_t\)、加权 JS 最大权匹配、梯度共同中心和 exact proximal。

## 2. 先验证统一配置

```bash
python -m unittest \
  tests.test_fcpc_grad_runner \
  tests.test_full_comparison_summary -v

python -m scripts.run_cifar10_full_comparison \
  --methods all \
  --seeds 45 \
  --rounds 2 \
  --dry-run
```

## 3. 服务器正式执行

建议在 `tmux` 中运行，避免 SSH 断开后任务退出：

```bash
tmux new -s fcpc-baselines

cd ~/FCPC-re/TII_R1/fcpc/fcpc
conda activate fcpc-core

CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_cifar10_full_comparison \
  --methods fedavg,fedprox,moon,feddyn_dynamicreg,fblg,fedcfa,fcpc_grad \
  --seeds 45,46,47 \
  --rounds 200 \
  --continue-on-error \
  2>&1 | tee outputs/cifar10_full_comparison_run.txt
```

按 `Ctrl-b`，再按 `d` 可离开 tmux；使用下面命令重新进入：

```bash
tmux attach -t fcpc-baselines
```

这会顺序执行 21 个 200 轮任务，预计需要较长时间。若只做第一轮探索，可先使用 `--seeds 45`；确认所有方法数值稳定后再补 46、47。

## 4. 查看进度

```bash
pgrep -af "run_cifar10_full_comparison|src.main"

for f in outputs/cifar10_full_comparison/logs/*.csv; do
  echo "$(basename "$f"): $(($(wc -l < "$f") - 1)) rounds"
done

tail -n 30 outputs/cifar10_full_comparison_run.txt
```

runner 会跳过已经完成的 CSV；任务中断后重新执行同一命令即可从未完成的方法继续，但单个不足 200 轮的任务会重新开始，而不是从模型检查点续训。

## 5. 汇总结果

```bash
python -m scripts.summarize_full_comparison \
  --seeds 45,46,47 \
  --rounds 200 \
  --clients-per-round 6 \
  --thresholds 0.50,0.60,0.65,0.70 \
  | tee outputs/cifar10_full_comparison/summary.txt
```

汇总文件为：

```text
outputs/cifar10_full_comparison/comparison_summary.csv
```

主要比较 Val-AUC@50、Val-AUC@100、达到各准确率阈值的轮数、时间和累计通信量，以及验证集选择的测试准确率、最后一轮准确率、每轮时间、CPU/GPU利用率和峰值内存。

## 6. 结果边界

当前五个外部基线使用项目中已经接入的固定默认参数；这可以作为统一协议下的第一轮比较，但还不是最终的超参数公平性证据。正式投稿前，应在开发种子 42 上给每个方法相同数量的候选配置、只用验证指标选参，再将固定配置用于种子 45、46、47。不得根据测试准确率重新选择方法参数。
