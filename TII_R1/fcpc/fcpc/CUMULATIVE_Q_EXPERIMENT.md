# FCPC-grad 累计整体收益与收敛加速实验

## 1. 要回答的两个问题

1. **机制问题**：在固定状态和相同 mini-batch 重放下，开启梯度共同中心相对关闭代理，累计是否产生正的额外下降收益？
2. **训练问题**：在相同种子、初始模型、数据划分和训练协议下，FCPC-grad 是否比各基线具有更高的前期验证 AUC，或更少的达标轮数？

两者不能混成一个指标。第一个实验解释机制，第二个实验验证端到端收敛加速。

## 2. 单轮变量

在被审计状态 (w^t) 上，令：

- (B_t=\Delta_t^{\mathrm{off}})：关闭梯度代理、保留相同 proximal 更新时的服务器更新；
- (\Delta_t^{\mathrm{grad}})：开启 FCPC-grad 时的服务器更新；
- (Z_t=\Delta_t^{\mathrm{grad}}-B_t)：梯度共同中心真正改变的服务器更新；
- (g_t=\nabla F(w^t))：同一目标和 BatchNorm 模式下的经验全局梯度。

对给定光滑常数 (L)，理论额外下降证书为：

\[
Q_t(Z_t;B_t)
=-\langle g_t,Z_t\rangle
-L\langle B_t,Z_t\rangle
-\frac{L}{2}\|Z_t\|^2.
\]

同时记录无需指定 (L) 的经验反事实收益：

\[
G_t^{\mathrm{obs}}
=\widehat F(w^t+B_t)-\widehat F(w^t+B_t+Z_t).
\]

其中两次目标评估必须使用相同数据批次。

## 3. 每 5 轮累计

在 (t_k=5,10,\ldots,200) 记录一次。审计网格上的直接累计量是

\[
\sum_k Q_{t_k}.
\]

脚本还报告右端点近似：

\[
\widehat{\mathcal Q}_{1:T}
=\sum_k(t_k-t_{k-1})Q_{t_k}
=5\sum_kQ_{5k}.
\]

它是全轮累计收益的数值近似，不是未观测 160 个轮次的精确求和。因此论文中应写“every-5-round quadrature estimate”，不能写成逐轮精确值。

归一化累计系数为：

\[
\widehat q_{1:T}
=\frac{\widehat{\mathcal Q}_{1:T}}
{\sum_k(t_k-t_{k-1})\|g_{t_k}\|^2}.
\]

## 4. 统计单位与判据

同一轨迹上的 40 个时间点高度相关，不能作为 40 个独立样本。统计顺序是：

1. 在每个 `model_seed × batch_seed` 内先完成时间累计；
2. 跨 `batch_seed` 给出条件均值、标准误和 95% 置信下界；
3. 每个 `model_seed` 先对重放种子取均值，再跨至少 3 个模型种子给出模型种子级置信下界。

主要判据：

- 经验机制成立：最终 `cumulative_observed_gain_lcb95_replay > 0`；
- 理论证书成立：只有在 (L) 已被独立验证为有效光滑上界时，才可用 `cumulative_Q_lcb95 > 0`；
- (L=0) 仅隔离一阶投影，(L=0.1,1) 在未验证光滑常数前都只是敏感性分析；
- 收敛加速成立：跨模型种子的配对 `AUC@50/AUC@100` 差值置信下界大于 0，或“节省的达标轮数”置信下界大于 0。

## 5. 运行方法

先做 CPU 合成数据冒烟测试：

```bash
python -u -m scripts.run_fcpc_grad_oracle_audit \
  --config configs/lemma456/smoke_fcpc_grad_cumulative.yaml

python -m scripts.summarize_cumulative_q \
  outputs/fcpc_grad_cumulative_smoke/oracle_metrics.csv \
  --expected-step 1 --smoothness-L 0 \
  --output-dir outputs/fcpc_grad_cumulative_smoke/summary_L0
```

生成三个模型种子的配置：

```bash
python -m scripts.generate_cumulative_q_configs --seeds 42,43,44
```

单个种子运行（流式审计不会保存四十份巨大的客户端历史检查点）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_oracle_audit \
  --config configs/lemma456/generated_cumulative_q/cifar10_fcpc_grad_cumulative_seed42.json \
  2>&1 | tee outputs/fcpc_grad_cumulative_seed42_console.txt
```

另外两个种子把命令中的 `42` 改为 `43`、`44`。不要传 `--reuse-checkpoints`，因为配置启用了 `stream_checkpoints`。

合并三个种子并输出累计曲线、均值和置信下界：

```bash
python -m scripts.summarize_cumulative_q \
  outputs/fcpc_grad_cumulative_seed42/oracle_metrics.csv \
  outputs/fcpc_grad_cumulative_seed43/oracle_metrics.csv \
  outputs/fcpc_grad_cumulative_seed44/oracle_metrics.csv \
  --panel raw --strategy optimal --method ungated \
  --step-scale 0.5 --smoothness-L 0 \
  --expected-step 5 \
  --output-dir outputs/fcpc_grad_cumulative_summary_L0
```

对 (L=0.1) 和 (L=1) 重复汇总，只需要改变 `--smoothness-L`，不需要重新训练。

端到端收敛比较先用现有统一脚本生成逐种子明细，再做配对差值：

```bash
python -m scripts.summarize_full_comparison \
  --seeds 45,46,47 --rounds 200 --clients-per-round 6 \
  --output outputs/cifar10_full_comparison/comparison_seeds45_47_detail.csv

python -m scripts.summarize_paired_convergence \
  outputs/cifar10_full_comparison/comparison_seeds45_47_detail.csv \
  --candidate fcpc_grad \
  --output outputs/cifar10_full_comparison/paired_convergence_lcb.csv
```

## 6. 结论边界

当前审计以冻结参考轨迹做匹配反事实，能回答“相同状态和随机批次下，梯度共同中心是否带来累计额外收益”。它与真实 FCPC-grad 训练轨迹的 AUC 结果互相补充，但不能把参考轨迹的累计量冒充成真实训练轨迹逐轮精确求和。若论文需要这一更强结论，应把同样的反事实钩子嵌入 FCPC-grad 主训练循环。
