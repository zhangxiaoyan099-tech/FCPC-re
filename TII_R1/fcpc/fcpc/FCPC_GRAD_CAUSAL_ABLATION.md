# FCPC-grad 收敛加速因果消融方案

## 1. 目标

不再预设“历史更新近似当前负梯度”是唯一原因，而是在完全相同的数据划分、客户端采样、模型初始化、优化器和评估协议下，逐项替换FCPC-grad组件，回答哪些机制真正提高前50轮AUC。

主要终点：`Val-AUC@50`。辅助终点：最佳验证准确率、达到50%/60%准确率的轮数，以及客户端更新几何指标。

## 2. 预注册实验组

| 方法 | 唯一用途 |
|---|---|
| `fedavg` | 无FCPC基线 |
| `global_prox` | 中心固定为当前全局模型，只测试proximal收缩 |
| `pair_center` | 共同历史参数中心，不使用更新外推 |
| `grad_real` | 完整FCPC-grad |
| `grad_shuffled` | 保持当轮代理方向范数集合，但把方向轮换给错误配对 |
| `grad_reversed` | 保持代理范数和配对，把方向取反 |
| `grad_random_pairing` | 保留真实代理和proximal，改用随机配对 |
| `grad_constant_beta` | 保留其余机制，取消beta余弦衰减 |
| `grad_local_end` | 保持总收缩系数rho^H，只在本地训练末尾拉回一次 |
| `grad_no_clip` | 保留其余机制，取消共同中心5%信赖域裁剪 |

第一阶段只运行以上10组。若互补配对、beta调度或逐batch近端在第一阶段显示正效应，再补运行3个低成本交互单元：`pair_center_random`、`pair_center_constant_beta`、`pair_center_local_end`。它们与已有结果组成差分中的差分，不需要运行完整的全因子实验。

关键配对比较：

- `global_prox - fedavg`：纯漂移收缩是否加速；
- `pair_center - global_prox`：配对历史参数中心是否有效；
- `grad_real - pair_center`：历史更新外推的增量；
- `grad_real - grad_shuffled/reversed`：真实方向信息是否必要；
- `grad_real - grad_random_pairing`：互补配对是否必要；
- `grad_real - grad_constant_beta`：余弦释放是否必要；
- `grad_real - grad_local_end`：逐batch交互是否必要；
- `grad_real - grad_no_clip`：裁剪是否必要。

## 3. 同步记录的机制指标

每轮记录：

\[
S_t=\sum_i a_i\|\Delta_i^t\|^2,
\]

\[
V_t=\sum_i a_i\|\Delta_i^t-\bar\Delta_t\|^2,
\]

\[
C_t=1-\frac{\|\bar\Delta_t\|^2}{S_t},
\]

以及配对内更新分歧：

\[
D_t^{\rm pair}
=\frac1{|M_t|}\sum_{(i,j)\in M_t}\|\Delta_i^t-\Delta_j^t\|^2.
\]

CSV字段分别为：

- `client_update_second_moment`；
- `client_update_variance`；
- `server_update_norm`；
- `update_cancellation_fraction`；
- `mean_pair_update_disagreement`；
- `mean/min/max/std_effective_proximal_contraction`。

若某变体AUC更高，同时显著降低 `V_t`、`C_t` 或配对分歧，则支持“稳定化/方差降低”机制。若 `grad_real`稳定优于shuffle和reverse，才支持历史方向携带有效时序信息。

## 4. 服务器测试顺序

### 4.1 单元测试

```bash
python -m unittest discover -s tests -v
```

### 4.2 两轮GPU冒烟

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_causal_ablation \
  --methods screen \
  --seeds 45 \
  --rounds 2 \
  --continue-on-error \
  2>&1 | tee outputs/fcpc_grad_causal_ablation_smoke.txt
```

确认10种方法全部显示 `DONE`，并检查任一CSV包含新增几何字段。

### 4.3 三种子、50轮预注册筛选

建议在tmux中执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_causal_ablation \
  --methods screen \
  --seeds 45,46,47 \
  --rounds 50 \
  --continue-on-error \
  2>&1 | tee outputs/fcpc_grad_causal_ablation_r50.txt
```

已有完整CSV会自动跳过。中断中的不完整运行会从头覆盖，不能从中间轮次恢复。

### 4.4 第二阶段交互确认

第一阶段完成后执行。已有的5个重复单元会自动跳过，实际只需补3种方法：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_causal_ablation \
  --methods interactions \
  --seeds 45,46,47 \
  --rounds 50 \
  --continue-on-error \
  2>&1 | tee outputs/fcpc_grad_causal_ablation_interactions_r50.txt
```

## 5. 汇总

```bash
python -m scripts.summarize_causal_ablation \
  --seeds 45,46,47 \
  --rounds 50 \
  --clients-per-round 6 \
  --output outputs/fcpc_grad_causal_ablation/causal_ablation_r50_detail.csv \
  | tee outputs/fcpc_grad_causal_ablation/causal_ablation_r50_summary.txt
```

随后同时计算逐种子组件效应和交互效应：

```bash
python -m scripts.analyze_causal_ablation \
  outputs/fcpc_grad_causal_ablation/causal_ablation_r50_detail.csv \
  --metrics val_auc_50,best_val_acc,round_to_0.5,round_to_0.6 \
  --output outputs/fcpc_grad_causal_ablation/causal_effects_r50.csv \
  | tee outputs/fcpc_grad_causal_ablation/causal_effects_r50.txt
```

其中交互项以互补配对为例：

\[
I_{M\times G}
=\bigl[A(\mathrm{grad\_real})-A(\mathrm{grad\_random})\bigr]
-\bigl[A(\mathrm{pair\_center})-A(\mathrm{pair\_center\_random})\bigr].
\]

若该值大于0，说明互补配对在梯度共同中心中带来的收益超过它在普通历史中心中的收益，即二者存在正协同，而非两个孤立效果的简单相加。所有差值统一规定正数代表相应组件或交互更好。三种子样本量仍较小；若均值为正但置信下界跨零，应扩展到至少5个种子，而不是直接写成显著结论。

## 6. 决策规则

1. `global_prox`已接近`grad_real`：主要来自proximal稳定化；
2. `grad_real > pair_center`且优于shuffle/reverse：历史方向确有因果贡献；
3. `grad_real > grad_random_pairing`：JSD互补配对贡献成立；
4. `grad_real > grad_constant_beta`：余弦衰减避免后期过约束；
5. `grad_real > grad_local_end`：逐batch拉回与本地SGD交互、或数量偏斜下的重复拉回，是关键机制；
6. `grad_real > grad_no_clip`：信赖域裁剪对稳定性必要；
7. 若高AUC伴随低更新方差但方向消融差异不大，应把主理论转向偏差—方差权衡，而不是历史梯度对齐。
