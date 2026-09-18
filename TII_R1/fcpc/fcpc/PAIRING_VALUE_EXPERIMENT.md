# FCPC-grad 配对价值与 constant-beta 因果认证

## 1. 研究问题

本实验不再预设 JSD 配对一定加快收敛，而是把两个问题分开：

1. 在相同客户端选择轨迹、相同初始模型和相同训练协议下，JSD 配对是否比随机配对、相似配对更好地降低配对混合分布的全局 KL 残差？
2. JSD 配对是否改善类别性能和客户端标签偏斜下的性能均衡，同时不损害收敛 AUC？

三组处理均使用 FCPC-grad、constant `beta=0.2`、真实历史代理、batchwise exact proximal。唯一改变是：

- `pairing_jsd`：加权 JS 互补图上的精确最大权匹配；
- `pairing_random`：随机配对；
- `pairing_similar`：优先匹配分布相似客户端。

训练器在每轮记录 `selected_clients`。汇总脚本逐种子、逐轮验证三组选择轨迹完全一致；不一致时直接报错，不输出比较结论。

## 2. 配对分布指标

对配对 ((i,j))，令

\[
q_{ij}=\frac{n_i p_i+n_jp_j}{n_i+n_j},\qquad
a_i=\frac{n_i}{\sum_k n_k}.
\]

逐轮记录：

\[
R_t(M)=\sum_{(i,j)\in M_t}(a_i+a_j)
KL(q_{ij}\|\bar p),
\]

以及

\[
S_t(M)=\sum_{(i,j)\in M_t}
\left[a_iKL(p_i\|q_{ij})+a_jKL(p_j\|q_{ij})\right].
\]

代码同时检查恒等式：

\[
\sum_{(i,j)}
\left[a_iKL(p_i\|\bar p)+a_jKL(p_j\|\bar p)\right]
=S_t(M)+R_t(M).
\]

对应 CSV 字段：

- `pair_mixture_kl_residual_normalized`：按本轮已配对全局样本质量归一化的 (R_t)，越低越好；
- `pair_complementarity_gain_normalized`：归一化 (S_t)，越高越好；
- `pair_kl_identity_error`：恒等式数值误差，应接近 0；
- `mean/min_pair_label_entropy`：混合标签熵；
- `mean/min_pair_class_coverage`：每对覆盖类别比例；
- `mean_selected_pairing_score`：实际选中边在 LDP 配对矩阵中的平均分。

## 3. 分类与公平性指标

最佳验证模型在统一 CIFAR-10 测试集上记录：

- `test_macro_f1`；
- `test_macro_recall`；
- `test_worst_class_recall`。

客户端指标定义为标签偏斜代理，而不是真实本地测试准确率：

\[
\widetilde A_i=\sum_y p_i(y)\operatorname{Recall}_y.
\]

记录其均值、最小值、10% 分位、标准差和 Jain 指数。还记录客户端局部少数标签对应召回率的均值与最小值。

该代理只回答 label-shift 条件下的均衡性，不能证明 feature-shift 下的真实客户端准确率。论文中必须保持这一名称和限制。

## 4. 单元测试与配置检查

服务器进入项目和环境后运行：

```bash
cd ~/FCPC-re/TII_R1/fcpc/fcpc
conda activate fcpc-core

python -m unittest discover \
  -s tests \
  -p "test_pairing_value_diagnostics.py" \
  -v

python -u -m scripts.run_pairing_value_experiment \
  --seeds 45 \
  --rounds 2 \
  --dry-run
```

## 5. 先做 seed45、50轮筛查

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_pairing_value_experiment \
  --seeds 45 \
  --rounds 50 \
  2>&1 | tee outputs/fcpc_grad_pairing_value_seed45_r50.txt

python -m scripts.summarize_pairing_value_experiment \
  --seeds 45 \
  --rounds 50 \
  --clients-per-round 6 \
  | tee outputs/fcpc_grad_pairing_value/summary_seed45_r50.txt
```

## 6. 三种子、200轮确认

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_pairing_value_experiment \
  --seeds 45,46,47 \
  --rounds 200 \
  2>&1 | tee outputs/fcpc_grad_pairing_value_seeds45_47_r200.txt

python -m scripts.summarize_pairing_value_experiment \
  --seeds 45,46,47 \
  --rounds 200 \
  --clients-per-round 6 \
  | tee outputs/fcpc_grad_pairing_value/summary_seeds45_47_r200.txt
```

## 7. constant-beta 因果补充

以下 preset 包含：FedAvg、global-prox、完整 FCPC-grad、反向代理、随机配对、local-end proximal，所有 proximal 组均使用 constant `beta=0.2`。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_causal_ablation \
  --methods constant_beta \
  --seeds 45,46,47 \
  --rounds 200 \
  2>&1 | tee outputs/fcpc_grad_constant_beta_causal_r200.txt

python -m scripts.summarize_causal_ablation \
  --seeds 45,46,47 \
  --rounds 200 \
  --clients-per-round 6 \
  --output outputs/fcpc_grad_causal_ablation/constant_beta_r200_detail.csv \
  | tee outputs/fcpc_grad_causal_ablation/constant_beta_r200_summary.txt

python -m scripts.analyze_causal_ablation \
  outputs/fcpc_grad_causal_ablation/constant_beta_r200_detail.csv \
  --metrics val_auc_50,val_auc_100,val_auc_all,best_val_acc,selected_test_acc,round_to_0.5,round_to_0.6 \
  --output outputs/fcpc_grad_causal_ablation/constant_beta_r200_effects.csv \
  | tee outputs/fcpc_grad_causal_ablation/constant_beta_r200_effects.txt
```

## 8. 预注册解释规则

- 若 JSD 显著降低 (R_t)，并提高最差客户端代理、少数标签召回或 macro-F1，且 AUC 不下降：将 JSD 定位为分布互补与性能均衡组件。
- 若 JSD 只降低 (R_t)，但所有下游指标与随机配对无差别：只能证明配对目标实现，不能声称实际收益。
- 若 JSD 的 (R_t) 不低于随机/相似配对：首先检查配对实现、LDP 扰动和指标口径，不能写正面结论。
- 若 batchwise proximal 在 constant-beta 下仍优于 matched local-end：支持“逐 batch 方向传递/收缩”机制；否则原结论应降级为特定 schedule 下的现象。
- 任何客户端选择公平性结论均不在本实验范围内，因为当前流程是先采样、后配对。
