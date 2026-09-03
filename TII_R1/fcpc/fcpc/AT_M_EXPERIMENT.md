# \(A_t(M)\) 冻结检查点实验执行说明

## 实验回答的问题

该实验在同一个 FedAvg 检查点上重放一轮 FCPC，仅改变客户端匹配 \(M\)，比较：

\[
M_{\mathrm{optimal}},\quad
M_{\mathrm{random}},\quad
M_{\mathrm{similar}},\quad
M_{\mathrm{JSDN}}.
\]

输出四个核心量：

- \(R(M)\)：配对混合标签分布相对全局标签分布的加权 KL 残差；
- \(H_t(M)\)：轮初配对梯度相对全局梯度的加权偏差；
- \(A_t(M)\)：实际多步本地配对更新相对理想配对梯度步的执行误差；
- \(U_t(M)\)：实际全局聚合更新相对理想全局梯度步的误差。

脚本不会改动正常的 `src.main` 训练流程。

## 服务器执行顺序

在项目根目录执行。

### 1. 单元测试

```bash
python -m unittest tests.test_at_m_metrics -v
```

也可以执行全部测试：

```bash
python -m unittest discover -s tests -v
```

### 2. CPU 合成数据冒烟测试

```bash
python -u -m scripts.run_at_m_audit \
  --config configs/at_m/smoke_synthetic_at_m.yaml \
  2>&1 | tee outputs/at_m_smoke.txt
```

预期产生：

```text
outputs/at_m_smoke/at_m_metrics.csv
outputs/at_m_smoke/at_m_pairs.csv
outputs/at_m_smoke/at_m_summary.csv
outputs/at_m_smoke/at_m_correlations.csv
outputs/at_m_smoke/neutral_checkpoints/
```

### 3. CIFAR-10 快速 GPU 审计

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_at_m_audit \
  --config configs/at_m/cifar10_at_m_quick.yaml \
  2>&1 | tee outputs/at_m_quick.txt
```

快速配置只在 FedAvg 第 5 轮进行审计，并用每个客户端最多 20 个 batch 估计轮初梯度。它用于检查 GPU、显存、数值稳定性、CSV 字段和四种配对是否都能运行，不用于论文最终结论。

### 4. CIFAR-10 正式审计

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_at_m_audit \
  --config configs/at_m/cifar10_at_m_full.yaml \
  2>&1 | tee outputs/at_m_full_seed42.txt
```

正式配置审计 FedAvg 第 5、10、20、50 轮，使用完整客户端数据计算 \(g_i(w^t)\)，使用 5 个 batch seed 和 20 个随机配对 seed，并分别运行 raw 与 LDP 两个面板。

如果审计中断，但 `neutral_checkpoints` 已完整生成，可跳过 FedAvg 预热：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_at_m_audit \
  --config configs/at_m/cifar10_at_m_full.yaml \
  --reuse-checkpoints \
  2>&1 | tee outputs/at_m_full_seed42-reuse.txt
```

`--reuse-checkpoints` 会校验数据、模型、划分和 warmup 配置签名；不一致时拒绝复用，避免错误检查点污染实验。

## 两个实验面板

### raw 面板

全部方法都使用未扰动的客户端标签分布和样本数。该面板检查配对几何机制本身。

### LDP 面板

全部数据驱动的匹配方法都使用同一份 LDP 扰动元数据。该面板检查隐私噪声下的部署效果。

两个面板都使用真实未扰动分布计算 `R_M`，同时用方法实际观察到的元数据计算 `R_observed_M`。因此可以分辨“方法以为残差下降”和“真实残差确实下降”。

## 关键公平性约束

- 数据增强固定关闭；
- 所有反事实重放从完全相同的全局模型与客户端历史模型开始；
- 同一 `batch_seed` 下，每个客户端的样本索引和顺序完全相同；
- 重放统一执行 2 个本地 SGD 步；
- 重放使用 `momentum=0`、`weight_decay=0`，因此 \(\gamma=\eta E\) 与定义一致；
- 模型初始化种子、数据划分和 LDP 噪声固定；
- 随机匹配使用多个 seed，不拿一次偶然结果下结论；
- `optimal` 与 `similar` 使用加权 JS 矩阵，`jsdn` 使用原 JSDN 矩阵；
- 所有方法使用相同的共同中心、裁剪、proximal、\(\beta\) 和样本比例权重规则。

## 输出文件

`at_m_metrics.csv` 每行是一项检查点/面板/匹配/batch seed 实验，包含核心指标、中心距离、配对终点距离、损失和验证集变化。

`at_m_pairs.csv` 保存每一对客户端对 \(A_t\) 和 \(H_t\) 的贡献，可用于定位是哪一对放大了误差。

`at_m_summary.csv` 按检查点、面板和匹配策略汇总均值与标准差。

`at_m_correlations.csv` 输出 `R-H`、`R-A`、`A-U` 以及 `U-下一轮验证损失变化` 的 Spearman 相关性。常量序列无法定义相关系数，此时显示 `nan`，不应当按相关性为零解释。

汇总文件还直接给出相对随机匹配的 `R/H/A/U` 差值及 `A/U` 百分比变化；负值表示小于随机匹配均值。

`at_m_residual_angles.csv` 枚举任意两个客户端对执行残差之间的内积、余弦值和它们对全局误差交叉项的贡献。汇总文件中的关键字段为：

- `kappa_mean`：\(\kappa=U/A\)，即残差能量的全局保留比例；
- `residual_cosine_positive_fraction_mean`：有效余弦中大于零的比例；
- `residual_inner_product_nonnegative_fraction_mean`：交叉内积非负的比例；
- `residual_cross_term_mean`：\(U\) 展开式中的总交叉项；
- `alignment_assumption_hold_fraction`：所有交叉内积均非负的实验比例；
- `b_min_mean`：最小客户端对聚合权重。

脚本同时验证精确展开：

\[
U
=
\sum_k b_k^2\|r_k\|^2
+2\sum_{k<l}b_kb_l\langle r_k,r_l\rangle.
\]

只有当所有交叉内积均非负时，才能使用条件下界：

\[
b_{\min}A\le U\le A.
\]

### 5. 残差夹角正式诊断

该配置复用快速实验已经生成的第 5 轮检查点，使用 5 个 batch seed 和 20 个随机配对 seed：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_at_m_audit \
  --config configs/at_m/cifar10_residual_angles.yaml \
  --reuse-checkpoints \
  2>&1 | tee outputs/at_m_residual_angles.txt
```

完成后首先查看：

```bash
cat outputs/at_m_residual_angles/at_m_summary.csv
```

如果 `alignment_assumption_hold_fraction` 接近 1，且 `residual_inner_product_nonnegative_fraction_mean` 在不同 batch seed 和配对策略下稳定接近 1，才有依据继续讨论正的 \(\kappa_{\min}\)。如果这些比例明显低于 1，则无条件下界仍只能取 \(\kappa_{\min}=0\)。

## 先看哪些字段

按以下顺序检查：

1. `R_mean`：确认 optimal 是否按构造优于 random/similar；
2. `H_mean`：确认标签分布残差是否传递为梯度偏差；
3. `A_mean`：确认多步本地训练是否保留配对收益；
4. `U_mean`：确认执行误差是否进入实际全局更新；
5. `val_loss_change_mean` 与 `val_acc_change_mean`：确认更小的更新误差是否对应更好的单轮下降。

还要检查 `U_le_A_gap >= 0`。在完整参与、完整覆盖、统一本地步长的实验中，理论上应有：

\[
U_t(M)\le A_t(M).
\]

浮点误差可能造成极小的负数；明显为负则说明权重、客户端覆盖或指标实现存在错误。

## 结果解释

- `R、H、A、U` 都下降：分布—梯度—本地执行—全局更新链条得到较完整的机制支持；
- `R、H` 下降但 `A` 上升：中心陈旧、正则过强或多步漂移破坏了收益，应优化中心；
- `R` 下降但 `H` 不下降：纯标签偏斜假设不充分，需要把特征原型、实际梯度或更新方向加入关系图；
- `A` 下降但 `U` 不下降：不同客户端对误差向量的方向与抵消关系不可忽略；
- 四种匹配的 `A、U` 基本相同：当前正则太弱、检查点太早，或配对没有真正改变本地轨迹。

快速配置通过后，先把 `at_m_summary.csv` 和控制台末尾的 `summary:` 行发回来，再决定是否直接跑完整配置或先调整共同中心。
