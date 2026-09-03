# \(A_t(M)\) 的定义、跨配对实验与中心优化方案

## 1. 研究目标

本实验要回答的核心问题不是“加权 JSD 分数是否更大”，而是：

> 在相同模型状态、相同数据、相同随机性和相同训练协议下，配对策略 \(M\) 是否会减小本地多步训练产生的配对执行误差 \(A_t(M)\)，并进一步减小实际全局更新误差、改善收敛？

需要比较：

\[
A_t(M_{\mathrm{optimal}}),\qquad
A_t(M_{\mathrm{random}}),\qquad
A_t(M_{\mathrm{similar}}),\qquad
A_t(M_{\mathrm{JSDN}}).
\]

同时对每种匹配都计算：

\[
R(M),\qquad H_t(M),\qquad U_t(M).
\]

其中：

- \(M_{\mathrm{optimal}}\)：最大化加权 JS 互补收益的精确最大权匹配；
- \(M_{\mathrm{random}}\)：随机匹配，需要使用多个匹配随机种子；
- \(M_{\mathrm{similar}}\)：优先匹配分布最相似的客户端；
- \(M_{\mathrm{JSDN}}\)：原 FCPC 使用的 JSDN/LDP-JSDN 贪心差异配对。

这个实验的目标是检查“分布互补”是否真正传递到了“本地训练执行误差”和“全局更新误差”，而不是预先假定 \(A_t(M_{\mathrm{optimal}})\) 一定最小。

---

## 2. 配对层的定义

设客户端 \(i\) 的样本权重为：

\[
a_i=\frac{N_i}{N},
\qquad
a_{ij}=a_i+a_j,
\qquad
\theta_{ij}=\frac{a_i}{a_i+a_j}
=\frac{N_i}{N_i+N_j}.
\]

客户端标签分布分别为 \(p_i\) 和 \(p_j\)，配对后的混合标签分布为：

\[
q_{ij}=\theta_{ij}p_i+(1-\theta_{ij})p_j.
\]

全局标签分布为：

\[
\bar p=\sum_i a_i p_i.
\]

加权 JS 互补收益定义为：

\[
S_{ij}=a_{ij}JS_{\theta_{ij}}(p_i,p_j).
\]

对于完整匹配 \(M\)，存在恒等式：

\[
\sum_i a_iKL(p_i\|\bar p)
=
\sum_{(i,j)\in M}S_{ij}
+
\sum_{(i,j)\in M}a_{ij}KL(q_{ij}\|\bar p).
\]

左端与匹配方式无关，因此最大化总互补收益等价于最小化配对混合残差：

\[
R(M)
=
\sum_{(i,j)\in M}
a_{ij}KL(q_{ij}\|\bar p).
\]

所以，\(R(M)\) 检验的是“配对后标签分布是否更接近全局标签分布”。它是分布层指标，还不是实际训练误差。

---

## 3. 梯度层指标 \(H_t(M)\)

在标签偏斜分析中，若各客户端共享类别条件分布：

\[
P_i(x\mid y)=P(x\mid y),
\]

定义类别条件期望梯度：

\[
g_y(w)=
\mathbb E_{x\sim P(x\mid y)}
[\nabla_w\ell(w;x,y)].
\]

则配对目标和全局目标在同一点 \(w^t\) 上的梯度分别为：

\[
\nabla F_{ij}(w^t)
=\sum_yq_{ij}(y)g_y(w^t),
\qquad
\nabla F(w^t)
=\sum_y\bar p(y)g_y(w^t).
\]

定义匹配的配对梯度异质性：

\[
H_t(M)
=
\sum_{(i,j)\in M}
a_{ij}
\|\nabla F_{ij}(w^t)-\nabla F(w^t)\|^2.
\]

若 \(\|g_y(w)\|\le G\)，使用 Pinsker 不等式可得：

\[
H_t(M)\le 2G^2R(M)
\]

（上式按自然对数定义 KL；若使用以 2 为底的对数，需要调整常数。）

因此，\(R(M)\) 减小能够给出配对目标梯度偏差的上界，但它仍未直接说明多步本地训练后的实际模型更新一定更好。

---

## 4. 核心指标 \(A_t(M)\)

设第 \(t\) 轮开始时的全局模型为 \(w^t\)。客户端 \(i\) 完成本地训练后的模型为 \(w_i^{t,E}\)，客户端实际更新为：

\[
\Delta_i^t(M)=w_i^{t,E}(M)-w^t.
\]

之所以写成 \(\Delta_i^t(M)\)，是因为 FCPC 的配对中心和正则项会随匹配 \(M\) 改变，因而客户端最终更新也可能随匹配改变。

一对客户端的样本加权更新为：

\[
\Delta_{ij}^t(M)
=
\theta_{ij}\Delta_i^t(M)
+
(1-\theta_{ij})\Delta_j^t(M).
\]

在统一执行 \(E\) 个本地 SGD 步、学习率为 \(\eta_t\) 的受控实验中，记：

\[
\gamma_t=\eta_tE.
\]

若本地训练没有随机梯度噪声、参数漂移、动量和配对正则的影响，理想配对更新近似为：

\[
-\gamma_t\nabla F_{ij}(w^t).
\]

因此定义单对执行残差：

\[
r_{ij}^t(M)
=
\Delta_{ij}^t(M)
+
\gamma_t\nabla F_{ij}(w^t),
\]

并定义：

\[
\boxed{
A_t(M)
=
\sum_{(i,j)\in M}
a_{ij}\|r_{ij}^t(M)\|^2
}.
\]

它直接度量：实际的多步本地配对更新与“在轮初全局模型上执行理想配对梯度步”之间的差距。

为了便于跨轮比较，还应报告归一化版本：

\[
\widetilde A_t(M)
=
\frac{A_t(M)}
{\gamma_t^2\|\nabla F(w^t)\|^2+\varepsilon}.
\]

### 4.1 不同客户端本地步数不相同时

如果代码按 `local_epochs` 训练，而客户端样本量不同，那么各客户端实际 batch 数 \(E_i\) 不同，不能继续使用统一的 \(\gamma_t=\eta_tE\)。此时应定义：

\[
\gamma_i^t=\sum_{s=0}^{E_i-1}\eta_{t,s},
\]

并把理想配对更新改为：

\[
-\theta_{ij}\gamma_i^t\nabla F_i(w^t)
-(1-\theta_{ij})\gamma_j^t\nabla F_j(w^t).
\]

第一阶段的机制实验应强制所有客户端使用相同的本地步数，从而避免把样本量差异误当成配对策略差异。

---

## 5. 全局更新误差 \(U_t(M)\)

服务器实际聚合更新为：

\[
\Delta_{\mathrm{global}}^t(M)
=
\sum_{(i,j)\in M}a_{ij}\Delta_{ij}^t(M).
\]

定义相对于理想全局梯度步的误差：

\[
\boxed{
U_t(M)
=
\left\|
\Delta_{\mathrm{global}}^t(M)
+\gamma_t\nabla F(w^t)
\right\|^2
}.
\]

在完整参与、完整匹配和统一 \(\gamma_t\) 下：

\[
\sum_{(i,j)\in M}
a_{ij}\nabla F_{ij}(w^t)
=\nabla F(w^t).
\]

因此有更直接的关系：

\[
U_t(M)
=
\left\|
\sum_{(i,j)\in M}a_{ij}r_{ij}^t(M)
\right\|^2
\le
\sum_{(i,j)\in M}a_{ij}
\|r_{ij}^t(M)\|^2
=A_t(M).
\]

这个结论说明：在上述条件下，减小 \(A_t(M)\) 会收紧全局更新误差上界。

但也必须注意：

1. \(A_t(M)\) 是上界，较小的 \(A_t\) 不保证单次观测到的 \(U_t\) 严格更小，因为不同客户端对的误差向量可能相互抵消；
2. 对完整参与的一步梯度而言，所有完整匹配的加权配对梯度之和都等于全局梯度，因此仅改变分组本身不会改变理想平均梯度；
3. FCPC 的实际作用只能通过多步本地轨迹、配对中心、正则项、部分参与或其他非线性因素进入 \(r_{ij}^t(M)\)；
4. 所以不能只凭 \(R(M)\) 变小就宣称全局收敛加速，必须继续观察 \(A_t(M)\)、\(U_t(M)\) 和最终验证性能。

---

## 6. 为什么减小 \(U_t(M)\) 与收敛有关

把一轮实际更新写成：

\[
w^{t+1}
=w^t-\gamma_t\nabla F(w^t)+e_t,
\]

其中：

\[
e_t
=
\Delta_{\mathrm{global}}^t(M)
+\gamma_t\nabla F(w^t),
\qquad
\|e_t\|^2=U_t(M).
\]

在 \(F\) 为 \(L\)-smooth 且取 \(\gamma_t=1/L\) 的简化情形下：

\[
F(w^{t+1})
\le
F(w^t)
-\frac{1}{2L}\|\nabla F(w^t)\|^2
+\frac{L}{2}U_t(M).
\]

因此，更小的 \(U_t(M)\) 会减小递推式中的误差项，使下降界更紧，并可能减少达到目标精度所需的通信轮数。

严谨表述应为“改善有限轮收敛上界或降低误差底”，不能仅凭这一项宣称改善了渐近收敛阶。若要证明严格更快的收缩率，还需要得到类似：

\[
U_t(M)
\le
\kappa_M\|\nabla F(w^t)\|^2
\]

并证明新匹配对应的 \(\kappa_M\) 更小。

---

## 7. 第一阶段：冻结检查点的单轮反事实实验

### 7.1 为什么要冻结检查点

如果分别从头训练四条曲线，那么第 2 轮以后四个方法的模型状态、客户端历史模型和随机轨迹都会不同，无法判断观察到的差异究竟来自当前配对还是过去多轮累积差异。

因此先用一条不含配对正则的 FedAvg 轨迹生成公共检查点，例如：

\[
t\in\{5,10,20,50\}.
\]

每个检查点必须保存：

- 全局模型 \(w^t\)；
- 每个客户端的历史模型或构造中心所需状态；
- 优化器和学习率状态；
- 数据划分；
- 随机数状态或可复现随机种子。

不应只检查第 1 轮。初始化时所有客户端历史模型往往等于全局模型，各种配对产生的中心可能完全相同，此时配对无法体现真实作用。

### 7.2 公平控制变量

对每个冻结检查点，分别重放四种匹配的一轮训练。除 `pairing_strategy` 外，以下内容必须完全相同：

- 初始全局模型与客户端历史状态；
- 被选择的客户端集合；
- 每个客户端的 mini-batch 索引和顺序；
- 数据增强随机参数；
- 学习率、\(\beta_t\)、本地步数和优化器；
- LDP 噪声样本；
- 中心构造、裁剪和 proximal/penalty 规则；
- 聚合权重和验证集。

建议第一组机制审计使用：

```yaml
full_participation: true
num_clients: 10
local_steps: 2
batch_size: 64
augment: false
optimizer: sgd
learning_rate: 0.01
momentum: 0.0
weight_decay: 0.0
```

这样所有客户端都有共同的 \(E=2\)，可直接使用 \(\gamma_t=2\eta_t\)。机制确认后，再恢复论文主实验中的数据增强、动量和本地轮数。

### 7.3 两个公平性面板

为了避免把“是否使用隐私噪声”和“如何匹配”混在一起，应分成两个面板：

**面板 A：几何机制比较**

- 所有方法使用同一个 raw 标签分布矩阵；
- 比较 weighted-JS optimal、random、similar 和 greedy-dissimilar；
- 回答最大互补匹配本身是否有效。

**面板 B：隐私部署比较**

- 所有需要标签关系的方法都使用同一次 LDP 扰动后的分布；
- 比较 LDP weighted-JS optimal、LDP JSDN greedy、LDP similar 和 random；
- 回答加入 LDP 后结论是否仍成立。

不能让某个方法使用 raw 分布、另一个方法使用 LDP 分布后，把差异全部归因于匹配规则。

### 7.4 随机重复

随机匹配不能只采一次。建议：

- 公共模型种子：至少 \(42,43,44\)；
- 冻结检查点：\(5,10,20,50\)；
- mini-batch 重放种子：至少 5 个；
- 随机匹配种子：每个检查点至少 20 个。

先用种子 42 做代码和机制审计，再扩展到多种子。所有比较都使用配对差值，即在相同检查点和相同 batch seed 下作比较。

---

## 8. 指标的具体计算流程

在每个冻结检查点 \(w^t\) 上：

1. 对客户端 \(i\) 的固定探测数据计算轮初梯度：

   \[
   g_i^t=\nabla F_i(w^t).
   \]

2. 计算全局梯度：

   \[
   g^t=\sum_i a_i g_i^t.
   \]

3. 对匹配中的每一对计算：

   \[
   g_{ij}^t
   =\theta_{ij}g_i^t+(1-\theta_{ij})g_j^t.
   \]

4. 从同一检查点、同一 batch trace 分别重放 FCPC 本地训练，得到 \(\Delta_i^t(M)\)。

5. 计算：

   \[
   \Delta_{ij}^t(M)
   =\theta_{ij}\Delta_i^t(M)
   +(1-\theta_{ij})\Delta_j^t(M).
   \]

6. 计算单对执行残差和 \(A_t(M)\)：

   \[
   r_{ij}^t(M)=\Delta_{ij}^t(M)+\gamma_tg_{ij}^t,
   \]

   \[
   A_t(M)=\sum_{(i,j)\in M}a_{ij}\|r_{ij}^t(M)\|^2.
   \]

7. 计算 \(R(M)\)、\(H_t(M)\) 和真实聚合误差：

   \[
   U_t(M)=\|\Delta_{\mathrm{global}}^t(M)+\gamma_tg^t\|^2.
   \]

8. 用相同验证集测量重放一轮前后的变化：

   \[
   \Delta F_{\mathrm{val}}^t
   =F_{\mathrm{val}}(w^{t+1})-F_{\mathrm{val}}(w^t),
   \]

   以及验证准确率变化。

计算梯度时，应避免把训练 batch 的偶然噪声当作真实梯度。十个客户端时可以计算各客户端完整训练子集梯度；如果代价过高，则为每个客户端固定同一个较大的 probe subset，并在全部匹配方法之间复用。

---

## 9. 建议记录的 CSV 字段

每一行对应一个检查点、一个匹配方法和一个重放随机种子：

```text
model_seed
checkpoint_round
panel
pairing_strategy
pairing_seed
batch_seed
pair_list
R_M
H_t_M
A_t_M
A_t_normalized
U_t_M
U_t_normalized
val_loss_before
val_loss_after
val_acc_before
val_acc_after
mean_center_staleness
max_center_staleness
mean_pair_model_distance_before
mean_pair_model_distance_after
mean_task_loss
mean_pair_regularizer_loss
clipping_rate
round_time_s
communication_bytes
```

除了均值和标准差，还应报告相对随机匹配的成对差值：

\[
\Delta A_t
=A_t(M)-A_t(M_{\mathrm{random}}),
\qquad
\Delta U_t
=U_t(M)-U_t(M_{\mathrm{random}}).
\]

并计算以下 Spearman 相关性：

- \(R(M)\) 与 \(H_t(M)\)：检查分布到梯度的桥梁；
- \(R(M)\) 与 \(A_t(M)\)：检查分布互补是否改善多步执行；
- \(A_t(M)\) 与 \(U_t(M)\)：检查配对执行误差是否传递到全局更新；
- \(U_t(M)\) 与下一轮验证损失变化：检查更新误差是否影响实际下降。

---

## 10. 可能结果与解释

### 结果一：\(R\)、\(H\)、\(A\)、\(U\) 均降低

这是最理想的证据链：

\[
\text{加权 JS 匹配}
\Rightarrow R(M)\downarrow
\Rightarrow H_t(M)\downarrow
\Rightarrow A_t(M)\downarrow
\Rightarrow U_t(M)\downarrow.
\]

随后用多轮多种子实验检查是否表现为更快达到目标精度或更低误差底。

但实验相关性仍不是普适数学证明。论文中应将其称为机制证据，并把严格结论限制在已经证明的假设范围内。

### 结果二：\(R\)、\(H\) 降低，但 \(A\) 升高

说明分布层和轮初梯度层成立，但多步本地训练、陈旧中心或过强正则放大了漂移。此时主要优化中心和 proximal 执行方式，而不是继续修改 JSD 指标。

### 结果三：\(R\) 降低，但 \(H\) 不降低

说明标签分布不能充分解释实际梯度差异，常见原因是条件特征偏斜，即：

\[
P_i(x\mid y)\ne P_j(x\mid y).
\]

这时应把梯度、更新方向或类别特征原型加入关系图，而不能继续依靠纯标签 JSDN 声称控制梯度偏差。

### 结果四：\(A\) 降低，但 \(U\) 没有降低

说明不同客户端对的误差方向及其抵消关系很重要。此时不能只优化误差范数之和，还要检查：

\[
\left\|\sum_{(i,j)}a_{ij}r_{ij}^t\right\|^2
\]

以及不同 \(r_{ij}^t\) 之间的内积。

### 结果五：各匹配的 \(A\) 和 \(U\) 基本相同

说明当前中心或正则强度太弱，配对没有真正改变客户端轨迹；也可能是检查点过早，所有历史模型仍十分接近。此时应检查中心距离、正则梯度占比和配对后模型距离，不能直接得出“配对无效”。

---

## 11. 如果 \(A_t(M)\) 没有下降：中心消融实验

固定同一个 \(M_{\mathrm{optimal}}\)，只改变中心和更新方式，比较：

1. `beta = 0`：无配对正则；
2. 旧 partner 模型作为中心；
3. 样本加权共同中心 + penalty；
4. 样本加权共同中心 + proximal；
5. 全局锚定共同中心 + proximal；
6. 全局锚定、裁剪共同中心 + proximal。

历史配对中心可写成：

\[
c_{ij}^{t-1}
=
\theta_{ij}w_i^{t-1}
+(1-\theta_{ij})w_j^{t-1}.
\]

为了避免陈旧中心把客户端拉离当前全局模型，使用全局锚定中心：

\[
c_{ij,\lambda}^{t-1}
=
w^t
+\lambda_t(c_{ij}^{t-1}-w^t),
\qquad
\lambda_t\in\{0,0.25,0.5,0.75,1\}.
\]

也可以通过半径 \(\rho_t\) 自适应裁剪：

\[
\lambda_t
=
\min\left\{
1,
\frac{\rho_t}{\|c_{ij}^{t-1}-w^t\|+\varepsilon}
\right\}.
\]

新的安全中心为：

\[
\widehat c_{ij}^{t-1}
=w^t+\lambda_t(c_{ij}^{t-1}-w^t).
\]

对局部任务梯度完成一个临时步：

\[
\widetilde w_i^{s+1}
=w_i^s-\eta_t\nabla\ell_i(w_i^s;\xi_i^s),
\]

再执行共同中心的 proximal 更新：

\[
w_i^{s+1}
=
\frac{
\widetilde w_i^{s+1}
+2\eta_t\beta_t\widehat c_{ij}^{t-1}
}{1+2\eta_t\beta_t}.
\]

这一更新能够控制单步拉回强度，并在同一中心下收缩配对客户端的临时模型差异；但它仍不自动保证 \(A_t(M)\)、\(U_t(M)\) 或最终准确率下降，所以必须通过上述中心消融验证。

中心消融的选择标准应按以下优先级：

1. 验证集性能；
2. \(U_t(M)\)；
3. \(A_t(M)\)；
4. 中心陈旧距离和裁剪率；
5. 训练时间与通信量。

不能使用测试集准确率选择 \(\lambda_t\)、\(\rho_t\) 或 \(\beta_t\)。

---

## 12. 是否要在一轮内更新共同中心

第一阶段不建议一轮内不断重算中心，因为这会带来：

- 客户端执行顺序不对称；
- 同轮信息泄漏或额外通信；
- 不同配对方法的通信预算不公平；
- 理论中的“冻结中心”假设失效。

应先使用每轮开始时冻结的共同中心。若冻结中心仍造成较大的 \(A_t(M)\)，再单独比较：

- 每个本地 epoch 同步一次中心；
- 中心偏离阈值触发更新；
- 跨轮指数移动平均中心。

每种动态中心方案都必须单独记录额外通信量。

---

## 13. 第二阶段：多轮端到端公平消融

冻结检查点实验用于识别机制，最终仍需在相同初始化下完整训练：

- FedAvg；
- FCPC + \(M_{\mathrm{random}}\)；
- FCPC + \(M_{\mathrm{similar}}\)；
- FCPC + \(M_{\mathrm{JSDN}}\)；
- FCPC + \(M_{\mathrm{optimal}}\)。

至少使用种子 \(42,43,44\)，保持数据划分、模型初始化、客户端采样序列、batch 顺序、学习率和 \(\beta_t\) 调度一致。报告：

- 每轮 \(R(M)\)、\(H_t(M)\)、\(A_t(M)\)、\(U_t(M)\)；
- 最佳验证准确率及其轮次；
- 对应最佳验证模型的测试准确率；
- 最后若干轮验证均值；
- 达到固定准确率阈值所需轮数；
- 训练时间、GPU 内存和通信量；
- 均值、标准差及成对差值。

---

## 14. 最终可以支持和不能支持的结论

如果实验结果稳定支持 \(M_{\mathrm{optimal}}\) 减小 \(R\)、\(A\) 和 \(U\)，可以表述为：

> 加权 JS 最大权匹配在受控实验中不仅最小化了配对混合分布残差，而且降低了多步本地训练相对于理想配对梯度步的执行误差，并收紧了全局更新误差上界；多轮实验进一步显示其改善了有限轮收敛表现。

当前不能仅凭 \(R(M)\) 的恒等式直接声称：

- 对任意非凸模型都严格加快收敛；
- \(A_t(M_{\mathrm{optimal}})\) 必然最小；
- 最终测试准确率必然高于 FedAvg；
- 标签 JSDN 在存在条件特征偏斜时仍能完全控制梯度偏差。

本实验的关键价值，就是判断缺失的链条究竟断在 \(R\to H\)、\(H\to A\)，还是 \(A\to U\)，从而决定下一步应优化匹配指标、共同中心，还是本地更新规则。

---

## 15. CIFAR-10 第 5 轮快速审计结果

实验条件为种子 42、第 5 轮中立 FedAvg 检查点、完整 10 客户端参与、两个本地 SGD 步、相同 batch trace。当前快速实验只有 1 个 batch seed 和 3 个随机匹配 seed，因此属于机制诊断，不能作为最终统计结论。

raw 面板以随机匹配均值为基准：

| 配对策略 | \(R\) 变化 | \(H\) 变化 | \(A\) 变化 | \(U\) 变化 |
|---|---:|---:|---:|---:|
| optimal | \(-19.5\%\) | \(-27.6\%\) | \(-22.5\%\) | \(+0.033\%\) |
| similar | \(+5.7\%\) | \(+1.9\%\) | \(-6.5\%\) | \(-0.013\%\) |
| JSDN | \(-0.5\%\) | \(+29.4\%\) | \(+26.9\%\) | \(+0.076\%\) |

其中：

\[
A_{\mathrm{random}}=0.00457313,
\qquad
A_{\mathrm{optimal}}=0.00354263,
\]

但：

\[
U_{\mathrm{random}}=0.00118252,
\qquad
U_{\mathrm{optimal}}=0.00118290.
\]

因此，最大权加权-JS配对在该检查点明显降低了 \(R、H、A\)，但没有降低实际全局更新误差 \(U\)。四种策略的单轮验证准确率变化也完全相同：

\[
\Delta\operatorname{Acc}_{\mathrm{val}}=1.40625\%.
\]

该结果不否定 \(U\le A\)，而是说明这个上界在不同配对下具有不同松紧程度。

---

## 16. \(A\)、\(U\) 与残差抵消的精确关系

把所有客户端对及未配对单例统一记为组 \(k\)，组权重为 \(b_k\)，执行残差为：

\[
r_k=\Delta_k+\gamma g_k,
\qquad
\sum_kb_k=1.
\]

则：

\[
A=\sum_kb_k\|r_k\|^2,
\qquad
U=\left\|\sum_kb_kr_k\right\|^2.
\]

由加权方差恒等式：

\[
\boxed{
A=U+V
}
\]

其中：

\[
V=\sum_kb_k\left\|r_k-\sum_lb_lr_l\right\|^2\ge0.
\]

因此无条件成立：

\[
\boxed{0\le U\le A}.
\]

但不存在仅由正系数 \(A\) 和 \(R\) 构造的普适非零下界。原因是不同 \(r_k\) 可以完全反向抵消，使 \(A>0\) 而 \(U=0\)；同时也可以在 \(R>0\) 时完美执行配对梯度，使 \(A=U=0\)。所以 \(R\) 不能提供 \(U\) 的正下界。

为避免反复使用减法形式，定义残差保留系数：

\[
\boxed{
\kappa_t(M)=\frac{U_t(M)}{A_t(M)+\varepsilon}
}
\]

则：

\[
0\le\kappa_t(M)\le1,
\qquad
U_t(M)\approx\kappa_t(M)A_t(M).
\]

本次 raw 快速实验中：

\[
\kappa_{\mathrm{random}}\approx0.2586,
\qquad
\kappa_{\mathrm{optimal}}\approx0.3339.
\]

optimal 将 \(A\) 降低约 \(22.5\%\)，却使 \(\kappa\) 上升约 \(29\%\)，两种作用基本抵消，因此 \(U=\kappa A\) 几乎不变。

### 16.1 正下界成立的附加条件

展开 \(U\)：

\[
U
=
\sum_kb_k^2\|r_k\|^2
+2\sum_{k<l}b_kb_l\langle r_k,r_l\rangle.
\]

如果所有不同组的残差内积均非负：

\[
\langle r_k,r_l\rangle\ge0,
\qquad k\ne l,
\]

并记：

\[
b_{\min}=\min_kb_k,
\]

则：

\[
U
\ge
\sum_kb_k^2\|r_k\|^2
\ge
b_{\min}\sum_kb_k\|r_k\|^2
=b_{\min}A.
\]

从而得到条件夹击：

\[
\boxed{
b_{\min}A_t(M)
\le
U_t(M)
\le
A_t(M)
}.
\]

如果共有 \(m\) 个等权客户端对，则 \(b_{\min}=1/m\)。10 个客户端组成 5 个等权对时，条件下界为：

\[
0.2A_t(M)\le U_t(M)\le A_t(M).
\]

但是，“所有残差内积非负”尚未由当前算法保证，必须先通过残差夹角实验检查。没有这个方向性条件时，普适常数只能取 \(\kappa_{\min}=0\)。

---

## 17. 客户端对残差夹角实验

### 17.1 实验目的

检验下列条件在 CIFAR-10 上是否大部分或全部成立：

\[
\langle r_k,r_l\rangle\ge0.
\]

需要区分：

- “大部分内积为正”：只能作为经验现象；
- “每一次运行的所有内积都非负”：才能支持条件下界；
- 只要存在负内积，\(b_{\min}A\le U\) 在该次运行中就不能仅靠上述推导保证。

### 17.2 公平设置

- 复用快速实验生成的种子 42、第 5 轮 FedAvg 检查点；
- 仅使用 raw 面板，先排除 LDP 噪声干扰；
- 配对策略：optimal、random、similar、JSDN；
- batch seed：100、101、102、103、104；
- random pairing seed：0 至 19；
- 所有方法使用相同客户端历史、数据顺序、学习率、\(\beta\)、中心和 proximal 规则；
- 继续使用两个本地 SGD 步，以保持 \(\gamma=\eta E\) 的定义准确。

### 17.3 记录指标

对每两个不同客户端组 \(k,l\) 记录：

\[
\langle r_k,r_l\rangle,
\qquad
\cos(r_k,r_l)
=
\frac{\langle r_k,r_l\rangle}
{\|r_k\|\|r_l\|},
\]

以及它们对 \(U\) 的交叉贡献：

\[
2b_kb_l\langle r_k,r_l\rangle.
\]

每次重放汇总：

- 余弦均值、中位数和最小值；
- 正余弦比例；
- 非负内积比例；
- 总交叉项；
- \(\kappa=U/A\)；
- \(b_{\min}\)；
- 所有内积是否均非负；
- 直接核对 \(U\) 与残差展开重建值是否一致。

### 17.4 服务器命令

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_at_m_audit \
  --config configs/at_m/cifar10_residual_angles.yaml \
  --reuse-checkpoints \
  2>&1 | tee outputs/at_m_residual_angles.txt
```

主要输出：

```text
outputs/at_m_residual_angles/at_m_summary.csv
outputs/at_m_residual_angles/at_m_residual_angles.csv
outputs/at_m_residual_angles/at_m_metrics.csv
```

### 17.5 判断规则

重点检查：

- `residual_inner_product_nonnegative_fraction_mean`；
- `alignment_assumption_hold_fraction`；
- `residual_cosine_min`；
- `residual_cross_term_mean`；
- `kappa_mean`。

若 `alignment_assumption_hold_fraction = 1` 在多个检查点、模型种子和 batch seed 下稳定成立，才考虑在附加假设下使用 \(\kappa_{\min}=b_{\min}\)。如果只表现为“多数为正”但仍存在负内积，则只能报告经验对齐现象，不能把它写成无条件下界。

---

## 18. 2026-09-03 侧边栏讨论与实验结论汇总

本节集中记录本次讨论中完成的公式解释、理论修正和 CIFAR-10 残差夹角实验结果。最终目标始终是回答：配对策略是否不仅改变分布层指标，而且能够减小服务器真实更新相对于理想全局梯度步的误差，从而改善全局收敛递推界。

### 18.1 加权 JS 恒等式的逐项推导

对客户端对 \((i,j)\)，记：

\[
a_{ij}=a_i+a_j,
\qquad
\theta_{ij}=\frac{a_i}{a_i+a_j},
\qquad
q_{ij}=\theta_{ij}p_i+(1-\theta_{ij})p_j.
\]

在两个 KL 散度中插入混合分布 \(q_{ij}\)：

\[
\log\frac{p_i(y)}{\bar p(y)}
=
\log\frac{p_i(y)}{q_{ij}(y)}
+
\log\frac{q_{ij}(y)}{\bar p(y)},
\]

客户端 \(j\) 同理。因此：

\[
\begin{aligned}
&a_iKL(p_i\|\bar p)+a_jKL(p_j\|\bar p)\\
={}&a_iKL(p_i\|q_{ij})+a_jKL(p_j\|q_{ij})\\
&+\sum_y\big[a_ip_i(y)+a_jp_j(y)\big]
\log\frac{q_{ij}(y)}{\bar p(y)}.
\end{aligned}
\]

由混合分布的定义：

\[
a_ip_i(y)+a_jp_j(y)=a_{ij}q_{ij}(y).
\]

前两项为：

\[
a_{ij}
\left[
\theta_{ij}KL(p_i\|q_{ij})
+(1-\theta_{ij})KL(p_j\|q_{ij})
\right]
=a_{ij}JS_{\theta_{ij}}(p_i,p_j),
\]

后一项为：

\[
a_{ij}KL(q_{ij}\|\bar p).
\]

故：

\[
\boxed{
a_iKL(p_i\|\bar p)+a_jKL(p_j\|\bar p)
=a_{ij}JS_{\theta_{ij}}(p_i,p_j)
+a_{ij}KL(q_{ij}\|\bar p)
}.
\]

对完整匹配求和后，左侧与匹配方式无关，所以最大化加权 JS 互补收益等价于最小化混合分布残差 \(R(M)\)。这只证明了分布层的等价关系，尚未直接证明训练或收敛加速。

### 18.2 为什么引入类别条件期望梯度

本地损失本来就是直接对参数 \(w\) 求导。引入：

\[
g_y(w)=\mathbb E_{x\sim P(x\mid y)}
[\nabla_w\ell(w;x,y)]
\]

不是替代求导，而是在求导后按标签 \(y\) 使用全期望公式分组。在纯标签偏斜假设：

\[
P_i(x\mid y)=P(x\mid y)
\]

下，各客户端共享同一个 \(g_y(w)\)，差异只来自标签权重：

\[
\nabla L_i(w)=\sum_y p_i(y)g_y(w).
\]

因此配对梯度与全局梯度之差可以精确写成：

\[
\boxed{
\nabla F_{ij}(w)-\nabla F(w)
=\sum_y\big(q_{ij}(y)-\bar p(y)\big)g_y(w)
}.
\]

这个表示把“标签分布差异”连接到“梯度差异”。如果客户端之间还存在条件特征偏斜，即 \(P_i(x\mid y)\) 不同，就必须改写成客户端相关的 \(g_{i,y}(w)\)，上述简单桥梁不再充分，需要增加条件分布偏差项。

### 18.3 梯度差异的范数放缩

由向量和的三角不等式与范数齐次性：

\[
\begin{aligned}
\|\nabla F_{ij}(w)-\nabla F(w)\|
&=\left\|\sum_y(q_{ij}(y)-\bar p(y))g_y(w)\right\|\\
&\le\sum_y|q_{ij}(y)-\bar p(y)|\,\|g_y(w)\|.
\end{aligned}
\]

若对所有类别都有 \(\|g_y(w)\|\le G\)，则：

\[
\|\nabla F_{ij}(w)-\nabla F(w)\|
\le G\sum_y|q_{ij}(y)-\bar p(y)|
=G\|q_{ij}-\bar p\|_1.
\]

这里 \(q_{ij}(y)\) 是标签 \(y\) 对应的一个标量概率；\(q_{ij}\) 是由所有类别概率组成的完整向量。\(\|q_{ij}-\bar p\|_1\) 就是先对每个类别求绝对差，再对所有类别求和。

### 18.4 一对客户端的聚合更新及研究目标

客户端实际更新参数，但分析时把更新量记为：

\[
\Delta_i^t=w_i^{t,E}-w^t.
\]

一对客户端在服务器侧的样本加权更新为：

\[
\Delta_{ij}^t
=\theta_{ij}\Delta_i^t+(1-\theta_{ij})\Delta_j^t.
\]

它不是额外执行一次训练，而是先把两个客户端的模型变化按样本权重合并成一个“客户端对更新”，便于与该对的混合目标 \(F_{ij}\) 和混合分布 \(q_{ij}\) 对应。

以轮初模型 \(w^t\) 为共同起点，理想的配对梯度步为：

\[
-\gamma_t\nabla F_{ij}(w^t),
\qquad \gamma_t=\eta_tE.
\]

于是配对执行残差定义为：

\[
r_{ij}^t
=\Delta_{ij}^t+\gamma_t\nabla F_{ij}(w^t).
\]

研究 \(r_{ij}^t\)、\(A_t(M)\) 和 \(U_t(M)\) 的目的，不是只描述误差，而是确定 FCPC 是否能减小全局递推式中的误差项：

\[
F(w^{t+1})
\le F(w^t)
-\frac{1}{2L}\|\nabla F(w^t)\|^2
+\frac{L}{2}U_t(M).
\]

在其余条件相同时，更小的 \(U_t(M)\) 会收紧有限轮下降界、降低误差底；但要声称更快的渐近收缩率，还需要更强的相对误差界，不能只依据一次实验或单独一个上界。

### 18.5 四类误差及其关系

从分布配对到服务器更新，至少要区分四类来源：

1. 配对梯度偏差：\(\nabla F_{ij}(w^t)-\nabla F(w^t)\)，由 \(H_t(M)\) 衡量，\(R(M)\) 只给它提供分布代理和上界；
2. 本地多步漂移：后续 SGD 步是在移动后的本地参数上计算，而不是一直在 \(w^t\) 上；
3. 随机梯度噪声：mini-batch 梯度与期望梯度不同；
4. 共同中心与历史滞后误差：FCPC 中心、伙伴模型和本地轨迹可能偏离当轮理想目标。

其中 \(A_t(M)\) 汇总“实际配对更新相对理想配对梯度步”的执行残差，而 \(U_t(M)\) 才是服务器最终收到的整体更新相对理想全局梯度步的真实误差。减少 \(R\) 或 \(H\) 只解决第一段桥梁，不自动解决后三类误差。

### 18.6 (A\)、(U\) 与残差方向的精确关系

把每个客户端对编号为 \(k\)，其聚合权重为 \(b_k\)，残差为 \(r_k\)，且 \(\sum_kb_k=1\)。定义：

\[
A=\sum_kb_k\|r_k\|^2,
\qquad
U=\left\|\sum_kb_kr_k\right\|^2.
\]

加权方差恒等式给出：

\[
A=U+
\sum_kb_k
\left\|r_k-\sum_lb_lr_l\right\|^2,
\]

所以无条件只有：

\[
\boxed{0\le U\le A}.
\]

不同残差可以完全反向抵消，因此不存在只依赖 \(A\) 和 \(R\) 的普适正常数 \(\kappa_{\min}>0\)，使 \(U\ge\kappa_{\min}A\)。无附加条件时只能取 \(\kappa_{\min}=0\)。

展开 \(U\)：

\[
U=\sum_kb_k^2\|r_k\|^2
+2\sum_{k<l}b_kb_l\langle r_k,r_l\rangle.
\]

只有在所有交叉内积均非负时，才能推出条件下界：

\[
\boxed{b_{\min}A\le U\le A},
\qquad b_{\min}=\min_kb_k.
\]

“大部分内积为正”并不足够；少数大权重、大幅度的负内积仍可能让总交叉项为负。

### 18.7 残差夹角实验实际结果

实验使用 CIFAR-10、种子 42、第 5 轮冻结检查点、raw 客户端元数据、5 个 batch seed；随机配对额外使用 20 个 pairing seed，因此 random 共 100 次重放，其他策略各 5 次。

| 配对策略 | 正/非负内积比例 | 平均余弦 | 最小余弦 | 平均交叉项 | 条件成立比例 |
|---|---:|---:|---:|---:|---:|
| JSDN | 60.0% | 0.0053 | -0.599 | \(-1.117\times10^{-4}\) | 0% |
| Optimal | 34.0% | -0.0329 | -0.589 | \(-9.955\times10^{-5}\) | 0% |
| Random | 41.1% | -0.0528 | -0.650 | \(-4.132\times10^{-5}\) | 0% |
| Similar | 32.0% | -0.0846 | -0.479 | \(-1.614\times10^{-4}\) | 0% |

JSDN 是唯一正内积比例超过一半的策略，说明它在该检查点上提高了残差方向的一致性；但平均余弦几乎为零，仍存在约 \(-0.6\) 的明显负余弦，而且总交叉项为负。四种策略的 `alignment_assumption_hold_fraction` 全部为 0，因此实验不支持把 \(b_{\min}\) 当成普适或稳定的 \(\kappa_{\min}\)。

### 18.8 (A\) 显著变化而 (U\) 几乎不变

与随机配对相比：

| 配对策略 | \(A\) 相对变化 | \(U\) 相对变化 | 单轮验证准确率变化 |
|---|---:|---:|---:|
| Optimal | \(-26.81\%\) | \(+0.012\%\) | 约 \(+1.406\%\) |
| Similar | \(-11.55\%\) | \(-0.025\%\) | 约 \(+1.422\%\) |
| JSDN | \(+17.31\%\) | \(+0.051\%\) | 约 \(+1.406\%\) |
| Random | 基准 | 基准 | 约 \(+1.415\%\) |

对应的主要数值为：

\[
A_{\mathrm{random}}=0.00494872,
\qquad
A_{\mathrm{optimal}}=0.00362205,
\]

而：

\[
U_{\mathrm{random}}=0.00120514,
\qquad
U_{\mathrm{optimal}}=0.00120529.
\]

这说明 optimal 虽然显著收紧了上界 \(A\)，但没有让服务器的真实更新误差 \(U\) 同步下降。

### 18.9 为什么配对可以改变 (A)，却几乎不改变 (U)

若客户端更新 \(\Delta_i\) 本身不依赖伙伴，对任意完整匹配都有：

\[
\begin{aligned}
\sum_{(i,j)\in M}a_{ij}\Delta_{ij}
&=\sum_{(i,j)\in M}(a_i+a_j)
\left(
\frac{a_i}{a_i+a_j}\Delta_i
+\frac{a_j}{a_i+a_j}\Delta_j
\right)\\
&=\sum_{(i,j)\in M}(a_i\Delta_i+a_j\Delta_j)\\
&=\sum_i a_i\Delta_i.
\end{aligned}
\]

因此单纯重新分组能够改变每组残差及其平方范数之和 \(A_t(M)\)，但不会改变服务器加权总更新，也就不会改变 \(U_t(M)\)。FCPC 只有通过伙伴相关的共同中心、正则项和多步本地轨迹，使 \(\Delta_i^t(M)\) 真正依赖匹配 \(M\)，才可能改变全局更新。当前 \(U\) 的差异只有万分之一量级，表明该实验设置下这种伙伴作用很弱。

### 18.10 当前能够写入论文和不能写入论文的结论

当前可以报告：

- 加权 JS 恒等式严格成立，最大化互补收益等价于最小化分布层残差 \(R(M)\)；
- 在纯标签偏斜假设下，\(R(M)\) 可以通过条件期望梯度连接到配对梯度异质性 \(H_t(M)\)；
- 无条件恒有 \(0\le U_t(M)\le A_t(M)\)；
- 在种子 42、第 5 轮检查点上，JSDN 提高了正残差内积的比例，但没有保证所有残差同向；
- optimal 显著减小 \(R、H、A\)，但未减小 \(U\)，说明上界收紧不等于真实更新改善。

当前不能声称：

- 最大化 JSD/JSDN 一定加快全局收敛；
- \(A_t(M)\) 下降必然导致单次观测的 \(U_t(M)\) 下降；
- 存在不加假设的正常数 \(\kappa_{\min}\)，使 \(U\ge\kappa_{\min}A\)；
- 残差“大部分同向”足以证明 \(b_{\min}A\le U\)。

### 18.11 下一步指标与算法检查方向

下一步应直接度量配对机制到底改变了多少服务器更新。建议定义伙伴作用量：

\[
\boxed{
D_t(M)=
\left\|
\sum_i a_i
\big[
\Delta_i^t(M)-\Delta_i^t(\mathrm{FedAvg})
\big]
\right\|
}.
\]

需要在相同冻结检查点和随机批次下比较不同配对的 \(D_t(M)\)、\(U_t(M)\) 及更新与负全局梯度的余弦：

\[
\cos\left(
\Delta_{\mathrm{global}}^t(M),
-\nabla F(w^t)
\right).
\]

判定逻辑为：

1. 若 \(D_t(M)\) 很小，说明当前共同中心或 \(\beta\) 对客户端轨迹影响太弱，问题不主要在匹配算法；
2. 若 \(D_t(M)\) 明显但 \(U_t(M)\) 不降，说明中心确实改变了更新，却没有把更新推向理想全局梯度方向；
3. 若 \(D_t(M)\) 明显、\(U_t(M)\) 下降且方向余弦提高，才能形成“配对—伙伴正则—更新改善—递推项收紧”的完整证据链。
