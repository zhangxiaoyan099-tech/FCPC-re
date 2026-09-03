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
