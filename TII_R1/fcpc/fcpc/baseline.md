# FCPC 对比基线原理、关键疑问与理论启示

记录日期：2026-08-26  
适用任务：FCPC 重构、CIFAR-10 基线复现与论文理论修订

## 1. 文档目的

本文档整理 FCPC 论文涉及的六个主要对比基线：FedAvg、FedProx、MOON、FedDyn、FBLG 和 FedCFA，并集中回答以下问题：

1. MOON 中特征表示 $z$ 与模型参数 $w$ 有什么区别？
2. 特征余弦相似度与模型参数欧氏距离有什么区别？
3. 参数接近或特征接近是否意味着客户端梯度方向对齐？
4. MOON、FedDyn 和当前 FCPC 都使用历史状态，是否存在滞后性？
5. FBLG 选择相似客户端，而 FCPC 选择 JSDN 最大的客户端，FBLG 的证明能否用于 FCPC？
6. FCPC 应该怎样解释“选择差异最大的客户端”，并进一步建立收敛分析？

文档中的结论分为三类：

- **已有论文明确提出的机制**：可作为基线实现依据；
- **根据当前 FCPC 代码得到的事实**：可通过代码复核；
- **拟建立的理论方向**：目前只是证明路线，不能作为已经成立的定理写入论文。

## 2. 统一的联邦学习问题

设系统包含 $K$ 个客户端，客户端 $k$ 拥有数据集 $D_k$，样本数为 $n_k$，本地目标为

$$
F_k(w)=\frac{1}{n_k}\sum_{(x,y)\in D_k}\ell(w;x,y).
$$

服务器希望优化全局目标

$$
F(w)=\sum_{k=1}^{K}p_kF_k(w),
\qquad
p_k=\frac{n_k}{\sum_j n_j}.
$$

在 CIFAR-10 dual-skew 场景下，客户端同时存在：

- **标签偏斜**：不同客户端拥有不同的类别比例；
- **数量偏斜**：不同客户端的样本量差异明显。

例如，一个客户端可能有 3378 个样本，主要包含猫、青蛙和卡车；另一个客户端只有 668 个样本，主要包含汽车和狗。六种基线的主要区别，是它们在不同位置处理这种客户端异质性。

## 3. 六种基线的原理

### 3.1 FedAvg：本地训练后进行样本量加权平均

第 $t$ 轮中，服务器向客户端发送全局模型 $w^t$。每个客户端从相同的 $w^t$ 出发，在本地执行若干步 SGD，得到 $w_k^{t+1}$。服务器随后聚合：

$$
w^{t+1}
=
\sum_{k\in S_t}
\frac{n_k}{\sum_{j\in S_t}n_j}
w_k^{t+1}.
$$

FedAvg 不显式计算客户端关系，也不约束本地模型漂移。它的优点是简单、通信协议清晰；缺点是客户端在非 IID 数据上执行多步本地训练后，可能沿不同方向偏离全局目标。

在 FCPC 实验中，FedAvg 是最重要的基础参照。FCPC 必须在相同数据划分、模型、优化器、训练轮数和随机种子下证明新增配对机制相对 FedAvg 有效。

原始论文：[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

### 3.2 FedProx：在参数空间限制本地漂移

FedProx 在本地任务损失中加入近端项：

$$
\min_w
F_k(w)
+
\frac{\mu}{2}\|w-w^t\|_2^2.
$$

其中：

- $F_k(w)$ 是客户端分类损失；
- $w^t$ 是服务器本轮下发的全局模型；
- $\mu$ 控制本地模型不能离全局模型过远。

当 $\mu=0$ 时，FedProx 退化为 FedAvg。FedProx 的服务器聚合通常仍采用 FedAvg 的样本量加权平均。

FedProx 处理的是“本地模型偏离全局参数过远”，但它不判断客户端之间谁和谁更适合交互，也不直接利用标签分布差异。

原始论文：[Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)

### 3.3 MOON：在特征空间进行模型对比学习

MOON 将模型表示为

$$
f_w(x)=h_w(R_w(x)),
$$

其中 $R_w(x)$ 是编码器和投影头产生的特征表示，$h_w$ 是分类输出部分。

客户端 $i$ 在第 $t$ 轮对同一输入 $x$ 计算：

$$
z=R_{w_i^t}(x),
\qquad
z_{\mathrm{glob}}=R_{w^t}(x),
\qquad
z_{\mathrm{prev}}=R_{w_i^{t-1}}(x).
$$

MOON 使用模型对比损失：

$$
\ell_{\mathrm{con}}
=
-\log
\frac{
\exp(\operatorname{sim}(z,z_{\mathrm{glob}})/T)
}{
\exp(\operatorname{sim}(z,z_{\mathrm{glob}})/T)
+
\exp(\operatorname{sim}(z,z_{\mathrm{prev}})/T)
}.
$$

本地总损失为

$$
\mathcal L_i
=
\mathcal L_{\mathrm{cls}}
+
\mu\mathcal L_{\mathrm{con}}.
$$

它希望当前本地模型产生的特征靠近当前全局模型，并远离上一轮本地模型产生的特征。这里的 $T$ 是对比学习温度，不是 FCPC 中的样本权重 $\tau$。

MOON 需要保存上一轮本地模型，并为当前本地模型、全局模型和上一轮本地模型执行额外前向计算。它改变的是本地表示学习过程，服务器仍可使用 FedAvg 聚合。

原始论文：[Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257)

### 3.4 FedDyn：通过动态历史修正使局部与全局驻点一致

标准联邦优化文献中的 FedDyn 使用动态正则项修正客户端长期漂移。其客户端目标可以用如下代表性形式理解：

$$
F_k(w)
-
\langle h_k^t,w\rangle
+
\frac{\alpha}{2}\|w-w^t\|_2^2,
$$

其中 $h_k^t$ 是客户端跨轮次维护的历史修正状态。

FedProx 只限制“当前这一轮不要离全局模型太远”；FedDyn 还记录客户端过去持续向哪个方向偏移，并使用历史项修正这种长期偏差。服务器也需要维护相应状态。

FedDyn 的目标不是让每个客户端梯度都趋近于零，而是使最终一致模型满足全局驻点条件：

$$
\sum_k p_k\nabla F_k(w^*)=0.
$$

标准 FedDyn 原始论文：[Federated Learning Based on Dynamic Regularization](https://arxiv.org/abs/2111.04263)

#### FedDyn 引用歧义

FCPC 原稿参考文献中的 FedDyn 并不是上述标准动态正则化 FedDyn，而是推荐系统领域的联邦蒸馏方法：*FedDyn: A Dynamic and Efficient Federated Distillation Approach on Recommender System*。

文献记录：[DBLP 条目](https://dblp.org/rec/conf/icpads/JinCGL22.html)

由于 FCPC 实验是 CIFAR-10 图像分类，这里可能存在引用错误或方法命名错误。在没有核实原作者代码和实验记录之前，不能直接把标准 FedDyn 的复现结果当作原稿表格中 FedDyn 的严格复现。后续文档和表格必须区分：

- `FedDyn-DynamicReg`：标准动态正则化版本；
- `FedDyn-DistillRec`：原稿引用的推荐系统联邦蒸馏版本。

### 3.5 FBLG：构建客户端关系图并选择相似客户端

FBLG 主要修改服务器端的客户端选择过程。其流程为：

1. 客户端接收全局模型并进行本地训练；
2. 客户端上传本地模型和本地损失；
3. 服务器选择本地损失较大的候选客户端；
4. 服务器向客户端模型输入依据验证数据统计生成的高斯噪声；
5. 根据不同模型产生的网络嵌入计算 JS divergence；
6. 构建客户端关系图；
7. 选择损失较大、相似度较高且样本较多的客户端；
8. 对选中模型进行样本量加权聚合。

其图边权可概括为

$$
V_{ij}=\frac{1}{1+S_{ij}},
$$

其中 $S_{ij}$ 是客户端模型嵌入之间的 JS divergence。客户端越相似，$V_{ij}$ 越大。

FBLG 通过选择相似客户端降低聚合冲突，同时通过本地损失和样本量考虑训练不足客户端及数量偏斜。它属于“客户端选择和聚合”方法，而不是“配对正则训练”方法。

原始论文：[FBLG: A Local Graph Based Approach for Handling Dual Skewed Non-IID Data in Federated Learning](https://www.ijcai.org/proceedings/2024/585)

### 3.6 FedCFA：使用全局平均信息构造反事实特征

FedCFA 关注非 IID 联邦学习中的 Simpson's paradox，即客户端局部观察到的特征—标签关系可能与全局关系不同，甚至方向相反。

例如某客户端中的飞机大多具有蓝色背景，模型可能错误地把蓝色背景作为飞机类别依据。其他客户端的数据关系不同，聚合后这种局部伪相关会影响全局模型。

FedCFA 的主要流程为：

1. 客户端把本地数据划分为若干子集并计算局部平均样本；
2. 服务器聚合得到全局平均数据；
3. 客户端使用编码器提取特征因子；
4. 根据特征梯度判断可替换和不可替换因子；
5. 使用全局平均特征替换部分本地特征，构造反事实正、负样本；
6. 联合优化分类损失、反事实损失和特征解耦损失。

反事实特征可写为

$$
F_{\mathrm{pos}}
=
M_{\mathrm{pos}}\odot F
+
(1-M_{\mathrm{pos}})\odot\bar F_g,
$$

$$
F_{\mathrm{neg}}
=
M_{\mathrm{neg}}\odot F
+
(1-M_{\mathrm{neg}})\odot\bar F_g.
$$

FedCFA 修改了客户端数据表示、网络模块和本地损失，实现成本明显高于 FedAvg、FedProx 和 MOON。

原始论文：[FedCFA: Federated Counterfactual Learning with Aggregation](https://ojs.aaai.org/index.php/AAAI/article/view/33942)

## 4. 六种基线与 FCPC 的定位对比

| 方法 | 主要干预位置 | 核心机制 | 额外状态或信息 | 是否显式利用客户端关系 |
|---|---|---|---|---|
| FedAvg | 服务器聚合 | 本地训练后样本量加权平均 | 无 | 否 |
| FedProx | 本地参数损失 | 本地模型靠近当前全局模型 | 当前全局模型 | 否 |
| MOON | 本地特征损失 | 当前本地特征靠近全局、远离历史本地特征 | 上一轮本地模型、投影头 | 间接 |
| FedDyn | 本地动态目标和服务器状态 | 历史修正抵消长期客户端漂移 | 客户端和服务器历史状态 | 间接 |
| FBLG | 客户端选择和聚合 | 选择损失大、相似且样本多的客户端 | 模型嵌入、关系图、验证统计 | 是，选择相似客户端 |
| FedCFA | 本地表示和数据构造 | 全局平均信息指导反事实特征学习 | 全局平均数据、特征掩码 | 间接 |
| FCPC | 客户端配对和本地参数正则 | JSDN 最大差异配对并使用伙伴历史模型约束 | JSDN 矩阵、配对模型快照 | 是，选择差异客户端 |

这些方法代表不同技术路线：

$$
\text{直接平均}
\rightarrow
\text{参数约束}
\rightarrow
\text{特征约束}
\rightarrow
\text{历史漂移修正}
\rightarrow
\text{关系图选择}
\rightarrow
\text{反事实表示学习}.
$$

## 5. 问题一：MOON 中的特征 $z$ 和模型 $w$ 有什么区别？

模型参数 $w$ 是整个神经网络的全部可训练变量，包括卷积核、全连接权重和偏置。特征 $z$ 是模型对某个具体输入进行前向计算后得到的中间表示：

$$
w
\xrightarrow[\text{输入 }x]{\text{前向传播}}
z=R_w(x).
$$

例如，一个模型可能有 1100 万个参数：

$$
w\in\mathbb R^{11\,000\,000},
$$

而一张图片经过编码器后可能得到一个 256 维特征：

$$
z\in\mathbb R^{256}.
$$

因此：

- $w$ 描述“模型本身是什么”；
- $z$ 描述“模型如何表示某个输入”；
- 同一个 $w$ 面对不同输入会产生不同 $z$；
- 不同 $w$ 面对同一输入也可能产生相同或相似的 $z$。

MOON 所谓模型级对比学习，实际比较的是不同模型对同一输入产生的特征，而不是直接比较两个完整参数向量。

## 6. 问题二：$\operatorname{sim}(z_k,z_g)$ 与 $\|w_k-w^t\|$ 有什么区别？

### 6.1 参数距离

FedProx 或当前 FCPC 使用的参数距离类似：

$$
d_w=\|w_k-w^t\|_2^2.
$$

它比较所有对应参数的数值差异，与具体输入无关。

神经网络存在参数对称性。例如交换两个隐藏神经元并同步交换下一层连接，模型的参数向量可能变化很大，但函数输出保持不变。ReLU 网络还可能通过相邻层的放大和缩小保持相同函数。因此可能出现：

$$
\|w_1-w_2\|\text{ 很大},
\qquad
f_{w_1}(x)=f_{w_2}(x).
$$

所以参数距离不是严格的函数行为距离。

### 6.2 特征余弦相似度

MOON 使用

$$
\operatorname{sim}(z_k,z_g)
=
\frac{z_k^\top z_g}{\|z_k\|\,\|z_g\|}.
$$

它衡量同一输入经过两个模型后，特征方向是否相似，更接近“两个模型对该输入的理解是否一致”。但余弦相似度只看方向，不看幅值。例如 $z_k=100z_g$ 时，余弦相似度仍为 1。

三种常见度量回答不同问题：

| 度量 | 表示含义 |
|---|---|
| $\|w_k-w_g\|$ | 两个模型的参数数值是否接近 |
| $\operatorname{sim}(z_k,z_g)$ | 两个模型对同一输入产生的特征方向是否接近 |
| $\operatorname{sim}(g_k,g_g)$ | 两个模型的优化更新方向是否接近 |

特征相似比参数接近更接近模型行为，但仍不等于梯度对齐。

## 7. 问题三：参数或特征接近是否意味着梯度方向对齐？

不意味着。

对样本 $(x,y)$，梯度可以写成

$$
\nabla_w\ell(w;x,y)
=
J_w(x)^\top\nabla_z\ell(z,y),
$$

其中 $J_w(x)$ 是特征对模型参数的雅可比矩阵。梯度同时取决于：

- 特征 $z$；
- 标签 $y$；
- 当前预测误差；
- 分类器参数；
- 特征对参数的雅可比矩阵；
- 客户端数据分布。

即使两个模型产生相同特征，如果标签或雅可比矩阵不同，梯度仍可能不同。

更直接的反例是：每轮开始时所有客户端参数相同，满足

$$
w_1=w_2=w^t,
$$

但由于客户端数据分布不同，通常仍有

$$
\nabla F_1(w^t)\neq\nabla F_2(w^t).
$$

在 $L$-smooth 条件下，对同一个目标函数只能得到

$$
\|\nabla F(w)-\nabla F(w')\|
\leq
L\|w-w'\|.
$$

对两个不同客户端应分解为

$$
\begin{aligned}
\|\nabla F_i(w_i)-\nabla F_j(w_j)\|
\leq{}&
L\|w_i-w_j\|\\
&+
\|\nabla F_i(w_j)-\nabla F_j(w_j)\|.
\end{aligned}
$$

第一项来自参数位置不同，第二项来自客户端目标函数和数据分布不同。参数正则最多直接控制第一项，不能自动消除第二项。

这也是 FCPC 原理论的重要不足：原附录试图从 JSDN 或参数差异直接跳到梯度关系和收敛改善，但缺少中间机制。

## 8. 问题四：使用历史状态是否存在滞后性？

答案是存在，但“使用历史信息”不等于“算法一定错误”。关键在于历史量的作用以及是否能控制陈旧误差。

### 8.1 MOON 的历史模型

MOON 把上一轮本地模型产生的特征 $z_{\mathrm{prev}}$ 当作负样本。它不是把旧模型当成当前正确答案，而是把它作为客户端过去局部偏斜轨迹的参照物。

可能的问题包括：

1. 客户端多轮未参与时，历史模型可能很旧；
2. 如果上一轮模型本来较好，强制远离它可能有害；
3. 如果历史模型已经很远，对比任务过于容易，额外梯度可能很弱；
4. 对比损失权重过大时，可能干扰前期分类学习。

MOON 主要依靠实验验证该设计，并未给出与 FedDyn 同等级的完整收敛理论。

### 8.2 FedDyn 的历史状态

FedDyn 的历史状态是递推更新的优化修正量，类似控制变量或累计误差。它试图估计并抵消客户端长期偏移，而不是简单要求当前模型靠近或远离某个旧模型。

未参与客户端的状态确实会保持不变，因此同样存在陈旧性。但标准 FedDyn 的分析在随机部分参与等假设下，把这种状态递推包含在收敛证明中。在收敛点附近，模型变化逐渐减小，历史状态带来的额外变化也随之减小。

### 8.3 当前 FCPC 的历史伙伴模型

当前代码中的 FCPC 本地目标为

$$
\mathcal L_i^t
=
F_i(w_i)
+
\beta_t\tau_{i\leftarrow j}
\|w_i-w_j^{t-1}\|_2^2.
$$

其中 $w_j^{t-1}$ 是配对客户端上一轮冻结的模型快照。实现见：

- [`src/fcpc/regularizer.py`](src/fcpc/regularizer.py)
- [`src/federated/trainer.py`](src/federated/trainer.py)

冻结上一轮快照可以避免顺序模拟造成不对称：第二个训练的客户端不能读取第一个客户端刚刚更新完成的当前轮模型。但它也引入了一轮时间滞后。

使用历史伙伴和理想当前伙伴时，两种正则梯度的差为

$$
\begin{aligned}
&\|2\beta_t\tau_{ij}(w_i-w_j^{t-1})
-2\beta_t\tau_{ij}(w_i-w_j^t)\|\\
&=2\beta_t\tau_{ij}\|w_j^t-w_j^{t-1}\|.
\end{aligned}
$$

若每轮模型移动有界：

$$
\|w_j^t-w_j^{t-1}\|\leq\eta EG,
$$

则滞后误差满足

$$
\text{staleness error}
\leq
2\beta_t\tau_{ij}\eta EG.
$$

这可以作为余弦衰减 $\beta_t$ 的一个理论依据：随着 $\beta_t\to0$，历史伙伴造成的渐近偏差也趋于零。但这只能帮助证明 FCPC 后期不会一直被旧模型拖住，不能单独证明 FCPC 比 FedAvg 收敛更快。

## 9. 问题五：FBLG 选择相似客户端，FCPC 为什么选择最大差异？

两者处理客户端冲突的方式相反：

- FBLG 在聚合前避免冲突，选择更相似的客户端；
- FCPC 主动找到差异较大的客户端，希望通过配对正则交换互补信息。

FBLG 的理论分析使用了很强的条件：被选中的相似客户端可以近似认为来自相同分布。其结论更接近“同分布客户端聚合后，更容易接近该分布对应的最优模型”。

因此，FBLG 的核心结论不能直接反过来写成：

$$
\text{差异越大}
\Longrightarrow
\text{FCPC 收敛越快}.
$$

可以借鉴的只有外层证明方法：

1. 使用 $L$-smooth 下降引理；
2. 展开一轮全局目标变化；
3. 把误差分解为随机梯度方差、客户端异质性、选择误差和额外正则误差；
4. 证明新增机制减小了其中某个明确项。

FCPC 还需要单独证明：差异配对经过正则化后，究竟降低了聚合更新的偏差、方差还是客户端漂移。

## 10. 最大差异配对更合适的理论解释

令全局梯度为 $g$，客户端梯度偏差为

$$
\delta_i=g_i-g,
\qquad
\delta_j=g_j-g.
$$

对一对客户端设置归一化权重 $a+b=1$，存在方差分解：

$$
a\|\delta_i\|^2+b\|\delta_j\|^2
=
\|a\delta_i+b\delta_j\|^2
+
ab\|\delta_i-\delta_j\|^2.
$$

其中：

- 左侧表示两个客户端独立偏移的平均强度；
- 第一项表示把两个更新组合后剩余的偏移；
- 第二项表示组合过程中潜在可以消除的组内差异。

当 $\|\delta_i-\delta_j\|$ 较大时，可消除的组内差异具有更大上限。因此最大差异配对更准确的表述应当是：

> 最大差异配对试图寻找“潜在可消除漂移”最大的客户端组合；FCPC 正则负责把这部分差异转化为互补更新。

这里必须强调“潜在”二字。差异大本身不是收益，只有 FCPC 正则真正实现知识传递、漂移抵消或者全局覆盖改善时，它才可能带来更快优化。

## 11. JSDN 大为什么仍不等于梯度互补？

即使 $\operatorname{JSDN}_{ij}$ 很大，也只能直接说明客户端标签统计和样本量不同，不能保证

$$
\langle\delta_i,\delta_j\rangle<0.
$$

真正有利的互补情况要求两个客户端的梯度偏差方向相互抵消。因为

$$
\|a\delta_i+b\delta_j\|^2
=
a^2\|\delta_i\|^2
+b^2\|\delta_j\|^2
+2ab\langle\delta_i,\delta_j\rangle,
$$

只有当交叉项较小或为负时，组合后漂移才会明显下降。

高 JSDN 可能对应以下不同情形：

- 标签互补，配对混合分布更接近全局分布；
- 标签不同，但两个客户端梯度在当前模型上并不互补；
- 一个客户端是极端小样本异常点，差异大但信息不可靠；
- 数量差异很大，大客户端可作为低方差知识来源，但标签差异同时带来迁移偏差。

因此，不能继续使用未经证明的线性链条：

$$
\operatorname{JSDN}_{ij}
\Longrightarrow
\|\nabla F_i-\nabla F_j\|
\Longrightarrow
\text{收敛加速}.
$$

建议修改为：

$$
\text{JSDN 排序选择}
\Longrightarrow
\text{潜在互补客户端}
\Longrightarrow
\text{FCPC 降低组内漂移}
\Longrightarrow
\text{减小收敛界中的异质性项}.
$$

其中第一步需要实验校准或额外分布条件，第二、三步需要新的理论引理。

## 12. 建议建立的三个理论引理

以下内容是后续理论工作方案，目前尚未完成证明。

### 12.1 引理一：JSDN 的代理排序有效性

不再假设 JSDN 与梯度差异线性相关，而是研究排序关系：

$$
\mathbb E\left[
\|\delta_i-\delta_j\|^2
\mid \operatorname{JSDN}_{ij}\text{ 较大}
\right]
>
\mathbb E\left[
\|\delta_i-\delta_j\|^2
\mid \operatorname{JSDN}_{ij}\text{ 较小}
\right].
$$

该关系可以在额外的标签偏移、共享类条件分布等假设下推导，也可以先通过训练过程中的梯度审计验证。

### 12.2 引理二：配对正则降低组内局部漂移

目标是证明在适当学习率和正则强度下：

$$
\mathcal D_{ij}^{t+1}
\leq
(1-c\eta\beta_t)\mathcal D_{ij}^{t}
+
\text{随机梯度误差}
+
\text{伙伴滞后误差},
$$

其中 $\mathcal D_{ij}^t$ 表示一对客户端在参数、更新或梯度残差意义下的漂移量。

这一引理是 FCPC 正则与 JSDN 配对之间最关键的桥梁。

### 12.3 引理三：历史伙伴误差有界且渐近消失

利用

$$
\|\nabla R_{\mathrm{stale}}-\nabla R_{\mathrm{current}}\|
\leq
2\beta_t\tau_{ij}\|w_j^t-w_j^{t-1}\|,
$$

在模型每轮移动有界且 $\beta_t\to0$ 的条件下，证明历史配对引入的平均误差不会破坏收敛到一阶驻点。

## 13. 目标收敛界的结构

后续希望得到的不是“JSDN 越大必然越快”这种无条件结论，而是一条带有收益条件的界：

$$
\begin{aligned}
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb E\|\nabla F(w^t)\|^2
\leq{}&
\underbrace{\text{FedAvg 基础项}}_{\text{优化误差、方差、客户端异质性}}\\
&-
\underbrace{c_1\eta\beta_t\mathcal C_t}_{\text{配对互补收益}}\\
&+
\underbrace{c_2\eta\beta_t^2\mathcal S_t}_{\text{正则偏差和滞后代价}}.
\end{aligned}
$$

只有满足

$$
c_1\eta\beta_t\mathcal C_t
>
c_2\eta\beta_t^2\mathcal S_t
$$

时，才能声称 FCPC 的理论上界比 FedAvg 更紧。

这个结构也能解释当前实验现象：

- 固定且较大的 $\beta$ 可能使后期正则代价持续存在；
- 余弦衰减允许前期进行伙伴知识迁移、后期释放历史锚点；
- 样本权重 $\tau_{i\leftarrow j}$ 可能控制大样本伙伴向小样本客户端传递知识的强度；
- 但如果 JSDN 配对没有形成真实梯度互补，减小 $\beta$ 只能降低伤害，不能创造收益。

## 14. 理论验证所需实验

为了判断上述证明路线是否成立，训练代码需要增加以下审计量。

### 14.1 客户端对级别指标

每隔若干轮，在同一全局模型和统一参考批次上记录：

- $\operatorname{JSDN}_{ij}$；
- $\|w_i-w_j\|_2$；
- $\|g_i-g_j\|_2$；
- $\operatorname{cos}(g_i,g_j)$；
- $\langle\delta_i,\delta_j\rangle$；
- 配对混合标签分布与全局标签分布的 JS divergence；
- 伙伴模型的一轮陈旧距离 $\|w_j^t-w_j^{t-1}\|$。

### 14.2 需要报告的相关性

- JSDN 与梯度差异的 Spearman 排序相关；
- JSDN 与梯度余弦相似度的关系；
- JSDN 与配对混合分布全局覆盖改善的关系；
- 伙伴陈旧距离与 FCPC 正则损失的关系；
- 每种配对策略实际减少的组内漂移。

### 14.3 配对消融

在完全一致的训练条件下比较：

1. FedAvg：无 JSDN、无配对、无 FCPC 正则；
2. Random pairing：随机配对加相同正则；
3. Similar pairing：选择 JSDN 较小客户端；
4. Dissimilar pairing：当前 JSDN 最大差异配对；
5. Complementary pairing：直接选择配对混合分布更接近全局的客户端；
6. Gradient-oracle pairing：仅作为诊断，按梯度互补性选择，不作为隐私可部署算法。

如果最大 JSDN 配对确实优于随机和相似配对，并且接近 gradient-oracle pairing，才能有力支持 JSDN 是有效代理指标。

## 15. 基线复现优先级与注意事项

建议复现顺序：

1. **FedAvg**：已经作为统一训练主干；
2. **FedProx**：改动小，可优先检查现有实现和超参数；
3. **MOON**：适合 CIFAR-10，但需要投影头、上一轮本地模型和额外前向计算；
4. **FedDyn**：先解决原稿引用歧义，再决定实现标准动态正则化版本还是另行处理；
5. **FBLG**：需要服务器验证统计、模型嵌入、JS 图和客户端选择优化；
6. **FedCFA**：需要较大的模型结构和本地训练改造，实现成本最高。

所有基线必须遵循统一实验协议：

- 相同 CIFAR-10 训练、验证和测试划分；
- 相同 dual-skew 客户端数据；
- 相同模型主干和初始化；
- 相同客户端参与率、本地 epoch 和通信轮数；
- 分别调节每种方法特有超参数；
- 使用多个随机种子报告均值和标准差；
- 严格区分原论文报告值 `reported` 与本项目实际运行值 `reproduced`。

## 16. 当前结论

1. 参数空间距离、特征空间相似度、梯度方向和数据分布差异是四个不同概念，不能直接相互替代。
2. FedProx 控制参数漂移；MOON控制特征表示漂移；FedDyn使用历史修正状态对齐局部与全局驻点。
3. MOON、FedDyn 和当前 FCPC 都使用历史信息，但 MOON 使用历史负样本，FedDyn 使用递推控制状态，FCPC 使用历史伙伴参数，三者的滞后性质不同。
4. FBLG 通过选择相似客户端避免冲突，其同分布假设不能直接用于证明 FCPC 的最大差异配对。
5. JSDN 最大只能表示分布统计差异较大，不能自动推出梯度互补或收敛加速。
6. FCPC 更合理的理论目标是证明：最大差异配对找到潜在可消除漂移较大的客户端，配对正则将这种差异转化为互补更新，同时余弦 $\beta_t$ 控制历史正则偏差。
7. 当前 CIFAR-10 结果尚未证明 FCPC 稳定超过 FedAvg，因此理论中不能预设“FCPC 必然加速”，必须先完成 JSDN—梯度互补性审计和多随机种子实验。

## 17. 主要参考资料

1. [FedAvg](https://arxiv.org/abs/1602.05629)
2. [FedProx](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)
3. [MOON](https://arxiv.org/abs/2103.16257)
4. [FedDyn：标准动态正则化版本](https://arxiv.org/abs/2111.04263)
5. [FCPC 原稿引用的推荐系统 FedDyn](https://dblp.org/rec/conf/icpads/JinCGL22.html)
6. [FBLG](https://www.ijcai.org/proceedings/2024/585)
7. [FedCFA](https://ojs.aaai.org/index.php/AAAI/article/view/33942)
8. [SCAFFOLD：client drift 与控制变量分析](https://arxiv.org/abs/1910.06378)
9. [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/abs/1907.02189)
