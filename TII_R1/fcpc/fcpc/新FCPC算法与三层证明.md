# 新 FCPC 算法与三层证明

> 文档状态：理论设计与实现说明（2026-09-01）  
> 对应实现：提交 `1ac731b` 之后的新算法版本  
> 目的：解释新配对指标为什么成立、共同配对中心如何进入局部训练，以及“三层证明”分别证明了什么、没有证明什么。

---

## 1. 为什么这次 CIFAR-10 训练很久

正式配置 `configs/cifar10_pair_complementarity.yaml` 不是冒烟配置。它当前包含：

- 100 个通信轮次；
- 每轮选择 10 个客户端，即全客户端参与；
- 每个客户端执行 5 个本地 epoch；
- 模型为 ResNet-18；
- 每轮都执行验证集评估；
- 当前模拟器在一张 GPU 上依次训练各客户端，而不是同时训练 10 个客户端。

由于 10 个客户端的数据划分合起来近似构成一遍训练集，每轮的 10 客户端乘以 5 个本地 epoch，约等于 5 次完整训练集遍历。因此 100 轮约等于：

\[
100\times 5=500
\]

次完整 CIFAR-10 训练集遍历。此外，每轮还需要：

1. 为每个客户端建立和加载本地 ResNet-18；
2. 顺序执行 10 次本地训练；
3. 进行服务器聚合；
4. 进行验证集推理；
5. 保存统计信息并监控 CPU、GPU 和内存。

所以训练时间长主要来自正式实验规模，而不是加权 JS 或最大权匹配。10 个客户端时，配对矩阵只有 \(10\times10\)，最大权匹配的时间相对于 ResNet-18 训练可以忽略。

当前程序把逐轮结果写入 CSV，而控制台只在训练结束时打印最终结果。因此 `nohup` 日志长时间只有

```text
nohup: 忽略输入
```

不代表程序卡住。应通过以下方式检查：

```bash
ps -fp <PID>
nvidia-smi
tail -n 5 outputs/logs/cifar10_pair_complementarity.csv
```

正式实验前应依次完成：单元测试、两轮合成数据测试、CIFAR-10 冒烟测试，最后才运行 100 轮配置。

---

## 2. 原 FCPC 的关键缺口

原方法包含两个相互独立但没有被严格连接的部分。

### 2.1 原配对指标

原 JSDN 为

\[
J_{ij}
=(1-\lambda)\operatorname{JSD}(p_i,p_j)
+\lambda\frac{|N_i-N_j|}{N_i+N_j},
\]

其中 \(p_i,p_j\) 是标签分布，\(N_i,N_j\) 是样本量。

该指标能够描述标签偏斜和数量偏斜，但“指标值大”本身不能推出：

- 两个客户端的梯度方向互补；
- 两个客户端相互约束后更接近全局最优点；
- 最大 JSDN 配对一定加速收敛；
- 最大 JSDN 配对一定提高测试准确率。

特别是数量差项

\[
\frac{|N_i-N_j|}{N_i+N_j}
\]

只说明两端样本量不平衡，不能说明二者混合以后更接近全局分布。极端情况下，大客户端会完全支配加权混合，小客户端贡献很弱。

### 2.2 原交叉模型正则

原算法让客户端 \(i\) 直接靠近客户端 \(j\) 的历史模型：

\[
\mathcal L_i(w)+\beta_t\|w-w_j^{t-1}\|^2.
\]

相应地，客户端 \(j\) 又靠近 \(w_i^{t-1}\)。这产生两个问题：

1. 两端使用不同参考点，正则目标不对称；
2. “标签差异大”与“伙伴历史参数值得靠近”之间没有直接数学联系。

因此，原算法缺少下面的逻辑桥梁：

\[
\text{配对分数大}
\Longrightarrow
\text{配对后形成更有代表性的共同信息}
\Longrightarrow
\text{局部正则目标更有利}.
\]

新算法的目标不是继续假设 JSDN 线性控制梯度，而是重新定义一个具有精确分布解释的配对目标，并让参数正则与这个“共同混合”结构一致。

---

## 3. 统一符号

设共有 \(K\) 个客户端。

- \(N_i\)：客户端 \(i\) 的样本数；
- \(N=\sum_{i=1}^K N_i\)：总样本数；
- \(a_i=N_i/N\)：全局聚合权重；
- \(p_i\in\Delta^{C-1}\)：客户端 \(i\) 的标签分布；
- \(\bar p=\sum_i a_i p_i\)：全局标签分布；
- \(\mathcal L_i(w)\)：客户端 \(i\) 的本地任务损失；
- \(\mathcal L(w)=\sum_i a_i\mathcal L_i(w)\)：全局任务损失；
- \(w^\star\in\arg\min_w\mathcal L(w)\)：全局最优点；
- \(t\)：通信轮次；
- \(s\)：第 \(t\) 轮内的本地 SGD 步；
- \(w^{t-1}\)：第 \(t\) 轮开始时的全局模型；
- \(w_i^{t,s}\)：客户端 \(i\) 在第 \(t\) 轮第 \(s\) 个本地步的模型；
- \(v_i^{t-1}\)：客户端 \(i\) 上一次完成训练后保存的历史模型；
- \(G_i^{t,s}\)：客户端 \(i\) 在 mini-batch 上计算的随机梯度；
- \(\eta_t\)：第 \(t\) 轮学习率；
- \(\beta_t\ge0\)：配对正则系数；
- \(P_t\)：第 \(t\) 轮的客户端匹配集合。

本文把 mini-batch 梯度统一记为 \(G_i^{t,s}\)，避免同时使用多个容易混淆的梯度符号。

---

## 4. 新配对指标：加权 JS 互补收益

### 4.1 两个客户端的样本权重和混合分布

对客户端 \(i,j\)，定义

\[
\theta_{ij}=\frac{N_i}{N_i+N_j},
\]

以及样本量加权标签混合

\[
q_{ij}
=\theta_{ij}p_i+(1-\theta_{ij})p_j.
\]

定义广义加权 Jensen--Shannon 散度

\[
\operatorname{JS}_{\theta_{ij}}(p_i,p_j)
=\theta_{ij}\operatorname{KL}(p_i\|q_{ij})
+(1-\theta_{ij})\operatorname{KL}(p_j\|q_{ij}).
\]

再定义一条配对边的互补收益

\[
\boxed{
S_{ij}
=\frac{N_i+N_j}{N}
\operatorname{JS}_{\theta_{ij}}(p_i,p_j).
}
\]

与原 JSDN 不同，样本量不再作为一个独立的“差异越大越好”项。它有两个明确作用：

1. 决定配对内部的混合权重 \(\theta_{ij}\)；
2. 决定该配对在全局数据中的质量权重 \((N_i+N_j)/N\)。

### 4.2 KL--JS 精确恒等式

对任意全局标签分布 \(\bar p\)，有

\[
\boxed{
\theta_{ij}\operatorname{KL}(p_i\|\bar p)
+(1-\theta_{ij})\operatorname{KL}(p_j\|\bar p)
=\operatorname{JS}_{\theta_{ij}}(p_i,p_j)
+\operatorname{KL}(q_{ij}\|\bar p).
}
\]

证明如下。先展开左边：

\[
\begin{aligned}
&\theta_{ij}\sum_c p_{i,c}\log\frac{p_{i,c}}{\bar p_c}
+(1-\theta_{ij})\sum_c p_{j,c}\log\frac{p_{j,c}}{\bar p_c}.
\end{aligned}
\]

在两个对数中分别插入 \(q_{ij,c}\)：

\[
\log\frac{p_{i,c}}{\bar p_c}
=\log\frac{p_{i,c}}{q_{ij,c}}
+\log\frac{q_{ij,c}}{\bar p_c},
\]

\[
\log\frac{p_{j,c}}{\bar p_c}
=\log\frac{p_{j,c}}{q_{ij,c}}
+\log\frac{q_{ij,c}}{\bar p_c}.
\]

于是，含第一部分对数的两项组成

\[
\operatorname{JS}_{\theta_{ij}}(p_i,p_j).
\]

剩余部分为

\[
\begin{aligned}
&\sum_c
\left[\theta_{ij}p_{i,c}+(1-\theta_{ij})p_{j,c}\right]
\log\frac{q_{ij,c}}{\bar p_c}\\
&=\sum_c q_{ij,c}\log\frac{q_{ij,c}}{\bar p_c}\\
&=\operatorname{KL}(q_{ij}\|\bar p).
\end{aligned}
\]

恒等式得证。

### 4.3 为什么应该最大化总互补收益

把恒等式乘以配对质量权重

\[
a_{ij}=\frac{N_i+N_j}{N},
\]

得到

\[
\begin{aligned}
&\frac{N_i}{N}\operatorname{KL}(p_i\|\bar p)
+\frac{N_j}{N}\operatorname{KL}(p_j\|\bar p)\\
&=S_{ij}
+a_{ij}\operatorname{KL}(q_{ij}\|\bar p).
\end{aligned}
\]

若参与客户端数为偶数，且 \(P\) 是覆盖所有参与客户端的完全匹配，则对所有配对求和：

\[
\boxed{
\sum_i a_i\operatorname{KL}(p_i\|\bar p)
=\sum_{(i,j)\in P}S_{ij}
+\sum_{(i,j)\in P}a_{ij}\operatorname{KL}(q_{ij}\|\bar p).
}
\]

左边与如何配对无关，是常数。因此：

\[
\boxed{
\arg\max_P\sum_{(i,j)\in P}S_{ij}
=\arg\min_P
\sum_{(i,j)\in P}
a_{ij}\operatorname{KL}(q_{ij}\|\bar p).
}
\]

这给出了“为什么选择差异较大的客户端”的更准确回答：

> 新算法不是无条件追求统计差异最大，而是选择能够使配对混合标签分布整体最接近全局标签分布的客户端组合；最大化加权 JS 收益与最小化该剩余全局 KL 偏差严格等价。

这个结论比原 JSDN 的启发式解释更强，但仍有边界：

- 它是标签分布层面的严格结论；
- 它没有自动变成神经网络梯度定理；
- 它没有自动证明测试准确率提高；
- 完全等价式要求偶数客户端的完全匹配；奇数客户端需要固定未配对客户端、引入虚拟节点或显式保留未配对项；
- 使用 LDP 扰动统计量时，恒等式针对扰动后的分布和样本量成立，而不一定针对真实分布成立。

---

## 5. 新共同配对中心

### 5.1 参数空间中的共同中心

若客户端 \(i,j\) 被配对，使用与标签混合相同的样本权重构造共同历史中心：

\[
\boxed{
c_{ij}^{t-1}
=\theta_{ij}v_i^{t-1}
+(1-\theta_{ij})v_j^{t-1}.
}
\]

两个客户端都使用同一个中心，而不是相互使用对方的历史模型：

\[
c_i^t=c_j^t=c_{ij}^{t-1}.
\]

客户端 \(i\) 的本地正则化目标为

\[
\boxed{
\Phi_i^t(w)
=\mathcal L_i(w)
+\beta_t\|w-c_{ij}^{t-1}\|^2.
}
\]

其梯度为

\[
\nabla\Phi_i^t(w)
=\nabla\mathcal L_i(w)
+2\beta_t(w-c_{ij}^{t-1}).
\]

因此本地 SGD 更新为

\[
\boxed{
w_i^{t,s+1}
=w_i^{t,s}
-\eta_t
\left[
G_i^{t,s}
+2\beta_t(w_i^{t,s}-c_{ij}^{t-1})
\right].
}
\]

初始化仍然是

\[
w_i^{t,0}=w^{t-1}.
\]

### 5.2 第一轮的定义

在第 1 轮没有客户端历史模型时，代码用全局初始化代替两端历史模型：

\[
v_i^0=v_j^0=w^0,
\qquad
c_{ij}^0=w^0.
\]

所以本地训练开始的第一个 SGD 步正则梯度为零。执行第一个任务梯度更新以后，模型离开 \(w^0\)，后续 mini-batch 的正则项一般不再为零。

合成冒烟配置每客户端只执行一个 batch，因此第 1 轮记录的正则损失接近零；这不是正则失效，而是统计发生在第一次参数更新之前。

---

## 6. 三层证明总览

三层证明回答三个不同问题，不能混为一个定理。

| 层次 | 研究对象 | 回答的问题 | 不能推出的结论 |
|---|---|---|---|
| 第一层 | \(w_i^{t,s}\to u_i^t\) | 本地 SGD 是否收敛到本轮正则化目标的最优点 | 不说明该点就是全局最优点 |
| 第二层 | \(u_i^t\) 与 \(w^\star\) | 正则化本地最优点离全局最优点多远 | 不说明最大 JS 配对一定最好 |
| 第三层 | \(p_i,p_j,q_{ij},c_{ij}\) | 为什么新的配对目标具有全局分布互补意义 | 不无条件证明深度网络收敛加速 |

完整逻辑应写成：

\[
\underbrace{w_i^{t,s}\to u_i^t}_{\text{第一层}}
\quad\Longrightarrow\quad
\underbrace{\|u_i^t-w^\star\|\text{ 的偏差分解}}_{\text{第二层}}
\quad\Longleftarrow\quad
\underbrace{\text{选择更有代表性的配对混合与共同中心}}_{\text{第三层}}.
\]

第三层指向第二层，但二者之间需要明确的模型稳定性条件或实验验证，不能凭空跨越。

---

## 7. 第一层证明：本地 SGD 向正则化最优点收缩

### 7.1 假设

在固定通信轮次 \(t\) 内，中心 \(c_{ij}^{t-1}\) 保持不变。为得到参数距离收缩式，暂时作以下强凸分析假设：

1. \(\mathcal L_i\) 是 \(\mu_i\)-强凸函数；
2. \(\mathcal L_i\) 的梯度是 \(L_i\)-Lipschitz；
3. mini-batch 梯度条件无偏：
   \[
   \mathbb E_s[G_i^{t,s}]
   =\nabla\mathcal L_i(w_i^{t,s});
   \]
4. mini-batch 方差有界：
   \[
   \mathbb E_s
   \|G_i^{t,s}-\nabla\mathcal L_i(w_i^{t,s})\|^2
   \le \frac{\sigma_i^2}{B}.
   \]

这里的期望只针对当前 mini-batch 抽样。LDP 噪声不加入 \(G_i^{t,s}\)，因为当前算法的 LDP 只扰动配对元数据，不扰动训练梯度。

加入二次正则后，\(\Phi_i^t\) 的强凸和光滑常数分别为

\[
\mu_{\Phi,i}^t=\mu_i+2\beta_t,
\qquad
L_{\Phi,i}^t=L_i+2\beta_t.
\]

定义本轮正则化最优点

\[
u_i^t=\arg\min_w\Phi_i^t(w),
\]

因此

\[
\nabla\Phi_i^t(u_i^t)=0.
\]

### 7.2 距离平方展开

定义

\[
d_i^{t,s}=w_i^{t,s}-u_i^t.
\]

把更新式写成

\[
d_i^{t,s+1}
=d_i^{t,s}
-\eta_t
\left[
G_i^{t,s}+2\beta_t(w_i^{t,s}-c_{ij}^{t-1})
\right].
\]

使用向量恒等式

\[
\|x-\eta y\|^2
=\|x\|^2-2\eta\langle x,y\rangle+\eta^2\|y\|^2,
\]

得到

\[
\begin{aligned}
\|d_i^{t,s+1}\|^2
={}&\|d_i^{t,s}\|^2\\
&-2\eta_t
\left\langle
d_i^{t,s},
G_i^{t,s}+2\beta_t(w_i^{t,s}-c_{ij}^{t-1})
\right\rangle\\
&+\eta_t^2
\left\|
G_i^{t,s}+2\beta_t(w_i^{t,s}-c_{ij}^{t-1})
\right\|^2.
\end{aligned}
\]

令随机梯度噪声

\[
\delta_i^{t,s}
=G_i^{t,s}-\nabla\mathcal L_i(w_i^{t,s}).
\]

则

\[
G_i^{t,s}+2\beta_t(w_i^{t,s}-c_{ij}^{t-1})
=\nabla\Phi_i^t(w_i^{t,s})+\delta_i^{t,s}.
\]

由于条件无偏性 \(\mathbb E_s[\delta_i^{t,s}]=0\)，交叉项在条件期望下消失：

\[
\mathbb E_s
\|\nabla\Phi_i^t(w_i^{t,s})+\delta_i^{t,s}\|^2
=\|\nabla\Phi_i^t(w_i^{t,s})\|^2
+\mathbb E_s\|\delta_i^{t,s}\|^2.
\]

因此

\[
\begin{aligned}
\mathbb E_s\|d_i^{t,s+1}\|^2
\le{}&\|d_i^{t,s}\|^2
-2\eta_t
\langle d_i^{t,s},\nabla\Phi_i^t(w_i^{t,s})\rangle\\
&+\eta_t^2
\|\nabla\Phi_i^t(w_i^{t,s})\|^2
+\eta_t^2\frac{\sigma_i^2}{B}.
\end{aligned}
\]

### 7.3 使用强凸性和光滑性

因为 \(\nabla\Phi_i^t(u_i^t)=0\)，强凸性给出强单调性：

\[
\langle
w_i^{t,s}-u_i^t,
\nabla\Phi_i^t(w_i^{t,s})-\nabla\Phi_i^t(u_i^t)
\rangle
\ge
\mu_{\Phi,i}^t\|d_i^{t,s}\|^2.
\]

即

\[
\langle d_i^{t,s},\nabla\Phi_i^t(w_i^{t,s})\rangle
\ge
\mu_{\Phi,i}^t\|d_i^{t,s}\|^2.
\]

光滑性给出

\[
\|\nabla\Phi_i^t(w_i^{t,s})\|
\le
L_{\Phi,i}^t\|d_i^{t,s}\|.
\]

代入得到

\[
\boxed{
\mathbb E_s\|d_i^{t,s+1}\|^2
\le
\rho_i^t\|d_i^{t,s}\|^2
+\eta_t^2\frac{\sigma_i^2}{B},
}
\]

其中

\[
\rho_i^t
=1-2\eta_t\mu_{\Phi,i}^t
+\eta_t^2(L_{\Phi,i}^t)^2.
\]

若

\[
0<\eta_t
<\frac{2\mu_{\Phi,i}^t}{(L_{\Phi,i}^t)^2},
\]

则 \(0\le\rho_i^t<1\)。递推 \(S\) 步后：

\[
\boxed{
\mathbb E\|d_i^{t,S}\|^2
\le
(\rho_i^t)^S\|d_i^{t,0}\|^2
+\frac{\eta_t^2\sigma_i^2}{B}
\frac{1-(\rho_i^t)^S}{1-\rho_i^t}.
}
\]

结论是：本地 SGD 收缩到 \(u_i^t\) 附近的随机噪声邻域。

### 7.4 第一层究竟证明了什么

第一层证明：

- 正则化本地目标在强凸条件下有唯一最优点；
- 适当学习率下，mini-batch SGD 向该点收缩；
- mini-batch 方差决定稳态邻域大小；
- \(2\beta_t\) 同时增加强凸常数和光滑常数。

第一层没有证明：

- \(u_i^t=w^\star\)；
- \(\beta_t\) 越大收敛越快；
- 新配对一定优于随机配对；
- ResNet-18 的全局损失是强凸的。

二次正则会改善形式上的条件数

\[
\frac{L_i+2\beta_t}{\mu_i+2\beta_t},
\]

使其趋近于 1，但允许学习率、随机噪声邻域和目标偏差也会变化。因此不能仅凭 \(\mu_i+2\beta_t\) 增大就宣称训练必然加速。

---

## 8. 第二层证明：正则化最优点与全局最优点的偏差

第一层只说明模型接近 \(u_i^t\)。第二层研究 \(u_i^t\) 是否接近真正关心的全局最优点 \(w^\star\)。

### 8.1 最优性条件

由 \(u_i^t\) 的定义：

\[
\nabla\Phi_i^t(u_i^t)=0.
\]

由 \(\Phi_i^t\) 的强凸性，梯度具有强单调性：

\[
\langle
\nabla\Phi_i^t(u_i^t)-\nabla\Phi_i^t(w^\star),
u_i^t-w^\star
\rangle
\ge
\mu_{\Phi,i}^t\|u_i^t-w^\star\|^2.
\]

代入 \(\nabla\Phi_i^t(u_i^t)=0\)，并使用 Cauchy--Schwarz 不等式：

\[
\mu_{\Phi,i}^t\|u_i^t-w^\star\|^2
\le
\|\nabla\Phi_i^t(w^\star)\|
\|u_i^t-w^\star\|.
\]

若 \(u_i^t\ne w^\star\)，约去一个距离因子：

\[
\boxed{
\|u_i^t-w^\star\|
\le
\frac{
\|\nabla\Phi_i^t(w^\star)\|
}{\mu_{\Phi,i}^t}.
}
\]

展开 \(\nabla\Phi_i^t(w^\star)\)：

\[
\boxed{
\|u_i^t-w^\star\|
\le
\frac{
\|\nabla\mathcal L_i(w^\star)
+2\beta_t(w^\star-c_{ij}^{t-1})\|
}{\mu_i+2\beta_t}.
}
\]

### 8.2 两类偏差

上式出现两个不同来源：

1. 本地目标异质性
   \[
   \nabla\mathcal L_i(w^\star),
   \]
   因为全局最优只保证
   \[
   \sum_i a_i\nabla\mathcal L_i(w^\star)=0,
   \]
   不保证每个本地梯度分别为零。

2. 配对中心误差
   \[
   w^\star-c_{ij}^{t-1}.
   \]

使用 Young 不等式，对任意 \(\tau>0\)：

\[
\|x+y\|^2
\le(1+\tau)\|x\|^2
+\left(1+\frac1\tau\right)\|y\|^2,
\]

可以进一步得到

\[
\boxed{
\begin{aligned}
\|u_i^t-w^\star\|^2
\le{}&
\frac{1+\tau}{(\mu_i+2\beta_t)^2}
\|\nabla\mathcal L_i(w^\star)\|^2\\
&+
\frac{4\beta_t^2(1+1/\tau)}{(\mu_i+2\beta_t)^2}
\|w^\star-c_{ij}^{t-1}\|^2.
\end{aligned}
}
\]

这就是第二层真正得到的结果：共同中心离全局最优点越远，正则化本地最优点可能产生的偏差越大。

### 8.3 正则项可能有利，也可能有害

在分析到全局最优点的距离时，正则梯度对应的内积为

\[
\langle
w_i^{t,s}-w^\star,
2\beta_t(w_i^{t,s}-c_{ij}^{t-1})
\rangle.
\]

使用四点/三点距离恒等式：

\[
2\langle a-b,a-c\rangle
=\|a-b\|^2+\|a-c\|^2-\|b-c\|^2,
\]

得到

\[
\boxed{
\begin{aligned}
&\langle
w_i^{t,s}-w^\star,
2\beta_t(w_i^{t,s}-c_{ij}^{t-1})
\rangle\\
&=\beta_t\Big(
\|w_i^{t,s}-w^\star\|^2
+\|w_i^{t,s}-c_{ij}^{t-1}\|^2
-\|c_{ij}^{t-1}-w^\star\|^2
\Big).
\end{aligned}
}
\]

前两项为正，但最后一项为负。若中心离 \(w^\star\) 太远，整个内积不一定为正，正则项就不一定帮助模型靠近全局最优点。

因此第二层不是证明“正则项必然有效”，而是把算法设计问题转化为：

> 如何选择配对并构造共同中心，使 \(\|c_{ij}^{t-1}-w^\star\|\) 尽可能小，同时控制历史模型陈旧性？

这正是第三层要回答的方向。

---

## 9. 第三层证明：配对互补收益与共同中心的联系

### 9.1 严格成立的分布层结论

第 4 节已经严格证明：最大化

\[
\sum_{(i,j)\in P}S_{ij}
\]

等价于最小化

\[
\sum_{(i,j)\in P}
a_{ij}\operatorname{KL}(q_{ij}\|\bar p).
\]

所以新配对能够找到“标签混合后整体更接近全局标签分布”的组合。

### 9.2 从标签混合到参数中心仍需要条件

神经网络参数平均不等于数据分布混合：

\[
\theta w_i+(1-\theta)w_j
\not\equiv
w(\theta p_i+(1-\theta)p_j).
\]

因此不能直接声称最小化标签 KL 就必然最小化参数中心误差。若希望形成条件性理论桥梁，需要显式加入以下局部稳定性条件。

设 \(W(p)\) 表示在标签分布 \(p\) 下理想训练得到的参数表示。假设：

1. 训练映射对分布变化局部稳定：
   \[
   \|W(q)-W(\bar p)\|
   \le K\operatorname{TV}(q,\bar p);
   \]
2. 历史参数中心能够近似混合分布的理想模型：
   \[
   \|c_{ij}^{t-1}-W(q_{ij})\|
   \le\delta_{ij}^t;
   \]
3. 全局最优参数满足 \(w^\star=W(\bar p)\)。

则由三角不等式和 Pinsker 不等式，在 KL 使用以 2 为底的对数时：

\[
\operatorname{TV}(q,\bar p)^2
\le\frac{\ln2}{2}\operatorname{KL}_2(q\|\bar p),
\]

于是

\[
\boxed{
\|c_{ij}^{t-1}-w^\star\|
\le
\delta_{ij}^t
+K\sqrt{
\frac{\ln2}{2}
\operatorname{KL}_2(q_{ij}\|\bar p)
}.
}
\]

再使用 \((x+y)^2\le2x^2+2y^2\)：

\[
\boxed{
\|c_{ij}^{t-1}-w^\star\|^2
\le
2(\delta_{ij}^t)^2
+K^2\ln2\,
\operatorname{KL}_2(q_{ij}\|\bar p).
}
\]

将其代回第二层上界，可以得到一个条件性的逻辑链：

\[
\max_P\sum S_{ij}
\Longleftrightarrow
\min_P\sum a_{ij}\operatorname{KL}(q_{ij}\|\bar p)
\Longrightarrow
\text{减小共同中心误差的一个上界}
\Longrightarrow
\text{减小正则化本地最优点偏差的一个上界}.
\]

### 9.3 必须诚实说明的边界

上面的前一个等价箭头是严格恒等式；后两个箭头依赖训练映射稳定性和中心近似条件。这些条件对一般深度网络并非自动成立，必须通过实验检查，例如：

- \(\operatorname{KL}(q_{ij}\|\bar p)\) 与 \(\|c_{ij}-w^t\|\) 的相关性；
- 配对中心与全局聚合模型的距离；
- 配对前后局部更新的余弦相似度；
- 不同配对策略下的客户端漂移；
- 新分数与实际精度收益的 Spearman 排序相关性。

因此论文中可以写“在稳定性条件下得到配对收益上界”，不能写成“加权 JS 无条件保证深度网络更快收敛”。

---

## 10. 三层证明合并后的总误差结构

使用三角不等式：

\[
\|w_i^{t,S}-w^\star\|^2
\le
2\|w_i^{t,S}-u_i^t\|^2
+2\|u_i^t-w^\star\|^2.
\]

第一项由第一层控制：

\[
\mathbb E\|w_i^{t,S}-u_i^t\|^2
\le
(\rho_i^t)^S\|w^{t-1}-u_i^t\|^2
+\frac{\eta_t^2\sigma_i^2}{B}
\frac{1-(\rho_i^t)^S}{1-\rho_i^t}.
\]

第二项由第二层控制，并在第三层条件成立时进一步受混合分布 KL 控制。因此总结构可写成：

\[
\boxed{
\begin{aligned}
\mathbb E\|w_i^{t,S}-w^\star\|^2
\lesssim{}&
\underbrace{(\rho_i^t)^S
\|w^{t-1}-u_i^t\|^2}_{\text{有限本地步优化误差}}\\
&+\underbrace{
\frac{\eta_t^2\sigma_i^2}{B(1-\rho_i^t)}
}_{\text{mini-batch 随机误差}}\\
&+\underbrace{
\frac{\|\nabla\mathcal L_i(w^\star)\|^2}
{(\mu_i+2\beta_t)^2}
}_{\text{本地目标异质性}}\\
&+\underbrace{
\frac{\beta_t^2}
{(\mu_i+2\beta_t)^2}
\left[
(\delta_{ij}^t)^2
+K^2\operatorname{KL}(q_{ij}\|\bar p)
\right]
}_{\text{配对中心误差与分布残差}}.
\end{aligned}
}
\]

这里的 \(\lesssim\) 隐藏了 Young 不等式产生的常数。该式适合作为理论结构说明，不应在未补齐所有常数、客户端抽样和服务器聚合递推之前直接命名为完整的联邦收敛定理。

---

## 11. 强凸分析与 ResNet-18 的关系

CIFAR-10 上的 ResNet-18 是非凸模型，因此不能宣称其损失满足全局 \(\mu\)-强凸性。强凸三层证明的作用是：

1. 清楚展示正则项怎样改变局部最优点；
2. 分离本地优化误差与正则偏差；
3. 揭示共同中心误差为何关键；
4. 为凸模型、最后一层优化或局部强凸邻域提供严格结果。

对于正文中的深度网络，更稳妥的是补充光滑非凸分析。若 \(\Phi_i^t\) 为 \(L_{\Phi}\)-smooth，mini-batch 梯度无偏且方差有界，则下降引理给出

\[
\begin{aligned}
\mathbb E_s[\Phi_i^t(w_i^{t,s+1})]
\le{}&
\Phi_i^t(w_i^{t,s})
-\eta_t\left(1-\frac{L_{\Phi}\eta_t}{2}\right)
\|\nabla\Phi_i^t(w_i^{t,s})\|^2\\
&+\frac{L_{\Phi}\eta_t^2}{2}
\frac{\sigma_i^2}{B}.
\end{aligned}
\]

当 \(0<\eta_t<2/L_{\Phi}\) 时，对 \(S\) 步求和可得平均驻点界：

\[
\frac1S\sum_{s=0}^{S-1}
\mathbb E\|\nabla\Phi_i^t(w_i^{t,s})\|^2
\le
\frac{
\Phi_i^t(w_i^{t,0})-\Phi_{i,\inf}^t
}{
\eta_t S(1-L_{\Phi}\eta_t/2)
}
+
\frac{
L_{\Phi}\eta_t\sigma_i^2
}{
2B(1-L_{\Phi}\eta_t/2)
}.
\]

这个非凸结果证明局部训练接近正则化目标的一阶驻点，但仍需进一步处理多客户端聚合、中心跨轮变化和部分参与，才能成为完整的全局 FCPC 收敛定理。

---

## 12. 新 FCPC 算法流程

### 12.1 初始化和配对元数据

每个客户端计算标签直方图 \(h_i\) 和样本量 \(N_i\)。当前隐私流程对这些元数据执行 LDP：

\[
(h_i,N_i)
\longrightarrow
(\widetilde h_i,\widetilde N_i).
\]

服务器归一化得到 \(\widetilde p_i\)，并使用扰动后的样本量构造

\[
\widetilde\theta_{ij}
=\frac{\widetilde N_i}
{\widetilde N_i+\widetilde N_j},
\]

\[
\widetilde S_{ij}
=\frac{\widetilde N_i+\widetilde N_j}
{\sum_k\widetilde N_k}
\operatorname{JS}_{\widetilde\theta_{ij}}
(\widetilde p_i,\widetilde p_j).
\]

注意：LDP 噪声只影响配对矩阵，不作为梯度噪声加入 SGD。

### 12.2 精确最大权匹配

把参与客户端作为顶点，\(\widetilde S_{ij}\) 作为非负边权，求解

\[
P_t^*
\in
\arg\max_{P_t\text{ 为匹配}}
\sum_{(i,j)\in P_t}\widetilde S_{ij}.
\]

代码使用 NetworkX 的 `max_weight_matching(..., maxcardinality=True)`。相比原贪心算法，它直接求小规模客户端图上的精确最大权匹配。

复杂度为：

- 构造分数矩阵：\(O(K^2C)\)；
- 一般图最大权匹配：多项式时间，常见实现量级约 \(O(K^3)\)；
- 10 个客户端时，这部分远小于一次神经网络训练。

### 12.3 每轮共同中心和本地训练

第 \(t\) 轮：

1. 服务器选择参与客户端并在子图上匹配；
2. 冻结所有参与客户端的轮前历史状态；
3. 对每个配对 \((i,j)\) 构造共同中心
   \[
   c_{ij}^{t-1}
   =\theta_{ij}v_i^{t-1}
   +(1-\theta_{ij})v_j^{t-1};
   \]
4. 两个客户端都从全局模型初始化
   \[
   w_i^{t,0}=w_j^{t,0}=w^{t-1};
   \]
5. 两端分别最小化
   \[
   \mathcal L_i(w)+\beta_t\|w-c_{ij}^{t-1}\|^2,
   \]
   \[
   \mathcal L_j(w)+\beta_t\|w-c_{ij}^{t-1}\|^2;
   \]
6. 服务器对训练后的客户端模型执行样本量加权 FedAvg；
7. 保存本轮本地模型，供下一轮中心构造使用。

当前实现中，配对分数使用 LDP 扰动计数，而共同参数中心使用真实样本数计算权重。这与服务器执行样本量加权 FedAvg 的现有可见信息一致，但论文必须明确相应威胁模型，不能同时声称服务器完全不知道真实样本量。

### 12.4 伪代码

```text
Input:
    clients {1,...,K}
    initial global model w^0
    metadata privacy budget epsilon
    learning rates {eta_t}
    regularization schedule {beta_t}

Metadata stage:
    each client i releases perturbed histogram and count
    server computes weighted-JS complementarity matrix S

For communication round t = 1,...,T:
    A_t <- select participating clients
    P_t <- exact maximum-weight matching on S[A_t, A_t]
    snapshot every selected client's previous local state

    For every pair (i,j) in P_t:
        theta_ij <- N_i / (N_i + N_j)
        center_ij <- theta_ij * previous_i
                     + (1-theta_ij) * previous_j

        For k in {i,j}:
            w_k^{t,0} <- w^{t-1}
            For local step s = 0,...,E-1:
                compute mini-batch gradient G_k^{t,s}
                w_k^{t,s+1} <- w_k^{t,s}
                    - eta_t [G_k^{t,s}
                    + 2 beta_t (w_k^{t,s} - center_ij)]

    Unpaired selected client:
        train without FCPC regularization

    w^t <- sample-count-weighted average of selected local models
    store local states for the next round

Output: w^T
```

---

## 13. 代码与公式的对应关系

| 文件 | 关键实现 | 对应公式或作用 |
|---|---|---|
| `src/fcpc/jsdn.py` | `weighted_js_divergence` | \(\operatorname{JS}_{\theta}(p_i,p_j)\) |
| `src/fcpc/jsdn.py` | `pair_complementarity_score` | \(S_{ij}=a_{ij}\operatorname{JS}_{\theta}\) |
| `src/fcpc/jsdn.py` | `build_pair_complementarity_matrix` | 向量化构造完整边权矩阵 |
| `src/fcpc/pairing.py` | `optimal_high_dissimilarity_pairing` | 精确最大权匹配 |
| `src/fcpc/regularizer.py` | `weighted_state_center` | \(c_{ij}=\theta v_i+(1-\theta)v_j\) |
| `src/fcpc/regularizer.py` | `fcpc_regularization` | 可传梯度的 \(\|w-c_{ij}\|^2\) |
| `src/federated/server.py` | `pairing_metric` | 在旧 JSDN 与新分数之间切换 |
| `src/federated/trainer.py` | `reference_strategy` | 在伙伴历史模型与共同中心之间切换 |
| `configs/cifar10_pair_complementarity.yaml` | 正式实验配置 | 新分数、新中心、精确匹配 |

新算法配置的核心字段是：

```json
"fcpc": {
  "enabled": true,
  "metric": "pair_complementarity",
  "reference_strategy": "pair_center",
  "pairing_strategy": "optimal",
  "beta": 0.2,
  "beta_schedule": "cosine",
  "min_beta": 0.0,
  "epsilon": 1.0
}
```

旧 FCPC 仍可通过下面的组合保留：

```json
"fcpc": {
  "enabled": true,
  "metric": "jsdn",
  "reference_strategy": "partner",
  "pairing_strategy": "greedy_dissimilar"
}
```

---

## 14. 为什么使用余弦衰减的 \(\beta_t\)

若 \(\beta_t\) 固定且共同中心长期存在误差，第二层上界中的中心偏差项不会自动消失，算法可能只能收敛到原始全局目标附近的邻域。

当前正式配置使用

\[
\beta_t
=\beta_{\min}
+\frac12(\beta_0-\beta_{\min})
\left[1+\cos\left(\frac{\pi t}{T-1}\right)\right],
\]

并设置 \(\beta_{\min}=0\)。其设计含义是：

- 训练前期使用共同中心抑制过度本地漂移；
- 训练后期逐渐释放历史中心约束；
- 降低陈旧中心造成永久渐近偏差的风险。

这能解释为什么要衰减 \(\beta_t\)，但不能提前宣称余弦形式是最优日程。仍需与固定、线性衰减和不同初始 \(\beta_0\) 做消融。

---

## 15. 必须进行的公平对照

仅比较“原 FCPC”和“新 FCPC”不足以判断提升来自哪一部分。至少应采用二乘二消融：

| 组别 | 配对分数 | 正则参考 | 目的 |
|---|---|---|---|
| A：原 FCPC | JSDN | 伙伴历史模型 | 原方法 |
| B：只改分数 | 加权 JS 互补收益 | 伙伴历史模型 | 检验新配对目标 |
| C：只改中心 | JSDN | 共同加权中心 | 检验共同中心 |
| D：新 FCPC | 加权 JS 互补收益 | 共同加权中心 | 完整方法 |
| E：FedAvg | 无 | 无 | 基础基线 |
| F：随机配对 | 随机 | 共同加权中心 | 检验配对选择必要性 |
| G：相似配对 | 最小分数 | 共同加权中心 | 检验互补配对方向 |

所有组必须固定：

- 相同数据划分；
- 相同随机种子；
- 相同模型初始化；
- 相同客户端参与集合；
- 相同本地 epoch、batch size、学习率日程；
- 相同 \(\beta_t\) 日程（无正则组除外）；
- 相同验证和早停规则。

至少运行 3 个随机种子，报告均值和标准差。机制指标至少包括：

1. 测试准确率和损失；
2. 达到固定准确率所需轮数；
3. 每轮训练时间和总时间；
4. 配对计算时间；
5. 配对混合 KL 残差；
6. 共同中心到全局模型的距离；
7. 客户端更新余弦相似度；
8. 客户端漂移范数；
9. 正则原始损失与加权损失；
10. 通信字节数和显存峰值。

只有当新分数在混合 KL、中心误差和最终精度上形成一致证据时，才能把第三层的条件桥梁写成可信的机制解释。

---

## 16. 当前可以写进论文的结论

可以严格陈述：

1. 加权 JS 互补收益等于配对混合前后相对全局标签分布的 KL 减少量；
2. 对偶数客户端的完全匹配，最大化总互补收益等价于最小化配对混合分布的总加权 KL 残差；
3. 精确最大权匹配求解了所定义的组合优化目标；
4. 共同中心正则的梯度是 \(2\beta_t(w-c_{ij})\)，且当前实现可以正常反向传播；
5. 在强凸、光滑、随机梯度无偏和方差有界条件下，本地 SGD 收缩到正则化本地最优点的噪声邻域；
6. 正则化本地最优点与全局最优点的距离受本地异质性和共同中心误差共同控制。

暂时不能无条件陈述：

1. 加权 JS 越大，神经网络梯度越互补；
2. 共同参数中心等价于在混合数据上训练的模型；
3. 新 FCPC 必然比 FedAvg 或原 FCPC 收敛更快；
4. 新 FCPC 必然提高最终准确率；
5. ResNet-18 损失满足全局强凸性；
6. LDP 扰动不会改变最优配对；
7. \(\beta_0=0.2\) 或余弦衰减是理论最优选择。

---

## 17. 与原附录相比删除的错误跳跃

新分析不再使用：

- “JSDN 与梯度差异线性相关”的未验证假设；
- 凭空出现的 \(e^{-\gamma\operatorname{JSDN}t}\) 收敛奖励项；
- 实际算法没有加入的 LDP 梯度噪声方差；
- 从参数距离直接推出梯度方向一致；
- 用深度网络实验却声称目标全局强凸；
- 从“最大化指标”直接跳到“测试精度最高”。

新逻辑改为：

\[
\text{精确的分布恒等式}
+\text{明确的局部优化分析}
+\text{带条件的参数桥梁}
+\text{可证伪的机制实验}.
\]

---

## 18. 理论和算法依据

1. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017. FedAvg 与本地多步、服务器模型平均的基础：  
   https://proceedings.mlr.press/v54/mcmahan17a.html

2. Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020. FedProx 说明二次邻近项是处理异质联邦优化的重要理论工具，但其参考点和本文共同配对中心不同：  
   https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html

3. Karimireddy et al., *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*, ICML 2020. 客户端漂移及异质性影响的标准参考：  
   https://proceedings.mlr.press/v119/karimireddy20a.html

4. Ghadimi and Lan, *Stochastic First- and Zeroth-order Methods for Nonconvex Stochastic Programming*, SIAM Journal on Optimization, 2013. 光滑非凸随机优化与平均梯度范数分析依据：  
   https://arxiv.org/abs/1309.5549

这些文献支持标准联邦优化和随机优化工具，但新加权 JS 恒等式、配对目标以及三层之间的条件桥梁仍必须由本文自行证明并通过实验验证。

