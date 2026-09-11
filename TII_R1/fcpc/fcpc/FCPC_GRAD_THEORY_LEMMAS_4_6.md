# FCPC-grad 理论主线：从加权 JS 配对到条件收敛加速

> **状态：理论主线（唯一引用入口）。**
> 本文统一算法、变量、引理1—6、条件加速定理、实验对应关系与结论边界。论文正文和附录应以本文件为准。
> [At(M).md](At(M).md) 与 [新FCPC算法与三层证明.md](新FCPC算法与三层证明.md) 是研究草稿；若与本文冲突，以本文为准。
> 文件名暂时保留 LEMMAS_4_6 以兼容已有引用，但本版已经覆盖引理1—6。

## 0. 理论主线

FCPC-grad 的逻辑不是“JSD 越大，所以收敛越快”，而是：

\[
\boxed{
\begin{aligned}
&\text{最大化加权 JS 互补收益}\\
&\Longleftrightarrow
\text{最小化配对混合分布的加权 KL 残差}\\
&\Longrightarrow
\text{在纯标签偏移条件下控制配对梯度偏差}\\
&\Longrightarrow
\text{在陈旧、噪声和步长失衡可控时，历史更新代理是下降方向}\\
&\Longrightarrow
\text{精确 proximal 以非负系数把该方向传入客户端更新}\\
&\Longrightarrow
\text{服务器聚合保留其沿负全局梯度的投影}\\
&\Longrightarrow
\text{当一阶收益超过二阶平滑代价时，获得条件单轮加速。}
\end{aligned}
}
\]

第一步是精确恒等式；第二步需要标签偏移模型；后续步骤需要明确的充分条件。因此，本文证明的是**可检验的条件加速**，不是任意深度网络、任意轮次下的无条件加速。

---

## 1. 为什么需要新证明

### 1.1 原 FCPC 的断点

原文使用

\[
J_{ij}
=(1-\lambda)\operatorname{JSD}(p_i,p_j)
+\lambda\frac{|N_i-N_j|}{N_i+N_j}
\]

选择高差异客户端，再让客户端靠近伙伴历史模型。这里有两个未证明的跳跃：

1. $J_{ij}$ 大不自动表示两端混合后更接近全局分布；
2. 参数距离小不自动表示更新与全局负梯度同向。

因此，不能直接从最大 JSDN 推出梯度误差上界变小，更不能凭空加入指数收敛奖励。

### 1.2 上一版新 FCPC 的断点

上一版用加权 JS 互补收益和共同历史参数中心修复了分布解释，但冻结检查点实验发现：配对可明显改变客户端对残差上界 $A_t(M)$，服务器真实更新误差 $U_t(M)$ 却几乎不变。客户端侧干预在聚合时大量抵消，而且历史参数中心没有显式的下降方向。

FCPC-grad 因此把历史客户端更新作为梯度代理加入共同中心，在不要求客户端额外上传当前梯度的前提下，显式增强服务器更新的下降投影。

---

## 2. 统一符号

设共有 $K$ 个客户端，第 $t$ 轮参与集合为 $\mathcal S_t$。

| 符号 | 含义 |
|---|---|
| $N_i$ | 客户端 $i$ 的样本数 |
| $N=\sum_iN_i$ | 全部样本数 |
| $a_i=N_i/N$ | 全局目标权重 |
| $N_{\mathcal S_t}=\sum_{i\in\mathcal S_t}N_i$ | 当轮参与样本数 |
| $a_{i,t}=N_i/N_{\mathcal S_t}$ | 当轮服务器聚合权重 |
| $p_i\in\Delta^{C-1}$ | 客户端标签分布 |
| $\bar p=\sum_i a_ip_i$ | 全局标签分布 |
| $F_i(w)$ | 客户端期望任务损失 |
| $F(w)=\sum_i a_iF_i(w)$ | 全局目标 |
| $w^t$ | 第 $t$ 轮开始时的全局模型 |
| $g_t=\nabla F(w^t)$ | 当前全局梯度 |
| $M_t$ | 当轮参与客户端的匹配 |
| $p=(i,j)$ | 一个客户端对 |
| $\theta_p=N_i/(N_i+N_j)$ | 配对内部样本权重 |
| $\omega_{p,t}=(N_i+N_j)/N_{\mathcal S_t}$ | 配对在当轮聚合中的权重 |
| $q_p=\theta_pp_i+(1-\theta_p)p_j$ | 配对混合标签分布 |
| $g_p^t=\theta_p\nabla F_i(w^t)+(1-\theta_p)\nabla F_j(w^t)$ | 配对轮初梯度 |
| $v_i^{\tau_i(t)}$ | 客户端最近一次参与后的本地终点 |
| $b_i^{\tau_i(t)}$ | 该次参与时服务器下发的起点 |
| $d_i^t=v_i^{\tau_i(t)}-b_i^{\tau_i(t)}$ | 客户端历史更新代理 |
| $H_i^{\tau_i(t)}$ | 该次历史本地训练的实际优化步数 |
| $\gamma_i^t=\sum_{h=0}^{H_i^{\tau_i(t)}-1}\eta_{i,h}$ | 历史累计学习率 |
| $\beta_t$ | proximal 强度 |
| $\xi_t\in[0,1]$ | 历史中心和代理中心的混合比例 |
| $s_t\ge0$ | 历史更新代理的外推尺度 |
| $\chi_{p,t}\in[0,1]$ | 中心裁剪后保留的比例 |

$w^t$ 始终表示当前全局模型。$b_i^{\tau_i(t)}$ 表示客户端上一次参与时收到的历史广播模型，通常等于当时的 $w^{\tau_i(t)}$，而不是当前 $w^t$。

完整参与和完全匹配时，

\[
\sum_{p\in M_t}\omega_{p,t}=1,
\qquad
\sum_{p\in M_t}\omega_{p,t}g_p^t=g_t.
\]

部分参与时，右端变为

\[
g_{\mathcal S_t}^t
=\sum_{i\in\mathcal S_t}a_{i,t}\nabla F_i(w^t),
\]

必须保留抽样残差 $g_{\mathcal S_t}^t-g_t$。奇数参与客户端还需保留未配对残差。

---

## 3. FCPC-grad 算法

### 3.1 加权 JS 配对

对候选边 $p=(i,j)$，定义

\[
q_p=\theta_pp_i+(1-\theta_p)p_j,
\]

\[
\operatorname{JS}_{\theta_p}(p_i,p_j)
=\theta_p\operatorname{KL}(p_i\|q_p)
+(1-\theta_p)\operatorname{KL}(p_j\|q_p),
\]

\[
S_{ij}=(a_i+a_j)\operatorname{JS}_{\theta_p}(p_i,p_j).
\]

服务器把客户端视为顶点、$S_{ij}$ 视为边权，求解

\[
M_t^*\in\arg\max_{M_t}
\sum_{(i,j)\in M_t}S_{ij}.
\]

当前代码使用一般图最大权匹配，而不是逐边贪心。固定参与集合下，统一缩放所有边权不会改变最优匹配。

### 3.2 梯度代理共同中心

历史参数中心为

\[
c_{p,t}^{\mathrm{hist}}
=\theta_pv_i^{\tau_i(t)}
+(1-\theta_p)v_j^{\tau_j(t)}.
\]

客户端和配对历史更新为

\[
d_i^t=v_i^{\tau_i(t)}-b_i^{\tau_i(t)},
\qquad
d_p^t=\theta_pd_i^t+(1-\theta_p)d_j^t.
\]

从当前全局模型外推代理中心：

\[
c_{p,t}^{\mathrm{proxy}}=w^t+s_td_p^t.
\]

未裁剪中心为

\[
c_{p,t}^{\mathrm{raw}}
=(1-\xi_t)c_{p,t}^{\mathrm{hist}}
+\xi_tc_{p,t}^{\mathrm{proxy}}.
\]

令

\[
h_p^t=c_{p,t}^{\mathrm{hist}}-w^t,
\]

则

\[
c_{p,t}^{\mathrm{raw}}-w^t
=(1-\xi_t)h_p^t+\xi_ts_td_p^t.
\]

实际代码进行径向裁剪。用 $\chi_{p,t}\in[0,1]$ 表示裁剪比例：

\[
\boxed{
\widetilde c_{p,t}
=w^t+\chi_{p,t}
\big[(1-\xi_t)h_p^t+\xi_ts_td_p^t\big].
}
\]

当前实验最优配置使用 $\xi_t=1$，因此历史参数中心项消失，中心主要由经过裁剪的历史更新代理决定。一般形式仍用于覆盖 mix0 消融和其他配置。

### 3.3 本地 proximal 与服务器聚合

客户端 $i\in p$ 从 $x_{i,0}=w^t$ 开始。每一步先执行任务优化器更新

\[
u_{i,h}=x_{i,h}-\eta_t\widehat v_{i,h},
\]

其中 $\widehat v_{i,h}$ 是优化器实际采用的方向；普通 SGD 时它是 mini-batch 梯度，带动量和权重衰减时还包含相应优化器状态。

随后执行精确 proximal 映射：

\[
x_{i,h+1}
=\arg\min_x
\left\{
\frac{1}{2\eta_t}\|x-u_{i,h}\|^2
+\beta_t\|x-\widetilde c_{p,t}\|^2
\right\}.
\]

服务器进行样本量加权聚合：

\[
w^{t+1}
=w^t+\sum_{i\in\mathcal S_t}a_{i,t}\Delta_i^t,
\qquad
\Delta_i^t=x_{i,H_i^t}-w^t.
\]

---

## 4. 分析条件

1. **全局光滑性：**
   \[
   F(w+z)\le F(w)+\langle\nabla F(w),z\rangle+\frac L2\|z\|^2.
   \]
2. **纯标签偏移桥梁：** 引理2假设 $P_i(x\mid y)=P(x\mid y)$。
3. **类别条件梯度有界：** $\|g_y(w)\|\le G$。
4. **历史代理误差可控：** 历史噪声、本地漂移、优化器变换、旧正则和陈旧性形成的残差有界。
5. **当前本地执行误差可控：** 当前多步优化方向与轮初真实客户端梯度之差有界。
6. **代理下降条件：** 引理4的余量为正，或使用门控剔除明显有害的代理。
7. **有限中心强度：** $\beta_t,s_t,\xi_t$ 与裁剪半径使二阶平滑代价不超过一阶下降收益。

引理1是分布恒等式；引理2依赖标签偏移；引理3和引理5是代数分解；引理4、引理6及加速定理是条件结论。

---

## 5. 引理1：加权 JS 恒等式与最大权匹配

**引理1。** 对任意配对 $p=(i,j)$ 和参考分布 $\bar p$，

\[
\boxed{
a_i\operatorname{KL}(p_i\|\bar p)
+a_j\operatorname{KL}(p_j\|\bar p)
=S_{ij}+(a_i+a_j)\operatorname{KL}(q_p\|\bar p).
}
\]

**证明。** 在两个 KL 的对数中插入 $q_p$：

\[
\log\frac{p_i(y)}{\bar p(y)}
=\log\frac{p_i(y)}{q_p(y)}
+\log\frac{q_p(y)}{\bar p(y)},
\]

客户端 $j$ 同理。第一组项组成 $S_{ij}$；第二组项利用

\[
a_ip_i(y)+a_jp_j(y)=(a_i+a_j)q_p(y)
\]

化为 $(a_i+a_j)\operatorname{KL}(q_p\|\bar p)$。

对固定参与集合的完全匹配求和，左端与配对方式无关，因此

\[
\boxed{
\arg\max_{M_t}\sum_{p\in M_t}S_p
=\arg\min_{M_t}R(M_t),
}
\]

\[
R(M_t)
=\sum_{p\in M_t}(a_i+a_j)
\operatorname{KL}(q_p\|\bar p).
\]

所以目标不是“差异本身越大越好”，而是让每对混合以后留下的全局标签残差总量最小。最大权匹配精确优化总边权；贪心通常没有全局最优保证。

若使用 LDP 扰动后的 $\widetilde p_i,\widetilde N_i$ 构造分数，恒等式对应扰动后的量，不自动保证真实分布上的最优匹配不变。

---

## 6. 引理2：标签混合到配对梯度偏差

在纯标签偏移条件下定义

\[
g_y(w)
=\mathbb E_{x\sim P(x\mid y)}
[\nabla_w\ell(w;x,y)].
\]

则

\[
\nabla F_p(w)=\sum_yq_p(y)g_y(w),
\qquad
\nabla F(w)=\sum_y\bar p(y)g_y(w).
\]

**引理2。** 若 $\|g_y(w)\|\le G$，则

\[
\boxed{
\|\nabla F_p(w)-\nabla F(w)\|
\le G\|q_p-\bar p\|_1
\le G\sqrt{2\operatorname{KL}(q_p\|\bar p)}.
}
\]

**证明。**

\[
\begin{aligned}
\|\nabla F_p(w)-\nabla F(w)\|
&=\left\|\sum_y[q_p(y)-\bar p(y)]g_y(w)\right\|\\
&\le\sum_y|q_p(y)-\bar p(y)|\,\|g_y(w)\|\\
&\le G\|q_p-\bar p\|_1\\
&\le G\sqrt{2\operatorname{KL}(q_p\|\bar p)}.
\end{aligned}
\]

最后一步使用自然对数版本的 Pinsker 不等式。完整参与时定义

\[
H_t(M_t)
=\sum_{p\in M_t}\omega_{p,t}\|g_p^t-g_t\|^2,
\]

便有

\[
\boxed{H_t(M_t)\le2G^2R_t(M_t).}
\]

引理2的用途是把引理1的分布残差转换成引理4中的梯度偏差。可取

\[
\delta_p^t=G\|q_p-\bar p\|_1.
\]

如果 $P_i(x\mid y)$ 也不同，必须引入客户端相关的 $g_{i,y}$ 和条件特征偏移残差；标签直方图不能控制全部梯度偏差。

---

## 7. 引理3：含数量偏斜的历史更新代理分解

### 7.1 单客户端严格分解

客户端 $i$ 最近一次参与发生在 $\tau_i(t)<t$。它从

\[
x_{i,0}=b_i^{\tau_i(t)}=w^{\tau_i(t)}
\]

出发，执行 $H_i^{\tau_i(t)}$ 个实际优化器步：

\[
x_{i,h+1}=x_{i,h}-\eta_{i,h}\widehat v_{i,h}.
\]

望远镜求和得到

\[
d_i^t
=v_i^{\tau_i(t)}-b_i^{\tau_i(t)}
=-\sum_{h=0}^{H_i^{\tau_i(t)}-1}
\eta_{i,h}\widehat v_{i,h}.
\]

令

\[
\gamma_i^t
=\sum_{h=0}^{H_i^{\tau_i(t)}-1}\eta_{i,h}.
\]

在和式中加入并减去当前轮初梯度 $\nabla F_i(w^t)$：

\[
\boxed{
d_i^t=-\gamma_i^t\nabla F_i(w^t)+r_i^t,
}
\]

\[
r_i^t
=-\sum_h\eta_{i,h}
\big[\widehat v_{i,h}-\nabla F_i(w^t)\big].
\]

$r_i^t$ 包含 mini-batch 噪声、历史轮内多步漂移、从 $w^{\tau_i(t)}$ 到 $w^t$ 的陈旧误差、动量、权重衰减及历史 FCPC 正则的影响。因此 $d_i^t$ 是优化方向代理，不是当前无偏任务梯度。

### 7.2 $H_i$ 和 $\gamma_i$ 不一致产生的额外残差

当前实验按 local epochs 训练。数量偏斜导致客户端 batch 数不同，通常有

\[
H_i^{\tau_i(t)}\ne H_j^{\tau_j(t)},
\qquad
\gamma_i^t\ne\gamma_j^t.
\]

不能把二者静默替换成共同的 $\gamma_p^t$。定义

\[
\bar\gamma_p^t
=\theta_p\gamma_i^t+(1-\theta_p)\gamma_j^t,
\]

\[
r_p^t=\theta_pr_i^t+(1-\theta_p)r_j^t.
\]

则

\[
\boxed{
d_p^t
=-\bar\gamma_p^tg_p^t
+r_p^t+r_{p,t}^{\mathrm{hist\mbox{-}step}},
}
\]

其中

\[
\boxed{
r_{p,t}^{\mathrm{hist\mbox{-}step}}
=-\theta_p(1-\theta_p)
(\gamma_i^t-\gamma_j^t)
\big[\nabla F_i(w^t)-\nabla F_j(w^t)\big].
}
\]

直接展开右端即可验证该恒等式。其范数满足

\[
\|r_{p,t}^{\mathrm{hist\mbox{-}step}}\|
\le
\theta_p(1-\theta_p)|\gamma_i^t-\gamma_j^t|
\|\nabla F_i(w^t)-\nabla F_j(w^t)\|.
\]

因此，数量偏斜不仅改变样本聚合权重，还通过 $H_i$ 和累计学习率 $\gamma_i$ 改变历史代理的尺度与方向。固定 local steps 并使用相同学习率可令该项为零；按 local epochs 训练时必须保留并测量。

记

\[
\|r_p^t\|\le\varepsilon_p^t,
\qquad
\|r_{p,t}^{\mathrm{hist\mbox{-}step}}\|
\le\zeta_p^t.
\]

---

## 8. 引理4：历史代理成为全局下降方向的条件

设

\[
\|g_p^t-g_t\|\le\delta_p^t.
\]

**引理4。** 若

\[
\boxed{
\|g_t\|
>
\delta_p^t+
\frac{\varepsilon_p^t+\zeta_p^t}{\bar\gamma_p^t},
}
\]

则

\[
\langle g_t,d_p^t\rangle<0.
\]

**证明。** 由引理3，

\[
\begin{aligned}
\langle g_t,d_p^t\rangle
&=-\bar\gamma_p^t\|g_t\|^2
-\bar\gamma_p^t\langle g_t,g_p^t-g_t\rangle\\
&\quad+\langle g_t,
r_p^t+r_{p,t}^{\mathrm{hist\mbox{-}step}}\rangle\\
&\le
-\bar\gamma_p^t\|g_t\|
\left[
\|g_t\|-\delta_p^t
-\frac{\varepsilon_p^t+\zeta_p^t}{\bar\gamma_p^t}
\right]
<0.
\end{aligned}
\]

定义有利余量

\[
m_p^t=-\langle g_t,d_p^t\rangle,
\]

则

\[
\boxed{
m_p^t
\ge
\bar\gamma_p^t\|g_t\|
\left[
\|g_t\|-\delta_p^t
-\frac{\varepsilon_p^t+\zeta_p^t}{\bar\gamma_p^t}
\right]
>0.
}
\]

引理1—2通过减小 $\delta_p^t$ 帮助该条件成立；历史陈旧性和数量偏斜通过 $\varepsilon_p^t,\zeta_p^t$ 阻碍条件成立。接近驻点时 $\|g_t\|$ 变小，条件可能失效，因此中心裁剪、后期 $\beta_t$ 衰减或代理门控具有理论动机。

---

## 9. 引理5：精确 proximal 把代理方向传入客户端

假设当轮 $\eta_t,\beta_t$ 在本地步内固定。proximal 一阶最优条件给出

\[
x_{i,h+1}
=(1-\rho_t)u_{i,h}+\rho_t\widetilde c_{p,t},
\]

\[
\rho_t
=\frac{2\eta_t\beta_t}{1+2\eta_t\beta_t}
\in[0,1).
\]

令 $e_{i,h}=x_{i,h}-w^t$，则

\[
\begin{aligned}
e_{i,h+1}
={}&(1-\rho_t)e_{i,h}
-\eta_t(1-\rho_t)\widehat v_{i,h}\\
&+\rho_t\chi_{p,t}(1-\xi_t)h_p^t
+\rho_t\chi_{p,t}\xi_ts_td_p^t.
\end{aligned}
\]

**引理5。** 执行 $H_i^t$ 步后，

\[
\boxed{
\Delta_i^t
=-\alpha_{i,t}\bar v_i^t
+\kappa_{i,t}h_p^t
+\lambda_{i,t}d_p^t,
}
\]

其中

\[
\alpha_{i,t}
=\eta_t(1-\rho_t)
\frac{1-(1-\rho_t)^{H_i^t}}{\rho_t},
\]

\[
\kappa_{i,t}
=\chi_{p,t}(1-\xi_t)
\big[1-(1-\rho_t)^{H_i^t}\big],
\]

\[
\boxed{
\lambda_{i,t}
=\chi_{p,t}\xi_ts_t
\big[1-(1-\rho_t)^{H_i^t}\big]\ge0,
}
\]

\[
\bar v_i^t
=
\frac{
\sum_{h=0}^{H_i^t-1}
(1-\rho_t)^{H_i^t-1-h}\widehat v_{i,h}
}{
\sum_{h=0}^{H_i^t-1}
(1-\rho_t)^{H_i^t-1-h}
}.
\]

当 $\rho_t=0$ 时取极限：

\[
\alpha_{i,t}=\eta_tH_i^t,
\qquad
\kappa_{i,t}=\lambda_{i,t}=0.
\]

所以共同中心不是在服务器聚合以后额外加入，而是在每个本地优化器步后通过 proximal 映射进入客户端轨迹。历史中心和代理方向分别以非负系数 $\kappa_{i,t}$、$\lambda_{i,t}$ 进入最终更新。

再加入并减去 $\nabla F_i(w^t)$：

\[
\Delta_i^t
=-\alpha_{i,t}\nabla F_i(w^t)
+\kappa_{i,t}h_p^t
+\lambda_{i,t}d_p^t
+r_{i,t}^{\mathrm{loc}},
\]

\[
r_{i,t}^{\mathrm{loc}}
=-\alpha_{i,t}
\big[\bar v_i^t-\nabla F_i(w^t)\big].
\]

### 9.1 当前轮数量偏斜的第二个步数失衡项

按 local epochs 训练时，$H_i^t\ne H_j^t$，因而 $\alpha,\kappa,\lambda$ 均可能不同。定义

\[
\bar\alpha_{p,t}
=\theta_p\alpha_{i,t}+(1-\theta_p)\alpha_{j,t},
\]

\[
\bar\kappa_{p,t}
=\theta_p\kappa_{i,t}+(1-\theta_p)\kappa_{j,t},
\qquad
\bar\lambda_{p,t}
=\theta_p\lambda_{i,t}+(1-\theta_p)\lambda_{j,t}.
\]

配对更新严格满足

\[
\boxed{
\Delta_p^t
=-\bar\alpha_{p,t}g_p^t
+\bar\kappa_{p,t}h_p^t
+\bar\lambda_{p,t}d_p^t
+\bar r_{p,t}^{\mathrm{loc}}
+r_{p,t}^{\mathrm{current\mbox{-}step}},
}
\]

\[
r_{p,t}^{\mathrm{current\mbox{-}step}}
=-\theta_p(1-\theta_p)
(\alpha_{i,t}-\alpha_{j,t})
\big[\nabla F_i(w^t)-\nabla F_j(w^t)\big].
\]

必须区分：

- $r_{p,t}^{\mathrm{hist\mbox{-}step}}$ 来自构造历史代理时过去的 $H_i,\gamma_i$ 不同；
- $r_{p,t}^{\mathrm{current\mbox{-}step}}$ 来自当前训练时的 $H_i^t,\alpha_{i,t}$ 不同。

二者都能用固定本地步数消除；在当前 local epochs 双偏斜实验中都不能省略。

---

## 10. 引理6：服务器保留有利下降投影

定义服务器更新中由梯度代理中心显式产生的分量

\[
P_t
=\sum_{p\in M_t}
\omega_{p,t}\bar\lambda_{p,t}d_p^t.
\]

**引理6。** 若每个被采用的配对满足引理4，且 $\bar\lambda_{p,t}\ge0$，则

\[
\boxed{
-\langle g_t,P_t\rangle
=\sum_{p\in M_t}
\omega_{p,t}\bar\lambda_{p,t}m_p^t
>0.
}
\]

**证明。**

\[
\begin{aligned}
-\langle g_t,P_t\rangle
&=-\left\langle
g_t,\sum_p\omega_{p,t}\bar\lambda_{p,t}d_p^t
\right\rangle\\
&=\sum_p\omega_{p,t}\bar\lambda_{p,t}
[-\langle g_t,d_p^t\rangle]\\
&=\sum_p\omega_{p,t}\bar\lambda_{p,t}m_p^t>0.
\end{aligned}
\]

结合引理4，

\[
\boxed{
\begin{aligned}
-\langle g_t,P_t\rangle
\ge
\sum_p\omega_{p,t}\bar\lambda_{p,t}\bar\gamma_p^t\|g_t\|
\left[
\|g_t\|-\delta_p^t
-\frac{\varepsilon_p^t+\zeta_p^t}{\bar\gamma_p^t}
\right].
\end{aligned}
}
\]

该结论解决的是“下降投影是否在聚合中抵消”。只要每项投影非负，它们作为标量相加时不会抵消。其他正交分量仍可能抵消或增大更新范数，所以引理6不声称完整向量范数被保留。

当前实现没有访问真实 $g_t$ 逐对门控。因此，“每一对都满足引理4”是理论充分条件和待验证机制条件，不是代码无条件保证。

---

## 11. 条件单轮加速定理

### 11.1 同一运行内部的代理贡献

引理5给出的 $P_t$ 是 FCPC-grad 当前实际更新中的显式代理分量。定义同一运行内部的其余分量

\[
B_t^{\mathrm{int}}
=\Delta_t^{\mathrm{grad}}-P_t.
\]

于是严格有

\[
\Delta_t^{\mathrm{grad}}
=B_t^{\mathrm{int}}+P_t.
\]

注意：$B_t^{\mathrm{int}}$ 不一定等于单独重跑得到的 mix0 更新，因为移除代理中心后，本地模型轨迹和后续任务梯度也会改变。

定义光滑下降收益

\[
\mathcal G_t(\Delta)
=-\langle g_t,\Delta\rangle
-\frac L2\|\Delta\|^2.
\]

由 $L$-smooth 性，

\[
F(w^t+\Delta)
\le F(w^t)-\mathcal G_t(\Delta).
\]

**定理1（内部代理分量的条件单轮增益）。**

\[
\boxed{
\begin{aligned}
\mathcal G_t(B_t^{\mathrm{int}}+P_t)
-\mathcal G_t(B_t^{\mathrm{int}})
={}&-\langle g_t,P_t\rangle\\
&-L\langle B_t^{\mathrm{int}},P_t\rangle
-\frac L2\|P_t\|^2.
\end{aligned}
}
\]

因此，只要

\[
\boxed{
-\langle g_t,P_t\rangle
>
L\langle B_t^{\mathrm{int}},P_t\rangle
+\frac L2\|P_t\|^2,
}
\]

代理分量就在该轮提高了光滑下降保证。

引理6控制左侧的一阶收益；中心裁剪和 $\beta_t$ 控制 $P_t$ 的幅度。方向正确仍不够，因为过大的 $P_t$ 会使二阶代价 $L\|P_t\|^2/2$ 超过一阶收益，表现为越过有效下降区域、振荡或不稳定。

### 11.2 相对 mix0 或任意基线

对实际参考算法 $B$，定义同一检查点和随机协议下的反事实差值

\[
Z_t^{(B)}
=\Delta_t^{\mathrm{grad}}-\Delta_t^{(B)}.
\]

这是精确差值。相对任意基线的收益差为

\[
\boxed{
\begin{aligned}
\mathcal G_t(\Delta_t^{\mathrm{grad}})
-\mathcal G_t(\Delta_t^{(B)})
={}&-\langle g_t,Z_t^{(B)}\rangle\\
&-L\langle\Delta_t^{(B)},Z_t^{(B)}\rangle
-\frac L2\|Z_t^{(B)}\|^2.
\end{aligned}
}
\]

相对 mix0，

\[
Z_t^{(\mathrm{mix0})}
=P_t+e_t^{\mathrm{trajectory}},
\]

其中 $e_t^{\mathrm{trajectory}}$ 表示中心改变以后，本地轨迹和后续任务梯度随之变化产生的响应。引理6只直接控制 $P_t$，所以要把引理6传到真实 mix0 对比，还需控制或实测 $e_t^{\mathrm{trajectory}}$。

该定理不局限于 FedAvg。将 $B$ 依次取为 FedAvg、FedProx、MOON、FedDyn、FBLG 或 FedCFA，可得到同一比较准则。但理论因果归因最清楚的是 mix0，因为它只移除梯度代理中心；不同算法之间的最终比较主要由公平实验支持。

---

## 12. 非凸有限轮与 PL 条件加速

### 12.1 光滑非凸有限轮

若存在 $c_t>0$ 和 $\epsilon_t\ge0$，使

\[
\mathbb E_t\mathcal G_t(\Delta_t^{\mathrm{grad}})
\ge c_t\|g_t\|^2-\epsilon_t,
\]

则

\[
\mathbb E_tF(w^{t+1})
\le F(w^t)-c_t\|g_t\|^2+\epsilon_t.
\]

求和得到

\[
\boxed{
\frac{
\sum_{t=0}^{T-1}c_t
\mathbb E\|\nabla F(w^t)\|^2
}{
\sum_{t=0}^{T-1}c_t
}
\le
\frac{
F(w^0)-F_{\inf}+\sum_t\epsilon_t
}{
\sum_{t=0}^{T-1}c_t
}.
}
\]

若 FCPC-grad 在一段训练区间内具有更大的 $c_t$，且没有增加更大的累计 $\epsilon_t$，则其有限轮平均梯度范数界更紧。这支持有限通信轮加速，但不改变一般非凸随机优化的渐近阶。

### 12.2 PL 条件下的收缩

若研究区域还满足

\[
\|\nabla F(w)\|^2
\ge2\mu[F(w)-F^*],
\]

则

\[
\mathbb E_t[F(w^{t+1})-F^*]
\le
(1-2\mu c_t)[F(w^t)-F^*]
+\epsilon_t.
\]

在同一误差水平下，更大的 $c_t$ 给出更小的条件收缩因子。对 ResNet-18 不能声称全局 PL 成立；该结论只能作为局部或条件加速定理。

---

## 13. 加权 JS 如何进入服务器完整误差

当 $\xi_t=1$ 时历史中心项消失。将引理3代入引理5，并把所有已列残差合并为 $e_p^t$：

\[
\Delta_p^t
=-\eta_{p,t}^{\mathrm{eff}}g_p^t+e_p^t,
\]

\[
\eta_{p,t}^{\mathrm{eff}}
=\bar\alpha_{p,t}
+\bar\lambda_{p,t}\bar\gamma_p^t.
\]

定义

\[
\bar\eta_t
=\sum_p\omega_{p,t}\eta_{p,t}^{\mathrm{eff}},
\qquad
V_{\eta,t}
=\sum_p\omega_{p,t}
(\eta_{p,t}^{\mathrm{eff}}-\bar\eta_t)^2.
\]

完整参与时，

\[
\Delta_t
=-\bar\eta_tg_t+e_t^{\mathrm{server}}.
\]

有效步长不一致和配对梯度偏差的耦合项满足

\[
\left\|
\sum_p\omega_{p,t}
(\eta_{p,t}^{\mathrm{eff}}-\bar\eta_t)
(g_p^t-g_t)
\right\|^2
\le V_{\eta,t}H_t(M_t)
\le2G^2V_{\eta,t}R_t(M_t).
\]

因此，加权 JS 控制的是服务器误差中的明确一项：**有效步长异质性与配对梯度偏差的耦合误差**。它不控制全部随机噪声、陈旧性、动量、历史中心偏差和部分参与误差。

部分参与时还需加入

\[
e_t^{\mathrm{sampling}}
=-\bar\eta_t(g_{\mathcal S_t}^t-g_t).
\]

奇数客户端还需加入未配对项。论文应先给出完整参与的清晰定理，再显式扩展 sampling、unpaired 和 step-imbalance residual。

---

## 14. $A_t(M),U_t(M),D_t(M)$ 的定位

这些量是发现算法问题的诊断工具，不再单独承担加速证明。

令

\[
r_p^t=\Delta_p^t+\gamma_tg_p^t,
\]

\[
A_t(M)=\sum_p\omega_{p,t}\|r_p^t\|^2,
\qquad
U_t(M)=\left\|\sum_p\omega_{p,t}r_p^t\right\|^2.
\]

加权方差恒等式给出

\[
\boxed{0\le U_t(M)\le A_t(M).}
\]

不同残差可能抵消，所以 $A_t$ 下降不保证单次观测的 $U_t$ 下降，也不存在无方向条件的正常数下界 $U_t\ge\kappa A_t$。

定义共同中心相对 FedAvg 改变服务器更新的大小

\[
D_t(M)
=\left\|
\sum_i a_{i,t}
[\Delta_i^t(M)-\Delta_i^t(\mathrm{FedAvg})]
\right\|.
\]

归一化 $D_t(M)$ 实验显示，旧共同中心对服务器更新的改变不足百分之一，且方向收益不稳定。这个负结果促成了 FCPC-grad：研究重点从“只缩小残差范数上界”转为“让服务器保留有利下降投影”。

---

## 15. 实验与理论的对应关系

### 15.1 mix0 因果消融

两种子、200轮实验中，加入梯度代理后：

\[
\operatorname{AUC}_{50}:
30.21\%\pm2.51\%
\longrightarrow
39.07\%\pm3.17\%,
\]

\[
\text{selected test accuracy}:
79.52\%\pm0.37\%
\longrightarrow
80.50\%\pm0.78\%.
\]

两组只改变 $\xi$，因此主要支持“梯度代理中心有用”。目前只有两个种子，仍需补充。

### 15.2 三种子完整基线比较

CIFAR-10、dual-skew $\alpha=0.1$、每轮6客户端、200轮、种子45—47下：

\[
\operatorname{AUC}_{50}^{\mathrm{FCPC\mbox{-}grad}}
=36.29\%\pm3.29\%,
\]

\[
\operatorname{AUC}_{100}^{\mathrm{FCPC\mbox{-}grad}}
=45.42\%\pm3.61\%.
\]

FedAvg 对应为

\[
27.29\%\pm1.25\%,
\qquad
36.95\%\pm2.44\%.
\]

FCPC-grad 在三个种子的 AUC50 和 AUC100 上均优于当前比较中的全部基线。修正 FedAvg seed46 日志后，FCPC-grad 与 FedAvg 的所选测试准确率约为

\[
77.60\%\pm3.40\%,
\qquad
76.64\%\pm4.01\%.
\]

现有证据支持：FCPC-grad 改善了**按通信轮数衡量的前中期收敛，并保持可比的最终精度**。三个种子不足以声称最终精度具有显著优势。

### 15.3 时间与通信代价

FCPC-grad 每轮约 $13.06$ 秒，FedAvg 约 $8.90$ 秒；200轮通信量约为 $161.04$ GB 与 $107.36$ GB。因此应写 communication-round convergence 或有限轮收敛改善，不能笼统声称墙钟时间和通信字节都减少。

---

## 16. 结论边界与待办

### 16.1 可以严格写入

1. 加权 KL–JS 恒等式和最大权匹配等价目标；
2. 纯标签偏移下的标签分布—配对梯度偏差上界；
3. 包含 $H_i\ne H_j,\gamma_i\ne\gamma_j$ 的历史代理严格分解；
4. 历史代理成为全局下降方向的充分条件；
5. 精确 proximal 以显式非负系数传递历史中心和代理方向；
6. 逐对下降条件成立时，服务器保留代理的有利投影；
7. 一阶收益超过二阶平滑代价时的条件单轮增益；
8. 光滑非凸有限轮界和 PL 条件下的局部收缩解释。

### 16.2 不能写成无条件结论

1. JSD 或 JSDN 越大就必然收敛越快；
2. 标签分布混合等价于神经网络参数平均；
3. 历史更新是当前全局梯度的无偏估计；
4. 所有配对在所有轮次都满足引理4；
5. ResNet-18 目标全局强凸或全局满足 PL；
6. 三个种子证明最终准确率显著提高；
7. 通信轮数减少等价于墙钟时间或通信字节减少；
8. LDP 扰动不会改变真实最优配对。

### 16.3 提交前必须补齐

1. 记录 $H_i^t,\gamma_i^t$，量化历史和当前两个 step-imbalance residual；
2. 在冻结检查点记录 $\langle g_t,d_p^t\rangle$、有利配对比例和代理余量；
3. 记录 $\chi_{p,t}$、$\|P_t\|$、轨迹响应和单轮收益差；
4. 把 mix0 补到至少三个种子；
5. 明确各基线的来源、适配差异和调参预算；
6. 同时报告 AUC、阈值轮数、最终精度、秒/轮和通信量；
7. 将完整参与主定理扩展为部分参与的条件期望版本；
8. 对奇数参与客户端给出未配对策略和残差。

---

## 17. 论文推荐结构

方法部分：

1. 双偏斜问题与原 JSDN 缺口；
2. 加权 JS 互补收益；
3. 最大权匹配；
4. 历史更新代理与梯度共同中心；
5. 精确 proximal；
6. 样本量加权服务器聚合。

理论附录：

1. 引理1：分布恒等式；
2. 引理2：梯度桥梁；
3. 引理3：含 quantity skew 的历史代理分解；
4. 引理4：代理下降充分条件；
5. 引理5：proximal 传递；
6. 引理6：服务器投影保留；
7. 条件单轮加速；
8. 非凸有限轮与 PL 推论；
9. sampling、unpaired、LDP 和 step-imbalance residual。

实验部分：

1. mix0 因果消融；
2. 配对策略消融；
3. $\beta,\xi,s$ 与裁剪消融；
4. 三种子基线比较；
5. 轮数、墙钟时间和通信代价分开报告；
6. $R,H,D,U$ 与代理下降投影机制指标。

---

## 18. 使用的标准理论工具

- FedAvg：本地多步训练与服务器加权聚合；
- proximal 方法：二次邻近项和精确 proximal 映射；
- Pinsker 不等式：由 KL 控制标签分布的 $L_1$ 距离；
- 条件期望分解：纯标签偏移下按类别重写客户端梯度；
- 光滑下降引理：将更新方向和范数转化为单轮下降保证；
- Polyak–Łojasiewicz 条件：将梯度范数下降转化为条件收缩；
- 一般图最大权匹配：精确优化加权 JS 总互补收益。

这些工具只支持各自对应的推导。FCPC-grad 的加权 JS 目标、历史代理分解、proximal 传递和方向保留链必须由本文完整证明，不能用引用替代。
