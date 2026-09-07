# FCPC-grad 引理4—6：梯度代理、客户端传递与服务器保留

## 1. 目标与符号

本文件补齐以下证明链：

\[
\text{历史更新代理对全局目标有利}
\Longrightarrow
\text{有利方向进入客户端更新}
\Longrightarrow
\text{服务器保留其下降投影}
\Longrightarrow
\text{条件单轮加速}.
\]

第 \(t\) 轮全局模型与全局梯度记为

\[
w^t,\qquad g_t=\nabla F(w^t).
\]

匹配 \(M_t\) 中一对客户端 \(p=(i,j)\) 的内部权重、全局权重和配对梯度为

\[
\theta_p=\frac{N_i}{N_i+N_j},\qquad
b_p=\frac{N_i+N_j}{N},
\]

\[
g_p^t=\theta_p\nabla F_i(w^t)+(1-\theta_p)\nabla F_j(w^t).
\]

完整参与和完整匹配下有

\[
\sum_{p\in M_t}b_p=1,
\qquad
\sum_{p\in M_t}b_pg_p^t=g_t.
\]

以下结论使用：梯度 Lipschitz、随机梯度方差有界、类别条件梯度有界，以及历史更新陈旧性和本地漂移可控。纯标签偏移条件下，前两层证明给出

\[
H_t(M_t):=\sum_{p\in M_t}b_p\|g_p^t-g_t\|^2
\le 2G^2R(M_t).
\]

部分参与、未配对客户端和不同累计本地步长应分别加入 sampling、unpaired 和 step-imbalance residual；不能在证明中静默删除。

## 2. 历史更新代理分解

客户端 \(i\) 最近一次参与时，从广播模型 \(x_{i,0}=w^{\tau_i(t)}\) 出发执行 \(H_i\) 步 SGD：

\[
x_{i,h+1}=x_{i,h}-\eta_{i,h}\widehat\nabla F_i(x_{i,h};\xi_{i,h}).
\]

模型增量望远镜求和得到严格恒等式

\[
d_i^t=x_{i,H_i}-x_{i,0}
=-\sum_{h=0}^{H_i-1}\eta_{i,h}\widehat\nabla F_i(x_{i,h};\xi_{i,h}).
\]

令 \(\gamma_i^t=\sum_h\eta_{i,h}\)，加入再减去 \(\nabla F_i(w^t)\)，可写为

\[
d_i^t=-\gamma_i^t\nabla F_i(w^t)+r_i^t,
\]

\[
r_i^t=-\sum_h\eta_{i,h}
\left[\widehat\nabla F_i(x_{i,h};\xi_{i,h})-\nabla F_i(w^t)\right].
\]

其中 \(r_i^t\) 同时包含随机梯度噪声、本地多步漂移和从 \(w^{\tau_i(t)}\) 到 \(w^t\) 的陈旧误差。若配对两端累计步长相同为 \(\gamma_p^t\)，则

\[
d_p^t=\theta_pd_i^t+(1-\theta_p)d_j^t
=-\gamma_p^tg_p^t+r_p^t,
\]

\[
r_p^t=\theta_pr_i^t+(1-\theta_p)r_j^t.
\]

假设或由光滑性、方差与陈旧性界推出

\[
\|r_p^t\|\le\varepsilon_p^t.
\]

## 3. 引理4：配对历史更新的全局下降性

**引理4。** 若

\[
\|g_p^t-g_t\|\le\delta_p^t,
\qquad
\|r_p^t\|\le\varepsilon_p^t,
\]

且

\[
\|g_t\|>\delta_p^t+\frac{\varepsilon_p^t}{\gamma_p^t},
\]

则 \(d_p^t\) 是全局目标在 \(w^t\) 处的一阶下降方向：

\[
\langle g_t,d_p^t\rangle<0.
\]

**证明。** 由 \(d_p^t=-\gamma_p^tg_p^t+r_p^t\)，

\[
\begin{aligned}
\langle g_t,d_p^t\rangle
&=-\gamma_p^t\|g_t\|^2
-\gamma_p^t\langle g_t,g_p^t-g_t\rangle
+\langle g_t,r_p^t\rangle\\
&\le-\gamma_p^t\|g_t\|^2
+\gamma_p^t\|g_t\|\delta_p^t
+\|g_t\|\varepsilon_p^t\\
&=-\gamma_p^t\|g_t\|
\left(\|g_t\|-\delta_p^t-\frac{\varepsilon_p^t}{\gamma_p^t}\right)<0.
\end{aligned}
\]

定义配对代理的有用性余量

\[
m_p^t:=-\langle g_t,d_p^t\rangle.
\]

于是

\[
m_p^t\ge
\gamma_p^t\|g_t\|
\left(\|g_t\|-\delta_p^t-\frac{\varepsilon_p^t}{\gamma_p^t}\right)>0.
\]

纯标签偏移下可取

\[
\delta_p^t=G\|q_p-\bar p\|_1.
\]

引理4是充分条件而非无条件保证；训练接近驻点时 \(\|g_t\|\) 变小，该条件可能失效，这为后期衰减 \(\beta_t\) 或使用方向门控提供依据。

## 4. 引理5：精确 proximal 将代理方向传入客户端

配对梯度共同中心固定为

\[
c_p^t=w^t+s_td_p^t.
\]

客户端 \(i\in p\) 每一步先执行任务更新

\[
u_{i,h}=x_{i,h}-\eta_t\widehat g_{i,h},
\]

再执行精确 proximal 映射

\[
x_{i,h+1}=\arg\min_x
\left\{
\frac{1}{2\eta_t}\|x-u_{i,h}\|^2
+\beta_t\|x-c_p^t\|^2
\right\}.
\]

**引理5。** 令

\[
\rho_t=\frac{2\eta_t\beta_t}{1+2\eta_t\beta_t}\in[0,1),
\]

则执行 \(H_i\) 步后的客户端更新严格满足

\[
\Delta_i^t=x_{i,H_i}-w^t
=-\alpha_{i,t}\bar g_i^t+\lambda_{i,t}d_p^t,
\]

其中

\[
\alpha_{i,t}=\eta_t(1-\rho_t)
\frac{1-(1-\rho_t)^{H_i}}{\rho_t},
\]

\[
\lambda_{i,t}=s_t[1-(1-\rho_t)^{H_i}]\ge0,
\]

\[
\bar g_i^t=
\frac{\sum_{h=0}^{H_i-1}(1-\rho_t)^{H_i-1-h}\widehat g_{i,h}}
{\sum_{h=0}^{H_i-1}(1-\rho_t)^{H_i-1-h}}.
\]

当 \(\rho_t=0\) 时按极限解释：\(\alpha_{i,t}=\eta_tH_i\)，\(\lambda_{i,t}=0\)。

**证明。** proximal 一阶最优条件给出

\[
x_{i,h+1}=(1-\rho_t)u_{i,h}+\rho_tc_p^t.
\]

令 \(e_{i,h}=x_{i,h}-w^t\)，则 \(e_{i,0}=0\)，并且

\[
e_{i,h+1}=(1-\rho_t)e_{i,h}
-\eta_t(1-\rho_t)\widehat g_{i,h}
+\rho_ts_td_p^t.
\]

递推展开 \(H_i\) 次后，

\[
\begin{aligned}
e_{i,H_i}
={}&-\eta_t(1-\rho_t)
\sum_{h=0}^{H_i-1}(1-\rho_t)^{H_i-1-h}\widehat g_{i,h}\\
&+s_t[1-(1-\rho_t)^{H_i}]d_p^t,
\end{aligned}
\]

代入上述定义即得结论。

为了进入全局分析，再加入并减去轮初真实梯度：

\[
\Delta_i^t=-\alpha_{i,t}\nabla F_i(w^t)
+\lambda_{i,t}d_p^t+r_{i,t}^{\mathrm{loc}},
\]

\[
r_{i,t}^{\mathrm{loc}}
=-\alpha_{i,t}[\bar g_i^t-\nabla F_i(w^t)].
\]

因此，引理4的下降方向以非负强度 \(\lambda_{i,t}\) 进入客户端最终更新，其有利投影为

\[
-\langle g_t,\lambda_{i,t}d_p^t\rangle
=\lambda_{i,t}m_p^t\ge0.
\]

### 4.1 数量偏斜导致不同本地步数时的精确配对分解

当前实验按本地 epoch 训练，因此数量偏斜会导致 \(H_i\ne H_j\)，进而通常有

\[
\alpha_{i,t}\ne\alpha_{j,t},
\qquad
\lambda_{i,t}\ne\lambda_{j,t}.
\]

不能为了得到简洁公式而假装两端系数相同。令

\[
\bar\alpha_{p,t}=\theta_p\alpha_{i,t}+(1-\theta_p)\alpha_{j,t},
\]

\[
\bar\lambda_{p,t}=\theta_p\lambda_{i,t}+(1-\theta_p)\lambda_{j,t}.
\]

对两个客户端更新加权后严格有

\[
\begin{aligned}
\Delta_p^t
={}&\theta_p\Delta_i^t+(1-\theta_p)\Delta_j^t\\
={}&-\bar\alpha_{p,t}g_p^t+\bar\lambda_{p,t}d_p^t
+\bar r_{p,t}^{\mathrm{loc}}+r_{p,t}^{\mathrm{step}},
\end{aligned}
\]

其中

\[
\bar r_{p,t}^{\mathrm{loc}}
=\theta_pr_{i,t}^{\mathrm{loc}}+(1-\theta_p)r_{j,t}^{\mathrm{loc}},
\]

\[
\boxed{
r_{p,t}^{\mathrm{step}}
=-\theta_p(1-\theta_p)(\alpha_{i,t}-\alpha_{j,t})
[\nabla F_i(w^t)-\nabla F_j(w^t)].
}
\]

因此

\[
\|r_{p,t}^{\mathrm{step}}\|
\le\theta_p(1-\theta_p)|\alpha_{i,t}-\alpha_{j,t}|
\|\nabla F_i(w^t)-\nabla F_j(w^t)\|.
\]

这一项明确表示 quantity skew 如何通过不同本地步数放大配对梯度差异。固定 local steps 会令该项消失；按 local epochs 训练时则必须保留并估计它。

## 5. 引理6：服务器保留的有利投影

先定义仅由梯度共同中心产生的服务器代理分量

\[
P_t=\sum_{p\in M_t}b_p\lambda_{p,t}d_p^t.
\]

**引理6（方向保留）。** 若每个配对满足引理4且 \(\lambda_{p,t}\ge0\)，则

\[
-\langle g_t,P_t\rangle
=\sum_{p\in M_t}b_p\lambda_{p,t}m_p^t>0.
\]

**证明。** 由内积对加法的线性性，

\[
\begin{aligned}
-\langle g_t,P_t\rangle
&=-\left\langle g_t,\sum_pb_p\lambda_{p,t}d_p^t\right\rangle\\
&=\sum_pb_p\lambda_{p,t}[-\langle g_t,d_p^t\rangle]\\
&=\sum_pb_p\lambda_{p,t}m_p^t.
\end{aligned}
\]

所有项非负时，沿 \(-g_t\) 的有利投影不会在服务器聚合中抵消。其他正交方向的向量分量仍可能抵消，所以本引理证明的是“下降投影保留”，不是完整向量范数保留。

结合引理4可得显式下界

\[
\boxed{
-\langle g_t,P_t\rangle
\ge
\sum_pb_p\lambda_{p,t}\gamma_p^t\|g_t\|
\left(\|g_t\|-\delta_p^t-\frac{\varepsilon_p^t}{\gamma_p^t}\right).
}
\]

## 6. 从方向保留到服务器完整更新

将上一节的不同步数分解与代理分解合并，一对客户端的有效更新可写成

\[
\Delta_p^t=-\eta_{p,t}^{\mathrm{eff}}g_p^t+e_p^t,
\qquad
\eta_{p,t}^{\mathrm{eff}}
=\bar\alpha_{p,t}+\bar\lambda_{p,t}\gamma_p^t,
\]

其中 \(e_p^t\) 显式包含 \(\bar r_{p,t}^{\mathrm{loc}}\)、\(r_{p,t}^{\mathrm{step}}\)、\(\bar\lambda_{p,t}r_p^t\) 以及累计历史步长不一致时的代理 step-imbalance residual。

定义

\[
\bar\eta_t=\sum_pb_p\eta_{p,t}^{\mathrm{eff}},
\qquad
V_{\eta,t}=\sum_pb_p(\eta_{p,t}^{\mathrm{eff}}-\bar\eta_t)^2.
\]

服务器聚合为

\[
\Delta_t=-\bar\eta_tg_t+e_t^{\mathrm{server}},
\]

\[
e_t^{\mathrm{server}}
=\sum_pb_pe_p^t
-\sum_pb_p(\eta_{p,t}^{\mathrm{eff}}-\bar\eta_t)(g_p^t-g_t).
\]

有效步长不一致导致的误差满足

\[
\left\|
\sum_pb_p(\eta_{p,t}^{\mathrm{eff}}-\bar\eta_t)(g_p^t-g_t)
\right\|^2
\le V_{\eta,t}H_t(M_t)
\le2G^2V_{\eta,t}R(M_t).
\]

该式说明加权 JS 匹配控制的是服务器误差中的“有效步长异质性乘配对梯度偏差”部分，而不是无条件控制所有误差。若所有配对的有效步长相同，则该项为零，理想配对梯度的加权和本来就等于全局梯度；此时匹配主要通过降低非线性本地轨迹和代理残差发挥作用。

## 7. 与 FedAvg 的条件单轮加速

定义一轮更新的下降收益

\[
\mathcal G_t(\Delta)
=-\langle g_t,\Delta\rangle-\frac{L}{2}\|\Delta\|^2.
\]

由 \(L\)-smooth 性，

\[
F(w^t+\Delta)\le F(w^t)-\mathcal G_t(\Delta).
\]

因此，FCPC-grad 相对 FedAvg 的直接充分条件是

\[
\boxed{
\mathbb E_t\mathcal G_t(\Delta_t^{\mathrm{grad}})
>
\mathbb E_t\mathcal G_t(\Delta_t^{\mathrm{FA}})>0.
}
\]

它同时要求：

1. 引理6保留的有利投影足够大；
2. proximal 对原任务步的衰减不能超过代理带来的收益；
3. 更新幅度不能因代理注入而过大；
4. proxy、local、step-imbalance 和 sampling residual 受控。

因此正确结论是“满足上述可检验条件时，FCPC-grad 的单轮下降下界严格优于 FedAvg”，而不是无条件宣称所有轮次、所有任务都更快。

## 8. 对应实验诊断量

为了验证证明条件，应在相同冻结检查点下记录：

\[
m_p^t=-\langle g_t,d_p^t\rangle,
\qquad
\cos(-d_p^t,g_t),
\]

\[
-\langle g_t,P_t\rangle,
\qquad
\frac{-\langle g_t,P_t\rangle}{\sum_pb_p\lambda_{p,t}\|d_p^t\|\|g_t\|+\varepsilon},
\]

以及

\[
\mathcal G_t(\Delta_t^{\mathrm{grad}})
-\mathcal G_t(\Delta_t^{\mathrm{FA}}).
\]

这些量分别验证“代理有用”“服务器保留”和“实际单轮下降界改善”，比只看模型参数距离或最终准确率更直接。
