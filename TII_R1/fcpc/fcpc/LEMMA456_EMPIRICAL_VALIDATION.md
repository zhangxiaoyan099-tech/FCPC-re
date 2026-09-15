# 引理 4–6 运行条件与条件加速不等式验证

## 1. 本实验回答什么

本实验不把最终准确率当成理论条件，也不声称有限次神经网络实验能够证明定理。它在同一个冻结检查点上逐项审计：

1. 历史配对代理是否满足引理 4 的充分下降条件；
2. exact proximal 是否以非负系数把代理方向传入客户端更新；
3. 有利标量投影经过服务器加权聚合后是否被保留；
4. 与完全相同随机协议下的 `mix0` 反事实相比，FCPC-grad 是否具有正的单轮净收益；
5. 多个随机 mini-batch 轨迹的均值是否支持条件不等式。

需要验证的量纲一致形式是

\[
\boxed{
\mathbb E_t[Q_t]
\ge q\|\nabla F(w^t)\|^2-\varepsilon_t^P.
}
\]

右侧必须是梯度范数平方。因为单轮光滑下降项为“有效步长乘梯度范数平方”，使用未平方的梯度范数不能直接进入现有非凸递推。

## 2. 冻结检查点与公平反事实

中性轨迹采用全参与 FedAvg。为暂时排除数量偏斜造成的步数差异，历史轮和回放轮均采用：

- 固定本地步数；
- 相同学习率；
- SGD，且 momentum 与 weight decay 均为 0；
- 中性预热正常更新 BatchNorm 运行统计；冻结检查点后的两次反事实回放才冻结这些统计，使两条回放路径使用完全相同的缓冲量；
- 所有客户端覆盖一次且服务器按样本量聚合。

检查点同时保存上一轮客户端终点 \(v_i^{\tau_i}\) 和当时服务器广播起点 \(b_i^{\tau_i}\)，因此历史更新代理可以被直接重建：

\[
d_i^t=v_i^{\tau_i}-b_i^{\tau_i}.
\]

在同一 \(w^t\)、同一配对和同一批次顺序下回放两次：

- `mix0`：\(\xi=0\)，只使用历史参数共同中心；
- `grad`：\(\xi=1\)，使用 \(w^t+s_td_p^t\) 梯度代理中心。

两次服务器更新之差是可观测反事实量：

\[
Z_t=\Delta_t^{\mathrm{grad}}-\Delta_t^{\mathrm{mix0}}.
\]

CIFAR-10 配置按照正式训练的 200 轮 cosine schedule，在每个冻结检查点使用下一轮对应的 \(\beta_t\)，而不是把 0.2 强行固定到所有阶段。

## 3. 引理 4：逐对运行条件

对配对 \(p=(i,j)\)，令

\[
d_p^t=\theta_pd_i^t+(1-\theta_p)d_j^t,
\qquad
g_p^t=\theta_p\nabla F_i(w^t)+(1-\theta_p)\nabla F_j(w^t).
\]

控制实验中历史累计步长是预先确定的

\[
\gamma=\eta_{\mathrm{hist}}H_{\mathrm{hist}},
\]

而不是根据观测到的 \(d_p^t\) 事后拟合。记录

\[
\delta_p^t=\|g_p^t-g_t\|,
\qquad
\varepsilon_p^t=\|d_p^t+\gamma g_p^t\|.
\]

本次固定步数实验令历史步数失衡项 \(\zeta_p^t=0\)。逐对检查

\[
\|g_t\|>
\delta_p^t+\frac{\varepsilon_p^t}{\gamma}.
\]

同时直接测量并校验

\[
m_p^t=-\langle g_t,d_p^t\rangle
\ge
\gamma\|g_t\|
\left(
\|g_t\|-\delta_p^t-\frac{\varepsilon_p^t}{\gamma}
\right).
\]

充分条件可能不成立，但 \(m_p^t>0\) 仍可能成立；前者是保守的可证门槛，后者是代理方向实际有利的必要观测。

## 4. 引理 5：proximal 传递系数

一次 exact proximal 的收缩因子为

\[
1-\rho_t=\frac{1}{1+2\eta_t\beta_{i,t}}.
\]

执行 \(H\) 步后，代理方向进入客户端最终更新的系数为

\[
\lambda_{i,t}
=\xi_ts_t\chi_{p,t}
\left[1-(1-\rho_{i,t})^H\right]\ge0,
\]

其中 \(\chi_{p,t}\in[0,1]\) 是中心裁剪比例。实验记录每个客户端的 `lambda_i`、`lambda_j` 以及配对平均 `lambda_bar`。这一步主要验证公式和实现一致；非负性来自 \(\eta,\beta,s,\xi,\chi\ge0\)，不是统计结论。

## 5. 引理 6：服务器保留有利投影

显式代理分量为

\[
P_t=\sum_p\omega_{p,t}\bar\lambda_{p,t}d_p^t.
\]

实验同时从向量直接计算左侧、从逐对余量计算右侧：

\[
-\langle g_t,P_t\rangle
=\sum_p\omega_{p,t}\bar\lambda_{p,t}m_p^t.
\]

`lemma6_identity_gap` 应只剩浮点误差。`lemma6_projection > 0` 表示服务器保留了有利的一阶投影；它不等于完整单轮净收益为正，因为还存在交叉项和二阶代价。

## 6. 单轮净收益与 \(\varepsilon_t^P\)

给定候选光滑常数 \(L\)，相对 `mix0` 的光滑代理收益为

\[
Q_t(Z_t)
=-\langle g_t,Z_t\rangle
-L\langle\Delta_t^{\mathrm{mix0}},Z_t\rangle
-\frac L2\|Z_t\|^2.
\]

把实际反事实修正写成

\[
Z_t=P_t+e_t^P.
\]

这里 \(e_t^P\) 不仅包含本地轨迹响应，也包含 `mix0` 历史中心被梯度中心替换产生的其余差异。代码采用三角不等式给出可计算上界

\[
\boxed{
\varepsilon_t^P=
\big(\|g_t\|+L\|\Delta_t^{\mathrm{mix0}}\|+L\|P_t\|\big)\|e_t^P\|
+\frac L2\|e_t^P\|^2.
}
\]

于是

\[
Q_t(Z_t)\ge Q_t(P_t)-\varepsilon_t^P.
\]

若一个在开发集上预先冻结的 \(q>0\) 满足

\[
Q_t(P_t)\ge q\|g_t\|^2,
\]

便得到目标式。CSV 同时记录 `proxy_q_condition_holds` 和 `inequality_holds`，避免仅仅因为 \(\varepsilon_t^P\) 很大而把目标不等式判为成立。

此外直接在与梯度探针相同的固定训练目标上计算

\[
Q_t^{\mathrm{obs}}
=F(w^t+\Delta_t^{\mathrm{mix0}})
-F(w^t+\Delta_t^{\mathrm{grad}}).
\]

`Q_observed_loss_gain > 0` 表示实际单轮训练目标更低，但它与基于未知全局 \(L\) 的理论下界必须分开报告。验证集损失和准确率另行记录，只用于观察泛化，不代替 \(F\)。

## 7. 两阶段验证，避免循环论证

第一阶段使用 seed 42，配置中的 `q_candidate: 0.0` 只用于测量可行范围。重点查看每个检查点的 `Q_proxy_vs_mix0 / ||g||^2`、`Q_over_grad_sq` 及其波动。不能在 seed 42 上选择 \(q\) 后又把同一批结果当作独立验证。

第二阶段在 seed 43、44 等未参与定标的数据上冻结同一个正 \(q\)。建议取开发阶段条件成立样本中低分位数的保守折扣值，而不是最大值。例如：

\[
q=0.5\times
Q_{0.1}\left(
\frac{Q_t(P_t)}{\|g_t\|^2}
\right).
\]

随后报告：

- 引理 4 充分条件成立的配对比例；
- 实际有利代理比例；
- 引理 6 投影为正的检查点比例；
- `proxy_q_condition_holds` 比例；
- 目标不等式在固定 \(q\) 下的成立比例；
- \(Q_t\) 和 \(Q_t^{\mathrm{obs}}\) 的均值、标准差和置信区间。

只有当 \(q\) 在独立种子上仍为正且余项没有吞没收益时，才能把它写成“实验支持运行条件”；仍不能写成对 ResNet-18 全局光滑常数或 PL 条件的无条件证明。

## 8. 服务器运行命令

先运行单元测试和合成冒烟：

```bash
cd ~/FCPC-re/TII_R1/fcpc/fcpc
conda activate fcpc-core

python -m unittest tests.test_lemma456_metrics -v
python -u -m scripts.run_lemma456_audit \
  --config configs/lemma456/smoke_synthetic_lemma456.yaml
```

再运行 CIFAR-10 seed 42 快速定标实验：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_lemma456_audit \
  --config configs/lemma456/cifar10_lemma456_quick_seed42.yaml \
  2>&1 | tee outputs/lemma456_quick_seed42_console.txt
```

汇总主方法 `raw + optimal`：

```bash
python -m scripts.summarize_lemma456 \
  outputs/lemma456_quick_seed42/lemma456_metrics.csv \
  --panel raw --strategy optimal
```

输出文件：

- `lemma456_pairs.csv`：逐对引理 4 条件与 \(\lambda\)；
- `lemma456_metrics.csv`：逐检查点、随机轨迹和 \(L\) 的引理 6 与 \(Q_t\)；
- `lemma456_summary.csv`：对 batch seeds 近似条件期望后的汇总。

## 9. 目前限制

1. `gradient_max_batches: 20` 是快速梯度估计；正式实验应改为 `null` 并使用完整客户端训练集。
2. 正值 \(L\) 目前只是敏感性网格。若没有独立成立的局部光滑上界，不能选择“最有利的 \(L\)”作为证明。
3. 条件期望目前通过固定检查点下改变 batch seed 近似；正式版本应增加 batch seeds 和模型 seeds。
4. 本实验先固定 \(H_i=H_j\) 与 \(\gamma_i=\gamma_j\)。恢复按 local epochs 训练后，必须重新加入历史与当前步数失衡项。
5. `mix0` 是实际算法中的 \(\xi=0\) 历史中心版本。因此 \(e_t^P\) 包含“替换旧中心”以及轨迹变化，不能只称为纯 SGD 噪声。
6. 任一检查点、客户端状态或梯度出现 NaN/Inf 时脚本会立即失败，不再把 NaN 比较产生的 `0.0%` 写成可解释结果。
