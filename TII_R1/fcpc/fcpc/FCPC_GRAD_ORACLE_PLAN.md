# FCPC-grad-oracle：干净反事实与方向门控实验

## 1. 定位

`FCPC-grad-oracle` 是机制诊断，不是最终可部署算法。它使用当前冻结检查点上的经验全局梯度

\[
g_t=\sum_i a_i\widehat\nabla F_i(w^t)
\]

判断历史配对代理是否有利。真实联邦服务器通常拿不到这个量，所以不能把 oracle 的结果直接当作可部署贡献。

本实验先回答一个更基础的问题：**如果能够排除方向错误的配对代理，FCPC-grad 的代理分量能否稳定改善服务器单轮下降？**

## 2. 为什么需要新的干净对照

旧引理审计比较：

\[
c^{\mathrm{mix0}}_{p,t}=c^{\mathrm{hist}}_{p,t},
\qquad
c^{\mathrm{grad}}_{p,t}=w^t+s_td_p^t.
\]

它同时改变了两件事：删除历史参数中心、加入梯度代理中心。因此

\[
Z_t=\Delta_t^{\mathrm{grad}}-\Delta_t^{\mathrm{mix0}}
\]

不能只归因于梯度代理。

新实验的三条回放都保留相同的 proximal 训练、\(\beta_t\)、配对、批次和初始模型，仅改变中心相对 \(w^t\) 的位移：

\[
c^{\mathrm{off}}_{p,t}=w^t,
\]

\[
c^{\mathrm{ungated}}_{p,t}=w^t+s_td_p^t,
\]

\[
c^{\mathrm{oracle}}_{p,t}
=w^t+s_t\phi^{\mathrm{oracle}}_{p,t}d_p^t.
\]

`proxy_off` 仍然执行 proximal，只是中心为当前全局模型。因此 oracle 与 off 的唯一干预是历史代理位移。

## 3. oracle 门控

对配对 \(p=(i,j)\)，历史更新代理为

\[
d_p^t=\theta_pd_i^t+(1-\theta_p)d_j^t,
\qquad
d_i^t=v_i^{\tau_i}-b_i^{\tau_i}.
\]

因为 \(d_p^t\) 近似负梯度方向，所以定义真实下降余量

\[
m_p^t=-\langle g_t,d_p^t\rangle.
\]

默认 oracle 门控为

\[
\boxed{
\phi^{\mathrm{oracle}}_{p,t}
=\mathbf 1\{m_p^t>0\}
\mathbf 1\{\cos(d_p^t,-g_t)\ge0\}.
}
\]

配置中的 `min_margin` 与 `min_cosine` 可以提高门槛。默认都为 0，只拒绝明确有害或退化的方向。

## 4. \(P_t\)：服务器收到的显式代理分量

exact proximal 把中心位移传入客户端更新的系数为

\[
\lambda_{i,t}
=s_t\phi_{p,t}\chi_{p,t}
\left[1-\left(\frac{1}{1+2\eta_t\beta_{i,t}}\right)^H\right]\ge0,
\]

其中 \(\phi_{p,t}\in[0,1]\) 是方向门控，\(\chi_{p,t}\in[0,1]\) 是中心裁剪比例。配对平均传递系数为

\[
\bar\lambda_{p,t}
=\theta_p\lambda_{i,t}+(1-\theta_p)\lambda_{j,t}.
\]

服务器按样本量聚合后，显式由代理中心引入的更新向量是

\[
\boxed{
P_t=\sum_p\omega_{p,t}\bar\lambda_{p,t}d_p^t.
}
\]

它不是配对集合，也不是概率。它是一个与模型参数同维的向量。`P_t_norm` 是其模长；`lemma6_projection` 是

\[
-\langle g_t,P_t\rangle.
\]

该值大于 0，只说明 \(P_t\) 的一阶投影有利，尚未计入步子过大造成的二阶代价。

## 5. \(Q_t\)：加入一个修正向量后的净下降保证增量

对任意基线服务器更新 \(B_t\) 和额外修正 \(X_t\)，由 \(L\)-smooth 下降式定义

\[
\boxed{
Q_t(X_t;B_t)
=-\langle g_t,X_t\rangle
-L\langle B_t,X_t\rangle
-\frac L2\|X_t\|^2.
}
\]

它是“加上 \(X_t\) 后，光滑下降下界相对基线增加了多少”。

本实验区分两个 \(Q_t\)：

1. 显式代理收益

\[
Q_t(P_t;\Delta_t^{\mathrm{off}}),
\]

对应 CSV 的 `Q_proxy_vs_baseline`。它直接检查引理 6 给出的 \(P_t\) 是否足以覆盖交叉项和二阶代价。

2. 真实反事实收益

\[
Z_t
=\Delta_t^{\mathrm{method}}-\Delta_t^{\mathrm{off}},
\]

\[
Q_t(Z_t;\Delta_t^{\mathrm{off}}),
\]

对应 `Q_counterfactual`。因为中心改变会改变后续本地 SGD 轨迹，一般 \(Z_t\ne P_t\)。令

\[
e_t^{\mathrm{traj}}=Z_t-P_t,
\]

便能用 `trajectory_error_norm` 判断理论显式分量在实际训练轨迹中保留了多少。

当 \(L=0\) 时，\(Q_t\) 只检查一阶方向：

\[
Q_t(X_t;B_t)=-\langle g_t,X_t\rangle.
\]

当 \(L>0\) 时才同时惩罚与基线更新的交互及过大的步长。当前正 \(L\) 只是敏感性分析，不能被称作已证明的 ResNet-18 光滑常数。

## 6. 如何解释四种结果

### 6.1 oracle 好，ungated 差

历史代理中存在方向错误项，门控是主要缺口。下一步研究只使用历史服务器更新的可部署门控，并证明代理误差条件下的方向安全性。

### 6.2 两者都好

原引理 4 的充分界过于保守；0% 的充分条件成立率不代表真实代理机制无效。可将理论主线改为门控余量条件。

### 6.3 两者的 \(P_t\) 有利，但 \(Z_t\) 无利

问题不在配对方向，而在 proximal 传递后的本地轨迹响应。需要减小 \(s_t\)、\(\beta_t\) 或增加信赖域裁剪。

### 6.4 oracle 也无利

即使删除有害配对，剩余代理的幅度或相互交叉项仍吞没一阶收益；暂时不能推进“方向门控即可条件加速”的论断。

## 7. 运行顺序

当前第二阶段配置在完全相同的冻结检查点上使用 20 条随机本地轨迹，并公平比较

\[
s_t\in\{0.1,0.25,0.5\}.
\]

汇总会额外输出

\[
q_P=\frac{Q_t(P_t)}{\|g_t\|^2},
\qquad
q_Z=\frac{Q_t(Z_t)}{\|g_t\|^2},
\]

以及 $q_Z$ 的近似 95% 下置信界。若某个 $s_t$ 在多个检查点上具有正的下置信界，且 `obs>0` 比例没有明显下降，它才进入完整梯度和独立模型种子验证。

单元测试与合成冒烟：

```bash
python -m unittest discover -s tests -v
python -u -m scripts.run_fcpc_grad_oracle_audit \
  --config configs/lemma456/smoke_fcpc_grad_oracle.yaml
```

CIFAR-10 seed 42（复用已经生成且 BatchNorm 目标一致的中性检查点）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_oracle_audit \
  --config configs/lemma456/cifar10_fcpc_grad_oracle_seed42.yaml \
  --reuse-checkpoints \
  2>&1 | tee outputs/fcpc_grad_oracle_seed42_console.txt
```

汇总：

```bash
python -m scripts.summarize_fcpc_grad_oracle \
  outputs/fcpc_grad_oracle_seed42/oracle_metrics.csv \
  --panel raw --strategy optimal
```

第二阶段步长筛选写入独立目录，不覆盖上述结果：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_oracle_audit \
  --config configs/lemma456/cifar10_fcpc_grad_oracle_stepscale_seed42.yaml \
  --reuse-checkpoints \
  2>&1 | tee outputs/fcpc_grad_oracle_stepscale_seed42_console.txt

python -m scripts.summarize_fcpc_grad_oracle \
  outputs/fcpc_grad_oracle_stepscale_seed42/oracle_metrics.csv \
  --panel raw --strategy optimal
```

## 8. 当前不足

1. oracle 使用真实经验全局梯度，不可直接部署；
2. 当前步长筛选仍只用每客户端 20 个梯度探针 batch；选出 $s_t$ 后，正式审计需把 `gradient_max_batches` 改为 `null`；
3. 当前先固定相同本地步数和学习率，尚未恢复 quantity skew 下的 \(H_i\)、\(\gamma_i\) 失衡项；
4. 正 \(L\) 尚无独立估计，只能先以 \(L=0\) 判断方向；
5. 单轮审计不能替代多轮收敛实验；oracle 成功后仍需设计可部署历史门控并单独训练。
