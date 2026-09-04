# FCPC-grad：梯度代理共同中心与实验方案

## 1. 研究目标

原 FCPC 的核心目标是在标签分布偏斜和样本数量偏斜同时存在时，通过高差异客户端配对和模型互约束改善全局分类性能。上一版新 FCPC 将配对指标改成加权 JS 互补收益，并将伙伴模型锚点改成历史共同中心，但归一化 \(D_t(M)\) 实验显示：共同中心只轻微改变客户端更新，而且大部分干预在服务器聚合时抵消。

FCPC-grad 的目标是检验：在保持加权 JS 配对、客户端共同中心和 FedAvg 聚合不变的前提下，为共同中心加入下降方向代理，能否提高有限轮收敛速度，同时不显著牺牲最终测试准确率。

目前 FCPC-grad 是待验证方案，不能预先声称一定加速收敛或提高准确率。

## 2. 三种 FCPC

### 2.1 原文 FCPC

原文使用 JSDN 贪心最大差异配对，并让客户端 \(i\) 受到伙伴上一轮模型约束：

\[
\mathcal L_i(w)=F_i(w)+\beta\|w-w_j^{t-1}\|^2.
\]

### 2.2 上一版新 FCPC

定义样本权重：

\[
\theta_{ij}=\frac{N_i}{N_i+N_j},
\]

历史共同中心为：

\[
c_{ij}^{\mathrm{hist}}
=\theta_{ij}w_i^{t-1}+(1-\theta_{ij})w_j^{t-1}.
\]

配对使用加权 JS 互补收益和精确最大权匹配，本地使用共同中心 penalty 或 proximal。该中心可以压缩配对分歧，但本身不包含全局下降方向。

### 2.3 FCPC-grad

客户端上次被选中训练时，服务器已经知道它收到的全局模型 \(b_i^{t-1}\) 和上传的本地终点 \(w_i^{t-1}\)。定义历史更新：

\[
d_i^{t-1}=w_i^{t-1}-b_i^{t-1}.
\]

该方向近似一次本地优化产生的下降方向，但它还可能包含动量、权重衰减和上一轮 FCPC 正则的作用；在部分参与时也可能陈旧。因此本文称其为梯度代理，而不称为当前精确或无偏任务梯度。

配对更新代理为：

\[
d_{ij}^{t-1}
=\theta_{ij}d_i^{t-1}+(1-\theta_{ij})d_j^{t-1}.
\]

从当前全局模型外推梯度代理中心：

\[
c_{ij}^{\mathrm{proxy}}
=w^t+s\,d_{ij}^{t-1},
\]

其中 \(s\ge0\) 是 `grad_center_step_scale`。最终共同中心为：

\[
\boxed{
c_{ij}^{\mathrm{grad}}
=(1-\xi)c_{ij}^{\mathrm{hist}}
+\xi c_{ij}^{\mathrm{proxy}}
},
\]

其中 \(\xi\in[0,1]\) 是 `grad_center_mix`。本地目标为：

\[
\mathcal L_i^{\mathrm{FCPC-grad}}(w)
=F_i(w)+\beta_i^t\|w-c_{ij}^{\mathrm{grad}}\|^2.
\]

当前实现使用精确 proximal 映射，并将最终中心裁剪到当前全局模型附近。第 1 轮没有历史更新时，梯度代理更新自动取零；不会把缺失历史误当成大幅更新。

## 3. 与上一版的唯一核心变化

为了清楚归因，第一阶段保持以下组件一致：

- 相同 CIFAR-10 数据与 dual-skew 划分；
- 相同加权 JS 互补收益；
- 相同精确最大权匹配；
- 相同共同中心 proximal；
- 相同中心裁剪；
- 相同 FedAvg 加权聚合；
- 相同模型、优化器、学习率、客户端采样和随机种子。

唯一核心变化是：

\[
c_{ij}^{\mathrm{hist}}
\quad\longrightarrow\quad
(1-\xi)c_{ij}^{\mathrm{hist}}+\xi c_{ij}^{\mathrm{proxy}}.
\]

当 \(\xi=0\) 时，`pair_grad_center` 严格退化为匹配的历史共同中心对照组；这组实验用于验证提升是否确实来自梯度代理中心。

## 4. 当前实现的优点和不足

优点：

- 不改变全局模型结构；
- 不改变 FedAvg 聚合；
- 不需要客户端额外上传当前梯度；
- 利用服务器已经持有的广播模型和本地终点重构更新；
- \(\xi=0\) 提供严格的中心消融；
- 可继续使用中心裁剪和余弦衰减 \(\beta_t\)。

不足：

- 历史更新只是梯度代理，部分参与时存在滞后；
- 更快降低训练或验证损失不保证更高测试准确率；
- 大客户端可能主导配对更新方向；
- 较大的 \(s、\xi、\beta\) 可能导致过冲或抑制有益的客户端多样性；
- 当前版本不提供无条件的深度非凸收敛加速保证；
- 若需要当前轮精确梯度中心，将增加一次探测梯度通信阶段，不再是当前轻量版本。

## 5. 配置字段

```json
{
  "fcpc": {
    "enabled": true,
    "metric": "pair_complementarity",
    "reference_strategy": "pair_grad_center",
    "update_rule": "proximal",
    "beta": 0.05,
    "beta_schedule": "cosine_decay",
    "min_beta": 0.0,
    "pairing_strategy": "optimal",
    "partner_weighting": "uniform",
    "grad_center_mix": 0.5,
    "grad_center_step_scale": 0.5,
    "center_max_relative_distance": 0.05
  }
}
```

新增日志字段：

- `grad_center_mix`：梯度代理中心混合比例 \(\xi\)；
- `grad_center_step_scale`：历史更新外推倍数 \(s\)；
- `mean_grad_proxy_distance`：梯度代理中心到当前全局模型的平均距离；
- `grad_proxy_available_fraction`：本轮配对客户端中具有完整历史更新的比例；
- `mean_center_clip_scale`：最终中心裁剪比例，小于 1 表示发生裁剪。

## 6. 实验顺序

### 6.1 单元测试

```bash
python -m unittest discover -s tests -v
```

### 6.2 合成数据冒烟测试

```bash
python -u -m src.main \
  --config configs/fcpc_grad/smoke_synthetic_fcpc_grad.yaml \
  2>&1 | tee outputs/fcpc_grad_smoke.txt
```

检查最后两轮 CSV：

```bash
tail -n 3 outputs/fcpc_grad_smoke/smoke_synthetic_fcpc_grad.csv
```

预期第 1 轮 `grad_proxy_available_fraction=0`，之后逐步大于 0；所有损失、模型状态和中心距离必须有限。

### 6.3 开发种子超参数筛选

默认使用种子 42、50 轮，仅依据验证曲线归一化面积选择参数：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_tuning \
  --betas 0.01,0.05,0.1,0.2 \
  --mixes 0,0.5,1.0 \
  --scales 0.5 \
  --rounds 50 \
  2>&1 | tee outputs/fcpc_grad_tuning.txt
```

默认共 12 次训练。其中 `mix=0` 是上一版历史中心的匹配对照，不含梯度代理作用。选参阶段设置 `evaluation.evaluate_test=false`：不评估测试集，只使用验证曲线选参。完成后生成：

```text
outputs/fcpc_grad_tuning/tuning_summary.csv
outputs/fcpc_grad_tuning/selected_hparams.json
outputs/fcpc_grad_tuning/logs/
outputs/fcpc_grad_tuning/console/
```

如果初筛选择了非零 \(\xi\)，再围绕选中参数细化 \(s\)。例如选中 `beta=0.05, mix=0.5`：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_tuning \
  --betas 0.05 \
  --mixes 0.5 \
  --scales 0.25,0.5,1.0 \
  --rounds 50 \
  2>&1 | tee outputs/fcpc_grad_scale_refine.txt
```

注意：第二条命令会按本次候选重写 `selected_hparams.json`，因此应在确认细化结果后再进入正式实验。

### 6.4 独立种子收敛比较

种子 42 已经用于选参，正式比较使用未参与选择的种子 43、44、45：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fcpc_grad_convergence \
  --selected outputs/fcpc_grad_tuning/selected_hparams.json \
  --seeds 43,44,45 \
  --rounds 200 \
  2>&1 | tee outputs/fcpc_grad_convergence.txt
```

默认比较：

1. FedAvg；
2. 原文 FCPC；
3. 上一版 `new_fcpc`；
4. 与 FCPC-grad 使用相同 \(\beta\) 和 proximal、但强制 \(\xi=0\) 的 `fcpc_grad_mix0`；
5. 验证集选出的 FCPC-grad。

结果汇总：

```bash
python -m scripts.summarize_fcpc_grad_convergence \
  --thresholds 0.50,0.60,0.65
```

## 7. 判断标准

“收敛更快”至少需要以下有限轮指标支持：

- 前 50 轮验证准确率 AUC；
- 前 100 轮验证准确率 AUC；
- 达到 50%、60%、65% 验证准确率所需轮数；
- 相同通信轮数下的验证准确率；
- 相同通信字节数和相同墙钟时间下的验证准确率。

“没有明显牺牲准确率”需要同时检查：

- 最佳验证轮选择出的测试准确率；
- 第 200 轮测试准确率；
- 至少 3 个未用于调参的随机种子均值与标准差。

如果 FCPC-grad 只改善早期 AUC、但最终测试准确率稳定下降，则只能报告速度和最终性能之间的权衡，不能声称全面优于原 FCPC 或 FedAvg。如果 `mix=0` 被验证集选中，则说明当前梯度代理中心没有贡献，应停止扩大实验，不应强行进入收敛加速结论。
