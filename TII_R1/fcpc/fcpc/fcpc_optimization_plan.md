# FCPC 优化与消融实验计划

记录日期：2026-08-26  
前置结果：见 `reconstruction_v1.md`  
当前状态：`reconstruction_v1` 中 FCPC 未超过 FedAvg；在完成本计划的协议修正与诊断实验前，不继续盲目运行新的 200 轮实验。

## 1. 本计划要回答的问题

当前负结果可能来自三个层面：

1. dual-skew 数据构造丢弃了大量训练样本；
2. LDP 扰动改变了 JSDN 矩阵及贪心匹配组合；
3. 最大差异客户端之间使用统一的参数距离正则可能损害分类学习。

本计划采用逐层隔离方式，避免一次修改多个因素后无法解释性能来源。

## 2. Raw JSDN 和 LDP-JSDN 的准确含义

### 2.1 Raw JSDN 仍然计算完整 JSDN 矩阵

Raw JSDN 使用客户端未经隐私扰动的真实标签直方图和样本量：

\[
J_{ij}^{raw}
=
(1-\lambda)\operatorname{JSD}(p_i,p_j)
+
\lambda\frac{|N_i-N_j|}{N_i+N_j}.
\]

随后仍然构造完整的 K×K 矩阵，并按照矩阵中的最大值进行贪心配对：

```text
真实标签直方图和样本量
        ↓
计算 Raw JSDN 矩阵
        ↓
最大差异贪心配对
        ↓
执行 FCPC 正则训练
```

因此，`raw JSDN` 的含义是“没有 LDP 噪声的 JSDN”，不是“不计算 JSDN”。

### 2.2 LDP-JSDN 先扰动统计量，再计算矩阵

LDP-JSDN 的流程为：

```text
真实标签直方图和样本量
        ↓
添加LDP噪声
        ↓
计算 LDP-JSDN 矩阵
        ↓
最大差异贪心配对
        ↓
执行 FCPC 正则训练
```

公式可以写为：

\[
J_{ij}^{LDP}
=
(1-\lambda)\operatorname{JSD}(\tilde p_i,\tilde p_j)
+
\lambda\frac{|\tilde N_i-\tilde N_j|}{\tilde N_i+\tilde N_j}.
\]

### 2.3 FedAvg 才是不计算 JSDN、不进行配对

三组实验的区别应明确写成：

| 实验 | 是否计算 JSDN | JSDN 输入 | 是否配对 | 是否有 FCPC 正则 |
|---|---|---|---|---|
| FedAvg | 否 | 无 | 否 | 否 |
| FCPC-Raw | 是 | 原始直方图和样本量 | 是 | 是 |
| FCPC-LDP | 是 | LDP 扰动后的统计量 | 是 | 是 |

三组比较分别回答：

\[
\text{FCPC-Raw}-\text{FedAvg}
\]

表示没有隐私噪声时，FCPC 配对正则本身是否有效；

\[
\text{FCPC-LDP}-\text{FCPC-Raw}
\]

表示加入 LDP 后的配对和性能代价；

\[
\text{FCPC-LDP}-\text{FedAvg}
\]

表示论文完整部署路径相对基础 FedAvg 的总效果。

## 3. 第一优先级：修正 dual-skew 数据构造

当前 `quantity_skew_resample` 从 45000 个候选训练样本中只保留 21902 个，丢弃 51.33%。新方案必须满足：

\[
\sum_{i=1}^{K}N_i=45000.
\]

新划分命名为：

```text
dual_skew_full
```

旧划分保留为：

```text
dual_skew_trimmed_v1
```

新方案应遵守：

1. 全部 45000 个训练样本均被使用；
2. 每个样本只属于一个客户端；
3. 标签偏斜由 Dirichlet alpha 控制；
4. 客户端目标容量不均匀，形成数量偏斜；
5. 不同 alpha 下总训练样本量完全一致；
6. 保存客户端索引、样本量、标签直方图和随机种子。

实现时可先生成总和为 45000 的不均匀客户端目标容量，再在按类别分配样本时同时满足标签概率和客户端剩余容量。不能先完成标签划分后直接裁减样本。

## 4. 第二优先级：Raw/LDP 隔离实验

修正数据划分后，冻结公共配置：

```text
dataset = CIFAR-10
model = ResNet-18
clients = 10
clients_per_round = 10
partition = dual_skew_full
alpha = 0.1
lr = 0.03
momentum = 0.9
weight_decay = 5e-4
local_epochs = 1
batch_size = 64
rounds = 50
seed = 42
```

执行：

| 编号 | 实验 | pairing source | privacy | 目的 |
|---|---|---|---|---|
| A | FedAvg | none | off | 新数据划分基础线 |
| B | FCPC-Raw | raw JSDN | off | 测试 FCPC 正则本身 |
| C | FCPC-LDP | LDP-JSDN | epsilon=1 | 测试隐私扰动代价 |

需要新增明确配置：

```json
"privacy": {
  "enabled": false,
  "epsilon": 1.0
}
```

不能使用一个极大的 epsilon 假装关闭隐私，因为明确开关更便于审计和复现。

### 判定规则

```text
B > A 且 C < B
```

说明 FCPC 正则可能有效，主要问题是 LDP 配对不稳定。

```text
B < A 且 C < A
```

说明主要问题在最大差异参数约束，而不是 LDP。

```text
B ≈ A 且 C ≈ A
```

说明当前正则作用有限，需要检查损失尺度和配对信息价值。

## 5. 样本量感知权重：假设、归一化和限制

### 5.1 为什么提出样本量感知

当前有向正则为：

\[
R_{i\leftarrow j}=\beta_0\|w_i-w_j^{t-1}\|^2.
\]

无论提供目标模型的客户端 j 有 136 个样本还是 4822 个样本，单步正则强度都相同。这促使我们提出“目标模型可靠性可能与其样本量有关”的待验证假设。

一个简单候选权重是：

\[
r_{i\leftarrow j}=\frac{N_j}{N_i+N_j}.
\]

它表达的是：提供目标模型的客户端 j 样本越多，i 受到它的约束越强。

这不是已证明的定理。它隐含了“局部模型估计方差随样本量增加而下降”等简化假设，忽略了类别覆盖、数据质量、优化程度和任务冲突。因此只能作为可检验的启发式方法。

### 5.2 分子乘 2 的真实作用

对一对客户端 i、j：

\[
r_{i\leftarrow j}+r_{j\leftarrow i}
=
\frac{N_j}{N_i+N_j}
+
\frac{N_i}{N_i+N_j}
=1.
\]

两个有向权重的平均值是 0.5。若直接使用 r，则在两个客户端样本量相等时，每个方向的有效正则只有原 beta 的一半。

乘 2 后：

\[
\tilde r_{i\leftarrow j}
=
\frac{2N_j}{N_i+N_j},
\]

两个方向的平均权重变为 1：

\[
\frac{
\tilde r_{i\leftarrow j}
+
\tilde r_{j\leftarrow i}
}{2}=1.
\]

因此，乘 2 只是尺度归一化，使加权版本的平均 beta 与原始统一 beta 大致可比。它不会改变两个方向的相对强弱，而且完全可以被重新调节 beta 吸收。

结论：

> 分子乘 2 有“保持平均尺度”的实验设计依据，但没有“它一定提高准确率”的理论保证。

### 5.3 如何严谨验证

样本量加权不能直接作为默认 FCPC。应比较：

```text
uniform：原始统一beta
partner_fraction：N_j/(N_i+N_j)
partner_fraction_normalized：2N_j/(N_i+N_j)
```

其中 `partner_fraction` 和 `partner_fraction_normalized` 的差别主要是整体尺度。若分别充分调 beta，两者理论上可以得到近似效果。因此，最有意义的比较是：

```text
uniform
vs.
normalized partner weighting
```

并同时记录每个方向的有效 beta、客户端样本量、任务损失和正则损失。

## 6. 余弦 beta 衰减

### 6.1 它是什么

余弦 beta 衰减是训练日程，不是余弦相似度损失。定义为：

\[
\beta_t
=
\beta_{min}
+
\frac{\beta_0-\beta_{min}}{2}
\left[
1+\cos\left(\frac{\pi t}{T}\right)
\right].
\]

当 `beta_min=0` 时：

\[
\beta_0=\beta_{t=0},
\qquad
\beta_T=0.
\]

训练中点为：

\[
\beta_{T/2}=\frac{\beta_0}{2}.
\]

### 6.2 为什么值得测试

它对应一个待验证假设：

```text
训练前期：客户端漂移大，配对约束可能帮助稳定训练
训练后期：分类模型接近收敛，持续拉向陈旧配对模型可能妨碍精细收敛
```

`reconstruction_v1` 中 FCPC 的 30 轮候选表现尚可，但 200 轮最终落后 FedAvg，这提示固定 beta 可能在后期产生持续干扰。不过这只是实验动机，不是收敛定理。

余弦形式相对线性衰减的优点是起点和终点变化更平滑：

\[
\left.\frac{d\beta_t}{dt}\right|_{t=0}
=
\left.\frac{d\beta_t}{dt}\right|_{t=T}
=0.
\]

它避免在训练开始或结束时突然改变正则强度。

### 6.3 公平消融方式

若 `beta_min=0`，余弦日程在整个训练区间的平均 beta 为：

\[
\overline\beta_{cos}=\frac{\beta_0}{2}.
\]

因此需要做两类比较：

#### 相同初始强度

```text
constant beta = 0.01
cosine beta = 0.01 → 0
```

该比较回答“训练后期释放约束是否有帮助”，但余弦版本的总正则暴露更小。

#### 相同平均强度

```text
constant beta = 0.005
cosine beta = 0.01 → 0
```

两者平均 beta 近似相同，该比较更能隔离“正则施加时机”的影响。

至少同时完成这两组比较，不能只拿 `0.01→0` 与固定 `0.01` 比较后，把提升全部解释为余弦日程；提升也可能仅仅来自平均正则变弱。

### 6.4 建议配置接口

```json
"fcpc": {
  "beta": 0.01,
  "beta_schedule": "constant",
  "min_beta": 0.0
}
```

以及：

```json
"fcpc": {
  "beta": 0.01,
  "beta_schedule": "cosine_decay",
  "min_beta": 0.0
}
```

CSV 应增加：

```text
effective_beta
train_fcpc_raw_loss
train_fcpc_weighted_loss
train_fcpc_weighted_loss / train_task_loss
```

## 7. 不要混淆两种“余弦”

### 7.1 余弦 beta 衰减

改变的是正则系数随时间的大小：

\[
\beta_t:\ \beta_0\rightarrow0.
\]

正则内容仍然可以是参数距离。

### 7.2 更新方向余弦损失

改变的是正则项定义。定义客户端更新：

\[
\Delta_i^t=w_i^t-w^t,
\qquad
\Delta_j^{t-1}=w_j^{t-1}-w^{t-1}.
\]

方向损失为：

\[
R_{cos}
=
1-
\frac{
\langle\Delta_i^t,\Delta_j^{t-1}\rangle
}{
\|\Delta_i^t\|\|\Delta_j^{t-1}\|+\epsilon
}.
\]

它直接约束更新方向，更接近论文“梯度方向对齐”的语言；但它属于新的 FCPC-v2 方法，不能与 beta 余弦衰减混为一谈。

当前优先测试 beta 余弦衰减。更新方向余弦损失应在原始参数距离机制被证明无效后单独实现和消融。

## 8. 分阶段实验顺序

### 阶段 A：协议修正

1. 实现 `dual_skew_full`；
2. 增加 `privacy.enabled`；
3. 保存数据划分、JSDN 矩阵和配对；
4. 增加逐客户端训练和有效 beta 日志；
5. 单元测试确认全部样本无重复、无丢失。

### 阶段 B：50 轮原因隔离

```text
A. FedAvg
B. FCPC-Raw
C. FCPC-LDP
```

若 B、C 都低于 A，进入阶段 C。

### 阶段 C：正则日程消融

在 Raw JSDN 下比较：

```text
D. constant beta=0.01
E. cosine beta=0.01→0      （相同初始值）
F. constant beta=0.005
G. cosine beta=0.01→0      （与F近似相同平均值）
```

实验 E 和 G 是同一次余弦运行，但分别与 D、F 回答不同问题。

### 阶段 D：样本量权重假设

若余弦日程仍不足，再比较：

```text
H. uniform beta
I. normalized partner-sample weighting
```

保持 beta 日程、数据划分和 JSDN 来源一致。样本量加权只是待验证启发式方法，不能提前作为论文理论结论。

### 阶段 E：更新方向正则

只有参数距离系列仍持续低于 FedAvg 时，才建立 FCPC-v2：

```text
parameter distance
update-vector distance
update-direction cosine
```

### 阶段 F：配对策略消融

在选定正则形式之后，再比较：

```text
最大差异配对
最相似配对
随机配对
最优最大权匹配
FedAvg无配对
```

否则配对策略和正则形式同时变化，无法判断性能来源。

## 9. 候选进入长训练的条件

所有候选先运行 50 轮。进入 200 轮前必须满足：

1. 最后 10 轮验证均值不低于同协议 FedAvg；
2. 优势不是单轮峰值；
3. task loss 正常下降；
4. FCPC/task 不持续超过 1；
5. 无 NaN、Inf 或明显准确率坍塌；
6. 至少再使用 3 个随机种子确认方向一致；
7. 超参数只根据验证集确定。

## 10. 当前优先实施内容

下一批代码只实现实验隔离能力，不立刻改变 FCPC 数学形式：

```text
1. dual_skew_full
2. privacy.enabled
3. JSDN和配对自动归档
4. per-client训练日志
5. beta_schedule（constant/cosine_decay）
```

样本量感知权重和更新方向余弦损失先保留为可审查的后续候选，等 Raw/LDP 隔离结果出来后再决定是否实现。

## 11. 2026-08-26 已实现的诊断实验入口

本次实现没有使用分子乘 2。样本量权重明确为：

\[
r_{i\leftarrow j}=\frac{N_j}{N_i+N_j},
\qquad
\beta_{i,t}^{effective}=\beta_t r_{i\leftarrow j}.
\]

配置项如下：

```json
"fcpc": {
  "beta": 0.01,
  "beta_schedule": "constant",
  "min_beta": 0.0,
  "partner_weighting": "sample_ratio"
}
```

四个 50 轮诊断配置为：

1. `cifar10_dual_a0p1_fcpc_uniform_beta0p01_r50.yaml`：原始统一权重；
2. `cifar10_dual_a0p1_fcpc_sample_ratio_nox2_beta0p01_r50.yaml`：样本比例权重，不乘 2；
3. `cifar10_dual_a0p1_fcpc_uniform_beta0p005_r50.yaml`：统一但较弱的固定正则；
4. `cifar10_dual_a0p1_fcpc_uniform_beta_cosine0p01_to0_r50.yaml`：统一权重，beta 从 0.01 余弦衰减至 0。

前两组回答样本比例分配是否有效；第 2、3 组帮助判断变化是否仅来自整体正则减弱；第 1、4 组回答训练后期逐步释放 FCPC 约束是否有效。所有配置保持随机种子、数据划分、JSDN/LDP、配对、模型和优化器一致。

训练 CSV 新增以下审计字段：

```text
beta_base
beta
beta_schedule
beta_min
partner_weighting
mean_partner_weight
mean_effective_beta
```

其中 `beta` 是当轮的 beta_t，`mean_effective_beta` 是当轮所有已配对客户端实际使用的 beta_t × partner_weight 的均值。该均值只用于实验审计，不作为新的理论量。

## 12. 样本比例乘 2 与余弦 beta 的组合消融

为避免预设乘 2 一定有效，将其作为显式实验组件 `sample_ratio_x2`：

\[
r_{i\leftarrow j}^{x2}=\frac{2N_j}{N_i+N_j}.
\]

在完全相同的余弦 beta 日程下比较：

1. `uniform + cosine`：现有基准；
2. `sample_ratio(no x2) + cosine`：非对称分配，同时降低系数总尺度；
3. `sample_ratio_x2 + cosine`：非对称分配，但保持一对客户端两个方向的系数和与 uniform 相同。

新增配置：

```text
cifar10_dual_a0p1_fcpc_sample_ratio_nox2_beta_cosine0p01_to0_r50.yaml
cifar10_dual_a0p1_fcpc_sample_ratio_x2_beta_cosine0p01_to0_r50.yaml
```

三者两两比较的含义是：

- `x2 cosine` 对 `uniform cosine`：相同成对系数总尺度下，非对称分配是否有用；
- `x2 cosine` 对 `no-x2 cosine`：将有效正则整体放大两倍的影响；
- `no-x2 cosine` 对 `uniform cosine`：非对称分配与整体减弱共同产生的效果。

这一实验只验证经验效果，不把乘 2 作为理论结论。
