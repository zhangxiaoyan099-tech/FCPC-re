# FCPC 重投稿审计与整改矩阵

审计日期：2026-07-29  
原稿编号：TII-26-0982  
审计范围：`编辑意见.txt`、论文及附录 PDF、LaTeX 源文件、上一轮回复信、`fcpc/fcpc` 现存代码。

## 1. 当前判断

本次决定属于允许在 1–6 个月内以新稿号重投的 reject-and-resubmit。编辑明确要求深度修改、逐点回复、在 cover letter 中注明 TII-26-0982，并突出新实验和贡献增量。

当前材料不能直接重投，原因不是语言问题，而是存在三类实质性风险：

1. **理论风险**：附录中的收敛证明包含未证明的 JSDN–梯度差异线性关系、非凸目标下未交代的局部强凸性、由 JSDN 产生的指数项，以及与实际机制不符的“梯度 LDP 噪声”。
2. **实验风险**：没有真实开销数据、规模扩展、动态加入、feature skew、concept drift、真实传感器/时间序列数据，也没有充分的配对策略对照。
3. **复现风险**：现存代码不是产生论文全部结果的完整代码；多个基线只有 stub。更严重的是，当前 FCPC 和 FedProx 正则项均通过 `model.state_dict()` 计算，所得张量不参与自动求导。实测 FCPC 项 `requires_grad=False`，因此该项不会改变训练梯度。

在修复实现并重跑关键实验以前，论文中的现有数值均标记为 **待复核**。

## 2. 审稿意见逐条整改矩阵

| 来源 | 核心问题 | 论文修改 | 必补实验/证据 | 代码任务 | 优先级 |
|---|---|---|---|---|---|
| R1-1 / R3-3 | 开销结论无定量支持 | 删除“negligible overhead”绝对表述，报告计算、内存和通信代价 | 每轮时间、端到端时间、峰值 RAM/VRAM、CPU/GPU 利用率、模型字节数和累计流量；与 FedAvg/FedProx/MOON 对比 | 加 profiler 和通信计数器 | P0 |
| R1-2 / R3-3 | 只验证 label + quantity skew | 扩展适用范围并明确边界 | feature skew；concept drift；至少一个传感器或时间序列 IoT 数据集 | 新增 skew 生成器和时序数据加载器 | P0 |
| R1-3 | λ、β 只在固定设置调参 | 增加跨规模、跨异质度敏感性和自适应策略 | 客户端数、α、λ、β 三维/分层敏感性；自适应参数与固定参数对比 | 配置 sweep 与自适应规则 | P1 |
| R1-4 | 缺少模型投毒与多策略防御讨论 | 增加威胁模型、FCPC 的攻击面与可组合防御讨论；不得宣称未经验证的鲁棒性 | 建议增加至少一种 model poisoning 下的补充实验 | 攻击注入与防御接口 | P2 |
| R1-5 | 可扩展性、动态加入不足 | 增加复杂度、部分参与、动态客户端流程 | 10/50/100/500/1000 客户端配对时间和内存；训练至少做到 10/50/100；动态加入曲线 | 向量化 JSDN、轮次级配对、新客户端注册 | P0 |
| R2 | 找不到上一轮回复 | 新回复信首页说明附件结构并逐条复述原意见 | 提交前核对上传文件名和系统槽位 | 生成独立、可检索的 response PDF/DOCX | P0 |
| R3-1 | 理论假设和证明不严谨 | 删除当前不可证项；改用标准 smooth non-convex 分析，或将结论降级为性质/复杂度命题 | 经验验证 JSDN 与梯度差异的相关性及置信区间 | 梯度差异测量脚本 | P0 |
| R3-2 | 贪心最大差异配对缺乏依据 | 给出非负边权最大权匹配的 1/2 近似保证；澄清最优目标 | greedy-max、optimal max-weight、random、similarity、无配对；报告质量与耗时 | 实现多配对策略 | P0 |
| R3-2 | 奇数客户端被“丢弃” | 改为“仍参与训练和聚合，只在该轮无配对正则”；轮换未配对客户端 | 奇偶规模和轮换公平性 | 轮换 bye / 公平性记录 | P0 |
| R3-3 | 配对效果和正则效果未隔离 | 重写消融逻辑 | no-reg、random-pair+reg、similar-pair+reg、dissimilar-pair+reg、optimal-pair+reg | 实验矩阵 | P0 |
| R3-3 | 基线比较公平性不足 | 统一数据划分、模型、优化器、预算、随机种子；报告均值±标准差 | 至少 3 个种子；同等通信轮数并补充同等字节预算结果 | 完整实现或采用官方基线代码 | P0 |
| R3-3 | “真实 IoT”只用图像基准 | 收缩未经实证的工程宣称，增加真实场景描述 | 至少一个真实传感器/时序数据集 | 数据预处理与适配模型 | P0 |

## 3. 已确认的论文—代码不一致

1. 论文称客户端将模型额外发送给配对客户端；模拟器只是直接读取内存中的 partner state，没有真实网络传输或字节统计。
2. 论文称奇数客户端会被“abandon”；代码仍让未配对客户端参与训练和聚合，只是不加 FCPC 项。代码行为更合理，论文应改成该行为。
3. 论文声称 FCPC 可集成 MOON、FBLG、FedCFA；现存代码对这些方法只有 `NotImplementedError` 占位实现，无法复现表格。
4. 论文的 LDP 仅用于标签直方图与样本量元数据；附录却把 LDP 方差加入随机梯度收敛式，机制不一致。
5. 配置文件中的 `alpha_grid`、`beta_grid`、`lambda_grid`、`algorithm_grid` 和 `dataset_grid` 未被训练入口消费，当前入口每次只运行一个固定设置。
6. 配置称使用数据增强，但当前 CIFAR/TinyImageNet loader 只有 `ToTensor`/`Resize`，与论文实验设置不一致。
7. 论文报告 10 个客户端且全部参与；现有代码可采样部分客户端，但固定全局 pairing 在部分参与时可能使用陈旧 partner state，尚无 staleness 处理。

## 4. 推荐的新实验主线

### 4.1 核心有效性

- 数据集：CIFAR-10、CIFAR-100、一个真实 IoT 传感器/时序数据集。
- 异质性：label skew、quantity skew、dual skew、feature skew、concept drift。
- 客户端数：10、50、100；配对算法微基准扩展到 500、1000。
- 随机种子：至少 3。
- 结果：最终精度、最佳精度、收敛轮数、AUC、均值±标准差。

### 4.2 公平消融

- FedAvg
- FedAvg + 无配对但相同形式正则控制
- FedAvg + random pairing
- FedAvg + similarity pairing
- FedAvg + greedy dissimilarity pairing
- FedAvg + exact maximum-weight matching

该设计直接隔离“正则项本身”和“dissimilarity-aware pairing”的贡献。

### 4.3 系统代价

- 训练耗时：每轮与全程 wall-clock。
- 计算资源：CPU 利用率、RSS、GPU 利用率、峰值 VRAM。
- 通信：上/下载字节、额外 partner-model 字节、达到目标精度所需累计字节。
- 配对：JSDN 构建与 matching 的单独时间/内存。

必须承认：若传完整模型给 partner，FCPC 的客户端上行通信量相对 FedAvg 约增加一个模型大小，不能再称通信开销“可忽略”。可以强调不增加通信轮数，并报告 accuracy-per-byte 权衡。

## 5. 理论修订原则

当前附录 A 建议整体撤下并重写，不能做局部修辞修补。

可保留并严格证明的内容：

1. JSDN 的有界性、对称性和极端单偏斜退化性质。
2. 贪心最大差异匹配对非负边权最大权匹配的 1/2 近似保证。
3. JSDN 矩阵、贪心匹配和最优匹配的时间/空间复杂度。
4. 在标准 L-smooth、无偏随机梯度、有界方差条件下，对“带时滞参考点的二次正则本地目标”给出不夸大作用的 stationarity bound；JSDN 只决定匹配，不应未经证明直接进入梯度差异上界。

若无法形成严谨的第 4 项，宁可删除强收敛主张，保留性质、复杂度与充分实验，也不要使用不可验证的指数改进项。

## 6. 实施顺序

1. 修复 FCPC/FedProx 自动求导和首轮参考状态。
2. 加入单元测试，证明正则项确实产生预期梯度。
3. 实现 random/similar/greedy/optimal pairing 和奇数客户端公平处理。
4. 加入 sweep、profiling、通信统计、动态加入和多随机种子入口。
5. 先跑小规模 sanity check，再跑正式实验。
6. 实验冻结后重写理论、正文、附录、cover letter 和逐点回复。

## 7. 2026-07-29 已完成

- 修复 FCPC 和 FedProx 正则项不参与自动求导的问题。
- 修复顺序模拟导致一对客户端分别读取“上一轮/本轮”不同 partner 状态的问题。
- 首轮 partner reference 明确定义为共同全局初始化。
- 配对限定在本轮实际参与客户端内，避免部分参与时配到缺席客户端。
- 实现 greedy-dissimilar、exact max-weight、random、similarity 四类配对。
- 实现奇数客户端 fair-bye 轮换；未配对客户端仍训练并参与聚合。
- 将 JSDN 矩阵改为分块向量化，将贪心匹配从实际 O(n³) 改为
  O(n² log n)。
- 加入每轮资源监测与精确模型字节统计。
- 加入不依赖外部数据的端到端 synthetic smoke run。
- 加入 UCI HAR 九通道原始惯性窗口 loader、自然 subject-client 划分和
  1D sensor CNN。
- 13 个回归测试全部通过。
- 单次配对微基准（仅作工程 sanity check，正式稿需重复实验）：
  - n=100：JSDN 0.0069 s，greedy 0.0179 s，greedy/optimal 权重比 0.9528；
  - n=500：JSDN 0.1343 s，greedy 0.7824 s；
  - n=1000：JSDN 0.5059 s，greedy 3.5657 s，Python 峰值内存约
    117.3 MiB。
- 建立独立 `resubmission` LaTeX 工作副本并完成主稿基线编译。
- 整体重写理论附录：保留可证明的 JSDN 性质、局部非凸 stationarity
  结果、贪心匹配 1/2 近似保证、复杂度与通信核算；删除原先未经证明的
  JSDN--梯度线性关系、局部强凸假设、指数改进项和错误的梯度 LDP 项。
- 增加 ScaleSign/MSGuard 的安全边界讨论，明确 FCPC 不是投毒防御。
