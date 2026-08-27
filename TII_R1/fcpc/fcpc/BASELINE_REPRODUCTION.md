# FCPC 基线复现记录

更新时间：2026-08-27

## 1. 当前结论

项目原有的 FedAvg、FedProx 可以运行；原来的 MOON、FedDyn、FBLG、FedCFA 只是占位接口。现在六种方法都已接入同一套 `Trainer`、数据划分、模型、优化器和评估代码。这样得到的结果使用相同 CIFAR-10 dual-skew 划分，不能把不同仓库中不可比的旧日志混在一起。

接入位置：

| 方法 | FCPC 项目实现 | 作者源码的核心位置 | 当前状态 |
|---|---|---|---|
| FedAvg | `src/algorithms/fedavg.py` | 标准服务器加权平均 | 已实现 |
| FedProx | `src/algorithms/fedprox.py` | `litian96/FedProx/flearn/trainers/fedprox.py` | 已实现，局部 proximal loss 可传梯度 |
| MOON | `src/algorithms/moon.py` | `Xtra-Computing/MOON/main.py::train_net_fedcon` | 已实现全局/历史特征对比损失 |
| FedDyn | `src/algorithms/feddyn.py` | `alpemreacar/FedDyn/utils_general.py`、`utils_methods.py` | 已实现动态正则、客户端历史量和 cloud aggregation |
| FBLG | `src/algorithms/fblg.py` | `YingLi-Y/FBLG/algorithm/FBLG.py` | 已实现首轮建图和图/损失联合选客户端；用枚举/贪心替代 Gurobi |
| FedCFA | `src/algorithms/fedcfa.py` | `hua-zi/FedCFA/alg/fedcfa.py` | 已实现均值数据池、正/负反事实特征损失 |

## 2. 上游版本固定

为保证以后能核对，下载脚本 `scripts/fetch_upstream_baselines.py` 固定到了以下提交：

- FedProx：`litian96/FedProx@d2a4501f319f1594b732d88315c5ca1a72855f50`，MIT。
- MOON：`Xtra-Computing/MOON@8fcba3c4efc9a47eb24687a91ba94cfb2005103f`，MIT。
- FedDyn：`alpemreacar/FedDyn@48a19fac440ef079ce563da8e0c2896f8256fef9`，MIT。
- FedCFA：`hua-zi/FedCFA@a1d01b548856f2224a011a099059e8c86fad0aec`，MIT。
- FBLG：`YingLi-Y/FBLG@8d78ac2ff695784c2dbba810a1db65893ecb952a`，仓库未提供许可证。

注意：这里的 FedDyn 是常用的 ICLR 2021 dynamic-regularization 方法。FCPC 原论文的参考文献如果实际指向同名的推荐系统蒸馏方法，必须在重投稿前统一论文引用、表格名称和代码，不能把两个 FedDyn 当成同一个方法。

FBLG 没有许可证，因此作者文件只保留在本机供阅读核对，不提交 GitHub；项目中的 `fblg.py` 是根据论文机制和接口独立编写的实现。

## 3. “208 条更新”检查结果

资源管理器显示的 208 条更新全部是 `third_party/baselines/` 中下载的上游源码快照，没有发现原项目文件被批量修改或删除。这些文件不应成为 FCPC 的正式代码，因此 `.gitignore` 已加入：

```gitignore
/third_party/baselines/
```

结果是：上游源码仍留在本机，便于核对；GitHub 和服务器只同步我们自己的适配器、测试、配置和文档。FBLG 的未授权源码也不会被再分发。

## 4. 公平实验协议

FBLG 的核心是选择客户端，若 10 个客户端每轮全部参加，FBLG 的选择机制没有发挥空间。作者代码默认每轮参与比例为 0.2、先按损失保留 0.5 的候选客户端，因此正式横向比较采用统一的 `10 个总客户端、每轮 2 个客户端`：FedAvg、FedProx、MOON、FedDyn、FBLG 和 FedCFA 全部使用同一协议。当前 10/10 的 FCPC 最强结果只能用于 FCPC 内部消融，不能直接作为这一组 2/10 横向表格的结果；需另跑一个 FCPC 2/10 配置。

生成的 CIFAR-10 配置位于 `configs/baselines/cifar10_dual_a0p1_*_cpr2_r200.yaml`。超参数是第一轮可运行的起点，不代表每个基线已经完成调参。论文主表应报告统一调参预算、多个随机种子下的均值和标准差。

## 5. 服务器测试顺序

先更新代码并运行单元测试：

```bash
cd ~/FCPC-re/TII_R1/fcpc/fcpc
git pull --rebase
conda activate fcpc-core
python -m unittest discover -s tests -v
```

然后逐个运行两轮 synthetic smoke；它不下载数据，只验证目标函数和历史状态是否能贯通：

```bash
for cfg in configs/baselines/smoke_*.yaml; do
  echo "=== $cfg ==="
  python -u -m src.main --config "$cfg" || break
done
```

所有 smoke 通过后，先在 CIFAR-10 上各跑 2 至 5 轮检查显存和损失，再启动 200 轮。正式配置示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.main \
  --config configs/baselines/cifar10_dual_a0p1_moon_cpr2_r200.yaml \
  2>&1 | tee outputs/cifar10_baselines/moon-cpr2-r200.txt
```

## 6. 尚未宣称完成的部分

- 本机环境没有 PyTorch，已完成语法检查、接口注册测试和纯 NumPy 的 FBLG 选择测试；涉及真实反向传播的完整 smoke 必须在 3090 服务器执行。
- FedCFA 公开代码没有实际加入论文描述中的 FDC 项，当前适配器忠实采用其可执行的分类、正反事实和负反事实三项损失；论文中必须披露这一点。
- FBLG 用开源可运行的精确枚举/确定性贪心替代商业 Gurobi。10 客户端规模下候选集合很小，会走精确枚举。
- 每个基线仍需在同等预算下调参；“代码跑通”不等于“论文数值已经复现”。
