# `forcefields` 抽象与 reduction 实施报告

> 状态：已实施并完成本地及 Python 3.9–3.14 回归；分支
> `refactor/abstract-ff-utils`。
> 本轮只消除实现重复、拆分职责和强化测试围栏，不改变力场选择、几何阈值、
> 环—键判据、候选选择或优化控制流。

## 1. 审计结论

原审计指出 `utils.py` 存在重复机械操作和过长 acceptance 主函数，这一方向成立；
但并非所有建议都适合落地。

| 结论 | 条目 | 理由 |
|---|---|---|
| 已实施 | A1–A6、B3、B4、B8、B9、acceptance 职责拆分 | 可建立单一事实来源，并保持业务等价 |
| 不实施 | A7 | `Energy()` 与 `Energy(True)` 语义不同；optimizer 还会预取换算因子，强行统一会隐藏梯度请求并增加热路径调用 |
| 不实施 | B1、B2 | 三段轨迹的选帧规则不同：开环按穿环数、配位恢复保留带 `NaN` 能量的拓扑帧、optimizer 按门控/能量且失败时保留末帧 |
| 不实施 | B5 | 两类 worker 的参数、错误类型和 diagnostics 契约不同；统一后需要宽泛 `object`/可变参数和大量模式开关 |
| 不实施 | B6、B7 | 会把明确的检查顺序换成间接样板；报告构造前也无法复用 `report.failures` |
| 不实施 | B10 | 为少量异常属性赋值引入 mixin 会增加 MRO 与 pickle 风险 |
| 不实施 | `AcceptanceCheck` 工厂 | 会扩张公开数据类 API，且不能实质减少不同检查的领域字段 |
| 拒绝原 §7 | 删除 `ff.py`/`ff39.py`/`utils39.py` | 与既定 Python 3.9/Open Babel 3.1 隔离策略冲突 |
| 折中实施 | façade 同名重导出 | 保留版本边界，仅消除无版本差异的薄转发函数 |

原审计中的数量也有偏差：坐标拷贝实际为 16 处；optional scan 的确认穿环计数
并非 6 处；预计“内部 helper 净删 230–400 行”不现实。职责拆分主要降低圈复杂度，
真正的净减行来自安全地压缩 façade。

## 2. 已落地的内部抽象

### 2.1 分子值与拓扑身份

| Helper | 唯一职责 |
|---|---|
| `_copy_coordinates()` | 创建独立、浮点型 Cartesian 坐标快照 |
| `_atom_index_map()` | 建立 atom 对象身份到分子原子表位置的映射 |
| `_bond_endpoint_indices()` | 从上述映射取得一根 bond 的两个端点位置 |
| `_atom_identity()` | 返回事务校验使用的 atom ID、原子序数和形式电荷 |
| `_bond_identity()` | 返回方向无关的端点、键级和键类型 |
| `_piercing_count()` | 读取 optional 几何扫描中的已确认穿环数 |
| `_unique_messages()` | 对 warning 文本作保序去重 |
| `_iter_metal_donor_pairs()` | 统一显式 metal–ligand bond 的金属/供体方向 |

`_topology_bond_signature()` 和 working-copy 事务现在共享 `_bond_identity()`，避免两套
拓扑身份定义发生漂移。三个 metal/donor 消费者仍各自负责索引、分组和化学用途，未被
强行合并。

### 2.2 Open Babel 与 conformer 基础操作

| Helper | 唯一职责 | 明确保留在调用方的策略 |
|---|---|---|
| `_setup_forcefield_backend()` | `Setup` 并生成统一的结构化错误 | VDW pair update、算法选择 |
| `_replace_conformer_trace()` | `clear → add → load` | 最佳帧、末帧、能量与保存策略 |

因此，本轮没有引入一个通用 `_MovieTrace.best()`；不同工作流的化学与诊断选帧规则仍然
清晰可见。

## 3. Acceptance 分层

`evaluate_structure_acceptance()` 已拆为四个只计算单一事实组的 helper：

```text
evaluate_structure_acceptance
├── _coordinate_acceptance_section
├── _topology_checks
├── _forcefield_acceptance_checks
├── _atom_pair_distance_acceptance_section
├── _bond_geometry_acceptance_section
└── _bond_ring_coordination_acceptance_section
```

| Section | 输出 |
|---|---|
| coordinate | 坐标形状、有限性与基础计数 |
| topology | 原子/键身份及允许新增氢的拓扑检查 |
| force-field | setup、能量、梯度、收敛与稳定历史 |
| atom-pair distance | 重叠、过近距离和最小原子间距 |
| bond geometry | 显式键距离、半径归一化键长与最短键 |
| bond-ring/coordination | 环—键关系及配位环境 metrics |

以下行为由 golden test 固定：检查顺序、每个 `AcceptanceCheck` 字段、metrics 的键和值、
`off/basic/standard/strict` 分层、early return，以及只有 `standard/strict` 执行环扫描。

## 4. 版本 façade 的 reduction

既定目录保留：

```text
forcefields/
├── __init__.py   # 唯一 Python 版本选择点
├── ff.py         # Python >= 3.10 / Open Babel 3.2 façade
├── ff39.py       # Python 3.9 / Open Babel 3.1 façade
├── utils.py      # 共享业务与现代 worker
└── utils39.py    # Open Babel 3.1 RNG/worker adapter
```

九个无版本差异的函数在两个 façade 中直接绑定到 `utils.py` 的同一函数对象：

```text
capture_topology                  perturb
evaluate_structure_acceptance     collect_coordination_environments
is_structure_accepted             prepare_coordination_geometry
optimize                          optimize_complex
auto_optimize
```

现代 `ff.py` 的四个 build workflow 也直接绑定共享实现。`ff39.py` 仅为以下四个
worker-sensitive API 保留显式、等签名 wrapper：

```text
build3d
build_complex3d
complexes_build
build_and_optimize
```

这些 wrapper 继续显式注入 `utils39._seeded_ob_build_worker` 或
`utils39._build_ligand_proxies_worker`。因此：

- Python 版本仍只在 package 入口选择；
- Open Babel 3.1 的 `ctypes/srand` 仍完全隔离在 `utils39.py`；
- worker 仍是模块顶层对象，可由 spawn pickle；
- 两套 façade 的 `__all__` 与公开签名保持一致；
- 不再维护无版本差异的薄转发函数副本。

共享函数的 `__module__` 现在正确指向真正实现位置 `forcefields.utils`。没有人为篡改该属性，
避免破坏 pickle；依赖文档工具时应启用 imported-member 展示。

## 5. Reduction 结果

| 范围 | 重构前 | 重构后 | 变化 |
|---|---:|---:|---:|
| `ff.py` | 450 行 | 64 行 | −386 |
| `ff39.py` | 451 行 | 267 行 | −184 |
| `utils.py` | 4481 行 | 4621 行 | +140（职责拆分与具名 helper） |
| `forcefields` Python 源码合计 | 5462 行 | 5032 行 | **−430 行（−7.9%）** |

新增测试不计入生产源码 reduction。

## 6. 行为与测试围栏

- acceptance golden test：锁定四级策略的完整检查序列、关键值和 metrics；
- façade identity test：共享函数必须是 `utils.py` 的同一对象；
- adapter test：Python 3.9 四个 build workflow 必须注入 `utils39` worker；
- signature test：`ff.py` 与 `ff39.py` 全部公开函数签名一致；
- import isolation：3.10+ 不加载 `ff39`、`utils39` 或 legacy `ctypes`；
- pickle test：公共函数和四个 worker target 均可 round-trip；
- force-field、complex、geometry、topology 与 conversion 回归必须通过；
- Python 3.9–3.14 compatibility runner 继续作为跨版本最终围栏。

本轮没有改动任何化学阈值、力场参数、几何关系算法、候选排序、环开结流程、配位键恢复
流程或最终帧保留策略。

## 7. 最终验收结果

| 验收项 | 结果 |
|---|---|
| 本轮 force-field / complex / geometry / topology / conversion 回归 | 300 passed |
| Python 3.9 / Open Babel 3.1.0 | 798 passed, 5 skipped, 3 xfailed；SMARTS 组 255 passed |
| Python 3.10–3.14 / Open Babel 3.2.1 | 每个版本均为 799 passed, 4 skipped, 3 xfailed, 49 subtests；SMARTS 组各 255 passed |
| Ruff | passed |
| Python 3.9 `compileall` | passed |
| `git diff --check` | passed |

Python 3.12–3.14 的测试中出现“多线程进程调用 `fork()`”弃用警告，但未造成失败；这是
后续 Python 版本需要处理的 multiprocessing 生命周期风险，不属于本轮 abstraction 的行为
回归。
