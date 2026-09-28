# Open Babel 原生力场后端实施报告

## 1. 结论

本轮已将 Hotpot 直接依赖 Open Babel 的力场构筑与数值优化路径迁移到 C++：Python 只负责把 `Molecule` 整理为类型和形状固定的连续 NumPy 数组，pybind11 从 buffer 读取数据，随后在单次 C++ 调用内完成 `OBMol` 构造、规则检查、`OBBuilder` 构筑、力场初始化、分 epoch 优化、收敛判断和结果收集。

本轮明确**没有设计或引入 `HpMol`，也没有新增与 `core.py` 并列的 `core.cpp`**。C++ 中的 `MoleculeData` 只是一次调用期间使用的只读传输值，不是 `hotpot.Molecule` 的替代后端。

迁移后的高层业务流程仍由 `hotpot.cheminfo.forcefields` 管理；`hotpot.cheminfo.obWrappers` 负责 Open Babel 原生执行、特例规则和可审计结果。旧的 Python/SWIG 规则规划器以标签 `obWrappers.old`（提交 `503f2a5`）保留为基线，生产代码不再保留两套执行路径。

## 2. 范围与非范围

### 2.1 已实施

- `Molecule` 到原生层的 typed NumPy buffer 边界。
- C++ 内从 buffer 构造 `OBMol`，不从 Python 传递 SWIG `OBMol`。
- C++ 内执行 `OBBuilder`、单段最速下降和完整分 epoch 力场优化。
- C++ 内执行 Open Babel 特例规则：P(V) 构筑规则与线性扭转奇点修复规则。
- 原生层输出能量、梯度、收敛状态、终止原因、选中帧、末帧、epoch 历史和规则证据。
- `forcefields.backend`、`forcefields.optimizer` 和 Python 3.9 适配路径改接原生后端。
- 构建系统按 Python 版本选择 Open Babel 3.1/3.2，并链接对应 wheel 中的头文件和动态库。

### 2.2 本轮不实施

- 不建立长期驻留的 C++ 分子对象，不实现 `HpMol`。
- 不迁移与力场无直接关系的 Hotpot 核心化学对象、图算法、SMARTS、芳香性或 CBond 业务。
- 不承诺用户直接操作 `_ob_native`；该模块是内部 ABI/API。
- 不替代 Open Babel 力场参数、能量模型和优化算法。
- 不在 Python 与 C++ 之间传递或长期保存 SWIG `OBMol`。

## 3. 代码结构

```text
hotpot/cheminfo/
├── forcefields/
│   ├── backend.py              # 高层构筑/短优化适配及错误翻译
│   ├── optimizer.py            # Hotpot 轨迹与原生批量优化结果适配
│   └── utils39.py              # Python 3.9 业务路径，复用原生随机种子入口
└── obWrappers/
    ├── __init__.py             # 稳定公开入口
    ├── contracts.py            # Python 只读结果、规则和帧数据类
    ├── packing.py              # Molecule -> typed contiguous NumPy arrays
    ├── native.py               # 延迟加载扩展、构造原生 MoleculeData
    ├── builder.py              # build(mol, ...)
    ├── forcefield.py           # optimize(mol, ...) 与内部短优化
    ├── registry.py             # available_rules()/inspect_rules()
    ├── reports.py              # 原生报告到 Python contract 的映射
    ├── settings.py             # 数值保护阈值
    ├── _ob_native.pyi          # 原生模块类型接口
    └── _native/
        ├── molecule_data.hpp/.cpp       # buffer 值对象与边界校验
        ├── openbabel_adapter.hpp/.cpp   # MoleculeData -> OBMol
        ├── native_engine.hpp/.cpp       # 构筑、力场循环和结果结构
        ├── rules.hpp                    # condition/action 规则协议
        ├── registry.hpp/.cpp            # 有序规则注册表
        ├── phosphorus_builder.cpp       # P(V) 构筑修正
        ├── degenerate_torsion.cpp       # 线性扭转奇点修正
        └── native_bindings.cpp          # pybind11 边界
```

旧的 `_ob_rules`、`snapshot.py` 及其重复执行路径已从源代码和打包清单中清理。

## 4. 数据边界与执行流

### 4.1 Buffer schema

`packing.py` 生成 schema version 1 的连续数组：

| 数据 | dtype | 形状 | 含义 |
|---|---:|---:|---|
| `atomic_numbers` | `int32` | `(N,)` | 原子序数 |
| `formal_charges` | `int32` | `(N,)` | 形式电荷 |
| `partial_charges` | `float64` | `(N,)` | 部分电荷 |
| `coordinates` | `float64` | `(N, 3)` | Å 坐标 |
| `atom_aromatic` | `uint8` | `(N,)` | 原子芳香标志 |
| `bond_indices` | `int32` | `(M, 2)` | 以零为起点的成键原子行号 |
| `bond_orders` | `float64` | `(M,)` | 键级 |
| `bond_kinds` | `uint8` | `(M,)` | Hotpot 键语义稳定编码 |
| `bond_aromatic` | `uint8` | `(M,)` | 键芳香标志 |
| `unit_cell` | `float64` | `(6,)` 或 `None` | 晶胞参数 |

数组在 Python 端完成一次显式、连续化复制；C++ 端读取 buffer 并构造短生命周期 `MoleculeData`。这种边界避免让 C++ 依赖 Python `Molecule` 的内部对象布局，也避免在每个优化 epoch 往返 Python/SWIG。

### 4.2 调用链

```text
hotpot.cheminfo.forcefields
        |
        | 调用稳定 Python facade
        v
obWrappers.build()/optimize()
        |
        | _pack_molecule(): typed contiguous NumPy arrays
        v
pybind11 _ob_native.MoleculeData
        |
        | C++ make_obmol()
        v
Open Babel OBMol
        |
        +--> PRE_BUILD rules --> OBBuilder
        |
        +--> PRE_FORCEFIELD_SETUP rules
        |       --> force-field setup
        |       --> initialization
        |       --> epoch/perturbation/stopping loop
        v
C++ value result
        |
        +--> coordinates/energy/gradient/history/rule evidence
        v
Python report + 更新 Hotpot Molecule 坐标/轨迹
```

## 5. 规则扩展框架

原生规则由四部分组成：

1. `RuleDescriptor`：稳定的 `rule_id`、版本、阶段和优先级；
2. `RuleCondition`：只判断某个结构是否满足规则条件；
3. `RuleAction`：执行有限、可审计的临时修改，并写入 `RulePlan`；
4. `Registry`：按阶段和优先级提供确定性的规则顺序。

新增 Open Babel 特例时，应新增独立的 C++ `condition/action`，注册到 `registry.cpp`，并补充规则证据、直接测试和力场集成测试。`openbabel_adapter` 是唯一的分子图到 `OBMol` 转换层；`native_engine` 负责执行生命周期；Python facade 不承载化学特例判断。该结构允许后续把更多“直接依赖 Open Babel 的力场操作”加入同一原生引擎，而无需引入 C++ 版 Hotpot 分子对象。

## 6. 行为与错误契约

- 构筑、力场 setup 和数值循环不设静默 Python fallback。
- 原生 setup、能量单位和帧读取错误分别映射为结构化异常。
- 力场结果统一以 `kJ/mol` 和 `kJ/(mol*angstrom)` 暴露，同时保留 Open Babel 原始能量单位字段。
- `retain_frames` 控制坐标帧传回；`retain_epoch_history` 控制标量 epoch 历史，避免不需要时累计大对象。
- 数值失败保留 terminal coordinates、终止原因与可用轨迹证据，供上层质量门控和人工检查。
- dative bond 不能无损映射到当前 Open Babel bond order 时明确拒绝，不静默降级为普通键。

## 7. 构建与版本兼容

- C++ 标准：C++17；绑定：pybind11。
- Python 支持声明保持 `3.9 <= Python < 3.15`。
- Python 3.9 构建依赖 `openbabel-wheel>=3.1.1.23,<3.2`；Python 3.10+ 使用 `openbabel>=3.2.1,<3.3`。
- `setup.py` 从已安装 Open Babel wheel 定位头文件和库，编译 `_ob_native`，并使用相对 rpath 指向同一环境的 `openbabel/lib`。
- 当前 wheel ABI 设置为 `_GLIBCXX_USE_CXX11_ABI=0`，与所用 Open Babel wheels 对齐。
- `MANIFEST.in` 包含类型 stub、README 和原生 C++ 源码，源码发行版可重建扩展。

已验证 Python 3.9/Open Babel 3.1 与 Python 3.11/Open Babel 3.2 的清洁强制编译和原生测试；完整验证记录见测试报告。

## 8. 提交脉络

| 提交 | 作用 |
|---|---|
| `7e2341d` | 建立 typed NumPy 分子 buffer 边界 |
| `1e00af7` | 将 Python facade 改为直接接受 Hotpot `Molecule`，删除旧 snapshot 路径 |
| `c758229` | 实现 C++ `MoleculeData` 与 `OBMol` adapter |
| `de3a5c9` | 实现原生 Open Babel 构筑和力场引擎 |
| `082bfe5` | 暴露轨迹保留控制 |
| `a3de55b` | 完成 pybind11 绑定、构建链接与类型 stub |
| `cb59294` | 增加运行诊断和有界历史 |
| `a227df8`、`ede42bb` | 将规则测试迁至原生后端并校准 inspect 行为 |
| `633f1d4`、`16f4819` | 将 optimizer 与阶段扫描测试迁至新接缝 |
| `1d603f0` | 生产 `forcefields` 路径改用原生循环，删除 Python/SWIG 数值循环 |
| `24f954f` | 加固参数、异常、能量单位、数值失败和线程契约 |
| `b68b10e`、`3755636` | 把原生构建/测试纳入兼容性与标准测试入口 |
| `3ac1f68` | 增加 dative bond 有损转换拒绝测试 |

## 9. 局限与后续边界

1. 本轮仍复制一次分子数组和一次结果数组；只有建立长期驻留 C++ 分子状态才可能消除此成本，而那属于未来独立架构决策。
2. 原生低层 `seed_random()` 不能在 Open Babel 3.2 的同一进程中重置 `randomUnitVector()` 内部首次初始化的静态 RNG；Hotpot 的受支持可复现工作流依赖全新 `spawn` 子进程在 Open Babel 首次初始化前设置 seed。不能把低层重复调用描述为跨版本可复现保证。
3. Open Babel 的力场覆盖范围与参数缺陷仍然存在；wrapper 只对已识别、已有测试证据的特定病理路径做显式规则处理。
4. `_ob_native` 是内部模块，稳定调用入口是 `hotpot.cheminfo.obWrappers`；更高层的络合物构筑、解结、质量门控和轨迹策略仍应通过 `hotpot.cheminfo.forcefields` 使用。
5. 构建依赖 Open Babel wheel 提供可链接的头文件和共享库；Python 3.11 clean wheel 已完成隔离安装验证，其他 Python/Open Babel 组合仍应在发布矩阵中分别构建 wheel。
