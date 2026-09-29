# `hotpot.cheminfo.forcefields` 接口、引用树与化学语义审计

## 1. 范围与结论

用户所称的 `ff.py` 在当前仓库中实际对应：

```text
hotpot/cheminfo/forcefields.py
```

业务审计基线为分支 `fix/complexes-build-pipeline`、提交 `51f70a0` 的
2,120 行实现；文中接口行号指本轮纯排布后的 2,193 行文件。排布前后 AST
对比确认全部既有函数体、签名、装饰器、类体和常量值均未改变。范围包括：

- 10 个非下划线模块函数；
- 38 个下划线内部函数；
- 12 个公开报告/数据类；
- 8 个公开异常类；
- 4 个内部数据类；
- `_OpenBabelOptimizer` 及其 8 个方法；
- `Molecule`、批量转换、示例、文档及测试中的实际引用。

嵌套在装饰器中的 `synchronized(*args, **kwargs)` 只是闭包实现，不作为独立模块接口列出。

### 1.1 核心结论

1. 当前生产代码只有两条直接入口：
   - `Molecule.build3d()` → `ff.build_and_optimize()`；
   - `Molecule.optimize()` → `ff.auto_optimize()`。
   其余非下划线函数属于文档承诺的高级 API，生产代码暂未直接使用。
2. 主体分层基本成立：公开分派、working-copy 事务、worker IPC、Open Babel 后端和 geometry 门控之间边界清楚；没有发现“捕获异常后无条件假装成功”的兜底。
3. 存在一个确定的死函数：`_ob_optimize()`；还存在一组仅为未来配位几何预留、当前未进入业务的类型和函数。
4. 存在明显的签名与语义漂移：同一个普通有机分子直接调用 `ff.optimize()` 默认使用 UFF，而调用 `ff.auto_optimize()`/`Molecule.optimize()` 默认使用 MMFF94s。
5. 当前最大的化学风险不是代码异常，而是 `build_complex3d()` 在中心配位几何尚未实现时仍以 `quality_level="off"` 提交结构。实测 `[Zn](N)` 得到 0.931 Å 的 Zn–N 键，但该入口自身报告通过。
6. 确实存在为了适配 Open Babel/保证流程运行而改变化学业务语义的行为，包括：络合物强制 UFF、忽略全部约束、临时隐藏金属—配体键、按硬编码 donor 集重算氢、临时开环、把配体逐组件独立构筑。多数是已批准且有报告的显式策略，但并非化学中性；其中“中心配位几何缺失仍允许 build-only 提交”需要优先处理。

## 2. 顶层引用树

### 2.1 从生产入口向下

```text
hotpot.works.convert._build3d
examples/*
    │
    ├─ Molecule.build3d(...)
    │    └─ ff.build_and_optimize(...)
    │         ├─ mol.has_metal == True
    │         │    └─ ff.complexes_build(...)
    │         │         └─ _complexes_build_impl(...)
    │         │              ├─ _build_complex_working(...)
    │         │              │    └─ spawn: _run_complexes_build(...)
    │         │              │         └─ _build_ligand_proxies(...)
    │         │              ├─ _run_optimizer_on_working(...)
    │         │              │    └─ _OpenBabelOptimizer.optimize(...)
    │         │              └─ _commit_working_copy(...)
    │         │
    │         └─ mol.has_metal == False
    │              ├─ outer working copy
    │              ├─ ff.build3d(...)
    │              │    ├─ _ob_build(...)                         [seed=None]
    │              │    └─ spawn: _run_seeded_ob_build(...)       [有 seed]
    │              ├─ ff.optimize(...)
    │              │    └─ _OpenBabelOptimizer.optimize(...)
    │              └─ _commit_working_copy(...)
    │
    └─ Molecule.optimize(...)
         └─ ff.auto_optimize(...)
              ├─ mol.has_metal == True  → ff.optimize_complex(...)
              └─ mol.has_metal == False → ff.optimize(...)
```

### 2.2 高级 API 的独立路径

```text
ff.build3d                  只做普通 OBBuilder 嵌入
ff.optimize                 只做普通 Open Babel 优化，可显式用于含金属分子
ff.build_complex3d          只做配体代理构筑并恢复金属—配体拓扑
ff.optimize_complex         只对现有络合物坐标按络合物策略优化
ff.complexes_build          build_complex3d 的内部阶段 + 全体系优化 + 一次最终提交
ff.perturb                  只扰动当前坐标
ff.collect_coordination_environments
                            只读取显式配位连接；当前未接入构筑主流程
ff.prepare_coordination_geometry
                            预留接口；当前必定抛 NotImplementedError
```

### 2.3 事务、后端与质量门控树

```text
输入 Molecule
  ├─ _capture_workflow_topology → geo.capture_topology
  ├─ _hydrogenated_working_copy
  │    ├─ copy(mol)
  │    ├─ 含金属时隐藏全部 metal–nonmetal bond
  │    ├─ donor valence/H 重算
  │    └─ 恢复 metal–nonmetal bond
  ├─ build/optimize working copy
  │    ├─ OBBuilder
  │    ├─ OBForceField
  │    └─ geo.evaluate_geometry_quality
  └─ _commit_working_copy
       ├─ _prepare_working_copy_commit
       ├─ _snapshot_molecule_for_commit
       ├─ 原子化写回
       └─ 失败时 _restore_failed_commit
```

## 3. 公开函数接口逐项审计

行号均指本轮纯排布后的 `hotpot/cheminfo/forcefields.py`。

### 3.1 `perturb(mol, *, sigma=0.5, seed=None) -> np.ndarray`（第 1780 行）

- 实际功能：使用局部 `numpy.random.Generator` 产生截断在 `±2σ` 的三维高斯位移，原地覆盖 `mol.coordinates`，同时返回新坐标。
- 引用：模块内部没有调用；文档将其列为高级 API；直接测试位于 `test_forcefield_api.py`。优化器内部直接调用 `_perturbed_coordinates()`，不会调用该公开包装。
- 副作用：修改当前构象坐标，不创建 working copy，也不执行质量门控。
- 结论：不是死接口；属于独立坐标工具。与内部 helper 的分离合理，因为优化器需要复用同一个 RNG，而公开函数按 seed 新建 RNG。

### 3.2 `collect_coordination_environments(mol) -> tuple[CoordinationEnvironment, ...]`（第 1791 行）

- 实际功能：从图中移除所有 metal–nonmetal 边；逐金属统计显式 donor、配位数、元素/电荷，并按移除配位边后的连通分量把 donor 分组为 `chelate_groups`。
- 引用：仅三个测试直接调用；主流程和 `prepare_coordination_geometry()` 当前都没有调用。
- 化学边界：`coordination_number` 等于显式 metal–nonmetal 边数，不处理 hapticity、键级、溶剂占位或隐式配位；同一连通配体上的所有 donor 会归为同一组，不代表它们一定形成真实螯合环。
- 结论：实现有效但处于“孤立的未来功能”状态；若短期不实现中心几何，应标注 experimental，而不是让名称暗示已参与构筑。

### 3.3 `prepare_coordination_geometry(mol, *, environments=None, strategy=None, seed=None) -> CoordinationGeometryResult`（第 1830 行）

- 实际功能：只验证输入具有显式 metal–ligand bond，然后无条件抛出 `NotImplementedError`。
- 引用：`_build_complex_working()` 仅在 `coordination_geometry is not None` 时调用；测试验证其明确失败；文档说明为预留接口。
- 未使用参数：`environments`、`strategy`、`seed` 均未消费。
- 结论：这是签名占位符，不是可运行功能。更严重的是高层流程先完成昂贵的 worker 配体构筑，之后才调用它并失败；应在构筑前验证非空策略，或在实现前不让高层签名暴露该选项。

### 3.4 `build3d(mol, *, add_hydrogens=True, seed=None, timeout=1000.0) -> Build3DReport`（第 1844 行）

- 实际功能：建立调用方拓扑快照，创建补氢 working copy，以 OBBuilder 生成初始坐标；有 seed 时使用 spawn worker；只执行 `quality_level="off"` 的不可关闭完整性检查；成功后原子化提交。
- 引用：`build_and_optimize()` 的普通分子分支；文档/测试也允许直接调用。
- 不做的事：不运行力场优化，不接受 forcefield，不执行 basic/standard 距离和穿环门控。
- 命名风险：`Molecule.build3d()` 的实际含义是“构筑并优化”，而 `ff.build3d()` 是“仅嵌入”。二者同名但语义不同，文档虽已说明，仍容易误用。

### 3.5 `optimize(mol, forcefield="UFF", *, algorithm="conjugate", epochs=1, steps_per_epoch=100, add_hydrogens=True, quality_level="standard", ...) -> ForceFieldRunReport`（第 1886 行）

- 实际功能：不重新嵌入坐标；创建补氢 working copy；按普通力场策略解析用户选择；执行分段 Open Babel 优化、可选扰动/VDW schedule、质量门控和最低能合格帧选择；成功后提交。
- 引用：`build_and_optimize()` 普通分支、`auto_optimize()` 普通分支，以及高级直接调用。
- 特殊契约：即使 `mol` 含金属，该函数仍不走配体代理流程；这是用户要求保留的“强制普通优化”捷径。
- 问题：默认 `forcefield="UFF"`，而 `auto_optimize(forcefield=None)` 的普通分支解析成 MMFF94s；同一个操作因入口不同改变后端。

### 3.6 `build_complex3d(mol, forcefield=None, *, candidate_count=5, max_attempts=50, ..., coordination_geometry=None) -> ComplexBuildReport`（第 1937 行）

- 实际功能：要求显式金属—非金属边；强制 effective forcefield 为 UFF；在 spawn worker 中拆开金属—配体边、逐配体生成/筛选/精修候选，再恢复完整拓扑；只以 `off` 门控验收并提交。
- 引用：没有模块内调用；属于高级直接入口，测试和文档直接引用。
- 不做的事：不执行全体系力场优化；默认也不执行中心配位几何放置。
- 严重风险：独立构筑的配体和原金属中心缺少相对放置，`off` 又不检查异常短键/过近。该入口可提交拓扑完整但几何明显不合理的络合物。

### 3.7 `optimize_complex(mol, forcefield=None, *, algorithm="conjugate", epochs=100, ..., quality_level="standard", ...) -> ForceFieldRunReport`（第 1989 行）

- 实际功能：要求显式金属—非金属边；不重建配体坐标；将所有受支持的 forcefield 请求解析为 UFF；按完整体系优化器运行并提交。
- 引用：`auto_optimize()` 的含金属分支；也作为高级直接入口。
- 与 `optimize()` 的差异：主要是输入域验证、强制 UFF 和默认 epoch 数；底层优化器完全相同。
- 结论：与 `optimize()` 不是纯重复，而是化学策略 wrapper；但大量重复参数容易发生默认值漂移。

### 3.8 `complexes_build(mol, forcefield=None, **options) -> ComplexBuildReport`（第 2041 行）

- 实际功能：兼容入口；先把旧参数名翻译成新参数，再调用 `_complexes_build_impl()` 完成“代理构筑 + 全体系 UFF 优化 + 门控 + 单次提交”。
- 引用：`build_and_optimize()` 含金属分支；文档、测试直接引用。
- 问题：`**options` 隐藏了真实签名，IDE、静态类型检查和自动文档无法获知可用参数。作为兼容入口可以接受，但不宜继续作为唯一公开完整流程签名。

### 3.9 `build_and_optimize(mol, forcefield="UFF", *, ..., coordination_geometry=None) -> ForceFieldWorkflowReport`（第 2054 行）

- 实际功能：按 `mol.has_metal` 自动分派。含金属直接走 `complexes_build()`；普通分子先建立外层 working copy，再依次调用公开 `build3d()`、公开 `optimize(add_hydrogens=False)`，最后整体提交。
- 引用：`Molecule.build3d()` 的唯一 forcefield 调用；测试和文档直接引用。
- 优点：普通分子组合流程维持整体事务；build 成功、optimize 失败时不会污染原始分子。
- 代价：普通分子会经历外层 working copy、`build3d` 内层 working copy、`optimize` 内层 working copy和三次提交准备，功能正确但复制/缓存重建冗余。
- 分派问题：仅以 `has_metal` 判定；`[Zn].N` 也先进入络合物分支，然后在下游因没有显式配位边报错。代码与文档所说的“有显式 metal–ligand bond 才走 complex”不完全一致。

### 3.10 `auto_optimize(mol, forcefield=None, *, ..., quality_level="standard", ...) -> ForceFieldRunReport`（第 2140 行）

- 实际功能：按 `mol.has_metal` 分派到 `optimize_complex()` 或 `optimize()`，不重新构筑初始坐标。
- 引用：`Molecule.optimize()` 的唯一 forcefield 调用；测试和文档直接引用。
- 化学差异：普通分子默认 `None` 最终成为 MMFF94s；含金属分子默认成为 UFF。
- 分派问题：与 `build_and_optimize()` 相同，金属存在即视为络合物；不区分游离盐、金属有机共价体系与显式配位络合物。

## 4. 报告、数据和异常接口

### 4.1 公开数据结构

| 接口 | 实际用途 | 生产引用 | 审计结论 |
|---|---|---|---|
| `ForceFieldRunReport`（第 73 行） | 优化选中帧、终止帧、步数、能量、梯度、质量和 movie 摘要 | `_OpenBabelOptimizer.optimize()` 构造；所有 optimize 路径返回 | 必要；`converged` 与 `terminal_converged` 的双语义需要继续保持文档化 |
| `Build3DReport`（第 113 行） | 普通 OBBuilder 嵌入的原子数、加氢数和 off-gate 报告 | `build3d()` 构造，普通组合报告包含 | 必要 |
| `CandidateRejection`（第 120 行） | 保存组件、attempt、拒绝原因和结构化失败证据 | `_build_ligand_proxies()` | 必要；便于解释为何构筑失败 |
| `ComplexBuildDiagnostics`（第 128 行） | 聚合 attempt、初筛接受数、拒绝项和耗时 | `_build_ligand_proxies()`、worker/error | 必要，但 `accepted_candidates` 实际是“通过初筛的候选数”，不保证精修成功，字段名偏强 |
| `BuildWorkerResult`（第 136 行） | worker IPC 成功/错误 envelope | 两个 worker target 与接收器 | 必要；`conformers` 当前仅测试构造，从未发送或消费 |
| `ForceFieldWorkflowReport`（第 147 行） | 普通/络合物组合报告的共同基类 | 只作父类与返回注解，不直接实例化 | 合理抽象，不是死代码 |
| `BuildAndOptimizeReport`（第 156 行） | 普通分子 build + optimize 两阶段报告 | `build_and_optimize()` | 必要 |
| `ComplexBuildReport`（第 162 行） | 络合物 proxy build + 可选 optimization 的统一报告 | `build_complex3d()`、`_complexes_build_impl()` | 必要；`optimization=None` 区分 build-only |
| `ForceFieldSetupReport`（第 167 行） | 区分 plugin lookup 与 backend setup 失败 | `_get_forcefield()`、两类 optimizer setup | 必要 |
| `CoordinationEnvironment`（第 175 行） | 显式配位连接描述 | 只由 `collect_coordination_environments()` 构造 | 有实现但未接主流程 |
| `CoordinationGeometryCandidate`（第 186 行） | 未来中心几何候选 | 没有实例化点 | 当前 dormant placeholder |
| `CoordinationGeometryResult`（第 193 行） | 未来中心几何结果 | 仅作为永远抛错函数的返回注解 | 当前 dormant placeholder |

### 4.2 异常树

```text
ForceFieldError
├─ ForceFieldSetupError(report)
├─ BuildWorkerError(error_type, message, traceback, diagnostics)
├─ BuildTimeoutError + TimeoutError
├─ ComplexBuildError(diagnostics)
│  ├─ ComplexBuildWorkerError(error_type, message, traceback, diagnostics)
│  └─ ComplexBuildTimeoutError + TimeoutError
└─ GeometryQualityError(report)
```

- `ForceFieldError`（第 202 行）是整个模块的业务异常基类，本身不携带额外字段。
- generic build 与 complex build 各有 worker/timeout 类型，字段高度相似，但允许调用者按业务域捕获异常。
- `_receive_worker_result()` 通过 `worker_error_type`/`timeout_error_type` 注入两套异常，因此它们虽无普通函数式调用点，并非死类。
- 可以未来收束为一个带 `operation`/`domain` 的 worker 异常族，但当前拆分具有兼容价值，不建议只为减少类数立即合并。

### 4.3 内部数据结构

| 接口 | 功能 | 引用 |
|---|---|---|
| `_CandidateOptimizationResult` | 候选快速优化的能量、单位和 explosion | `_single_ob_optimization()` → `_build_ligand_proxies()` |
| `_ObservedFrame` | 完整优化每个有效 epoch 的坐标、能量、梯度、门控和稳定历史 | `_OpenBabelOptimizer._observe_frame()`/`optimize()` |
| `_WorkingCopyCommit` | 预验证后的原子化写回 payload | `_prepare_working_copy_commit()` → `_commit_working_copy()` |
| `_MoleculeCommitSnapshot` | 失败回滚所需的对象、图、缓存和构象快照 | `_snapshot_molecule_for_commit()` → `_restore_failed_commit()` |

## 5. 内部函数与方法逐项引用

### 5.1 诊断、锁、力场策略和 Open Babel 基元

| 内部签名 | 实际功能 | 直接引用者 | 重复/风险判断 |
|---|---|---|---|
| `_format_geometry_checks(prefix, checks)` | 把结构化失败格式化为诊断字符串 | `_format_geometry_rejection`、候选穿环路径 | 合理公共 formatter |
| `_format_geometry_rejection(prefix, report)` | 从 `report.failures` 调用上一个 formatter | `_build_ligand_proxies` | 很薄但表达 report 适配语义，可保留 |
| `_serialized_forcefield_call(function)` | 用进程内 RLock 包裹 Open Babel FF plugin 操作 | `_get_forcefield`、`_single_ob_optimization`、`_OpenBabelOptimizer.optimize` 装饰器 | 看似重复加锁，但 RLock 允许嵌套；覆盖直接 lookup 与完整调用，两层均有用途 |
| `_serialized_builder_call(function)` | 按 lifecycle lock → FF lock 顺序串行 OBBuilder | `_ob_build` 装饰器 | 与上项锁域不同，防止 seed 环境窗口竞争，不应合并成普通锁装饰器 |
| `_resolve_complex_forcefield(requested)` | 校验名称后无条件返回 UFF | 三个 complex workflow | 明确改变用户请求；报告会保留 requested/effective |
| `_resolve_organic_forcefield(requested)` | `None`→MMFF94s，否则保留用户选择 | `optimize` | 功能清楚；与公开默认值共同造成入口差异 |
| `_require_explicit_complex(mol)` | 要求有金属且至少一个 metal–nonmetal bond | 三个 complex 入口和 coordination hook | 必要输入域检查；“配位键”定义仍过宽 |
| `_make_constraints(mol)` | 永远返回空 `OBFFConstraints` | 快速/完整 optimizer setup | 参数 `mol` 未使用；这是有意停用约束，不是有效 constraint adapter |
| `_iter_obmol_atoms(obmol)` | 一行包装 `OBMolAtomIter` | `_OpenBabelOptimizer._gradients` | 仅作为 Open Babel/测试 seam，轻微冗余 |
| `_energy_factor_to_kj(unit)` | kcal/mol 或 kJ/mol 转 kJ/mol | 两类 optimizer | 必要；未知单位显式失败 |
| `_get_forcefield(name)` | 加锁获取稳定 Open Babel plugin，失败附 setup report | 两类 optimizer | 必要；不得重新改回不稳定的 `MakeNewInstance()` |
| `_find_forcefield_prototype(name)` | 一行调用 `OBForceField.FindType` | `_get_forcefield`，测试 monkeypatch | 轻微冗余，但隔离 native lookup 便于测试 |
| `_single_ob_optimization(mol, forcefield, steps)` | 候选用的一次性 SteepestDescent、坐标写回和简报 | `_build_ligand_proxies`、死 wrapper `_ob_optimize` | 与完整 optimizer 有 setup/能量提取重复，但候选阶段不需要 movie/梯度/epoch；适合抽共享小 helper，不适合直接合并整条流程 |

### 5.2 working copy、补氢、随机数和原子化提交

| 内部签名 | 实际功能 | 直接引用者 | 重复/风险判断 |
|---|---|---|---|
| `_copy_molecule_metadata(source, target)` | 补拷 charge、properties、model、environment、crystal | 两条 copy 路径 | 暗示 `Molecule.__copy__` 契约不完整；当前必须，但根因更适合在 core copy 语义中统一 |
| `_recalculate_neutral_donor_valence(mol, donor_indices)` | 对中性 N/O/P/S/As/Se donor 重算 valence/implicit H | `_hydrogenated_working_copy` | 化学启发式，不是纯技术 helper |
| `_hydrogenated_working_copy(mol, *, add_hydrogens, seed)` | 创建副本；含金属时隐藏配位边、重算 donor、加氢、恢复边、重排新增 H id | 所有 build/optimize 路径 | 核心事务/化学边界；名称在 `add_hydrogens=False` 时略误导 |
| `_capture_workflow_topology(mol, *, allow_added_hydrogens)` | 委托 geometry 保存输入拓扑契约 | 五个工作流 | 合理 façade |
| `_structure_worker_proxy(mol)` | 创建可 pickle 的结构副本并将 atom id 改为位置 id | 两类 spawn build | 必要；worker 只回坐标，避免自定义 id 干扰位置映射 |
| `_seed_openbabel_random(seed)` | 同时设置 OB 3.2 环境 seed 和 OB 3.1 C RNG | 两个 worker target | 后端兼容行为，不改变力场模型；只影响随机路径 |
| `_atom_commit_signature(atom)` | 原子 id/元素/电荷身份 | `_prepare_working_copy_commit` | 必要小 helper |
| `_bond_commit_signature(bond, positions)` | 端点/键级/键类型身份 | `_prepare_working_copy_commit` | 必要小 helper |
| `_prepare_working_copy_commit(mol, working)` | 在写入前验证原始原子/键未改变，准备新增 H/键与构象 payload | `_commit_working_copy` | 必要事务阶段 |
| `_snapshot_molecule_for_commit(mol)` | 快照调用方对象、图、缓存和构象 | `_commit_working_copy` | 必要事务阶段 |
| `_restore_failed_commit(mol, snapshot)` | 写回失败后恢复上述状态 | `_commit_working_copy` | 必要；捕获 `BaseException` 仅限事务恢复边界，非业务兜底 |
| `_commit_working_copy(mol, working)` | 保持原 Atom/Bond identity 的原子化提交 | 六个工作流 | 必要；普通组合路径会嵌套调用，形成性能冗余 |
| `_perturbed_coordinates(coordinates, *, sigma, rng)` | 纯坐标扰动 | `perturb`、完整 optimizer | 合理复用，无重复逻辑 |

### 5.3 `_OpenBabelOptimizer` 方法

| 方法签名 | 实际功能 | 直接引用/说明 |
|---|---|---|
| `__init__(requested_forcefield, effective_forcefield, *, algorithm, epochs, ...)` | 校验控制参数、创建局部 RNG、取得 backend | 仅 `_run_optimizer_on_working()` 构造 |
| `_setup(mol, obmol)` | backend setup + 空 constraints；VDW 时更新 pair list | `optimize()` 在初始和每个重启 segment 调用 |
| `_set_vdw_cutoff(cutoff)` | 开启 cutoff、设置 VDW，并将 electrostatic cutoff 设为 `1e6` | VDW schedule | Open Babel 把两类 cutoff 绑定开启；`1e6` 是保持静电近似不截断的后端适配 |
| `_optimizer_methods()` | 根据 algorithm 返回 Initialize/TakeNSteps 方法对 | `optimize()` |
| `_initialize_with_budget(initialize, remaining_steps)` | 把共轭梯度初始化消耗计入预算，并用额外 sentinel 区分 limit 与 convergence | `optimize()` 每个 segment | 程序语义修正，不改变目标势能 |
| `_gradients(obmol, factor)` | 计算 RMS/max 梯度并转 kJ/(mol·Å) | `_observe_frame()` |
| `_observe_frame(...)` | 提取坐标、统一能量、梯度、explosion、稳定历史并调用 final-stage geometry gate | `optimize()` 每个有效 epoch |
| `optimize(mol, *, quality_level, topology_reference, quality_thresholds)` | 完整 epoch/segment 状态机，执行扰动、VDW schedule、选择最低能合格帧并生成报告 | `_run_optimizer_on_working()` |

### 5.4 络合物、worker 与工作流内部函数

| 内部签名 | 实际功能 | 直接引用者 | 重复/风险判断 |
|---|---|---|---|
| `_build_ligand_proxies(mol, *, candidate_count, max_attempts, ...)` | 隐藏 metal–ligand 边；逐非金属组件反复 OBBuilder、warmup、score、穿环处理、basic gate、按预评分精修；回填坐标 | `_run_complexes_build` | 核心化学算法；不是普通 helper |
| `_run_complexes_build(mol, connection, ..., seed)` | complex worker target，序列化成功或异常 envelope | `Process(target=...)` in `_build_complex_working` | 与 seeded worker 有少量 envelope 重复，可抽公共 runner，但要保持 spawn 可 pickle |
| `_run_seeded_ob_build(mol, connection, seed)` | 普通 seeded OBBuilder worker target | `Process(target=...)` in `_seeded_ob_build_coordinates` | 同上 |
| `_receive_worker_result(process, receive_connection, send_connection, *, timeout, ...)` | start、临时 seed 环境、receive-before-join、sentinel、exitcode、协议校验和强制清理 | 两类 worker parent | 已成功统一，不重复 |
| `_validated_worker_coordinates(result, *, expected_atom_count, worker_error_type)` | 验证 shape 与 finite | 两类 worker parent | 必要共享边界 |
| `_seeded_ob_build_coordinates(mol, seed, *, timeout)` | 组织普通构筑 spawn、IPC 和坐标校验 | `build3d(seed!=None)` | 必要 |
| `_build_complex_working(mol, *, ..., coordination_geometry)` | 参数校验、补氢 working copy、启动 complex worker、写入坐标、可选调用中心几何 hook | `build_complex3d`、`_complexes_build_impl` | 合理的非提交内部阶段；coordination option 检查位置过晚 |
| `_run_optimizer_on_working(working, *, ...)` | 构造 `_OpenBabelOptimizer` 并调用 | 三个优化路径 | 有效消除 optimizer 构造重复 |
| `_complexes_build_impl(mol, forcefield=None, *, 全部显式参数)` | 完整络合物事务实现 | 仅 `complexes_build` | 与兼容 wrapper 分离合理；它才是真正应供类型工具识别的 canonical signature |
| `_translate_legacy_complex_build_options(options)` | 翻译 10 个旧参数并检测冲突 | `complexes_build` | 兼容层专用，合理 |
| `_ob_build(mol)` | 加锁的内部 OBBuilder primitive | 普通 build、候选 build、seed worker | 必要私有后端入口 |
| `_ob_optimize(mol, ff="UFF", steps=100)` | 只返回 `_single_ob_optimization(...).energy` | **无任何引用** | 确定死函数，可直接删除并加“不得恢复旧入口”测试 |

## 6. 重复功能与冗余结论

### 6.1 确定可清理项

1. **`_ob_optimize()`：确定死代码。**
   - 定义之外没有源码、文档或测试引用。
   - 其全部功能只是丢弃 `_CandidateOptimizationResult` 的其他字段并返回 energy。
2. **`BuildWorkerResult.conformers`：未消费字段。**
   - 只有一个测试证明字段能存值；两个 worker 都不发送，接收端也不读取。
   - 若近期没有多构象 IPC 计划，应删除；若是已批准的协议预留，应明确 `reserved`。
3. **`CoordinationGeometryCandidate`/`CoordinationGeometryResult`：当前 dormant。**
   - 没有构造点，后一类型只作永远抛错函数的注解。
   - 可以保留为明确的后续设计，但不能计入“当前已有功能”。
4. **`collect_coordination_environments()` 与主流程断开。**
   - 算法和测试存在，但 `prepare_coordination_geometry()` 没有调用它。
   - 不是无用算法，却是当前不可达业务功能。

### 6.2 有重复但由事务或策略边界造成

1. **普通 `build_and_optimize()` 的三层 working copy/commit。**
   - 这是最明显的运行时冗余；优点是直接复用公开函数并保证外层原子性。
   - 推荐增加私有 `_build3d_on_working()` 与 `_optimize_on_working()`，公开函数只负责一次事务包装；组合流程直接组合私有阶段。
2. **`optimize()` 与 `optimize_complex()` 参数和主体高度相似。**
   - 差异是真实策略：输入验证、力场解析和默认 epoch；不应直接删除其一。
   - 推荐共享一个参数对象或内部 policy，而非继续复制长签名。
3. **`build_complex3d()` 与 `_complexes_build_impl()` 的 build 阶段重复。**
   - 已通过 `_build_complex_working()` 共享核心，所以当前重复主要是门面、报告和提交，属于合理重复。
4. **两个 worker target 的 envelope try/send/finally 重复。**
   - 可以抽可 pickle 的公共 runner，但收益较小，错误地引入闭包反而会破坏 spawn。
5. **generic/complex worker 异常字段重复。**
   - 可未来以 domain 字段合并；当前异常捕获兼容性比减少类更重要。
6. **`_iter_obmol_atoms()`、`_find_forcefield_prototype()` 是一行函数。**
   - 它们分别是 native iterator 和 plugin lookup 的测试/兼容 seam，属于可接受的小抽象。

### 6.3 不是重复的相似接口

- `ff.build3d()` 与 `ff.build_complex3d()`：前者普通 OBBuilder，后者代理配体构筑，化学流程不同。
- `ff.optimize()` 与 `ff.optimize_complex()`：底层 optimizer 相同，但 forcefield/input policy 不同。
- `_single_ob_optimization()` 与 `_OpenBabelOptimizer.optimize()`：前者是廉价候选评分，后者有 epoch、梯度、movie、严格门控和报告；只能抽公共 setup/energy helper，不能互相替换。
- 两个锁装饰器：锁域和锁顺序不同，合并会重新引入 seeded worker 环境竞争。
- `build3d`/`optimize`/`build_and_optimize`：分别提供 build-only、optimize-only 和原子化组合语义，三者都需要存在。

## 7. 是否为了程序运行而改变了化学业务逻辑

答案是：**有，而且不止一处；但需要区分“已批准且显式可见的适配策略”和“当前仍可能误导用户的化学缺口”。**

### 7.1 已批准、显式但并非化学中性的策略

| 行为 | 代码位置 | 为什么这样做 | 实际化学影响 | 结论 |
|---|---|---|---|---|
| 所有络合物请求映射为 UFF | `_resolve_complex_forcefield()` | 当前只有 UFF 能覆盖含金属体系 | 用户请求 MMFF94s/GAFF 等不会真正执行；势能面改变 | 已批准；requested/effective 均入报告，但最好增加显式 warning/policy 文档 |
| forcefield constraints 永远为空 | `_make_constraints()` | 旧实现把 Hotpot 0-based 索引错误传给 OB 1-based，约束错原子 | Atom/Bond/Angle/Torsion 上已有 constraint 完全不生效 | 比错误约束更安全，但属于功能停用；必须持续公开说明 |
| 补氢前隐藏全部 metal–nonmetal 边 | `_hydrogenated_working_copy()` | 防止配位边消耗普通配体价态 | 把所有金属—非金属关系都按配位处理，包括可能的共价 M–C/M–H/M–X | 对典型配位络合物合理；对有机金属/金属氢化物风险高 |
| 中性 N/O/P/S/As/Se donor 强制重算 valence/implicit H | `_recalculate_neutral_donor_valence()` | 消除“组装后加配位键”和“直接解析络合物”两条输入路径的氢数差异 | 以硬编码元素集合覆盖 parser 原状态；特殊价态、超价态、自由基可能被误判 | 有回归依据但只是启发式，需限定适用域 |
| 穿环时临时打开单键、非稠合环边 | `_build_ligand_proxies()` + geometry | 让 OBBuilder 下一次尝试有机会解除打结 | 暂时改变配体拓扑，可能改变构象/立体路径；之后恢复并重新评分 | 有边界、有恢复、有门控，属于可接受启发式；不能扩展到芳香/稠合环而无专项化学验证 |
| 每个配体组件独立 OBBuilder 构筑 | `_build_ligand_proxies()` | 把有机配体置于 Open Babel 的目标域，避开金属解析问题 | 丢失配体之间及配体—金属的初始相对取向 | 代理策略本身合理，但必须由中心放置和全体系优化补齐 |
| 选择“最低能且通过门控”的帧 | `_OpenBabelOptimizer.optimize()` | 避免数值低能但几何爆炸/穿环的帧 | 结果不一定是所有观测帧的绝对最低能，而是约束后的最低能 | 化学上比盲选最低能更合理，语义应保持 |
| VDW annealing 时把静电 cutoff 设为 `1e6` | `_set_vdw_cutoff()` | Open Babel 同时开启 VDW/静电 cutoff | 静电并非数学上的无限范围，只是工程上近似不截断 | 后端适配，影响很小且有明确理由 |

### 7.2 当前需要优先纠正或作出明确决策的行为

#### A. `build_complex3d()` 可提交明显不合理的中心配位几何（高）

- 代理配体各自独立构筑，中心金属没有按配位数/几何重新放置。
- `coordination_geometry=None` 时预留 hook 完全跳过。
- build-only 入口只运行 `quality_level="off"`，不会检查短键和 bond-length ratio。
- 成功报告因此只代表 shape/finite/topology 完整，不代表可用的络合物 3D 几何。

动态证据（Python 3.11.16、Open Babel 3.2.1）：

```text
input: [Zn](N)
ff.build_complex3d(..., seed=7)
returned off gate: passed=True
Zn–N distance: 0.9309909759777758 Å
subsequent standard gate: passed=False
failures: short_bond, bond_length_ratio
```

建议：在中心放置尚未实现前，至少不要把该入口包装成“可用络合物结构”。可选方案是：

1. 将它明确改名/标记为 raw proxy assembly，并默认不提交；或
2. 提交前至少执行 `basic/standard`，失败时返回 diagnostics 而不是坏结构；或
3. 优先实现 `prepare_coordination_geometry()`，再保留当前公开语义。

#### B. 普通优化默认力场随入口变化（高）

动态证据：

```text
ff.optimize(read_mol("CCO"), ...)      -> requested=UFF,  effective=UFF
ff.auto_optimize(read_mol("CCO"), ...) -> requested=None, effective=MMFF94s
Molecule.optimize()                     -> auto_optimize -> MMFF94s
```

这不是后端技术要求，而是签名默认值不统一。建议确定一个公开默认策略：

- 若普通有机优化默认 MMFF94s：把 `ff.optimize` 默认改为 `None`；
- 若全库默认 UFF：让 `auto_optimize`/`Molecule.optimize` 也显式为 UFF。

在决定前不能声称三个入口化学等价。

#### C. “含金属”与“显式络合物”分派条件不一致（中高）

- 自动分派使用 `mol.has_metal`。
- 专用实现又要求至少一条 metal–nonmetal bond。
- 实测 `[Zn].N` 由 `build_and_optimize()` 分派到 complex 后抛 `ValueError`。
- 文档声称“有显式 metal–ligand bond 才走 complex”，实际代码并非如此。

建议统一成 `_is_explicit_complex(mol)` policy，并为游离金属盐、金属单原子和真正有机金属共价体系分别定义行为。

#### D. 所有 metal–nonmetal bond 都被当作配位键（高，适用域风险）

`Bond.is_metal_ligand_bond` 的定义只检查“一端是金属、另一端不是”，不检查 bond kind、配位方向或化学类型。因此：

- 配位 N/O/S donor：符合预期；
- 金属—碳 σ 键、金属氢化物、金属卤化物：也被隐藏并进入同一代理语义；
- `collect_coordination_environments()` 也会把这些邻居计为 donor。

这不是本文件新创造的 core 定义，但本轮 forcefield 流程把它用作所有关键分派与补氢依据，使其成为实际化学业务规则。必须增加 bond-kind/体系类型策略，或明确当前仅支持经典配位图表示。

#### E. 候选选择不比较全部“精修后”能量（中）

当前算法：

1. 收集通过初筛的候选；
2. 按 `candidate_score_steps` 后能量排序；
3. 从低到高精修；
4. 遇到第一个精修后通过门控的候选立即停止。

因此最终候选是“预评分最低且首个精修成功”，不是“所有精修成功候选中精修后最低能”。这节省计算，但 `best_candidate_refine_steps` 的名称容易让人误认为做了全量最终比较。建议将策略和字段命名写清，或提供 `refine_all_candidates=True` 的严谨模式。

#### F. 默认补氢会改变成功返回分子的化学组成（中，已批准）

- `build3d`、`optimize`、complex 流程均默认 `add_hydrogens=True`。
- 成功提交会真实增加 H 原子和 X–H 键，而不是仅在后端临时使用。
- 这是已批准的正式契约，不是 bug；但它意味着 `optimize()` 不是纯坐标操作。
- 用户若要求原子数不变，必须显式传 `add_hydrogens=False`。

#### G. `quality_level="standard"` 会改变可接受结果与帧选择（中，已批准）

- 普通/完整优化默认只在通过 geometry gate 的帧中选最低能者。
- 一个后端能量更低但短键、爆炸或穿环的帧不会被选中。
- 这是化学质量控制而非程序兜底；应保留。
- 但 build-only 两个入口使用 `off`，尤其 `build_complex3d()` 与优化入口形成明显安全等级差异。

### 7.3 未发现的危险兜底模式

当前代码中没有发现以下行为：

- 请求的普通 forcefield setup 失败后偷偷切换到另一个 forcefield；
- geometry gate 失败后无条件采用最后一帧；
- worker 异常或 timeout 后返回伪成功报告；
- commit 中途失败后保留部分修改；
- 未知参数被静默忽略。

候选构筑中的 `try/except` 只记录失败并在同一算法内进行有界重试；达到上限后抛 `ComplexBuildError`，不构成无条件兜底。唯一的 backend 替换是 `_resolve_complex_forcefield()` 的显式 UFF policy。

## 8. 建议的处理优先级

### P0：先解决化学契约

1. 决定 `build_complex3d()` 在中心配位放置未实现时是否允许提交；当前实测可提交明显短键结构。
2. 统一 `ff.optimize`、`auto_optimize`、`Molecule.optimize` 的普通分子默认力场。
3. 将自动分派条件从单纯 `has_metal` 改为明确的体系分类策略。
4. 明确经典配位键与共价 M–C/M–H/M–X 的表示边界；不要长期依赖“所有 metal–nonmetal bond 都是 ligand bond”。

### P1：完善化学能力和可解释性

1. 实现或暂时下架 `coordination_geometry` 高层选项；若保留，应在昂贵构筑前 fail fast。
2. 将 donor 补氢规则从硬编码元素集合升级为可审计的价态/键类型 policy，并增加有机金属负例。
3. 明确候选选择是 pre-score-first-passing，或提供全候选精修比较模式。
4. 将 constraint adapter 的“完全停用”状态继续暴露在 API 文档/报告中，直到有正确索引映射测试后再启用。

### P2：清理程序冗余

1. 删除死函数 `_ob_optimize()`。
2. 决定删除或正式启用 `BuildWorkerResult.conformers`。
3. 把 dormant coordination 类型标记为 experimental/reserved，避免被视为已实现功能。
4. 为普通组合流程增加不提交的内部 build/optimize stage，减少三层 copy/commit。
5. 用配置数据类或共享 policy 收束多个长签名，防止默认参数再次漂移。
6. 保留 `complexes_build` 兼容 wrapper，但提供具有完整显式签名的 canonical 公开入口。

## 9. 整理后的文件布局

当前 `forcefields.py` 已按以下顺序排列：

```text
imports
__all__
type aliases and module constants
public report/data contracts
public exception hierarchy
internal workflow data contracts

diagnostic formatting helpers
synchronization and force-field policy helpers
low-level Open Babel primitives
working-copy and transactional commit helpers
stateful Open Babel optimizer
ligand-proxy construction
worker entry points and IPC lifecycle
non-committing workflow stages and compatibility translation

public force-field and coordination interfaces
```

公开运行接口集中在文件底部，并由坐标扰动、配位信息读取和单阶段操作，逐步上升到
`complexes_build()`、`build_and_optimize()` 与 `auto_optimize()` 等组合/自动分派入口。
数据类和异常按要求保留在顶部，因此不与“运行接口位于底部”的规则冲突。

## 10. 动态与结构验证记录

本次除静态调用树和测试代码审查外，使用 `hp-usage` 环境执行了独立探针：

```text
Python      3.11.16
Open Babel  3.2.1
```

探针验证了：

- `ff.optimize()` 与 `ff.auto_optimize()` 的默认 effective forcefield 确实不同；
- `[Zn].N` 自动进入 complex 分支后因缺显式配位键失败；
- `[Zn](N)` 的 build-only off gate 可通过，但 standard gate 随后因极短 Zn–N 键失败。

本轮仅移动既有定义并新增 `__all__` 和分区注释，没有删除接口或改动业务实现。
排布前后的 73 个顶层函数/类及全部既有顶层 AST 节点逐项比较、模块编译、导入及
32 项 `__all__` 导出校验均通过。与 geometry/forcefield 相关的 7 个定向测试模块
合计 `206 passed`。
