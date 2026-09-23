# forcefields/utils.py 可封装逻辑审计

> 审计状态（2026-09-22，复审于 `f559b64`「stage complex untangling workflow」及其后的工作树改动之上）：
> 只读审计，待评审。本文仅登记可封装点与复用统计，不含实施改动。
> 行号为复审快照时的 `hotpot/cheminfo/forcefields/utils.py`（4481 行），后续编辑会漂移，落地前以 `grep` 复核。
> 复审结论：本轮业务逻辑调整**未做任何抽象/reduce**——各 idiom 计数基本不变（仅 A1 由 15→14），下列条目全部依旧成立，本次仅刷新行号；无过时条目需删除。团队已顺手抽出 `_ligand_candidate_sort_key`(649) 并把 `_has_hard_acceptance_failure` 更名为 `_has_unreturnable_frame_failure`(637)，方向一致但均非本目录既有条目。

## 1. 审计动机

| 项 | 内容 |
|---|---|
| 背景 | `forcefields` 包共 5462 行，其中 `utils.py` 独占 4481 行，承载全部业务逻辑。 |
| 问题 | 大量机械 idiom（坐标拷贝、原子索引映射、piercing 计数、帧记录…）与中层逻辑块在多处逐字/近似重复，抬高行数并分散"同一件事"的实现。 |
| 目标 | 从现有代码中提取恰当抽象，用公共方法调用消除逻辑冗余，尽可能减少行数，同时提升可读性与"单一事实来源"。 |
| 约束 | 不改动算法控制流与化学/几何判据（受既有测试逐条锁定）；保持公开签名与行为。 |
| 方法 | 通读 `utils.py` 全文 + 对高频 idiom 做 `grep` 计数定位；对照已有良好抽象作为拆分范式。 |
| 范围 | 两层：(1) `utils.py` 内部封装（§3–§5）；(2) 版本分派三文件折叠（§7）。四维综合权衡见 §8。 |

## 2. 基线：已有的良好抽象（正面参照，不动）

| 抽象 | 说明 |
|---|---|
| `_bond_key`（复用 11 处） | 统一"排序端点 idx"，是其余 idiom 应效仿的范式。 |
| `_receive_worker_result` + `_validated_worker_coordinates` | IPC 生命周期/结果校验，两条 worker 路径共用。 |
| `_copy_molecule_metadata` | working-copy 元数据拷贝，已抽出。 |
| `_topology_checks` / `_forcefield_acceptance_checks` | 已把 `evaluate_structure_acceptance` 的两大块拆分为独立检查函数。 |
| `_OpenBabelOptimizer` | 单一优化引擎，所有工作流共用。 |

结论：作者已有拆分意识；下列条目属"同类拆分尚未做完"，非架构性缺陷。

## 3. 审计结果

### A. 工具函数级（micro-utilities）——纯机械、行为等价、可自由做

| # | 重复 idiom | 次数 | 出现行 | 建议 helper |
|---|---|---|---|---|
| A1 | `np.asarray(…, dtype=float).copy()`（防御性浮点拷贝） | 14 | 1277,1307,1311,1357,1360,1385,1393,1407,1604,1611,1641,1652,3213,3229 | `_float_copy(a)` |
| A2 | `{id(atom): index for index, atom in enumerate(atoms)}`（原子→行号映射） | 6 | 833,1055,1070,1948,1949,3531 | `_atom_index_map(atoms)` |
| A3 | `0 if report is None else len(report.piercings)` | 6 | 1274,1304,1354,1381,2657,3409 | `_piercing_count(report)` |
| A4 | `positions[id(bond.atom1)] / [id(bond.atom2)]`（成对取端点行号） | 5 对 | 756-757,1060-1061,1075-1076,1923-1924,1963-1964 | `_bond_endpoints(bond, positions)` |
| A5 | `(int(atom.id), int(atom.atomic_number), int(atom.formal_charge))`（原子身份三元组） | 3 | 785,1915,3535 | `_atom_identity(atom)` |
| A6 | `tuple(dict.fromkeys(messages))`（保序去重） | 3 | 1424,1704,3316 | `_dedup(messages)` |
| A7 | `float(backend.Energy(…)) * _energy_factor_to_kj(backend.GetUnit())` | 2 | 1817,2278（def 1771） | `_backend_energy_kj(backend, …)` |

### B. 逻辑块级（logic blocks）——中等收益，部分需小心

| # | 可封装逻辑块 | 次数 | 出现行 | 建议 | 风险 |
|---|---|---|---|---|---|
| B1 | 帧/movie 记录器（"按 `save_movie` 追加帧 + 跟踪最优帧 + 收尾去重末帧"三套独立实现；`save_movie` 全文引用 45 处） | 3 | untangle 1277-1424；restore 1604-1704；optimizer 2299-2470 | `_MovieTrace` 值对象（`append`/`finalize`/`best`） | 中 |
| B2 | "rescan → 更新 minimum/best → 记帧"三连（untangle 内近逐字重复） | 3 | 1300-1313,1350-1361,1376-1394 | 局部 `observe()`（配合 A1+A3+B1） | 中 |
| B3 | conformer 轨迹落盘 `conformer_clear + conformer_add + conformer_load(idx)` | 3 | 2462-2468,3153-3158,3250-3252 | `_store_conformer_trace(mol, coords, energies, index)` | 低 |
| B4 | Setup 或抛错 `backend.Setup(obmol, _make_constraints(mol)) else ForceFieldSetupError(…,"setup")` | 2 | 1808,2138 | `_setup_backend_or_raise(backend, mol, requested, effective)` | 低 |
| B5 | spawn worker 流水线 `get_context("spawn") + Pipe + Process + _receive_worker_result + _validated_worker_coordinates` | 2 | 3022,3092 | `_run_build_worker(target, args, *, timeout, seed, expected_atoms, …)` | 低 |
| B6 | "发失败检查，否则发一条通过检查"（acceptance 收尾模式） | 6 | 1159,3642,3685,3761,3770,3780 | `_checks_or_pass(failures, make_pass_check)` | 低 |
| B7 | `passed = all(c.passed or c.severity != "error" …)` | 3 | 3614,3626,3819 | 复用 `ForceFieldValidationReport.failures`（`passed = not failures`） | 低 |
| B8 | bond 签名 `_topology_bond_signature`(751-765) 与 `_bond_commit_signature`(1918-1928) 近重复 | 2 | 751,1918 | 统一底座，topology 版包一层构造 `BondTopologySignature` | 低 |
| B9 | 金属→配体供体分组（遍历 `is_metal_ligand_bond` 收集每金属供体） | 3 | 1065,1864,3859（各函数内 metal→donor 遍历） | `_metal_donor_pairs(mol)` 生成 `(metal, donor)` | 中 |
| B10 | worker 异常 `__init__` 重复：`BuildWorkerError`(454-469) 与 `ComplexBuildWorkerError`(485-499) 四属性赋值逐字相同 | 2 | 454,485 | 抽 `_WorkerErrorInit` mixin | 低 |

### C. 大函数分解（结构级，收益大但需 review）

| 目标 | 现状 | 建议 | 风险 |
|---|---|---|---|
| `evaluate_structure_acceptance`（3552-3821，约 270 行） | 已是"分类产出检查"形状，但检查产出内联 | 照 `_topology_checks`/`_forcefield_acceptance_checks` 范式，再拆 `_coordinate_checks`/`_pair_distance_checks`/`_bond_geometry_checks`/`_bond_ring_and_coordination_checks`；配合 B6/B7，主体可降至 ~60 行 | 中（核心策略，测试覆盖广，逐块回归） |
| `AcceptanceCheck(...)` 构造（32 处，775…3780） | 多为"失败检查(带 atom/bond_indices+message)"或"通过检查(仅 name/measured/threshold)"两形状 | 加工厂 `AcceptanceCheck.failure(...)` / `AcceptanceCheck.ok(...)` 削减样板 | 低 |

## 4. 复用统计总表（按出现次数排序）

| idiom / 逻辑块 | 次数 | 类别 |
|---|---|---|
| `AcceptanceCheck(...)` 构造 | 32 | C |
| `save_movie` 关注点（B1 覆盖） | 45 引用 / 3 套实现 | B1 |
| 防御性浮点拷贝 A1 | 14 | A |
| conformer 落盘 clear+add+load（B3） | 3 套 | B3 |
| 原子→行号映射 A2 | 6 | A |
| piercing 计数 A3 | 6 | A |
| "失败否则通过"检查 B6 | 6 | B |
| bond 端点行号 A4 | 5 对 | A |
| 原子身份三元组 A5 / 保序去重 A6 / 金属供体分组 B9 / passed 归约 B7 | 各 3 | A/B |
| 能量换算 A7 / Setup-或抛错 B4 / spawn 流水线 B5 / bond 签名 B8 / worker 异常 init B10 | 各 2 | A/B |

## 5. 优先级与预计收益

| Tier | 内容 | 性质 | 预计行数 |
|---|---|---|---|
| 1（先做） | A1–A7、B3、B4、B5、B8、B10、`AcceptanceCheck` 工厂 | 纯机械、行为等价，逐个引入 + 每步跑测试 | −150~250 |
| 2（次做） | B1(`_MovieTrace`)、B2、B6、B7、B9 | 需小心，测试钉死事件序列 | −80~150 |
| 3（独立 PR + review） | C 的 `evaluate_structure_acceptance` 分解 | 核心策略，逐块回归 | 主体 −200 级 |

落地原则：每个 helper 单独提交；不夹带化学参数/几何判据/优化算法变化；不重构 `_untangle_ring_piercings` 与 `_restore_coordination_bonds_incrementally` 的控制流本身。

## 6. 与版本分派方案的关系（边界说明）

| 项 | 说明 |
|---|---|
| 正交性 | 本文全部条目位于 `utils.py` 共享逻辑层，与 `ff.py`/`ff39.py`/`utils39.py` 的版本分派互不影响，可独立落地。 |
| 既有政策 | 版本分派是既定设计，见 `plan/reviews/forcefields_python39_module_split.md`（真实签名可见、`isinstance`/pickle 一致、adapter 显式注入、`utils.py` 保持无 `ctypes`、版本选择只在包入口）。 |
| 折叠议题 | 三文件折叠（约 −1000~1150 行）见 §7；它会触碰上述部分围栏，权衡与围栏修订建议见 §7.4 与 §8。 |

## 7. 版本折叠：ff / ff39 / utils39 → 单一实现

### 7.1 可折叠的事实细节（证据）

| 事实 | 证据 |
|---|---|
| `ff39.py` 与 `ff.py` 仅差 8 行 | `diff` 结果：docstring(L1)、一处 `import utils39`(L10)、及 worker 引用 `_utils39.*` vs `_utils.*`（ff.py 149/234/341/409-410）。两文件各 450/451 行。 |
| `ff.py` 是纯转发门面 | 非 worker 函数逐字转发 `_utils.X`；4 个 worker 函数调用 `_utils._X_workflow(..., worker_target=_utils._…)`，与 `_utils.X` 完全等价 ⇒ `ff.build3d ≡ utils.build3d` 等。 |
| `utils39.py` 唯一独有内容是 OB3.1 seed | 69 行中仅 `_seed_openbabel_random`(19，含 `ctypes` `srand` + `vector3.randomUnitVector`) 是版本差异；两个 worker 入口(30/58) 只是 `_utils._run_*_worker(..., seed_initializer=…)` 的薄包装。 |
| 唯一真实版本差异 = 一个函数 | `utils._seed_openbabel_random`(1796) 仅设环境变量；`utils39` 版多一段 ctypes srand hack。其余全部共享。 |
| 内部 `_*_workflow`/公共壳分裂只为注入 worker | `_build3d_workflow`(3914)/`build3d`(3958)、`_build_complex3d_workflow`(4026)/`build_complex3d`(4088)、`_complexes_build_workflow`(3437)/`complexes_build`(4287)、`_build_and_optimize_workflow`(4191)/`build_and_optimize`(4356)。 |
| worker 注入贯穿 6 处 + 3 个 Protocol | `worker_target` 于 3018/3066/3464/3920/4043/4218-4219；Protocol `_SeededBuildWorker`(119)/`_ComplexBuildWorker`(129)/`_SeedInitializer`(146) 仅作注入类型标注。 |
| seeding 只发生在子进程 | spawn 子进程重新 import `utils`；`_seed_openbabel_random` 从不在父进程调用 ⇒ 版本可在子进程内自判定，无需父→子注入。 |
| 存在潜在不一致（折叠可修正） | 测试多以 `import utils as ff` 直连，故 3.9 下走的是 utils 内**无 srand** 的 seed，而生产走 `ff39` 有 srand ⇒ 两路径 RNG 行为不同。 |

### 7.2 改动建议（合并关系）

| 动作 | 具体 | 行号 |
|---|---|---|
| 删门面 | 删 `ff.py`、`ff39.py`；`__init__.py` 改 `from .utils import *`（去掉版本分支） | ff/ff39 全文；__init__ 6-11 |
| 收敛 seed | `utils._seed_openbabel_random` 自选择：模块级不可变常量 `_OPENBABEL_NEEDS_LEGACY_SRAND = sys.version_info[:2] == (3, 9)`；为真时执行 OB3.1 分支 | 1796 |
| 隔离 ctypes | OB3.1 分支惰性 `from . import _ob31_rng`（即精简后的 `utils39`），`ctypes`/`srand`/`vector3` 只在 3.9 加载 | 新 `_ob31_rng.py` |
| 合并 worker | `_run_*_worker` 并入 `_*_worker`，去掉 `seed_initializer` 形参，直接调 `_seed_openbabel_random` | 2757/2785、2833/2847 |
| 去 worker_target | 从 6 处工作流函数移除该形参，直接引用模块级 worker | 3018,3066,3464,3920,4043,4218-4219 |
| 合并壳 | `_build3d_workflow`→`build3d`、`_build_complex3d_workflow`→`build_complex3d`、`_build_and_optimize_workflow`→`build_and_optimize`；`complexes_build` 与 `_complexes_build_workflow` 去 worker 后合一（后者因两处调用保留为内部 helper） | 见 7.1 |
| 删 Protocol | 删 `_SeededBuildWorker`/`_ComplexBuildWorker`/`_SeedInitializer` | 119/129/146 |
| 改测试 | 更新 `test_complex_untangling_workflow.py`（删 `import ff, ff39`；`test_public_staged_workflow_attempt_defaults_match_between_versions` 改测单一公共 API） | 该文件 6, 849-867 |

预计净删 ~1000~1150 行。

### 7.3 推荐设计（惰性隔离，兼顾鲁棒性）

```python
# utils.py 顶层不出现 ctypes；仅一个不可变常量承载版本判定
_OPENBABEL_NEEDS_LEGACY_SRAND = sys.version_info[:2] == (3, 9)

def _seed_openbabel_random(seed: int) -> None:
    os.environ["OB_RANDOM_SEED"] = str(seed)
    if _OPENBABEL_NEEDS_LEGACY_SRAND:
        from . import _ob31_rng          # 只在 3.9 运行时加载 ctypes/srand
        _ob31_rng.seed_legacy(seed)
```

要点：worker 仍是模块级函数（spawn 可 pickle）；dataclass/exception 仍单点定义于 `utils.py`（`isinstance`/pickle 一致）；`ctypes` 经惰性导入停留在 `_ob31_rng.py`，3.10+ 永不加载；版本判定是"一次算定的不可变常量"，非可变后端全局。

### 7.4 与既有围栏的关系（需评审 / 修订）

参见 `plan/reviews/forcefields_python39_module_split.md`。

| 既有围栏 | 折叠后的处置 |
|---|---|
| 真实签名可见 | 保留——`utils.py` 公共函数本就有显式签名；门面未提供额外价值，只带来漂移风险。 |
| `isinstance`/pickle 一致 | 增强——dataclass/exception 仍单点定义于 `utils.py`，删门面不影响。 |
| adapter 显式注入、不用可变全局 | 用**不可变常量**替代注入；非"可变的当前后端"，符合其精神但改变实现手段——需评审确认可接受。 |
| `utils.py` 无 `ctypes` | 经惰性 `from . import _ob31_rng` 保持顶层无 `ctypes`（仅 3.9 加载）；若接受直接内联则需修订此围栏。 |
| 版本选择只在包入口 | 下沉为 `_seed_openbabel_random` 内的单一常量；需把该围栏改述为"版本判定集中于单一常量/单一函数"。 |
| 双门面签名对拍测试(#2) | 随门面删除而失效；其防漂移目的因"只剩一份签名"而自然达成。 |

## 8. 四维综合考量（逻辑紧密性 / 可读性 / 复用性 / 鲁棒性）

| 维度 | 现状 | 本计划（§3–§5 封装 + §7 折叠） |
|---|---|---|
| 逻辑紧密性 | 一个 7 行版本差异摊到 3 文件 + 6 处穿参；"帧记录/piercing 计数"散落 3 套 | 版本差异收敛为单一自选择函数；机械 idiom 收敛为公共 helper |
| 可读性 | 门面重复 ~900 行（ff.py 450 + ff39.py 451）遮蔽真实结构；`evaluate_structure_acceptance` 270 行内联 | 单份公共 API + 分类检查函数；主体大幅缩短 |
| 复用性 | 共享逻辑已在 utils（好），但门面 + 内部壳重复约 1140 行 | helper/工厂集中；净删约 1000~1150（折叠）+ 230~400（封装） |
| 鲁棒性 | 显式注入、ctypes 隔离、入口选版、守卫测试（强项）；但双门面需人工保持同步（漂移风险） | 单点定义强化 `isinstance`/pickle；消除双门面漂移；修正 3.9 测试/生产 seed 不一致；`ctypes` 经惰性导入仍隔离；worker 仍模块级可 pickle。代价：版本判定手段从"注入"变"常量 + 惰性导入" |

综合：本计划在四维上净正——紧密性 / 可读性 / 复用性显著提升，鲁棒性净中性偏正（单点定义与去漂移 > `ctypes` 位置变化）。唯一需拍板项：是否接受把版本判定从"入口注入"改为"`utils` 内不可变常量 + 惰性隔离模块"，并相应修订 `forcefields_python39_module_split.md` 的两条围栏措辞。落地顺序建议：先做 §5 Tier 1（零风险封装）→ 再做 §7 折叠（单独 PR，跑满 3.9/3.10+ 矩阵）→ 最后 §5 Tier 2/3。

---

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
