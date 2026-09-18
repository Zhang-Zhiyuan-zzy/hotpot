# `hotpot.cheminfo.geometry` 包化与环—键空间关系重构计划

> 状态：设计与实施计划，尚未修改 `core`、`forcefields` 或几何业务代码  
> 审查基线：`fix/complexes-build-pipeline` 分支当前工作树  
> 核心目标：可靠报告有限键线段与平面/非平面环边界之间的空间关系  
> 数学实施契约：[relation.md](./relation.md)；必须先改契约，再改 `settings.py` / `relation.py`  
> 配套调用图（四模块旧草案，G00 重生成）：[HTML](./geo_refactor_call_graph.html) · [Archify 源文件](./geo_refactor_call_graph.architecture.json)

## 1. 最终架构决策

### 1.1 Geometry 只报告事实

`hotpot.cheminfo.geometry` 的职责严格限定为：

- 表示点、无限直线、有限线段、平面、三角形和有序闭合环边界；
- 从 Hotpot 化学对象中提取这些几何对象及稳定索引；
- 计算距离、投影、共面性、交点、接触类型和数值不确定性；
- 在明确声明的环面模型下，报告有限键段是否穿过环。

它不得回答：

- 当前结构是否“合理”“物理”“化学正确”；
- 某一距离是否是坏键长、原子是否太近；
- 一个候选是否通过质量门控；
- 是否应启动、继续或终止力场优化；
- 是否应断开某根环键，以及哪根键在化学上允许断开。

因此，旧版 `geometry.py` 中的 `passed`、`severity`、`reasonable`、force-field
convergence、energy、gradient、covalent-radius acceptance 和 opening-edge eligibility
均应迁出 geometry。`forcefields` 或 Core chemistry 读取几何事实后自行作出业务决定。

### 1.2 首轮只解决一个核心问题

首轮新增/修正的核心关系是：

```text
finite Bond Segment × ordered Ring Boundary
    -> PIERCES | DOES_NOT_PIERCE | UNDETERMINED
```

这三个结果仍是空间事实，不是价值判断：

- `PIERCES`：在声明并完成检查的 surface model 中，有限线段严格横穿环面内部；
- `DOES_NOT_PIERCE`：能够明确证明有限线段没有横穿环面内部；
- `UNDETERMINED`：数值容差、退化输入或非平面环的候选曲面分歧使数学结论不足。

这些状态与业务结论没有隐式映射：

| Geometry 事实 | 只能说明 | 不能由 Geometry 推断 |
|---|---|---|
| `PIERCES` | 在声明模型和覆盖范围内存在严格横穿 | 结构“不合理”、不物理、必须开环或优化必失败 |
| `DOES_NOT_PIERCE` | 在声明模型和覆盖范围内没有严格横穿 | 整体结构合理、力场一定能优化成功 |
| `UNDETERMINED` | 当前模型/数值证据不足以二分 | 结构不合理，或可以按“未穿环”放行 |

边界接触、顶点接触、端点接触和共面接触必须作为更细的 factual detail 保留。
在稳健谓词能够确定它们时，它们是“接触但不横穿”，不得被 geometry 擅自称为坏结构；
只有落入数值容差带、无法稳定区分内部与边界时才是 `UNDETERMINED`。

### 1.3 目标包包含五个职责明确的模块

```text
hotpot/cheminfo/geometry/
├── __init__.py   # 唯一公开入口；只组织导出
├── settings.py   # 所有几何数值容差、标准量倍率和算法预算
├── object.py     # 纯几何值对象
├── convert.py    # Hotpot 化学对象 -> 几何对象，并保留来源映射
└── relation.py   # 只接收纯几何对象的度量与空间关系分类
```

不创建 `geometry/quality.py`。现有质量门控并非 geometry 的职责；把它塞进新 package
只会把旧单文件的职责混杂原样复制到新目录。

`settings.py` 是 geometry 数值参数的唯一代码来源；`relation.md` 是这些参数及判定公式的
唯一规范来源。`relation.py` 不得声明阈值、评分、计算上限或未命名 magic number。

## 2. 当前实现审计

当前 [`geometry.py`](../hotpot/cheminfo/geometry.py) 共 1838 行，公开面包括 4 个类型别名、
1 个枚举、7 个数据类、4 个几何对象类和 17 个函数。它实际混合了四类职责：

1. 纯几何对象与数值关系；
2. Hotpot `Molecule` / `Atom` / `Bond` / `Ring` 的提取适配；
3. topology、元素半径和 force-field 质量策略；
4. 力场开环修复选择。

### 2.1 当前关键调用链

```text
core.py
├── Molecule.is_disorder
│   └── geometry.has_too_close_atoms
├── Molecule.has_bond_ring_intersection
│   └── geometry.has_bond_ring_intersection
├── Molecule.intersection_bonds_rings
│   └── geometry.find_bond_ring_intersections
├── Bond.bond_line
│   └── geometry.Line
├── Bond.bond_line_distance
│   └── geometry.Line.distance_to_line
│       └── geometry.calculate_line_distance
└── Ring
    ├── determine_aromatic -> geometry.points_on_same_plane
    ├── is_disorder -> geometry.has_too_close_atoms
    ├── is_bond_intersect_the_ring -> geometry.bond_intersects_ring
    ├── cycle_places -> geometry.CyclePlanes
    └── closest_edge_to_bond -> geometry.closest_ring_edge_to_bond

forcefields.py
├── workflow setup -> geometry.capture_topology
├── optimizer frame observation -> geometry.evaluate_geometry_quality
├── ligand candidate construction
│   ├── geometry.find_bond_ring_intersections
│   ├── geometry.bond_ring_intersection_checks
│   └── geometry.closest_ring_opening_edge
└── build / optimize public workflows -> geometry.evaluate_geometry_quality
```

源码位置：

- Core 导入和对象门面：[`core.py:34`](../hotpot/cheminfo/core.py#L34)、
  [`core.py:1626`](../hotpot/cheminfo/core.py#L1626)、
  [`core.py:4576`](../hotpot/cheminfo/core.py#L4576)、
  [`core.py:5277`](../hotpot/cheminfo/core.py#L5277)；
- Force-field 候选与逐帧门控：[`forcefields.py:22`](../hotpot/cheminfo/forcefields.py#L22)、
  [`forcefields.py:832`](../hotpot/cheminfo/forcefields.py#L832)、
  [`forcefields.py:1093`](../hotpot/cheminfo/forcefields.py#L1093)；
- 当前环面 kernel：[`geometry.py:318`](../hotpot/cheminfo/geometry.py#L318)、
  [`geometry.py:436`](../hotpot/cheminfo/geometry.py#L436)；
- 当前布尔键—环 API：[`geometry.py:1411`](../hotpot/cheminfo/geometry.py#L1411)；
- 当前综合质量门控：[`geometry.py:1584`](../hotpot/cheminfo/geometry.py#L1584)。

### 2.2 当前实现的核心问题

| 问题 | 当前事实 | 后果 |
|---|---|---|
| 事实与价值判断混合 | `evaluate_geometry_quality()` 同时读取坐标、拓扑、元素半径、FF setup/energy/gradient/convergence | geometry 事实上决定候选是否被接受 |
| 无限直线与有限线段未建模分离 | `Line` 同时承担 line/segment 语义，内部函数再传 `segment: bool`；当前 bond 路径实际传入 `segment=True`，且已有 $t\in(0,1)$ 门控 | 当前 bond 路径没有已证实的延长线误报，但 API 语义易被新调用者误用，且无法从类型上约束 |
| 非平面环使用任意中心扇面 | `_line_intersects_polygon()` 以算术中心连接每条环边 | 结果依赖任意 surface，不能代表唯一穿环事实 |
| 边界被压成穿环 | `_point_in_polygon_2d()` 将边界点归为 `True` | 擦边、顶点接触可能被当作严格横穿 |
| 混用不同量纲的容差 | determinant、cross product、距离和无量纲参数可读取同一 tolerance | 坐标同比缩放后可能改变结论；数值边界没有严格定义 |
| 布尔返回压缩证据 | 平行、共面、退化、数值不稳定和明确分离都可能成为 `False` | `forcefields` 无法区分“未穿环”和“算不清” |
| 共享端点直接短路 | `bond_intersects_ring()` 只要 bond 端点属于 ring 就立即返回 `False` | 折叠结构中从共有端点离开后再次穿面可能漏报 |
| 环覆盖范围未写进结论 | 默认只看 `max_ring_size=8` 且 Core 使用 `networkx.cycle_basis()` | 分子级“没有发现”不能扩张成全图数学证明 |
| 开环策略放错层 | `closest_ring_opening_edge()` 同时检查单键、稠合成员数和距离 | geometry 越权决定化学/修复资格 |
| `Any` 大量掩盖真实对象 | Molecule/Atom/Bond/Ring 和报告均被标作 `Any` | 难以检查调用边界和循环依赖 |

当前测试 [`test_nonplanar_ring_keeps_center_fan_surface_semantics`](../tests/test_cheminfo/test_geometry.py#L144)
只锁定历史行为，并未证明该行为是正确的非平面穿环定义。

## 3. 事实层与策略层的硬边界

| 输入或输出 | Geometry | Core chemistry / forcefields |
|---|---:|---:|
| 点坐标、向量、距离、夹角 | 计算并返回 | 可消费 |
| 最佳拟合平面、最大/RMS 面外偏差 | 计算并返回 | 决定是否满足芳香性等化学阈值 |
| 有限线段是否与给定三角面相交 | 计算并返回 | 可消费 |
| 环边界与键段的接触/横穿/未决事实 | 计算并返回 | 决定重构、警告、放行或复检 |
| `distance < caller_threshold` | 可返回比较事实，但阈值必须由调用方显式提供 | 定义阈值的化学意义 |
| 共价半径、键长是否正常、原子是否“太近” | 不决定 | chemistry / FF policy |
| topology 是否允许新增氢或改变键 | 不决定 | forcefields transaction policy |
| FF setup、energy、gradient、convergence | 不读取 | forcefields |
| 某根环键能否临时断开 | 只提供边—目标键距离 | forcefields + chemistry 决定单键/稠合等资格 |
| 整体结构是否 reasonable / accepted | 不提供 | forcefields / chemistry |

一个重要命名约束是：geometry 中不再出现 `reasonable`、`quality`、`passed`、
`severity`、`acceptable` 或 `opening_edge` 这类业务判断名称。

## 4. 目标接口布局树

下列是首轮完成后的完整目标布局。`public` 名称由 package `__init__.py` 重导出；
`private` helper 只在其所属模块内使用。
每个实现模块在顶部声明自身 `__all__`；Exception、Enum、DataClass 位于 helper 之前，公开函数
集中在文件底部并按“基础度量 -> 复合关系 -> 高级入口”排列。

```text
hotpot/cheminfo/geometry/
├── __init__.py
│   ├── __all__
│   ├── settings: NumericToleranceSettings, SurfaceEnumerationSettings,
│   │             GeometrySettings, DEFAULT_GEOMETRY_SETTINGS
│   ├── object: Point, Line, Segment, Plane, Triangle, Cycle
│   ├── convert: PairScope, RingScope, RingFamily,
│   │            AtomGeometry, AtomPairTarget, BondGeometry, RingGeometry,
│   │            BondRingTarget,
│   │            point_from_atom, segment_from_bond, cycle_from_ring,
│   │            iter_atom_geometries, iter_atom_pair_targets,
│   │            iter_ring_geometries, iter_bond_ring_targets,
│   │            measure_atom_pair_distances, determine_bond_ring_relation,
│   │            iter_bond_ring_findings, scan_bond_ring_relations,
│   │            determine_bond_ring_piercing_state,
│   │            AtomPairDistance, BondRingFinding, BondRingScanReport
│   └── relation: PlanarityKind, LineRelationKind, PointCycleLocation,
│                 SurfaceEmbeddingState, SurfaceSegmentState, PiercingState,
│                 SegmentCycleFeature, SegmentCycleIndeterminacy,
│                 CycleSurfaceModel,
│                 PlanarityMeasurement, LineRelation, PointPairDistance, ClosestCycleEdge,
│                 SurfaceFamilyEvidence, SegmentCycleRelation,
│                 measure_planarity, determine_line_relation, line_distance,
│                 point_segment_distance, segment_segment_distance,
│                 point_pair_distances, find_point_pairs_below_distance,
│                 locate_point_in_planar_cycle,
│                 determine_segment_cycle_relation,
│                 closest_cycle_edge
│
├── settings.py
│   ├── public immutable settings
│   │   ├── NumericToleranceSettings
│   │   ├── SurfaceEnumerationSettings
│   │   ├── GeometrySettings
│   │   └── DEFAULT_GEOMETRY_SETTINGS
│   └── no relation kernels or chemical policy
│
├── object.py
│   ├── public immutable value objects
│   │   ├── Point                 # 包含 from_coordinates() 规范构造器
│   │   ├── Line                 # origin + direction；无限直线
│   │   ├── Segment              # start + end；有限线段
│   │   ├── Plane                # point + unit normal
│   │   ├── Triangle             # 三个有序顶点
│   │   └── Cycle                # 有序闭合 1-D 边界，不隐含内部面
│   └── private construction helpers
│       ├── _coordinates3
│       └── _cycle_edges
│
├── convert.py
│   ├── public selection/data contracts
│   │   ├── PairScope = Literal["all", "bonded", "nonbonded"]
│   │   ├── RingScope = Literal["full_graph", "ligand_skeleton"]
│   │   ├── RingFamily           # NETWORKX_CYCLE_BASIS；记录 Core 的环集合算法
│   │   ├── AtomGeometry         # source Atom + Point + stable key
│   │   ├── AtomPairTarget       # 两个 AtomGeometry + bonded 图事实
│   │   ├── BondGeometry         # source Bond + Segment + stable key
│   │   ├── RingGeometry         # source Ring + Cycle + stable key
│   │   └── BondRingTarget       # 一个待判定的 RingGeometry/BondGeometry pair
│   ├── public factual mapping records
│   │   ├── AtomPairDistance     # Atom 来源映射 + PointPairDistance
│   │   ├── BondRingFinding      # BondRingTarget + SegmentCycleRelation
│   │   └── BondRingScanReport   # chemical selection coverage + factual findings
│   ├── public converters
│   │   ├── point_from_atom(atom: Atom) -> Point
│   │   ├── segment_from_bond(bond: Bond) -> Segment
│   │   ├── cycle_from_ring(ring: Ring) -> Cycle
│   │   ├── iter_atom_geometries(structure) -> Iterator[AtomGeometry]
│   │   ├── iter_atom_pair_targets(structure, pair_scope) -> Iterator[AtomPairTarget]
│   │   ├── iter_ring_geometries(mol: Molecule, ...) -> Iterator[RingGeometry]
│   │   ├── iter_bond_ring_targets(mol: Molecule, ...) -> Iterator[BondRingTarget]
│   │   ├── measure_atom_pair_distances(structure, pair_scope) -> Tuple[AtomPairDistance, ...]
│   │   ├── determine_bond_ring_relation(ring, bond, ...) -> BondRingFinding
│   │   ├── iter_bond_ring_findings(mol, ...) -> Iterator[BondRingFinding]
│   │   ├── scan_bond_ring_relations(mol, ...) -> BondRingScanReport
│   │   └── determine_bond_ring_piercing_state(mol, ...) -> PiercingState
│   └── private typed extraction helpers
│       ├── _AtomLike / _BondLike / _RingLike / _StructureLike Protocols
│       ├── _atom_key
│       ├── _bond_key
│       └── _ring_key
│
└── relation.py
    ├── public factual enums
    │   ├── PlanarityKind        # PLANAR / NONPLANAR / DEGENERATE / UNDETERMINED
    │   ├── LineRelationKind     # INTERSECTING / PARALLEL / COINCIDENT / SKEW /
    │   │                        # DEGENERATE / UNDETERMINED
    │   ├── PointCycleLocation   # INTERIOR / BOUNDARY / EXTERIOR / UNDETERMINED
    │   ├── SurfaceEmbeddingState
    │   │                        # EMBEDDED / PROVEN_NON_EMBEDDED /
    │   │                        # CONSTRUCTION_UNDETERMINED
    │   ├── SurfaceSegmentState  # INTERSECTING / NON_PIERCING /
    │   │                        # EVALUATION_UNDETERMINED
    │   ├── PiercingState        # PIERCES / DOES_NOT_PIERCE / UNDETERMINED
    │   ├── SegmentCycleFeature  # TRANSVERSE_INTERIOR / LINE_EXTENSION_INTERIOR /
    │   │                        # CYCLE_EDGE_CONTACT / CYCLE_VERTEX_CONTACT /
    │   │                        # SEGMENT_ENDPOINT_CONTACT / COPLANAR_CONTACT
    │   ├── SegmentCycleIndeterminacy
    │   │                        # NONFINITE_INPUT / NUMERIC_BAND / TOLERANCE_DOMAIN /
    │   │                        # DEGENERATE_* /
    │   │                        # SELF_INTERSECTION / SURFACE_DISAGREEMENT /
    │   │                        # INCOMPLETE_SURFACE_FAMILY
    │   └── CycleSurfaceModel    # PLANAR_POLYGON /
    │                            # VERTEX_TRIANGULATION_FAMILY
    ├── public factual records
    │   ├── PlanarityMeasurement
    │   ├── LineRelation          # kind + Optional[distance]
    │   ├── PointPairDistance
    │   ├── ClosestCycleEdge
    │   ├── SurfaceFamilyEvidence
    │   └── SegmentCycleRelation
    ├── public primitive measurements
    │   ├── measure_planarity(cycle: Cycle, ...) -> PlanarityMeasurement
    │   ├── determine_line_relation(first: Line, second: Line, ...) -> LineRelation
    │   ├── line_distance(first: Line, second: Line, ...) -> float
    │   ├── point_segment_distance(point: Point, segment: Segment) -> float
    │   ├── segment_segment_distance(first: Segment, second: Segment) -> float
    │   ├── point_pair_distances(points: Sequence[Point], ...)
    │   ├── find_point_pairs_below_distance(points, threshold, ...)
    │   └── locate_point_in_planar_cycle(point, cycle, plane, ...)
    ├── public segment—cycle facts
    │   ├── determine_segment_cycle_relation(segment, cycle, ...)
    │   └── closest_cycle_edge(cycle, segment) -> Optional[ClosestCycleEdge]
    └── private numerical kernels
        ├── _local_length_scale / _predicate_tolerances
        ├── _fit_plane_svd
        ├── _project_to_plane
        ├── _orient2d
        ├── _locate_point_in_polygon_2d
        ├── _segment_plane_intersection
        ├── _segment_triangle_relation
        ├── _enumerate_cycle_triangulations
        ├── _triangle_triangle_relation
        ├── _determine_surface_embedding
        ├── _surface_segment_relation
        └── _aabb_candidates
```

### 4.1 对象层约束

- `object.py` 不导入 Core、Open Babel、RDKit、NetworkX 或 forcefields。
- 几何对象是不可变值对象，只保存几何数据和无争议派生量。
- `Point`、`Line`、`Segment`、`Plane`、`Triangle`、`Cycle` 是可扩展几何层的基础大类；
  即使某类当前没有仓库消费者也不得仅以“零引用”为由删除。特别保留 `Point` 及其
  `Point.from_coordinates()` 规范构造器。
- `Line` 和 `Segment` 必须分离；所有 bond–ring 判断只接受 `Segment`。
- `Cycle` 只代表按顺序闭合的边界，不以名称或属性暗示其具有唯一平面或唯一内部面。
- 构造器只拒绝无法表达为目标对象的形状/类型；非有限坐标、零长 `Segment`、共线或自交
  `Cycle` 仍可表示，以便 `relation.py` 返回精确的退化/未决事实，而不是在对象层提前伪装成
  “输入非法”。
- `object.py` 不提供 `is_intersecting()`、`is_reasonable()` 等关系/判断方法，避免
  `object.py <-> relation.py` 循环依赖。

### 4.2 转换层约束

- `convert.py` 是唯一允许理解 Hotpot `Atom/Bond/Ring/Molecule` 形状的 geometry 模块。
- `PairScope`、`RingScope`、稳定 atom/bond/ring key 和分子级扫描覆盖率均属于该适配层；
  `relation.py` 不知道化学图、cycle basis、配体骨架或最大环尺寸。
- 使用最小结构 `Protocol`、`TypeVar` 和 postponed annotations 表达真实输入/来源类型；
  `TYPE_CHECKING` 仅用于校验 Hotpot concrete classes 是否满足协议。禁止用 `Any` 或宽泛
  `object` 绕过循环导入，也不依赖运行时无法解析的仅静态导入名称。
- 运行时不得导入 `hotpot.cheminfo.core`，因为当前导入链是
  `cheminfo.__init__ -> core -> forcefields -> geometry`。
- 转换不得经 SMILES、RDKit 或 Open Babel 往返，不得修改原分子、键、坐标、conformer 或 ring cache。
- 环识别继续由 Core/NetworkX 负责；geometry 不复制 cycle perception。
- Core 应把当前私有 `_uncached_rings(ligand_skeleton=...)` 收束为一个稳定的按 scope 查询入口；
  在该入口完成前，`convert.py` 不得同时保留多套属性/私有方法 fallback。
- 分子级报告必须声明 `ring_scope`、`max_ring_size`、`ring_family` 和 `scan_complete`；
  这些字段只陈述本次选择和覆盖范围，不宣称结构是否可接受。
- `max_ring_size` 由调用方显式传入。geometry 不以默认值表达“大环不重要”之类的业务假设；
  默认值如需保留，只能位于 Core/forcefields 的具体业务入口。
- `measure_atom_pair_distances()`、`determine_bond_ring_relation()` 与
  `scan_bond_ring_relations()` 是只做适配和事实聚合的一站式入口：它们调用 `relation.py`，
  不得在结果上应用“通过/失败/是否修复”的规则。
- `iter_bond_ring_findings()` 是惰性事实流；`determine_bond_ring_piercing_state()` 在首个
  `PIERCES` 时早退，否则扫描结束后按“见过未决则 `UNDETERMINED`，其余
  `DOES_NOT_PIERCE`”返回三态；`scan_bond_ring_relations()` 才负责生成稠密审计报告。

### 4.3 参数层约束

- `settings.py` 保存所有 geometry 自身的数值容差、标准量倍率、合并距离倍率、AABB padding
  和曲面枚举预算；不得保存共价半径、力场阈值、`max_ring_size` 等化学/业务选择。
- settings 均为不可变数据类；默认实例名为 `DEFAULT_GEOMETRY_SETTINGS`。
- 所有带判定的 relation 接口接收同一个 `GeometrySettings`；不得再并列暴露散装 tolerance 参数。
- 数学常数 $0,1,2,\pi$ 和由公式直接推导的数组维度不属于可配置阈值。
- 参数名称、默认值、单位、允许范围和唯一用途以 [relation.md](./relation.md) 第 2 节为准。

### 4.4 关系层约束

- `relation.py` 只依赖 `settings.py`、`object.py` 和 NumPy；不得导入 `convert.py`、Core 或 forcefields。
  SciPy 仅可用于有基准证据的距离加速。
- `relation.py` 的公开签名只接收纯几何对象或纯数值序列，不接收 `Atom/Bond/Ring/Molecule`。
- 返回值必须携带可并存的空间特征、使用的 surface model、数值容差、未决原因和
  surface-family 完成状态。
- `UNDETERMINED` 不得被内部或 façade 静默转换为 `False`。
- 每个判定分支、符号、公式和 settings 映射必须先写入 [relation.md](./relation.md)，再写 Python；
  Python 实现与测试不得自行发明文档外阈值。

依赖只能沿下列方向流动：

```text
settings <- relation
object <- relation
settings <- convert
object <- convert
relation <- convert       # convert 可引用纯关系结果类型；反向依赖禁止
```

`__init__.py` 只按 `settings -> object -> relation -> convert` 顺序重导出，不实现扫描、阈值或决策逻辑。
四个实现模块均从 `__future__` 导入 annotations，并只使用 Python 3.9 可解析的 annotation；
不得直接使用 `X | None`、`typing.Self`、`StrEnum` 或 `dataclass(slots=True)` 等较新语法/API。

## 5. 关键公开数据契约

### 5.1 `GeometrySettings`

配置对象只在 `settings.py` 定义；完整默认值、范围与唯一用途见 [relation.md](./relation.md#2-settingspy-唯一参数表)。
relation 根据本次对象的局部长度尺度 $L$ 推导分量纲容差：

$\epsilon_r=\max(\epsilon_{\mathrm{rel}},k_{\mathrm{mach}}\epsilon_{\mathrm{machine}})$

$\epsilon_L=\epsilon_{\mathrm{abs}}+\epsilon_rL$

$\epsilon_u=\epsilon_{\mathrm{param}}+\epsilon_L/L$

$\epsilon_A=\epsilon_LL$

$\epsilon_V=\epsilon_LL^2$

$\epsilon_L$、$\epsilon_A$、$\epsilon_V$ 分别只比较长度、面积、体积量，$\epsilon_u$ 只比较
无量纲参数。禁止再用一个 tolerance 同时比较 determinant、cross product、距离和 $t$。
这些容差只表达数值可分辨性，不表示原子碰撞半径、合理键长或化学评分。

### 5.2 `SurfaceFamilyEvidence` 与 `SegmentCycleRelation`

```python
@dataclass(frozen=True)
class ClosestCycleEdge:
    edge_index: int
    edge: Segment
    distance: float
```

```python
@dataclass(frozen=True)
class SurfaceFamilyEvidence:
    enumeration_complete: bool
    enumerated_surface_count: int
    embedded_surface_count: int
    proven_non_embedded_surface_count: int
    construction_undetermined_count: int
    intersecting_surface_count: int
    non_piercing_surface_count: int
    evaluation_undetermined_count: int
```

```python
@dataclass(frozen=True)
class SegmentCycleRelation:
    state: PiercingState
    features: FrozenSet[SegmentCycleFeature]
    indeterminacy_causes: FrozenSet[SegmentCycleIndeterminacy]
    surface_model: Optional[CycleSurfaceModel]
    intersection_points: Tuple[Point, ...]
    closest_boundary_edge: Optional[ClosestCycleEdge]
    surface_evidence: SurfaceFamilyEvidence
    settings: GeometrySettings
```

约束：

- `state` 只回答是否横穿，不回答结构好坏；
- `features` 是集合，因为“共享端点接触”和“线段内部再次横穿”可以同时发生；
- `indeterminacy_causes` 只解释数学结论为何未定，不承载 warning/severity；
- `surface_model` 是封闭枚举，不能用自由字符串；无法建立模型时为 `None` 并记录未决原因；
- `closest_boundary_edge` 同时保存 edge index、对应 `Segment` 和距离；
  `closest_cycle_edge()` 与关系计算复用同一 kernel，禁止维护两套最短距离实现；
- `LINE_EXTENSION_INTERIOR` 只是可选诊断事实：仅在本次平面/曲面求交已经得到无限线参数时记录，
  不允许为每个明确分离的 pair 再启动一轮昂贵扫描；
- 始终满足 `enumerated = embedded + proven_non_embedded + construction_undetermined`，以及
  `embedded = intersecting + non_piercing + evaluation_undetermined`；只有
  `enumeration_complete=True` 且两个 undetermined count 都为 0 时，候选面共识才能产生
  `PIERCES` 或 `DOES_NOT_PIERCE`；
- surface-family 未完成、没有可证明有效的曲面、存在未决候选或候选结论分歧时必须是
  `UNDETERMINED`。

### 5.3 `BondRingScanReport`（`convert.py` 的来源映射记录）

`RingFamily` 是封闭枚举，首轮只提供 `NETWORKX_CYCLE_BASIS`；它记录由谁、用哪类算法产生
环集合，不把普通字符串拼写当作覆盖率契约。

```python
@dataclass(frozen=True)
class BondRingScanReport:
    findings: Tuple[BondRingFinding, ...]
    ring_scope: RingScope
    ring_family: RingFamily
    max_ring_size: int
    selected_ring_count: int
    excluded_ring_count: int
    candidate_pair_count: int
    evaluated_pair_count: int
    piercing_pair_count: int
    does_not_pierce_pair_count: int
    undetermined_pair_count: int
    scan_complete: bool

    @property
    def piercings(self) -> Tuple[BondRingFinding, ...]: ...

    @property
    def undetermined(self) -> Tuple[BondRingFinding, ...]: ...
```

`scan_bond_ring_relations()` 的 `findings` 是稠密记录：每个已求值的候选 Ring×Bond pair
恰有一条 finding，包括 `DOES_NOT_PIERCE`。因此始终满足：

```text
evaluated_pair_count = len(findings)
evaluated_pair_count = piercing_pair_count
                     + does_not_pierce_pair_count
                     + undetermined_pair_count
0 <= evaluated_pair_count <= candidate_pair_count
```

候选 pair 的事实定义必须唯一：对每个已选 Ring，枚举 Molecule 中除该 Ring 自身边之外的
每条 Bond；共享一个环原子的 Bond 仍保留，因为它可能在端点接触之后再次穿面。AABB 只有在
cycle/surface model 已证明有效且枚举完整后才可跳过精确求交；该 pair 仍计为“已求值且明确不穿”。

`scan_complete=True` 仅表示在报告声明的 `ring_scope + ring_family + max_ring_size` 范围内，
所有 `candidate_pair_count` 均已求值。零候选、全部明确不穿和扫描中止由这些计数明确区分，
不提供会把空集或 `UNDETERMINED` 压成 `True/False` 的便利属性。当前 ring family 来自
`networkx.cycle_basis()`；报告不得把本次覆盖范围扩张为“分子全局不存在任何穿环”。

性能敏感的三态查询不构造该稠密报告：`iter_bond_ring_findings()` 逐 pair 惰性产出，
`determine_bond_ring_piercing_state()` 在首个 `PIERCES` 后立即停止；若未见 `PIERCES`，则必须
消费完候选以区分 `UNDETERMINED` 与 `DOES_NOT_PIERCE`。逐帧 forcefields 只在需要完整诊断时
调用稠密扫描。跨帧只能缓存 topology candidates 和组合 triangulation index family；AABB、
实际 triangle surface 与其他坐标派生量必须逐帧重算，或以明确的 coordinate revision 为 cache key。

## 6. 环—键三态判定方案

### 6.1 公共前提

1. 目标始终是有限 `Segment`，无限延长线仅可作为求解中间量；
2. 环是按拓扑顺序排列的 `Cycle` 边界；
3. 扫描层只排除与环边界完全相同的 Bond；共享一个端点的其他 Bond 必须进入关系计算，
   不能提前用 `False` 短路；
4. AABB 只用于已完成有效 surface 构造后的安全窄相位短路，不得掩盖退化、自交或枚举未决；
5. 所有状态均相对于报告中的 surface model、`GeometrySettings` 和 scan scope。

### 6.2 平面或数值上稳定近平面的环

流程：

```text
Cycle vertices
  -> SVD / Newell 得到平面与偏差度量
  -> 将环和候选交点投影到最稳定的二维坐标轴
  -> 计算有限 Segment 与平面的参数 t
  -> 分类交点位于 polygon interior / boundary / exterior
  -> 生成 SegmentCycleRelation
```

事实映射：

| 空间事实 | `PiercingState` | `features` / `indeterminacy_causes` |
|---|---|---|
| `t` 严格位于 `(0, 1)` 且交点稳定落在环内部 | `PIERCES` | `TRANSVERSE_INTERIOR` |
| 无限延长线命中，但 `t` 稳定落在有限线段之外 | `DOES_NOT_PIERCE` | `LINE_EXTENSION_INTERIOR` |
| 线段穿过平面但交点稳定落在环外部/凹口 | `DOES_NOT_PIERCE` | 空集合 |
| 稳定边界、顶点、端点或共面接触 | `DOES_NOT_PIERCE` | 一个或多个对应 `SegmentCycleFeature` |
| 交点、边界或共面关系落入数值容差带 | `UNDETERMINED` | `NUMERIC_BAND` |
| 坐标非有限、线段零长、环退化或自交且无法定义面 | `UNDETERMINED` | 对应的 `DEGENERATE_*` / `SELF_INTERSECTION` |

平面路径也必须填充 `SurfaceFamilyEvidence`：正常简单多边形固定为
`enumeration_complete=True`、`enumerated_surface_count=1`、`embedded_surface_count=1`，且
`intersecting_surface_count`、`non_piercing_surface_count`、`evaluation_undetermined_count`
恰有一个为 1。已证明自交的平面边界记录一个 `proven_non_embedded_surface_count`，最终为
`UNDETERMINED`。非有限、退化、模型选择未决和预算中止的全字段赋值严格采用
[relation.md 第 11 节](./relation.md#11-surfacefamilyevidence-全字段赋值)；不得以
`surface_evidence=None` 绕过统一契约。

这里特别避免两个旧错误：延长线命中不等于有限键穿环；接触事实也不等于 geometry 认为
结构不合理。后续 forcefields 可以独立决定是否把某类 contact 当作阻断条件。

### 6.3 非平面环

非平面闭合折线没有唯一内部曲面，任何成熟几何库也无法替 Hotpot 自动补上这一语义。
首轮采用有限、可审计的 surface-family 共识：

1. 对调用方允许的有序环尺寸枚举完整的 vertex-only triangulation family；
2. 把每个候选面的构造状态分为 `EMBEDDED`、`PROVEN_NON_EMBEDDED`、
   `CONSTRUCTION_UNDETERMINED`；仅有数学证据证明退化、自交或不保持同一边界的候选才可
   排除，不能把“算不清”当作无效后静默丢弃；
3. 对每个 `EMBEDDED` 候选面计算有限 Segment 的严格内部横穿，其结果仍可为未决；
4. 只有 family 枚举完整、没有 construction/evaluation 未决、至少存在一个有效面，且全部有效面
   一致横穿时返回 `PIERCES`；
5. 在同样的完整性前提下，只有全部有效面一致不横穿时返回 `DOES_NOT_PIERCE`；
6. 结论分歧、无有效面、family 未完整枚举、预算中止或数值落入容差带时返回
   `UNDETERMINED`。

`EMBEDDED` 不是未定义的实现占位符。候选必须满足三角形数/边出现次数/Euler 特征/唯一边界/
方向一致性，并对任意三角形对证明
$f(T_i)\cap f(T_j)=f(T_i\cap T_j)$。稳定出现额外交集为
`PROVEN_NON_EMBEDDED`；谓词落入保护带或交集维数无法确定为
`CONSTRUCTION_UNDETERMINED`。完整公式与允许交集表见
[relation.md 第 9 节](./relation.md#9-顶点三角剖分的-embedded-判据)。

8 元环的凸组合三角剖分上限为 Catalan 数 132；这在当前 forcefields 调用方明确选择的
`max_ring_size=8` 下可控。超过调用方预算的环不偷偷采用一个便宜 surface，而是在 scan report
中计入 `excluded_ring_count`；若环已进入候选集但枚举未完成，则该 pair 必须为
`UNDETERMINED` 且 `scan_complete=False`。

该结论是“对声明的候选 surface family 达成共识”，不是对自然界中某个唯一环膜的宣称。

### 6.4 共享端点

当前实现一旦 bond 与 ring 共享原子便直接返回 `False`。新算法应：

1. 把共有端点处的接触单独记录为 `SEGMENT_ENDPOINT_CONTACT`；
2. 从穿越统计中排除该端点本身；
3. 继续检查线段开区间是否在其他位置横穿环面；
4. 若只有共有端点接触，则为 `DOES_NOT_PIERCE + SEGMENT_ENDPOINT_CONTACT`；
5. 若线段内部重新穿面，则为 `PIERCES`；
6. 若两者在容差内无法分开，则为 `UNDETERMINED`。

## 7. 当前公开接口的逐项迁移

仓库生产代码中，当前 `geometry.py` 的外部消费者只有 `core.py` 和 `forcefields.py`；测试是直接
消费者但不构成业务 API。不能把“当前零外部引用”直接等同于“应删除”：基础几何大类按架构价值
保留，偶然暴露的 helper 才按调用链收束。

| 名称 | 当前生产调用事实 | 迁移判断 |
|---|---|---|
| `Point` | 仅在 `geometry.py` 内部构造/转换 | 基础公开值对象，必须保留 |
| `Plane` | 仅在 `geometry.py` 内部使用，但被 Core 平面性/环面路径间接消费 | 基础公开值对象，保留；关系方法移出 |
| `to_point()` | 仅内部构造 helper | 不作为公开 API；能力并入 `Point.from_coordinates()` |
| `LinesRelationship` | 仅内部 line-distance 调用链 | 由完整的 `LineRelationKind` 替换 |
| `get_line_relationship()` | 仅被 `calculate_line_distance()` 内部调用 | 迁入 relation，保留明确的公开关系接口 |
| `calculate_line_distance()` | 经 `Line.distance_to_line()` 被 `Bond.bond_line_distance` 间接调用 | 迁为 `line_distance()`；Core 明确 line/segment 语义 |
| `find_overlapping_atom_pairs()` | 全仓无生产消费者 | 旧价值命名删除；通用 point-pair measurement 能力保留 |

### 7.1 类型、数据类和对象

| 当前公开名称 | 当前职责 | 最终归属/动作 |
|---|---|---|
| `PairScope` | 原子对图范围 | `geometry.convert.PairScope`；bonded/nonbonded 依赖化学图 |
| `RingScope` | full graph / ligand skeleton | `geometry.convert.RingScope` |
| `QualityLevel` | 质量门控等级 | 迁至 forcefields policy |
| `ForceFieldStage` | candidate/final 阶段 | 迁至 forcefields |
| `LinesRelationship` | 无限直线关系 | `relation.LineRelationKind`，补足 coincident/degenerate/undetermined 事实 |
| `AtomPairGeometryIssue` | 把距离称为 issue | 拆为 `relation.PointPairDistance` 与 `convert.AtomPairDistance`；是否是 issue 由调用方决定 |
| `GeometryCheck` | passed/severity | 迁至 forcefields，建议名 `AcceptanceCheck` |
| `GeometryQualityThresholds` | 化学和 FF 阈值混合 | 化学/验收阈值迁至 forcefields；纯 numerical tolerance 进入 `geometry.settings` |
| `AtomTopologySignature` | 原子拓扑快照 | 迁至 forcefields transaction helper |
| `BondTopologySignature` | 键拓扑快照 | 同上 |
| `TopologyReference` | topology + 允许新增 H policy | 迁至 forcefields；快照与 policy 字段应拆分 |
| `GeometryQualityReport` | 通过/失败报告 | 迁至 forcefields，建议名 `ForceFieldValidationReport` |
| `Point` | 近乎空壳、当前无外部消费者 | 在 `object.py` 重建为不可变基础值对象；不能因当前零引用删除 |
| `Line` | 混用无限线和有限段 | 在 `object.py` 只表示无限线；新增 `Segment` |
| `Plane` | 三点平面及关系方法 | 在 `object.py` 只保存点/法向；关系方法迁 `relation.py` |
| `CyclePlanes` | 边界、扇面和关系混合 | 用 `object.Cycle` 替换；不再暗示唯一 surface |

### 7.2 函数

| 当前公开函数 | 最终动作 |
|---|---|
| `to_point()` | 删除；使用 `Point.from_coordinates()` 或 `point_from_atom()` |
| `get_line_relationship()` | 改为 `relation.determine_line_relation()`，返回精确枚举/记录 |
| `calculate_line_distance()` | 改为 `relation.line_distance()` |
| `points_on_same_plane()` | 改为 `relation.measure_planarity()`；Core chemistry 自行应用芳香性阈值 |
| `find_overlapping_atom_pairs()` | 纯关系层改为 `find_point_pairs_below_distance()`，`convert` 映射回 Atom；“overlap”政策迁出 |
| `has_overlapping_atoms()` | 不保留 geometry bool；调用方读取距离事实 |
| `find_too_close_atom_pairs()` | 删除该价值命名；绝对/半径阈值政策迁 Core/forcefields |
| `has_too_close_atoms()` | 同上；`Molecule/Ring.is_disorder` 由 Core chemistry 决策 |
| `bond_intersects_ring()` | 替换为 `convert.determine_bond_ring_relation()`；内部只适配并调用纯 relation，不再返回裸 bool |
| `find_bond_ring_intersections()` | 替换为 `convert.scan_bond_ring_relations()`，稠密保留全部三态 finding 与覆盖计数 |
| `bond_ring_intersection_checks()` | 迁至 forcefields；它把事实变成 passed/error |
| `has_bond_ring_intersection()` | 含混旧 bool 删除；hot path 改为 `determine_bond_ring_piercing_state()`，在首个 `PIERCES` 早退且不压缩未决；完整诊断读取 report |
| `closest_ring_edge_to_bond()` | 纯关系层只保留 `closest_cycle_edge()`；Core façade 将 edge index 映射回 Bond |
| `closest_ring_opening_edge()` | 迁至 forcefields repair helper；单键/非稠合是化学策略 |
| `capture_topology()` | 迁至 forcefields transaction helper |
| `evaluate_geometry_quality()` | 整体迁出并改名为 forcefield/structure acceptance API |
| `is_geometry_reasonable()` | 从 geometry 删除；不得保留同义 wrapper |

项目当前没有需要维护的旧 geometry API 包袱，因此最终状态不保留 deprecated alias、参数翻译
或静默 bool 投影。迁移提交中可以短暂保留可运行的中间态，但最终分支必须删除旧入口并同步更新
全部内部调用、测试和文档。

## 8. Core 接入方案

| 当前 Core 入口 | 改造后 |
|---|---|
| `Molecule.has_bond_ring_intersection` | 完整诊断改为 `Molecule.bond_ring_relations()`；hot path 改为 `Molecule.bond_ring_piercing_state()` 并返回三态 |
| `Molecule.intersection_bonds_rings` | 由 `report.piercings` 替代，并允许调用者同时读取 `undetermined` |
| `Bond.bond_line` | 改为 `bond_segment`；若确需无限轴线，另用显式名称 `bond_axis`，避免 bond 默认表达继续混淆 line/segment |
| `Bond.bond_line_distance` | 按真实意图改为 `bond_segment_distance` 或显式 `bond_axis_distance` |
| `Ring.cycle_places` | 改为 `Ring.geometry_cycle -> Cycle` |
| `Ring.is_bond_intersect_the_ring` | 改名 `Ring.relation_to_bond()` 并返回 `BondRingFinding`；非 bool 返回不得保留 `is_` 命名 |
| `Ring.closest_edge_to_bond` | 可保留纯几何门面，但返回距离证据；不判断可否开环 |
| `Ring.determine_aromatic` | 消费 `measure_planarity()` 的偏差，由 Core chemistry 选择芳香性阈值 |
| `Molecule/Ring.is_disorder` | 消费 atom-pair distance，由 Core chemistry 定义 0.5 Å 等业务阈值 |

Core 仍拥有分子图和 ring perception。建议把 `_uncached_rings()` 收束为一个稳定、无副作用的
按 scope 查询接口；算法首轮仍为 `networkx.cycle_basis()`，不在本计划中扩展为全部 simple cycles。

## 9. Force-fields 接入方案

当前 `_build_ligand_proxies()` 将检测、判断和动作混在同一条 geometry API 链中。改造后：

```text
forcefields candidate workflow
  -> working_mol.bond_ring_piercing_state(ring_scope="ligand_skeleton", max_ring_size=8)
     Core 委托 geometry.convert 的惰性三态适配器；适配器逐 pair 调用纯 geometry.relation
  -> 获得 PiercingState；需要诊断时另取完整 BondRingScanReport
  -> forcefields policy
       PIERCES          -> 进入开环/重构策略
       DOES_NOT_PIERCE  -> 继续候选流程
       UNDETERMINED     -> 不因未知而开环；继续普通优化并在后续/最终帧复检
  -> forcefields selector 根据键级、稠合关系等选择可断环键
  -> geometry 只提供候选边与目标键的 segment distance
  -> forcefields 执行 hide / optimize / restore / re-scan
```

首轮冻结的 FF 策略如下，避免三态改造留下行为空洞：

- `PIERCES`：可进入既定开环修复；是否允许断具体环键仍由 chemistry/FF policy 决定；
- `DOES_NOT_PIERCE`：本项不触发开环，继续普通流程；
- `UNDETERMINED`：既不视为 clear，也不单凭未知触发开环；继续普通优化，在下一观察帧和最终帧
  重新判定；最终仍未决时发出 warning 并返回最后一帧，`save_movie=True` 时保留全部帧；
- 未决不得造成无限 retry、异常终止或丢弃结构。上线前必须统计真实样本中
  `UNDETERMINED` 的比例与原因分布；比例异常时先修 predicate/fixture，不得靠 policy 吞掉。

逐帧门控调用 `determine_bond_ring_piercing_state()`：确认 `PIERCES` 时早退，未确认时继续扫描以
保留 `UNDETERMINED`。只有诊断、最终报告或需要读取未决原因时才构造完整
`BondRingScanReport`。`_observe_frame()` 的逐帧路径必须有单独性能基线。

从 geometry 迁出的现有内容按已规划的 forcefields package 归属：

```text
forcefields/
├── ff.py / ff39.py
│   └── 对外暴露一致的 validation / build / optimize 接口
├── utils.py
│   ├── topology snapshot 与事务比较
│   ├── typed force-field observation
│   ├── acceptance thresholds/check/report
│   ├── geometry fact -> FF decision 的 policy
│   └── 可开环键选择器
└── utils39.py
    └── 仅 Python 3.9 / Open Babel 3.1 的等价后端差异
```

`evaluate_geometry_quality()` 不应原名搬家，因为名称继续暗示 geometry 拥有价值判断。建议由
forcefields 公开为 `evaluate_structure_acceptance()`；`is_geometry_reasonable()` 对应改为
`is_structure_accepted()` 或直接读取 report，具体名称在 forcefields 接口审议时最终冻结。

## 10. 调用关系图

交互式 Archify 图：

- [目标调用关系图](./geo_refactor_call_graph.html)
- [可审查 JSON 图源](./geo_refactor_call_graph.architecture.json)

核心依赖方向为：

```text
                         geometry/__init__ (re-export only)
                           /        |        |        \
                          v         v        v         v
                    settings     object   relation   convert
                       ^           ^         ^        /  |
                       |           |         +-------+   |
                       +-----------+---------------------+

Core: chemical objects -> convert adapter -> pure objects -> relation -> factual report
forcefields tri-state check -> lazy findings -> stop at PIERCES; otherwise preserve UNDETERMINED
forcefields diagnostics -> dense scan report -> policy and repair actions
```

`relation.py` 只对纯几何对象工作。`convert.py` 可以引用 relation 的事实记录类型以维护来源映射，
但 `relation.py` 不得反向理解 converter 或 Hotpot 对象；`geometry/__init__.py` 只是公开 façade。
geometry 的任何模块都不得运行时导入 Core 或 forcefields。forcefields 读取报告后作业务决定，
不把判断逻辑回灌 geometry。

现有 HTML/JSON 图生成于四模块草案，不能作为本版实施依据；G00 必须按五模块结构、settings 参数流、
lazy early-exit 与 dense report 两条路径重新生成并通过同一 Archify 校验。

## 11. 第三方库评估

### 11.1 决策

首轮不新增强制依赖。NumPy 足以支持当前小环上的向量、SVD、投影、距离、AABB 和
segment–triangle predicates。当前困难的本质是非平面闭合边界没有唯一 spanning surface，
不是缺少一个更快的求交函数。

| 库 | 可薄包装能力 | 无法解决的部分 | 决策 |
|---|---|---|---|
| NumPy（已有） | 向量、SVD、投影、距离、Möller–Trumbore、批量 AABB | 不提供 adaptive exact predicates，也不定义非平面环面 | 首轮默认且唯一必需后端 |
| SciPy（已有） | `pdist/cdist/cKDTree` 等批量距离与空间索引 | Delaunay/ConvexHull 不能代表凹的非平面环面 | 仅在 benchmark 证明有益时用于大规模距离 broad phase |
| NetworkX（已有） | Core 的 cycle basis 和图筛选 | 不处理三维曲面 | 继续留在 Core；geometry 不直接复制 ring perception |
| `robust` | 二维 orientation/segment 等稳健谓词，体量小 | 不解决三维 surface model；纯 Python 性能需验证 | 可选候选；通过 Python 3.9–3.14 和性能矩阵后才接入 |
| `mapbox-earcut` | 投影二维多边形的成熟单次三角化 | 只给一个 surface，不能裁决非平面曲面分歧 | 仅可作测试 oracle/候选生成器，不作真值后端 |
| Shapely/GEOS | 成熟二维 polygon relation | 忽略 Z，仍需投影；增加二进制和版本维护 | 当前不引入 |
| trimesh / libigl | mesh、ray/segment–triangle、AABB | 必须先指定 mesh；不能定义非平面环的化学意义 | 未来通用 mesh 扩展可选，当前过重 |
| CGAL | filtered/exact predicates、复杂网格算法 | 构建、wheel、许可和跨 Python 维护成本高；仍需 surface model | 只有 exact predicate 成为硬需求时再评估 |
| Open Babel | 分子读取、构筑和力场优化 | 不是本几何事实层的数值后端 | 只属于 forcefields / I/O，不由 geometry 导入 |

不公开 `backend=` 参数。只有第二个后端通过同一事实验收矩阵并产生明确收益后，才引入内部
backend protocol；当前不为假想实现提前抽象。

## 12. 分阶段实施与 Git 节点

每个节点单独提交，任一节点可回退；禁止把 package move、算法语义、Core API 和 FF policy
一次性混入同一个 commit。

### G00 — 冻结数学与参数契约

- 审查并冻结 `relation.md` 中每个判定公式、保护带、状态映射和曲面共识；
- 冻结 `settings.py` 的字段、默认值、单位和适用范围；
- 确认没有 scoring/chemical threshold 混入 geometry；
- 按五模块架构与 lazy/dense 双路径重新生成 Archify 图；
- 不修改 Python 生产代码。

建议提交：`docs(geometry): freeze relation and settings contracts`

### G01 — 建立事实契约测试

- 新增纯几何对象、planarity、有限 segment 和三态关系的测试；
- 把 `relation.md` 中每个公式边界、保护带、embedded 三态与 surface 计数恒等式逐项转为测试；
- 加入坐标/`absolute_length` 同比缩放、不同量纲 predicate 和 planar evidence fixture；
- 分别锁定 lazy 首个 `PIERCES` 早退与 dense 全量计数；
- 固化当前平面凸/凹环正确样例；
- 把历史 center-fan 测试标记为待替换的缺陷证据，而非新算法 oracle；
- 不修改生产实现。

建议提交：`test(geometry): define factual relation contracts`

### G02 — 建立 `settings.py`

- 按 `relation.md` 原样实现不可变 settings 数据类及唯一默认实例；
- 建立字段默认值、范围、单位缩放与“relation 无 magic threshold”测试；
- 不在本提交迁移 relation 算法。

建议提交：`refactor(geometry): centralize numerical settings`

### G03 — 建立 `object.py`

- 实现不可变 `Point/Line/Segment/Plane/Triangle/Cycle`；
- 明确 Line/Segment 和 Cycle/bounded surface 的差异；
- 只迁数据表达与无争议派生量，不迁关系算法。

建议提交：`refactor(geometry): introduce geometric value objects`

### G04 — 建立 `convert.py` 与 Core ring 查询边界

- 实现 Atom/Bond/Ring converters 和稳定 key；
- 定义只用于来源映射的 `AtomGeometry/BondGeometry/RingGeometry/BondRingTarget` 记录；
- Core 提供唯一无副作用的按 scope ring 查询；
- 使用窄 `Protocol`/`TypeVar`，清除 geometry 中已知化学对象的 `Any`，并避免运行时 Core 导入；
- 验证转换不改变原分子和 cache。

建议提交：`refactor(geometry): isolate chemical object conversion`

### G05 — 迁移现有纯关系函数

- 把 line/plane/segment distance、planarity measurement、point-pair distance 移入
  `relation.py`；
- 距离等纯 measurement 先保持数值行为并建立旧/新对照；旧的混量纲判定不得作为新 oracle，
  新 predicate 严格按 `relation.md` 实施；
- 删除对象方法对 relation 的反向依赖。

建议提交：`refactor(geometry): isolate factual relation kernels`

### G06 — 实现环—键三态 kernel

- 平面环使用稳定投影和 interior/boundary/exterior 分类；
- 非平面环使用全部合法 vertex triangulations 共识；
- 实施并测试 `EMBEDDED / PROVEN_NON_EMBEDDED / CONSTRUCTION_UNDETERMINED`
  的组合拓扑与三角形对相交判据；
- 处理共享端点重新穿入、有限线段参数和 numeric band；
- 只实现 `Segment × Cycle` 关系，不在该层引入 Molecule/Bond/Ring。
- 新事实矩阵通过后删除旧的 center-fan 行为锁定测试；它只可在迁移中短暂作为已知缺陷
  fixture/xfail，不能留作最终 oracle。

建议提交：`feat(geometry): report tri-state bond ring relations`

### G07 — Core 改用事实接口

- 在 `convert.py` 完成 `measure_atom_pair_distances()`、
  `determine_bond_ring_relation()`、`iter_bond_ring_findings()`、
  `determine_bond_ring_piercing_state()` 与 `scan_bond_ring_relations()` 的薄适配；
- 更新 `Bond`、`Ring`、`Molecule` 的 geometry façade；
- 在 `Molecule.bond_ring_relations()` 中委托薄适配器，并原样返回 scope/coverage；
- 删除含混 bool 和 `CyclePlanes` 入口，不保留 deprecated alias；
- chemistry 属性自行消费 measurement 并作阈值决定。

建议提交：`refactor(core): consume factual geometry relations`

### G08 — 将判断与修复策略迁出 geometry

- 把 topology snapshot、force-field checks、quality report、opening-edge selector
  迁到 forcefields 当前/目标模块；
- `closest_ring_opening_edge` 拆成 geometry distance + FF eligibility policy；
- 所有 report 类型使用精确 annotation，不以 `Any` 连接层级。

建议提交：`refactor(forcefields): own validation and untangling policy`

### G09 — Force-field 三态接线

- candidate/refinement hot path 消费惰性 `PiercingState`；最终帧和诊断路径消费
  `BondRingScanReport`；
- 三种 state 的动作在 forcefields 测试中逐一锁定；
- `UNDETERMINED` 不得静默变成通过；
- 保留最后帧/电影和 warning 等既定 FF 行为，不在 geometry 中实现。

建议提交：`refactor(forcefields): consume bond ring relation evidence`

### G10 — 原子迁移为 package 并清理旧文件

- 添加 `geometry/__init__.py` 的最终 `__all__`；
- 删除旧 `geometry.py`，不得让同名 module/package 并存；
- 更新 `.github/workflows/inference_compatibility.yml` 的路径过滤器为
  `hotpot/cheminfo/geometry/**`；
- 构建 wheel，在源码树外校验导入。

建议提交：`refactor(geometry): replace monolith with package`

### G11 — 文档与性能基线

- 更新 API 文档和示例，不再使用 reasonable/bool intersection 叙述；
- 记录单 pair 和 molecule scan 性能；
- 明确 scope/coverage/surface model。

建议提交：`docs(geometry): document factual geometry contracts`

## 13. 测试布局与验收矩阵

所有测试继续放在 `tests/`：

```text
tests/test_cheminfo/
├── geometry/
│   ├── test_settings.py
│   ├── test_object.py
│   ├── test_convert.py
│   ├── test_relation_primitives.py
│   ├── test_segment_cycle_relation.py
│   └── test_metamorphic.py
├── test_geometry_imports.py
├── test_geometry_core_integration.py
└── test_geometry_forcefields_integration.py

tests/performance/
└── test_segment_cycle_relation_benchmark.py   # 非默认 CI 门禁
```

### 13.1 必须通过的事实测试

- 平面凸环、凹环内部严格横穿：`PIERCES`；
- 交点位于环外或凹口：`DOES_NOT_PIERCE`；
- 只有无限延长线穿面、有限 segment 未到达：`DOES_NOT_PIERCE`；
- 稳定的边/顶点/端点/共面接触：明确断言
  `state == DOES_NOT_PIERCE` 且保留对应 feature，不允许用 `UNDETERMINED` 含混带过；
- 容差带中的边界/共面关系：`UNDETERMINED`；
- 非有限坐标、零长 segment、退化或自交 cycle：`UNDETERMINED`；
- 共享端点但 segment 内部重新穿面：不得因共享端点漏报；
- 非平面 surface family 完整且全部有效 surfaces 命中：`PIERCES`；
- 非平面 surface family 完整且全部有效 surfaces 未命中：`DOES_NOT_PIERCE`；
- 三角剖分分别覆盖 `EMBEDDED`、已证实额外交集的 `PROVEN_NON_EMBEDDED`、保护带内的
  `CONSTRUCTION_UNDETERMINED`；
- 两个三角形只共享边或顶点（包括无额外交叠的折叠共享边）不得因组合必然产生的零
  `orient3d` 被误判为 `CONSTRUCTION_UNDETERMINED`；
- saddle ring 的 triangulation 结论分歧：`UNDETERMINED`；
- 任一候选面的构造/求交状态未决：整体 `UNDETERMINED`，不得通过丢弃形成假共识；
- 空候选集、全部已判为不穿和扫描未完成必须是三种可区分报告。

### 13.2 变换不变量

- 平移、正交旋转和镜像不改变 state/features/indeterminacy causes；
- 任一 `Line.direction` 乘非零标量不改变 line relation 或 distance；共点正交轴不得因局部
  origin scale 为 0 而退化；
- 任一 `Line.origin` 沿其自身直线平移不改变 line relation、distance 或 tolerance；
- Segment 两端逆序、Cycle 循环移位和 winding 逆序不改变结论；
- 远离容差边界时，小扰动不改变状态；进入 numeric band 时转为 `UNDETERMINED`；
- 远离容差边界、且坐标与 `GeometrySettings.tolerance.absolute_length` 按同一长度单位同比缩放时，
  dimensionless relation state 不变；
- 原子/键容器重排不改变 pair 事实，只允许最终输出按规范 key 重排。

### 13.3 集成和包装

- 分别在干净子进程中先 import `geometry`、先 import `core`、先 import `forcefields`；
- 确认没有 partially initialized module 或 multiprocessing spawn 导入环；
- 静态检查 `relation.py` 不导入 `convert/core/forcefields`，`object.py` 不导入 relation；
- 分别校验 `geometry.__all__`、各子模块直接导入以及公开对象 identity；
- 校验 value object 不可变；converter 不修改 molecule、coordinates、conformer 或 ring cache；
- 校验 stable key、PairScope/RingScope/RingFamily 和 report 全部计数恒等式；
- 校验平面路径始终返回完整 `SurfaceFamilyEvidence`；
- 校验 AABB 只有在 surface construction 完成后才能提交 `DOES_NOT_PIERCE`，且所有 AABB
  路径统一读取 `aabb_padding_factor`；
- 校验非法 settings 组合、$g\epsilon_u\ge1/2$、零长/近零 segment 均进入明确未决分支；
- 校验 lazy 三态 API 在首个 confirmed piercing 后不再消费候选；无 piercing 时必须消费完成并
  保留已见 `UNDETERMINED`；dense API 返回每个 pair；
- 静态检查 `relation.py` 不含 settings 之外的数值判定阈值；
- Core 不把 `UNDETERMINED` 压缩成 `False`；
- FF 测试分别锁定三种 state 的 policy 动作；
- monkeypatch 只 patch 公开 façade 或 helper 的真实所属模块，不要求 `__init__.py`
  重导出 `_iter_*` 私有函数；
- Python 3.9–3.14 均先执行 `compileall`，再 build wheel、安装到源码树外、运行 focused suite；
- 在隔离安装环境断言 `geometry.__file__` 指向 package `geometry/__init__.py`；
- wheel 中只能存在 `hotpot/cheminfo/geometry/`，不能残留 `geometry.py`。

### 13.4 性能边界

- 平面单 pair 对 ring size 应近似 `O(n)`，其 hot path 目标是不高于当前实现的 1.5 倍；
- 非平面 `n <= 8` 记录完整 surface-family 计数、`segment_triangle_tests_used` 和
  `triangle_pair_tests_used`；8 元环每 pair 的上限分别为
  $132\times6=792$ 次 segment–triangle 与 $132\times\binom{6}{2}=1980$ 次
  embedding triangle-pair 判定；后一计数每个 triangle pair 记 1，其内部固定 primitive 不重复
  计入前一预算；
- molecule scan 可先计算 AABB broad-phase 候选，但必须在 surface model 验证后才能据此提交
  `DOES_NOT_PIERCE`；一次扫描内缓存 Ring conversion/surface family；
- benchmark 记录 6/8 元环、典型 ligand ring 数和 bond 数的 median/p95；
- 单独记录 `_observe_frame()` 中 lazy tri-state check 与 dense diagnostic scan 的每帧耗时；
- 在真实络合物样本上记录 `UNDETERMINED` 总比例及各 cause 分布，作为三态上线门禁；
- 非平面新算法不与旧的错误单 center-fan 做 1.5 倍同比；先记录 6/8 元环绝对耗时基线。
  若性能不足，先优化重复转换和 broad phase，不允许用单一 center-fan 或静默丢弃
  `UNDETERMINED` 换速度。

## 14. 明确不在本轮实施的内容

- 环折叠是否“物理合理”；
- 原子拥挤伪能量、键角、扭转、环张力和统一 geometry score；
- 全部分子 knot/link invariant；
- 全部 simple cycles 或无限大环的完备扫描；
- 通用三角网格布尔运算与可视化；
- 新 force-field、配位数模板或化学价态规则；
- 为未知未来后端提前公开 plugin/backend API。

这些能力以后可以在不破坏 `settings/object <- relation`、`settings/object/relation <- convert`
依赖边界的前提下扩展，但不得进入当前
环—键互穿主链。

## 15. 完成标准

本重构只有同时满足下列条件才算完成：

1. `geometry` package 只包含 settings、事实对象、转换和关系计算；
2. `geometry.__all__` 不含 reasonable/quality/acceptance/opening-policy 名称；
3. 非平面环不再由单一 arithmetic-centroid fan 给出确定 bool；
4. Core 和 forcefields 都能读取 `PIERCES / DOES_NOT_PIERCE / UNDETERMINED` 及细粒度证据；
5. FF 决策与 geometry fact 在代码和测试中分层；
6. 没有 `Any` 被用来掩盖已知 Hotpot 化学对象；
7. 所有 relation 数值参数只来自 `settings.py`，实现与 [relation.md](./relation.md) 逐项对应；
8. lazy early-exit、dense report、逐帧性能和 `UNDETERMINED` 分布均有基线；
9. import-order、wheel、Python 3.9–3.14、事实矩阵和 FF 集成测试全部通过；
10. 五模块 Archify 调用图达到 showcase 9/9；有可用浏览器时必须完成视觉审查，无可用浏览器时
    必须把证据明确记录为 `skipped/pending`，不得宣称已完成。

## 16. 当前验证备注

本轮只精修计划并新增数学契约，没有修改生产代码。当前默认 shell 中曾尝试运行：

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_cheminfo/test_geometry.py \
  tests/test_cheminfo/test_geometry_quality.py
```

命令在当前 `Python 3.9.23`（`/home/zhangzhiyuan/usr/conda3/envs/pg`）环境的 collection 阶段因缺少
`cython` 而停止（2 个模块均为 `ModuleNotFoundError: cython`），因此本轮不声称现有业务测试通过。
正式实施时必须在项目支持环境中执行第 13 节矩阵。

现有四模块 Archify 草案曾通过 showcase `9/9`、0 error、0 warning；它尚未反映新增
`settings.py` 和 lazy/dense 双路径，必须在 G00 重生成。当前机器没有 Chrome/Chromium，既有
automated browser visual-check 为 `skipped`，人工视觉审查为 `pending`。
