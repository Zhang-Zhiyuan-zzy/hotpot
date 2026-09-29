# `hotpot.cheminfo.geometry` 接口、引用树与化学语义审计

## 1. 范围和摘要

审计对象为当前分支 `fix/complexes-build-pipeline` 上的：

```text
hotpot/cheminfo/geometry.py
```

本模块现有：

- 4 个公开类型别名；
- 1 个公开枚举；
- 7 个公开不可变数据类；
- 1 个内部数据类；
- 4 个公开几何对象类；
- 17 个公开函数；
- 28 个内部 helper。

本次同时完成了纯排布整理：数据类型位于顶部，内部实现按距离、面相交、环、拓扑和综合门控分区，公开接口集中在底部；新增 `__all__` 明确导出 33 个公开名称。AST 对比确认全部既有函数体、签名、装饰器、类体和常量值均未改变。

### 1.1 主要结论

1. 当前生产主链真正依赖的是：
   - `has_too_close_atoms()`；
   - `bond_intersects_ring()`；
   - `find_bond_ring_intersections()`；
   - `bond_ring_intersection_checks()`；
   - `has_bond_ring_intersection()`；
   - `closest_ring_edge_to_bond()`；
   - `closest_ring_opening_edge()`；
   - `capture_topology()`；
   - `evaluate_geometry_quality()`；
   - `Line`、`CyclePlanes` 和 `points_on_same_plane()` 的 core 兼容接口。
2. `Point` 是确定的未使用公开类；`to_point()`、`get_line_relationship()`、`calculate_line_distance()` 和 `Plane` 当前只有模块内引用，属于历史公开面。
3. `AtomPairGeometryIssue.kind` 声明了 `short_bond`，但代码从不创建这种 issue；短键直接生成为 `GeometryCheck`。`GeometryCheck.severity="info"` 当前也没有生产构造点。
4. 旧 `Plane`/`points_on_same_plane()` 与新 `_planar_polygon_normal()` 存在两套平面判定，容差语义不同；同一环可能在芳香性判断和穿环判断中得到不同“是否平面”结论。
5. `geometry.py` 没有修改分子或调用力场，整体保持只读；但门控阈值和图语义本身会影响哪些构象被接受，因此仍包含实质化学策略。
6. 最高优先级化学边界包括：默认只检测至 8 元环、非平面环采用中心扇面近似、金属—配体键使用通用共价半径比、配位角/配位数只记录不判定，以及 topology snapshot 不覆盖立体化学、同位素和价态等全部化学身份。

## 2. 模块引用树

### 2.1 生产引用

```text
Molecule.is_disorder / AtomSeq.is_disorder
    └─ has_too_close_atoms
         ├─ _pair_table
         └─ _iter_too_close_issues

Molecule.has_bond_ring_intersection
    └─ has_bond_ring_intersection
         └─ _iter_bond_ring_intersections
              └─ bond_intersects_ring

Molecule.intersection_bonds_rings
    └─ find_bond_ring_intersections

Bond.bond_line
    └─ Line

Ring.determine_aromatic
    └─ points_on_same_plane

Ring.is_bond_intersect_the_ring
    └─ bond_intersects_ring

Ring.cycle_places
    └─ CyclePlanes

Ring.closest_edge_to_bond
    └─ closest_ring_edge_to_bond
```

### 2.2 forcefield 主链引用

```text
forcefields._capture_workflow_topology
    └─ capture_topology

forcefields._build_ligand_proxies
    ├─ capture_topology
    ├─ find_bond_ring_intersections(ring_scope="ligand_skeleton")
    ├─ bond_ring_intersection_checks
    ├─ closest_ring_opening_edge
    └─ evaluate_geometry_quality(forcefield_stage="candidate")

forcefields._OpenBabelOptimizer._observe_frame
    └─ evaluate_geometry_quality(forcefield_stage="final")

forcefields.build3d / build_complex3d
    └─ evaluate_geometry_quality(level="off")
```

### 2.3 综合门控内部树

```text
evaluate_geometry_quality
├─ _resolve_thresholds
├─ _topology_checks                    [提供 topology_reference 时]
├─ _forcefield_checks                  [提供 forcefield_report 时]
├─ _pair_table
│  ├─ _overlap_issues
│  └─ _too_close_issues
├─ _bond_position_data
├─ find_bond_ring_intersections        [standard / strict]
│  └─ _iter_bond_ring_intersections
│     └─ bond_intersects_ring
├─ bond_ring_intersection_checks
└─ _coordination_metrics               [standard / strict，只记录]

is_geometry_reasonable
└─ evaluate_geometry_quality(...).passed
```

## 3. 顶部类型与数据接口

### 3.1 类型别名和枚举

| 接口 | 语义 | 使用位置 |
|---|---|---|
| `PairScope` | `all`、`bonded`、`nonbonded` 原子对范围 | too-close 查询与 pair mask |
| `RingScope` | `full_graph` 或移除金属配位边后的 `ligand_skeleton` | 穿环查询和安全开环 |
| `QualityLevel` | `off/basic/standard/strict` | 综合门控和报告 |
| `ForceFieldStage` | `candidate/final` | 区分快速候选报告与最终优化报告要求 |
| `LinesRelationship` | `INTERSECT/PARALLEL/SKEW` | 旧线几何函数；模块外无引用 |

### 3.2 数据类

| 数据类 | 实际功能 | 引用与状态 |
|---|---|---|
| `AtomPairGeometryIssue` | 保存 overlap/too-close 原子索引、距离和阈值 | 两个 pair iterator 构造；`kind="short_bond"` 分支未使用 |
| `GeometryCheck` | 单项门控结论、严重性、实测值、阈值和定位索引 | topology、forcefield、穿环和综合门控共同使用 |
| `GeometryQualityThresholds` | 所有默认距离、键比、梯度和稳定窗口 | `_resolve_thresholds()` 构造/替换 |
| `AtomTopologySignature` | 原始原子的顺序、id、元素和形式电荷 | `capture_topology()` |
| `BondTopologySignature` | 原始键端点、键级和 bond kind | `capture_topology()`/`_topology_checks()` |
| `TopologyReference` | 不可变拓扑快照及是否允许新增氢 | forcefield 全部事务入口 |
| `GeometryQualityReport` | level、总通过状态、checks、metrics；提供 failures/warnings/to_dict | 所有综合门控调用方 |
| `_AtomPairTable` | 一次性向量化缓存 atom、pair index、距离和 bonded pair | 所有原子对查询及综合门控 |

## 4. 公开几何对象与基础函数

### 4.1 `Point(x, y, z)`

- 实际功能：仅把三坐标保存到私有 `_pos`。
- 引用：定义外没有任何引用；自身也没有属性、运算或转换接口。
- 结论：确定冗余的历史公开类。由于本轮禁止删除，仍保留并列入 `__all__` 以维持现有公开面。

### 4.2 `to_point(p)`

- 实际功能：返回 `np.array(p)`。
- 引用：仅 `Line`、`Plane`、`CyclePlanes` 内部使用；模块外无引用。
- 结论：实际是内部转换 helper，但历史名称没有下划线。本轮仍作为公开接口保留。

### 4.3 `get_line_relationship(v1, v2, p1, p2) -> str`

- 实际功能：以叉积和标量三重积判断两条无限直线平行、相交或异面。
- 引用：仅 `calculate_line_distance()`。
- 签名问题：注解写 `str`，实际返回 `LinesRelationship` 枚举成员。
- 数值边界：没有拒绝零方向向量；零向量会被归入 parallel 分支。

### 4.4 `calculate_line_distance(v1, v2, p1, p2) -> (str, float)`

- 实际功能：根据上一函数计算两条无限直线最短距离。
- 引用：仅 `Line.distance_to_line()`。
- 签名问题：返回注解 `(str, float)` 不是规范 `Tuple[...]`，且第一项实际为枚举。
- 数值边界：平行分支会除以 `norm(v1)`，零向量可能产生无效值。

### 4.5 `Line(point1, point2)`

- 实际功能：表示无限直线兼有限线段端点，提供方向、单位方向、线长、参数点和线间距离。
- 引用：`Bond.bond_line` 返回该对象；`Plane`/`CyclePlanes` 和 geometry 测试使用。
- 边界：零长度线的 `identity_vector` 会除零；`get_param_t()` 使用固定 8 位 round 判断共线。

### 4.6 `Plane(p1, p2, p3)`

- 实际功能：由三点构造平面，提供法向量、点距、点在面上和直线交点。
- 引用：只由 `points_on_same_plane()` 与 `CyclePlanes` 使用；模块外无直接引用。
- 边界：`is_on_plane()` 用“点面距离/点到 point1 距离 < 0.03”的相对判据，与新 polygon helper 的尺度容差不同。

### 4.7 `points_on_same_plane(*points)`

- 实际功能：选择叉积最大的三点构造平面，再判断其他点。
- 引用：`Ring.determine_aromatic()` 和测试。
- 返回语义：少于 3 点返回 `None`；3 个共线点返回 `None`；3 个非共线点返回 `True`；更多点返回 bool。它不是严格的 bool predicate。
- 化学影响：直接参与 ring aromaticity 判断，因此其 0.03 相对容差会改变某些非平面环的芳香性结论。

### 4.8 `CyclePlanes(*points)`

- 实际功能：以环中心和相邻边构造一组扇形平面，提供法向量角度、边侧判断和线—环相交门面。
- 引用：`Ring.cycle_places` 和测试。
- 重复情况：`is_line_intersect_the_cycle()` 已委托新的 `_line_intersects_polygon()`；其余大量旧几何方法只服务该对象，和新的向量化环算法部分重叠。

## 5. 公开原子对接口

### 5.1 `find_overlapping_atom_pairs(mol, *, tolerance=1e-3)`

- 返回全部距离 `<= tolerance` 的 `AtomPairGeometryIssue`。
- 生产主链不直接调用；综合门控使用内部 tuple helper；测试直接覆盖。

### 5.2 `has_overlapping_atoms(mol, *, tolerance=1e-3)`

- 使用 generator 短路，只回答是否存在重叠。
- 当前仅测试直接调用；保留是为了廉价布尔 API。

### 5.3 `find_too_close_atom_pairs(mol, *, minimum_distance=0.50, covalent_radius_scale=None, pair_scope="all", include_overlaps=True, overlap_tolerance=1e-3)`

- 返回所有低于 `max(minimum_distance, scale × 两原子共价半径和)` 的选定原子对。
- 当前仅测试直接调用。
- 注意：默认 `pair_scope="all"` 包括成键原子，不能把默认结果直接解释成“非键碰撞”。

### 5.4 `has_too_close_atoms(...)`

- 与上一接口阈值相同，但以 iterator 短路返回 bool。
- `Molecule.is_disorder` 与 `AtomSeq.is_disorder` 使用固定 0.5 Å、全部原子对、无半径缩放。
- `find_*`/`has_*` 两套接口不是无意义重复：前者给诊断全集，后者避免构造全部 issue。

## 6. 公开环和穿环接口

### 6.1 `bond_intersects_ring(ring, bond, *, tolerance=1e-8) -> bool`

- 实际功能：若 bond 与 ring 共原子则直接 False；否则判断有限 bond segment 是否穿过环面。
- 引用：ring 对象门面、穿环枚举和测试。
- 平面凹多边形使用二维 point-in-polygon；非平面环使用中心扇面三角化。

### 6.2 `find_bond_ring_intersections(mol, *, ring_scope="full_graph", max_ring_size=8)`

- 稳定排序后返回全部 `(ring, bond)` 对。
- 引用：forcefield 候选/精修、综合门控、`Molecule.intersection_bonds_rings`。
- 默认忽略大于 8 元的环；这对大环配体是重要适用域限制。

### 6.3 `bond_ring_intersection_checks(mol, intersections)`

- 把对象型 `(ring, bond)` 转为可序列化 `GeometryCheck`；无交叉时返回一个 passed check。
- 引用：forcefield 候选拒绝和综合门控。
- 输入必须来自同一 `mol`；外部传入不属于该分子的 bond 会在索引查找时失败。

### 6.4 `has_bond_ring_intersection(mol, *, ring_scope="full_graph", max_ring_size=8)`

- 对同一 iterator 短路返回 bool。
- 引用：`Molecule.has_bond_ring_intersection` 和测试。

### 6.5 `closest_ring_edge_to_bond(ring, bond)`

- 在所有环边中按有限线段距离选择最近者，稳定键索引作为 tie-breaker。
- 引用：`Ring.closest_edge_to_bond` 和测试。
- 这是纯几何查询，可能返回芳香键、双键或稠合共享边，不能直接用于开环。

### 6.6 `closest_ring_opening_edge(mol, ring, bond, *, ring_scope="ligand_skeleton")`

- 只在单键且仅属于一个选定环的边中选择最近者；无安全边返回 `None`。
- 引用：仅 forcefield 解穿环流程和测试。
- 与上一函数不是重复：它额外编码了允许临时开环的化学资格。

## 7. 公开拓扑和综合门控接口

### 7.1 `capture_topology(mol, *, allow_added_hydrogens=True) -> TopologyReference`

- 实际功能：按原子顺序记录 id、元素、形式电荷，以及每条键的端点、键级和 bond kind。
- 引用：forcefield 所有 working-copy 流程和测试。
- 保护范围：保证原始原子顺序/身份和原始键不变，并可选择只允许新增 H/X–H。
- 未保护：同位素、手性、芳香标志、自由基、自旋、部分电荷、valence 等不在签名内；新增 H 的形式电荷及其所连原子类型也没有化学校验。

### 7.2 `evaluate_geometry_quality(mol, *, level="standard", topology_reference=None, forcefield_report=None, forcefield_stage="final", thresholds=None) -> GeometryQualityReport`

- 实际功能：唯一综合门控实现；聚合坐标 shape/finite、拓扑、后端报告、重叠、过近、键长、穿环和配位 metrics。
- 引用：forcefield 候选、完整优化、build-only 入口及测试。
- `off`：仍检查 shape、finite、传入的 topology 和传入的最终 forcefield report；不检查距离/穿环。
- `basic`：增加 overlap、全原子对 0.4 Å 下限、显式键 30 Å 上限和 backend explosion。
- `standard`：增加非键半径碰撞、键长半径比、短键、配体骨架穿环和配位 metrics；backend 未收敛只是 warning。
- `strict`：在 standard 上要求 backend convergence、梯度和 segment 内稳定历史。
- 副作用：函数本身只读，不增删原子/键，不改构象。

### 7.3 `is_geometry_reasonable(...) -> bool`

- 实际功能：完整转发到 `evaluate_geometry_quality(...).passed`。
- 引用：目前只有测试；设计为不需要诊断详情的高级布尔门面。
- 结论：虽然很薄，但避免未来出现第二套综合判断，适合保留。

## 8. 内部 helper 逐项审计

### 8.1 原子坐标和 pair table

| helper | 功能 | 直接引用 |
|---|---|---|
| `_atom_coordinates(atoms)` | 原子坐标转 `(N,3)` ndarray | `_pair_table`、`bond_intersects_ring` |
| `_atom_index(atom, fallback)` | 优先取 atom.idx，否则用位置 | pair/ring key、metrics、nonfinite 定位 |
| `_pair_table(obj, coordinates=None)` | 一次生成全部上三角 pair、距离和 bonded set | 四个 pair API、综合门控 |
| `_pair_scope_mask(table, pair_scope)` | 生成 all/bonded/nonbonded mask | `_iter_too_close_issues` |
| `_overlap_issues(table, tolerance)` | 把 overlap iterator materialize 成 tuple | find API、综合门控 |
| `_iter_overlap_issues(table, tolerance)` | 惰性生成 overlap issue | tuple helper、has API |
| `_too_close_issues(table, ...)` | 把 too-close iterator materialize 成 tuple | find API、综合门控 |
| `_iter_too_close_issues(table, ...)` | 计算绝对/半径阈值并惰性生成 issue | tuple helper、has API |

结论：tuple/iterator 双层是为“完整诊断”和“布尔短路”分别服务，不建议合并。

### 8.2 线段与多边形

| helper | 功能 | 直接引用 |
|---|---|---|
| `_segment_intersects_triangle(...)` | Möller–Trumbore 线/线段—三角形相交 | 非平面环中心扇面 |
| `_planar_polygon_normal(points, tolerance)` | Newell 型法向并判断全点共面 | `_line_intersects_polygon` |
| `_point_on_segment_2d(...)` | 二维边界点判断 | `_point_in_polygon_2d` |
| `_point_in_polygon_2d(...)` | 支持凹简单多边形的射线法 | 平面多边形相交 |
| `_line_intersects_planar_polygon(...)` | 求线面交点并投影到二维多边形 | `_line_intersects_polygon` |
| `_line_intersects_polygon(...)` | 平面 polygon 或非平面中心扇面统一入口 | `bond_intersects_ring`、`CyclePlanes` |
| `_point_segment_distance(...)` | 点到有限线段距离 | `_segment_distance` 的退化情况 |
| `_segment_distance(...)` | 两有限线段最短距离 | 两个 closest-edge API |

### 8.3 环、拓扑和质量门控

| helper | 功能 | 直接引用 |
|---|---|---|
| `_rings_for_scope(mol, ring_scope)` | 选择 full graph 或 ligand skeleton rings，优先使用 uncached API | 穿环 iterator、安全开环 |
| `_ring_key(ring)` | 稳定环原子索引 key | 穿环排序、结构化 check |
| `_bond_key(bond)` | 稳定 bond 端点 key | 排序、check、最近边 tie-break |
| `_iter_bond_ring_intersections(...)` | 稳定惰性遍历环×键 | find/has API |
| `_bond_kind(bond)` | 规范化 bond kind 字符串 | topology bond signature |
| `_topology_bond_signature(bond, positions)` | 创建不可变键身份 | capture/check |
| `_topology_checks(mol, reference)` | 比较原始原子/键并约束新增 H | 综合门控 |
| `_resolve_thresholds(thresholds)` | 默认阈值、原对象或 mapping override | 综合门控 |
| `_report_value(report, name)` | 同时读取 Mapping 或对象报告 | forcefield checks |
| `_forcefield_checks(report, level, thresholds, stage)` | 按 candidate/final 和 level 验证 setup、能量、梯度、爆炸、收敛和稳定性 | 综合门控 |
| `_bond_position_data(mol, atoms)` | 产生 bond index、bond、两端位置 | 综合键长检查 |
| `_coordination_metrics(mol, atoms, coordinates)` | 记录每个金属的 donor、距离和两两夹角 | standard/strict metrics；不影响 passed |

## 9. 重复功能与冗余

### 9.1 确定冗余或 dormant 项

1. `Point` 没有任何调用者，且只有 `_pos` 字段，没有实际对象行为。
2. `AtomPairGeometryIssue.kind="short_bond"` 没有构造路径；短键使用 `GeometryCheck`。
3. `GeometryCheck.severity="info"` 当前没有生产路径，只有 error/warning。
4. `CoordinationGeometry*` 位于 forcefields，而 `_coordination_metrics()` 位于 geometry；前者尚未启用，后者只记录。这两套协调环境概念尚未形成统一模型。

### 9.2 同模块重复

1. **两套平面判断：**
   - `points_on_same_plane()` → `Plane.is_on_plane()`，使用相对 0.03 容差；
   - `_planar_polygon_normal()`，使用 `tolerance × 几何尺度`。
   二者分别服务芳香性与穿环，可能对同一组点给出不同结论，应最终统一一个纯函数内核。
2. **旧对象式几何与新数组式几何并存：**
   - `Line`/`Plane`/`CyclePlanes` 是早期对象 API；
   - `_line_intersects_polygon()` 等是新门控内核。
   `CyclePlanes` 已部分委托新内核，但 `Plane` 的多项方法仍是独立实现。
3. **find/has 双接口不是冗余：** find 返回完整问题，has 使用 iterator 短路，性能和返回契约不同。
4. **两个 closest-edge 接口不是冗余：** 一个只回答几何最近，一个限定可安全临时打开的化学边。

### 9.3 与 forcefields 的跨模块重复

1. `capture_topology()`/`_topology_bond_signature()` 与 forcefields 的 `_atom_commit_signature()`/`_bond_commit_signature()` 都定义原子、键身份。
   - 前者用于质量门控；
   - 后者用于原子化提交前验证。
   两者目的不同，但“什么算身份变化”的规则可能漂移，建议未来共享不可变 signature 生成器。
2. `_coordination_metrics()` 与 `ff.collect_coordination_environments()` 都遍历 metal–ligand bond。
   - geometry 返回距离/角度 metrics；
   - forcefields 返回元素、电荷和 chelate grouping。
   建议最终由一个低层连接描述生成两类视图，避免对 donor 的定义分叉；geometry 不应反向 import forcefields。

## 10. 化学业务逻辑审视

### 10.1 门控中明确存在的化学假设

| 假设 | 实现 | 影响与风险 |
|---|---|---|
| 最大检测环为 8 元 | `find_bond_ring_intersections(..., max_ring_size=8)` | 大环、冠醚、卟啉外环等默认不参与穿环检查；可能漏报宏环打结 |
| 非平面环由几何中心扇形三角化 | `_line_intersects_polygon()` | 对强烈翘曲或凹非平面环，构造的扇面不一定代表合理环面，可能误报/漏报 |
| 标准非键下限为 `max(0.50 Å, 0.55×共价半径和)` | `_too_close_issues()` | 是通用碰撞启发式，不包含 vdW 半径、电荷、氢键或配位环境 |
| 普通共价键比范围 0.65–1.45 | `GeometryQualityThresholds` | 宽松结构 sanity check，不是键型精确标准 |
| metal–ligand 键比范围 0.65–1.60，仍用共价半径 | 同上 | 对不同氧化态、离子半径和配位数缺少化学分辨率 |
| 显式键最长 30 Å | `maximum_bond_distance` | 只用于捕捉爆炸，阈值不代表任何真实化学键 |
| 标准级 backend 未收敛仅 warning | `_forcefield_checks()` | standard 可接受未收敛但几何合理的帧；strict 才要求收敛 |
| 配位数、距离和角度只记录不判定 | `_coordination_metrics()` | 当前没有八面体、四面体、平方平面等中心几何质量门控 |
| ligand-skeleton ring 排除配位螯合环 | `_rings_for_scope()` | 避免把真实配位闭环误当作有机环，符合既定需求；但也意味着配位环自身不接受穿环检查 |

### 10.2 为程序稳健性而改变验收语义的行为

1. **`off` 不是真的关闭全部检查。**
   - shape、finite、传入的 topology、以及传入 final forcefield report 的 setup/有限能量/有限梯度仍强制检查。
   - 这是防止程序损坏结果伪装成成功，属于正确的不可关闭完整性边界。
2. **缺少 final backend 字段时 fail closed。**
   - 即使 level=off，只要调用者传了 final report，缺能量或梯度就失败。
   - candidate stage 不要求梯度，因为快速评分实际没有生成；这是基于阶段的合理差异。
3. **strict 的首 epoch 收敛特例。**
   - 后端第一 epoch 明确停止且梯度通过时，没有相邻帧可计算 stability history，代码允许通过。
   - 这是避免“为了满足数组长度而伪造历史”，不是放宽梯度标准。
4. **boolean API 采用短路。**
   - `has_*` 不生成全部诊断，与 `find_*` 结果内容不同但判断逻辑相同，不改变化学判据。

### 10.3 仍未覆盖的化学身份

`TopologyReference` 当前只保护：

- 原子顺序；
- atom id；
- 原子序数；
- 形式电荷；
- 键端点、键级、bond kind；
- 可选的新增 H/X–H 规则。

它不保护：

- 原子同位素；
- 手性标记和双键 E/Z；
- 芳香性 flag/芳香环模型；
- 自由基电子、自旋多重度；
- 显式 valence/implicit-H 元数据；
- 部分电荷；
- 新增 H 的形式电荷，以及新增 H 是否连接到合理的重原子；
- 周期性边界和晶胞等价拓扑。

因此报告中的 `topology passed` 应解释为“当前有限签名未变”，不能解释为完整化学身份严格等价。

## 11. 整理后的文件布局

当前 `geometry.py` 已按以下顺序排列：

```text
imports
__all__
type aliases
public enum and immutable data contracts
internal vectorized data

atom-coordinate and pair-distance helpers
line/segment/polygon intersection helpers
ring selection and identity helpers
topology comparison helpers
composite quality-gate helpers

public geometric primitives
public atom-pair queries
public bond-ring queries and repair selectors
public topology snapshot
public composite quality APIs
```

公开部分由低级几何对象逐步上升到最终 `evaluate_geometry_quality()` 和 `is_geometry_reasonable()`，符合“越高级越靠近文件底部”的要求。

## 12. 纯排布验证记录

- 以重排前文件为基准，对 58 个顶层函数/类逐项执行 AST 比较：函数体、签名、
  装饰器和类体均无变化；全部既有顶层 AST 节点完整保留，只新增 `__all__`。
- `py_compile`、模块导入、通配导入与 33 项 `__all__` 唯一性/完备性校验通过。
- 与 forcefield 相关的 7 个定向测试模块合计 `206 passed`，覆盖 geometry 基元、
  综合质量门控、公开 forcefield API、优化器、集成、络合物补氢和构筑流程。

## 13. 后续建议顺序

本轮不执行以下逻辑变更；建议在用户审查全部重构代码后按顺序决策：

1. 为 `build_complex3d()` 与中心配位几何建立强制验收契约。
2. 决定宏环穿环检查是否提高/取消默认 8 元上限。
3. 统一旧 `points_on_same_plane()` 与新 polygon planarity 内核。
4. 定义 metal–ligand 距离的配位化学阈值来源，不再只依赖共价半径。
5. 扩充 topology signature 的手性、同位素、芳香性和氢化学验证。
6. 合并 geometry/forcefields 的配位连接描述，避免 donor 语义漂移。
7. 最后再清理 `Point`、未用枚举分支和其他历史公开面；删除前需要给出兼容周期。
