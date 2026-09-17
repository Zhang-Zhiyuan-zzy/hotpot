# 环几何与环—键几何统一整改方案

本报告合并原 FF-Q004 和 FF-Q005。公开应用层只保留两个步骤：

```python
geo_status = determine_geo_status(target)
verdict = is_geo_reasonable(geo_status)
```

`determine_geo_status()` 负责计算和保留几何差异；`is_geo_reasonable()` 只把这些差异映射为
`REASONABLE / UNREASONABLE / UNCERTAIN`。环和 ring–bond pair 使用不同的详细状态类型，
但共享同一个三态 verdict。

## 1. 当前代码存在的问题

| 问题 | 当前实现 | 直接影响 |
|---|---|---|
| 平面和非平面环使用不同语义 | 平面环使用 point-in-polygon；非平面环使用 arithmetic-centroid fan | 极小坐标扰动即可翻转结果 |
| centroid 可能位于凹环外 | 仍以 centroid 和所有边组成扇面 | 产生不属于环孔的虚构面积并误报穿环 |
| 只返回 `bool` | 非法输入、不可判定、平行/共面和真正不相交都可能成为 `False` | 力场无法区分“合理”和“不确定” |
| 把无限直线和有限键段的措辞混淆 | 文档曾写成“端点之外穿过” | 容易把键延长线的交点错误理解为键本身穿环 |
| 状态层级过多 | 曾设计 ring 四态、surface 三态、ring–bond 五态 | 上层力场缺少统一参考线 |
| 判定作用域不清 | `BondRingGeometryStatus` 没有明确是 pair 还是 molecule | 无法正确聚合整分子结果 |
| 单个三角剖分具有任意性 | Earcut 或任意一条对角线只构造一个非平面环面 | 同一环边界可因选面不同得到相反答案 |
| 测试锁定旧行为 | `test_nonplanar_ring_keeps_center_fan_surface_semantics` | 缺陷被当成兼容目标保存 |

当前 `_line_intersects_polygon()` 对非法或退化输入直接返回 `False`，尤其危险：它把“程序
无法判定”伪装成“明确合理”。新结构必须做到：详细几何状态可以很多，但公开 verdict 始终
只有三个。

## 2. 新方法的几何状态与合理性映射

### 2.1 唯一的应用层三态

```python
class GeometryVerdict(Enum):
    REASONABLE = "reasonable"
    UNREASONABLE = "unreasonable"
    UNCERTAIN = "uncertain"
```

| Verdict | 化学应用含义 | 后续力场默认行为 |
|---|---|---|
| `REASONABLE` | 当前几何证据明确支持结构合理 | 作为正常候选进入优化 |
| `UNREASONABLE` | 当前几何证据明确证明结构不合理 | 拒绝并重新构筑 |
| `UNCERTAIN` | 现有几何证据不足以明确归入前两类 | 按合理度评分排序，优先优化高分候选并重新判定 |

本轮只使用坐标、图拓扑和无量纲几何阈值；不调用力场、不使用真实能量，也不依赖元素、杂化
或经验构象数据库。

### 2.2 环的详细几何状态

一个 `RingGeoStatus` 可以同时包含多个状态标记，不强迫不同问题互斥。定义：

\[
s=\operatorname{median}(\ell_i),\qquad
\epsilon=\epsilon_{abs}+\epsilon_{rel}s,
\]

\[
q_{clear}=\frac{\min(d_{edge-edge},d_{point-edge},d_{point-point})}{s},
\]

\[
q_{\tau}=\frac{1}{s}\min\left(\min_iR_{curv,i},\frac12d_{self}\right),
\quad
q_{rank}=\frac{\sigma_2}{\sigma_1},
\quad
q_{nonplanar}=\frac{\sigma_3}{\sigma_2},
\quad
q_{aperture}=\frac{4\pi A}{P^2}.
\]

其中 `sigma_1 >= sigma_2 >= sigma_3` 是中心化环坐标的奇异值，`A/P` 来自 best-fit plane
投影，距离计算排除共享端点和规定的局部拓扑邻居。

| `RingGeometryState` | 关键判别要点/公式 | Verdict |
|---|---|---|
| `REGULAR_PLANAR` | 全部硬检查和警戒检查均通过，且 `q_nonplanar <= delta_planar` | `REASONABLE` |
| `REGULAR_PUCKERED` | 全部硬检查和警戒检查均通过，且 `q_nonplanar > delta_planar`；非平面本身不是缺陷 | `REASONABLE` |
| `NEAR_SELF_CONTACT` | `hard_clearance < q_clear <= warning_clearance` | `UNCERTAIN` |
| `NEAR_COLLAPSE` | `q_tau`、`q_rank` 或 `q_aperture` 落在 hard 与 warning 阈值之间 | `UNCERTAIN` |
| `SURFACE_UNSTABLE` | 坐标小扰动后有效三角面集合或投影合法性发生变化 | `UNCERTAIN` |
| `SURFACE_UNDEFINED` | 无法从当前边界构造合格的 vertex-only 环面族，但环中心线未被证明不合理 | `UNCERTAIN` |
| `NUMERICALLY_UNRESOLVED` | 距离、方向或退化谓词位于数值容差带 | `UNCERTAIN` |
| `DEGENERATE_BOUNDARY` | 非有限坐标、少于 3 个有效顶点、重复连续顶点或 `min(ell_i) <= epsilon` | `UNREASONABLE` |
| `CENTERLINE_SELF_INTERSECTION` | 两条非相邻环边在各自有限内部发生明确横向相交 | `UNREASONABLE` |
| `SELF_COLLISION` | `q_clear <= hard_clearance`，且距离证据明确越过硬阈值 | `UNREASONABLE` |
| `COLLAPSED_RING` | `q_tau <= hard_thickness` 或 `q_rank <= hard_rank`，确认环已急折/近线状塌缩 | `UNREASONABLE` |

映射优先级固定为：

```text
存在任一 UNREASONABLE state -> UNREASONABLE
否则存在任一 UNCERTAIN state -> UNCERTAIN
否则                         -> REASONABLE
```

`REGULAR_PLANAR` 和 `REGULAR_PUCKERED` 只在没有不合理或不确定标记时写入。所谓“自打结”在
首轮实现中只按可直接判定的中心线自交、非相邻环段近接触和环孔塌缩处理；不扩展到复杂的
knot invariant。

### 2.3 单个环—键对的详细几何状态

`BondRingGeoStatus` **只针对一个环和一条目标键**。它不是整个分子的状态。目标键是有限线段：

\[
\mathbf b(t)=\mathbf a+t(\mathbf c-\mathbf a),\qquad 0\le t\le1.
\]

对一个三角面，若交点重心坐标为 `(lambda_1, lambda_2, lambda_3)`，明确穿越必须同时满足：

\[
\epsilon_t<t<1-\epsilon_t,
\qquad \min(\lambda_i)>\epsilon_b,
\qquad
\frac{|\mathbf d\cdot\mathbf n|}{\|\mathbf d\|\|\mathbf n\|}>\epsilon_{angle}.
\]

第一式明确表示交点位于两个键端点之间。若只有无限直线延长部分与环面相交，即 `t <= 0`
或 `t >= 1`，不属于键穿环。

| `BondRingGeometryState` | 关键判别要点 | Verdict |
|---|---|---|
| `ALL_SURFACES_CLEAR` | 所有有效环面都与有限键段分离，并有明确数值裕量 | `REASONABLE` |
| `LINE_EXTENSION_ONLY` | 仅无限直线延长部分相交；所有有限键段均无交点 | `REASONABLE` |
| `ALL_SURFACES_PIERCED` | 每个有效环面均存在满足上述三式的有限键段内部横向交点 | `UNREASONABLE` |
| `BOUNDARY_CONTACT` | 目标有限键段命中至少一个候选环面的三角形边或顶点 | `UNCERTAIN` |
| `COPLANAR_CONTACT` | 目标有限键段与至少一个候选三角面共面接触或重叠 | `UNCERTAIN` |
| `SURFACE_DISAGREEMENT` | 一部分合法环面被有限键段穿过，另一部分没有 | `UNCERTAIN` |
| `SURFACE_UNSTABLE` | 环面集合对数值扰动不稳定 | `UNCERTAIN` |
| `SURFACE_UNDEFINED` | 无可用环面，无法判断该 ring–bond pair | `UNCERTAIN` |
| `NUMERICALLY_UNRESOLVED` | `t`、重心坐标或交角位于容差带 | `UNCERTAIN` |
| `RING_UNREASONABLE_NOT_EVALUATED` | 环自身已经明确不合理，因此不再伪造 ring–bond 结论 | `UNREASONABLE`（继承环） |

原先“没有明确横穿，但至少一个有效环面出现擦边、顶点或共面接触”的准确含义是：目标有限
键段至少与一个候选环面的边、顶点或内部发生接触，但没有得到稳定的内部横穿证据。因此它
属于 `UNCERTAIN`，不是额外的第四种 verdict。

属于环本身或与环共享原子的键不进入 ring–bond pair 判定，记录
`skipped_reason=shared_ring_atom`。这不是 `REASONABLE` 状态，只是“不适用”。

pair 映射首先继承环的可判定性：环为 `UNREASONABLE` 时得到
`RING_UNREASONABLE_NOT_EVALUATED`；环为 `UNCERTAIN` 时，即使当前候选曲面都没有命中，
pair verdict 也只能是 `UNCERTAIN`。只有环自身 `REASONABLE` 时，pair 才能得到明确的
`REASONABLE/UNREASONABLE`。

分子级采用保守聚合：

```text
任一 ring 或 pair 为 UNREASONABLE -> molecule UNREASONABLE
否则任一 ring 或 pair 为 UNCERTAIN -> molecule UNCERTAIN
全部均为 REASONABLE                -> molecule REASONABLE
```

### 2.4 `determine_geo_status()` 与 `is_geo_reasonable()`

```python
@overload
def determine_geo_status(
    target: Ring,
    *,
    thresholds: GeometryThresholds,
) -> RingGeoStatus: ...

@overload
def determine_geo_status(
    target: BondRingPair,
    *,
    thresholds: GeometryThresholds,
) -> BondRingGeoStatus: ...

def is_geo_reasonable(
    status: RingGeoStatus | BondRingGeoStatus,
) -> GeometryVerdict: ...
```

- `determine_geo_status()` 不作化学业务决策；它返回状态集合、原始量、归一化量、阈值、裕量
  和环面共识证据。
- `is_geo_reasonable()` 不重新计算几何；它只执行本节两张映射表和明确的优先级。
- 两个函数都不接受 `Any`。公开 overload 精确表达 `Ring` 与 `BondRingPair` 的输入输出对应。

### 2.5 不确定结构的排序

纯几何 penalty 可写为：

\[
P_{geo}=w_c[\delta_c-q_{clear}]_+^2+
w_\tau[\delta_\tau-q_\tau]_+^2+
w_l\sum_i[|\ell_i/s-1|-\delta_l]_+^2+
w_a[\delta_a-q_{aperture}]_+^2.
\]

环的初始排序分数可取 `S_ring=exp(-P_geo)`。ring–bond pair 可同时记录：

\[
S_{surface}=\frac{N_{clear}}{N_{valid}},
\]

再结合最小数值裕量形成单调 `reasonableness_score`。整分子首轮采用最差组件分数进行保守
排序，而不是把多个分数相乘。

未经合理/不合理标注数据校准前，该量只能叫 `reasonableness_score`，不能声称是统计概率。
经过校准后才可增加 `reasonable_probability` 字段。

## 3. 重构后的代码结构树与职责

### 3.1 文件内结构

本轮保持 `geometry.py` 单模块，不为这一个问题新增 package。异常和 dataclass 位于顶部，
公开接口位于底部，并全部列入 `__all__`：

```text
hotpot/cheminfo/
└── geometry.py
    ├── __all__
    ├── exceptions / enums / type aliases
    ├── immutable dataclasses
    ├── generic vector and segment helpers
    ├── ring boundary metric helpers
    ├── ring surface enumeration helpers
    ├── finite segment–surface relation helpers
    ├── molecule-level aggregation helpers
    └── public API
        ├── determine_geo_status()
        ├── is_geo_reasonable()
        └── determine_molecule_geo_status()

tests/test_cheminfo/
├── test_ring_geometry_status.py
├── test_bond_ring_geometry_status.py
├── test_geometry_verdict.py
└── test_geometry_quality_integration.py
```

### 3.2 类型和数据类

| 名称 | 可见性 | 职责 |
|---|---|---|
| `GeometryVerdict` | public | 唯一业务三态：合理、不合理、不确定 |
| `GeometryTarget` | private type alias | `Ring | BondRingPair`，禁止使用 `Any` |
| `RingGeometryState` | public | 环的详细几何状态代码；允许一个报告包含多个状态 |
| `BondRingGeometryState` | public | 单个 ring–bond pair 的详细几何状态代码 |
| `GeometryThresholds` | public frozen dataclass | 保存环和 ring–bond 共用的尺度无关 hard/warning/numerical 阈值 |
| `RingGeometryMetrics` | public frozen dataclass | 保存距离、曲率、thickness、SVD、aperture 等量 |
| `BondRingPair` | public frozen dataclass | 明确绑定一个 `Ring` 和一个 `Bond`，消除作用域歧义 |
| `RingGeoStatus` | public frozen dataclass | 环状态集合、metrics、reasonableness score 和证据 |
| `BondRingGeoStatus` | public frozen dataclass | 单个 pair 状态、逐曲面事件、交点、裕量和 score |
| `RingSurface` | private frozen dataclass | 一个由原环顶点组成的三角盘 |
| `RingSurfaceEnsemble` | private frozen dataclass | 全部有效三角盘及构造诊断 |
| `MoleculeGeoStatus` | public frozen dataclass | 聚合分子的环和 pair 报告，给力场提供统一参考线 |

### 3.3 函数和接口

| 函数 | 可见性 | 输入 → 输出 | 职责 |
|---|---|---|---|
| `_geometry_scale()` | private | ring coordinates → `s, epsilon` | 建立无量纲尺度和数值容差 |
| `_ring_clearance_metrics()` | private | ring coordinates → clearance metrics | 复用 point/segment distance，检查近接触和自碰撞 |
| `_ring_shape_metrics()` | private | ring coordinates → curvature/SVD/aperture | 描述急折、塌缩、平面性和 puckering |
| `_determine_ring_geo_status()` | private | `Ring` → `RingGeoStatus` | 形成环的详细状态，不输出三态 verdict |
| `_enumerate_ring_surfaces()` | private | `RingGeoStatus` → `RingSurfaceEnsemble` | 枚举合法三角剖分、提升至 3D 并过滤无效曲面 |
| `_segment_triangle_event()` | private | finite segment + triangle → event | 区分无交点、内部横穿、边界/共面接触和数值不确定 |
| `_determine_bond_ring_geo_status()` | private | `BondRingPair` → `BondRingGeoStatus` | 对全部有效曲面取共识，作用域严格为一个 pair |
| `determine_geo_status()` | public | `Ring | BondRingPair` → 对应 status | 唯一详细几何判定入口，使用 overload 而非 `Any` |
| `is_geo_reasonable()` | public | detailed status → `GeometryVerdict` | 纯映射函数，不重复计算、不调用力场 |
| `determine_molecule_geo_status()` | public | `Molecule` → `MoleculeGeoStatus` | 收集所有 ring/pair；不合理优先，其次不确定 |

旧 `bond_intersects_ring()`、`find_bond_ring_intersections()` 和
`has_bond_ring_intersection()` 在生产调用迁移后删除，不保留把三态压回布尔值的兼容路径。

## 附录 A：当前缺陷的动态证据

审查基线中：

- `_segment_intersects_triangle()` 约在 `geometry.py:318-349`；
- `_line_intersects_polygon()` 约在 `geometry.py:436-476`；
- `bond_intersects_ring()` 约在 `geometry.py:1411-1430`；
- `tests/test_cheminfo/test_geometry.py:144-154` 锁定 historical center-fan。

动态测试采用 L 形凹六边界和穿过凹口外部的有限 probe：

```text
dz=0       branch=planar  hit=False
dz=2e-8    branch=planar  hit=False
dz=5e-8    branch=fan     hit=True
dz=1e-7    branch=fan     hit=True
dz=1e-3    branch=fan     hit=True
dz=0.2     branch=fan     hit=True
```

只增加 `5e-8 Å` 的 z 扰动就由合理翻转为不合理。原因是 centroid 位于 L 形凹口外，fan
产生虚构面积。该问题严重度为高，会直接拒绝本应进入优化的构筑候选。

## 附录 B：为什么需要环面集合

非平面闭合空间折线没有天然唯一的二维内部面。以下空间四边形具有同一个边界：

```text
p0=(-1,-1,0)  p1=( 1,-1,1)
p3=(-1, 1,1)  p2=( 1, 1,0)
```

使用 `p0-p2` 对角线时中心面高度为 `z=0`；使用 `p1-p3` 时为 `z=1`。短键段
`[(0,0,-0.1), (0,0,0.1)]` 只穿过前一个面。因此单个三角剖分只能给出一种工程约定。

当前最多检查八元环。凸 `n` 边形三角剖分数为 Catalan 数 `C_(n-2)`，八边形最多仅
`C_6=132` 种。可枚举全部合法剖分并提升回原始 3D 顶点：

```text
全部曲面无 finite-segment hit -> ALL_SURFACES_CLEAR
全部曲面有 transverse hit     -> ALL_SURFACES_PIERCED
曲面之间结论不同               -> SURFACE_DISAGREEMENT
```

该共识相较单个 Earcut 面消除了对某一条任意对角线的依赖。它严格作用于“由原始环顶点形成的
合法 vertex-only 三角盘集合”，不宣称枚举了所有含 Steiner 点的连续曲面。

## 附录 C：实施算法、依赖和验收

### C.1 算法流程

1. 计算环自身尺度、有限性、退化、非相邻边距离、clearance、curvature、thickness、SVD 和
   aperture metrics；
2. 用 SVD best-fit plane 建立二维参数化，验证 simple polygon；
3. 递归枚举全部内部非交叉三角剖分，要求 `n-2` 个三角形并验证面积守恒；
4. 提升回三维，过滤退化或三角形自交的曲面；
5. 用 AABB broad phase 和 finite segment–triangle narrow phase 判定逐曲面事件；
6. 去除内部对角线造成的重复命中并生成 `BondRingGeoStatus`；
7. `is_geo_reasonable()` 按固定映射表生成三态；
8. 分子级按“不合理优先、其次不确定”聚合。

投影自交、近零面积或不存在有效曲面时返回 `UNCERTAIN`，禁止回退到 centroid fan。首轮不在
投影失败后静默切换到另一种抽象曲面语义。

### C.2 复用和依赖

- 复用现有 `_point_segment_distance()` 和 `_segment_distance()`，不建立第二套距离 kernel；
- NumPy 继续承担 SVD 和向量运算；
- `n <= 8` 时自有递归穷举器最多处理 132 个剖分，性能足够；
- 已实测 `mapbox-earcut==2.1.0` 能修复 L 形凹环反例，约 `2.65 µs/call`，但它只返回一个
  三角剖分，因此只作测试 oracle；
- CGAL 可提供 exact predicates，但构建和 Python 3.9-3.14 分发成本过高，当前不作为依赖。

### C.3 分步迁移

1. 增加 enum、dataclass、threshold 和两项公开接口；
2. 先实现环详细状态及映射测试；
3. 实现环面枚举、三维过滤和表示不变性测试；
4. 实现单个 ring–bond pair 的有限键段判定；
5. 实现分子聚合并迁移 geometry quality / forcefield 调用者；
6. 删除 center-fan 和旧布尔 API；
7. 每一步单独提交。

### C.4 必须通过的测试

1. L 形反例在 `dz=0..1e-3` 内始终得到相同三态结果；
2. 平面凸/凹环、chair、puckered、急折、塌缩、自交和近接触；
3. 四顶点双对角线反例得到 `UNCERTAIN + SURFACE_DISAGREEMENT`；
4. 有限键段穿越得到 `UNREASONABLE + ALL_SURFACES_PIERCED`；
5. 只有无限延长线穿越得到 `REASONABLE + LINE_EXTENSION_ONLY`；
6. 擦边、顶点、共面和数值边界均得到 `UNCERTAIN`；
7. 循环移位、遍历反向、原子重编号、刚体变换和均匀缩放不改变 verdict；
8. ring–bond pair 与 molecule 两个作用域明确且聚合结果正确；
9. 非法/不可判输入绝不返回 `REASONABLE`；
10. 性能测试记录每个环、pair 和整分子的成本。
