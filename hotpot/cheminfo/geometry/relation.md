# `hotpot.cheminfo.geometry` API 参考

本文说明 `hotpot.cheminfo.geometry` 的全部公开接口、数学语义、数值参数和化学对象转换规则。
文档与 package 根目录的 `__all__` 对齐；用户可统一从以下入口导入：

```python
from hotpot.cheminfo import geometry as geo
```

## 1. Package introduction

`geometry` 是 Hotpot 的三维欧氏几何事实层。它提供四类能力：

1. 用不可变对象表示点、无限直线、有限线段、平面、三角形和有序闭合环边界；
2. 计算平面度、直线关系、有限对象间距离和点—环投影位置；
3. 在明确声明的环面模型下，给出有限线段与环的三态关系：
   `PIERCES`、`DOES_NOT_PIERCE` 或 `UNDETERMINED`；
4. 把 Hotpot 或满足相同结构协议的 Atom、Bond、Ring、Molecule 转换为纯几何对象，同时保留
   来源对象引用和稳定 key。

该 package 只报告空间事实，不判断结构是否化学合理、物理稳定或适合某种力场。诸如“是否接受
一个构象”“是否断开环键”“如何处理 `UNDETERMINED`”等决策属于调用方，当前由
`hotpot.cheminfo.forcefields` 等业务层承担。

模块职责和依赖方向为：

```text
settings.py  ─┐
object.py    ─┼─> relation.py ─┐
              └───────────────┼─> convert.py
                               └─> package __init__.py
```

- `settings.py`：数值容差和非平面环曲面枚举预算；
- `object.py`：不含关系算法的不可变几何值对象；
- `relation.py`：只接受纯几何对象的度量和关系分类；
- `convert.py`：唯一理解化学对象结构的薄适配层。

### 1.1 单位和输入约定

- 全文以 `CLU` 表示调用方采用的统一坐标长度单位。Hotpot 分子坐标通常以 Å 为 `CLU`；
- 同一次计算中的所有坐标和长度阈值必须使用相同单位；
- 数值 kernel 统一以 `float64` 计算；
- 几何对象允许携带非有限坐标，以便关系函数显式返回 `NaN` 或 `UNDETERMINED`；
- `Cycle` 仅表示一维有序闭合边界，不自行假定一个唯一的内部曲面。

### 1.2 快速示例

```python
from hotpot.cheminfo import geometry as geo

cycle = geo.Cycle([
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.0, 2.0, 0.0),
    (0.0, 2.0, 0.0),
])
segment = geo.Segment((1.0, 1.0, -1.0), (1.0, 1.0, 1.0))

result = geo.determine_segment_cycle_relation(segment, cycle)
assert result.state is geo.PiercingState.PIERCES
```

化学对象可直接走转换层：

```python
report = geo.scan_bond_ring_relations(
    mol,
    ring_scope="ligand_skeleton",
    max_ring_size=8,
)

for finding in report.piercings:
    print(finding.target.bond.source, finding.target.ring.source)
```

## 2. 全部公开 API 总览

### 2.1 Settings（4 项）

| 名称 | 种类 | 含义 |
|---|---|---|
| `NumericToleranceSettings` | frozen dataclass | 分量纲数值容差和保护带配置 |
| `SurfaceEnumerationSettings` | frozen dataclass | 非平面环候选曲面及谓词调用预算 |
| `GeometrySettings` | frozen dataclass | 聚合 numeric 与 surface settings |
| `DEFAULT_GEOMETRY_SETTINGS` | instance | 所有支持 `settings` 参数的接口默认配置 |

### 2.2 几何值对象（6 项）

| 名称 | 含义 |
|---|---|
| `Point` | 三维点 |
| `Line` | 由原点和方向定义的无限直线 |
| `Segment` | 含两个端点的有限闭线段 |
| `Plane` | 由平面上一点和单位法向定义的无限平面 |
| `Triangle` | 三个有序顶点构成的三角形；允许退化 |
| `Cycle` | 至少三个有序顶点构成的闭合一维边界 |

### 2.3 关系词汇和结果记录（15 项）

| 名称 | 种类 | 含义 |
|---|---|---|
| `PlanarityKind` | Enum | 平面度分类 |
| `LineRelationKind` | Enum | 两条无限直线的空间关系 |
| `PointCycleLocation` | Enum | 点在平面环投影中的位置 |
| `SurfaceEmbeddingState` | Enum | 候选三角曲面的嵌入状态 |
| `SurfaceSegmentState` | Enum | 线段与单个嵌入曲面的关系 |
| `PiercingState` | Enum | 线段—环最终三态结果 |
| `SegmentCycleFeature` | Enum | 已确认的穿越或接触事实 |
| `SegmentCycleIndeterminacy` | Enum | 不能二分为穿越/不穿越的原因 |
| `CycleSurfaceModel` | Enum | 判定环内部所采用的曲面模型 |
| `PlanarityMeasurement` | frozen dataclass | 平面度分类及 SVD 度量 |
| `LineRelation` | frozen dataclass | 直线关系、距离和平行度量 |
| `PointPairDistance` | frozen dataclass | 输入点序列中一对点的距离 |
| `ClosestCycleEdge` | frozen dataclass | 最近环边及距离 |
| `SurfaceFamilyEvidence` | frozen dataclass | 非平面候选曲面枚举和求交计数 |
| `SegmentCycleRelation` | frozen dataclass | 线段—环状态、证据、交点和数值配置 |

### 2.4 关系函数（11 项）

| 名称 | 作用 |
|---|---|
| `measure_planarity` | 测量有序环相对最佳拟合平面的偏离 |
| `determine_line_relation` | 分类两条无限直线 |
| `line_distance` | 返回两条无限直线的距离 |
| `point_segment_distance` | 计算点到有限线段的最短距离 |
| `segment_segment_distance` | 计算两条有限线段的最短距离 |
| `point_pair_distances` | 枚举全部无序点对距离 |
| `find_point_pairs_below_distance` | 按调用方阈值筛选点对 |
| `locate_point_in_planar_cycle` | 分类点在平面环投影中的位置 |
| `iter_segment_cycle_relations` | 对同一环批量、惰性判定多条线段 |
| `determine_segment_cycle_relation` | 判定一条有限线段是否穿过一个环 |
| `closest_cycle_edge` | 查找距目标线段最近的环边 |

### 2.5 转换词汇和结果记录（12 项）

| 名称 | 种类 | 含义 |
|---|---|---|
| `PairScope` | Literal alias | 原子对范围：全部、成键或未成键 |
| `RingScope` | Literal alias | 环范围：全图或配体骨架 |
| `RingFamily` | Enum | 扫描报告记录的环识别算法族标签；当前固定表示 Hotpot Core 的 NetworkX cycle basis |
| `AtomGeometry` | frozen generic dataclass | Atom、`Point` 和 atom key 的映射 |
| `AtomPairTarget` | frozen generic dataclass | 两个 atom mapping 及成键事实 |
| `BondGeometry` | frozen generic dataclass | Bond、`Segment` 和 bond key 的映射 |
| `RingGeometry` | frozen generic dataclass | Ring、`Cycle` 和 canonical ring key 的映射 |
| `BondRingTarget` | frozen generic dataclass | 一个待判定的 Ring × Bond 组合 |
| `AtomPairDistance` | frozen generic dataclass | 原子对来源与纯几何距离记录 |
| `BondRingFinding` | frozen generic dataclass | Ring × Bond 来源与线段—环关系 |
| `RingEdgeDistance` | frozen generic dataclass | 来源环键与最近边距离记录 |
| `BondRingScanReport` | frozen generic dataclass | 指定环范围内的稠密扫描报告 |

### 2.6 转换函数（12 项）

| 名称 | 作用 |
|---|---|
| `point_from_atom` | Atom-like 对象转为 `Point` |
| `segment_from_bond` | Bond-like 对象转为 `Segment` |
| `cycle_from_ring` | Ring-like 对象转为 `Cycle` |
| `iter_atom_geometries` | 逐原子产生来源—几何映射 |
| `iter_atom_pair_targets` | 按 `PairScope` 产生原子对 |
| `iter_ring_geometries` | 按 `RingScope` 和环大小产生环映射 |
| `iter_bond_ring_targets` | 产生候选 Ring × Bond 对 |
| `measure_atom_pair_distances` | 计算并保留来源对象的原子对距离 |
| `determine_bond_ring_relation` | 判定一个化学 Ring × Bond 对 |
| `iter_bond_ring_findings` | 惰性扫描指定范围内的 Ring × Bond 对 |
| `scan_bond_ring_relations` | 返回指定范围内的完整稠密扫描报告 |
| `determine_bond_ring_piercing_state` | 早退式聚合整个分子的穿环三态 |

## 3. 数学符号和数值约定

全文固定使用以下符号：

| 符号 | 含义 |
|---|---|
| `CLU` | 当前调用采用的统一坐标长度单位 |
| $\mathbf p$、$\mathbf q$、$\mathbf x$ | 三维点或其坐标向量 |
| $\mathcal L_i$ | 第 $i$ 条无限直线 |
| $\mathbf o_i$、$\mathbf d_i$ | $\mathcal L_i$ 的原点和方向向量 |
| $S=[\mathbf a,\mathbf b]$ | 端点为 $\mathbf a,\mathbf b$ 的有限线段 |
| $\Pi(\mathbf q,\hat{\mathbf n})$ | 经过 $\mathbf q$、单位法向为 $\hat{\mathbf n}$ 的平面 |
| $T=(\mathbf a,\mathbf b,\mathbf c)$ | 有序三角形 |
| $\mathcal C=(\mathbf v_0,\ldots,\mathbf v_{m-1})$ | 含 $m$ 个有序顶点的环边界 |
| $e_i=[\mathbf v_i,\mathbf v_{(i+1)\bmod m}]$ | 第 $i$ 条环边 |
| $L$ | 当前谓词的局部长度尺度 |
| $\epsilon_L,\epsilon_A,\epsilon_V$ | 长度、面积、体积量纲容差 |
| $\epsilon_u$ | 无量纲参数容差 |
| $g$ | 数值保护倍率 `predicate_guard_factor` |
| $\mathbf x_\cap$ | 已计算的交点 |

### 3.1 局部尺度

对当前谓词实际使用的点集 $X$、线段集 $E$ 和可选环 $\mathcal C$，局部尺度为：

$L=\max\left(\operatorname{diam}(X),\max_{e\in E}\lVert e\rVert,\operatorname{median}_{e_i\in\partial\mathcal C}\lVert e_i\rVert\right)$

不存在的项会被省略。该尺度不读取整个分子的包围盒，因此远端无关原子不会改变当前局部判定。

### 3.2 派生容差

令 float64 机器精度为 $\epsilon_{\mathrm{machine}}$：

$\epsilon_r=\max(\epsilon_{\mathrm{rel}},k_{\mathrm{mach}}\epsilon_{\mathrm{machine}})$

$\epsilon_L=\epsilon_{\mathrm{abs}}+\epsilon_rL$

$\epsilon_u=\epsilon_{\mathrm{param}}+\epsilon_L/L\qquad(L>0)$

$\epsilon_A=\epsilon_LL$

$\epsilon_V=\epsilon_LL^2$

$\epsilon_{\mathrm{AABB}}=k_{\mathrm{AABB}}\epsilon_L$

$\epsilon_{\mathrm{merge}}=k_{\mathrm{merge}}\epsilon_L$

其中 $\epsilon_L$、$\epsilon_A$、$\epsilon_V$ 分别只与长度、面积和体积量比较；
$\epsilon_u$ 只与参数或归一化量比较。

### 3.3 保护带和三态

对残差 $r$ 及与它同量纲的容差 $\epsilon$：

- $|r|\le\epsilon$：接触或位于定义边界；
- $\epsilon<|r|\le g\epsilon$：数值未决保护带；
- $|r|>g\epsilon$：可以稳定判断符号或分离。

需要把 $[0,1]$ 参数分成端点、内部和外部时，还要求 $g\epsilon_u<1/2$。不满足该条件时，
关系返回 `UNDETERMINED` 并记录 `TOLERANCE_DOMAIN`。

## 4. Settings API

### 4.1 `NumericToleranceSettings`

```python
NumericToleranceSettings(
    absolute_length: float = 1.0e-8,
    relative_length: float = 1.0e-10,
    parameter: float = 1.0e-10,
    machine_epsilon_factor: float = 64.0,
    predicate_guard_factor: float = 4.0,
    planarity_factor: float = 1.0,
    winding_residual: float = 1.0e-10,
    intersection_merge_factor: float = 4.0,
    aabb_padding_factor: float = 4.0,
)
```

冻结配置对象，定义所有关系 kernel 使用的数值宽容度。

| 参数 | 符号 | 默认值 | 单位/有效范围 | 用途 |
|---|---|---:|---|---|
| `absolute_length` | $\epsilon_{\mathrm{abs}}$ | $10^{-8}$ | `CLU`, $>0$ | 绝对长度分辨率和退化基线 |
| `relative_length` | $\epsilon_{\mathrm{rel}}$ | $10^{-10}$ | 无量纲, $\ge0$ | 随局部尺度增长的相对分辨率 |
| `parameter` | $\epsilon_{\mathrm{param}}$ | $10^{-10}$ | 无量纲, $0<\epsilon_{\mathrm{param}}<1/(2g)$ | 线段参数、重心坐标和角度量容差 |
| `machine_epsilon_factor` | $k_{\mathrm{mach}}$ | $64$ | 无量纲, $\ge1$ | float64 舍入误差下限倍率 |
| `predicate_guard_factor` | $g$ | $4$ | 无量纲, $>1$ | 确定区与未决区间的保护倍率 |
| `planarity_factor` | $k_{\mathrm{plane}}$ | $1$ | 无量纲, $>0$ | 平面偏差阈值相对 $\epsilon_L$ 的倍率 |
| `winding_residual` | $\epsilon_{\mathrm{winding}}$ | $10^{-10}$ | 无量纲, $0<\epsilon_{\mathrm{winding}}<1/2$ | winding number 到 $0$ 或 $\pm1$ 的允许残差 |
| `intersection_merge_factor` | $k_{\mathrm{merge}}$ | $4$ | 无量纲, $\ge1$ | 合并数值上相同交点 |
| `aabb_padding_factor` | $k_{\mathrm{AABB}}$ | $4$ | 无量纲, $\ge g$ | AABB 稳定分离检查的安全扩张 |

所有参数必须是有限数；无效设置在构造时抛出 `ValueError`。

### 4.2 `SurfaceEnumerationSettings`

```python
SurfaceEnumerationSettings(
    maximum_cycle_vertices: int = 8,
    maximum_surface_count: int = 132,
    maximum_segment_triangle_tests: int = 792,
    maximum_triangle_pair_tests: int = 1980,
)
```

冻结配置对象，限制非平面环完整候选曲面枚举的计算量。

| 参数 | 符号 | 默认值 | 有效范围 | 用途 |
|---|---|---:|---|---|
| `maximum_cycle_vertices` | $B_{\mathrm{vertex}}$ | $8$ | 整数, $\ge3$ | 允许完整顶点三角剖分的最大环顶点数 |
| `maximum_surface_count` | $B_{\mathrm{surface}}$ | $132$ | 正整数 | 候选三角曲面数量预算；$132=\operatorname{Cat}_6$ |
| `maximum_segment_triangle_tests` | $B_{S\triangle}$ | $792$ | 正整数 | 一次线段—环关系中 segment–triangle 谓词预算 |
| `maximum_triangle_pair_tests` | $B_{\triangle\triangle}$ | $1980$ | 正整数 | 构造候选曲面时 triangle-pair 谓词预算 |

预算耗尽不会用部分结果给出确定结论；关系会记录不完整证据并返回 `UNDETERMINED`。

### 4.3 `GeometrySettings`

```python
GeometrySettings(
    tolerance: NumericToleranceSettings = NumericToleranceSettings(),
    surface: SurfaceEnumerationSettings = SurfaceEnumerationSettings(),
)
```

将数值容差和曲面预算组合为一个不可变配置。两个字段均通过 `default_factory` 创建，实例之间不共享
可变状态。

### 4.4 `DEFAULT_GEOMETRY_SETTINGS`

```python
DEFAULT_GEOMETRY_SETTINGS: GeometrySettings
```

模块级默认实例。所有带 `settings` 的公开关系和转换接口默认引用该对象。自定义配置通常使用
`dataclasses.replace()`：

```python
from dataclasses import replace
from hotpot.cheminfo import geometry as geo

settings = replace(
    geo.DEFAULT_GEOMETRY_SETTINGS,
    tolerance=replace(
        geo.DEFAULT_GEOMETRY_SETTINGS.tolerance,
        absolute_length=1.0e-7,
    ),
)
```

## 5. 几何值对象

下列类均为冻结 dataclass。`PointInput` 在签名中表示 `Point | Iterable[float]`，三维坐标会被转换为
三个 `float`；坐标数量不是 3 时抛出 `ValueError`。

### 5.1 `Point`

```python
Point(coordinates: Iterable[float])
Point.from_coordinates(coordinates: Iterable[float]) -> Point
```

表示三维点 $\mathbf p=(p_x,p_y,p_z)$。

- 字段：`coordinates: tuple[float, float, float]`；
- 属性：`x`、`y`、`z`；
- 可迭代，顺序为 `x, y, z`；
- `from_coordinates()` 与构造器语义相同，是显式命名构造器。

### 5.2 `Line`

```python
Line(origin: PointInput, direction: Iterable[float])
Line.from_points(first: PointInput, second: PointInput) -> Line
```

表示无限直线：

$\mathcal L(\mathbf o,\mathbf d)=\{\mathbf o+t\mathbf d\mid t\in\mathbb R\}$

- 字段：`origin: Point`、`direction: tuple[float, float, float]`；
- `from_points(first, second)` 令 $\mathbf o=\mathbf p_1$、
  $\mathbf d=\mathbf p_2-\mathbf p_1$；
- 构造器不归一化方向，也不拒绝零方向；关系函数将零方向报告为 `DEGENERATE`。

### 5.3 `Segment`

```python
Segment(start: PointInput, end: PointInput)
```

表示有限闭线段：

$S=[\mathbf a,\mathbf b]=\{\mathbf a+t(\mathbf b-\mathbf a)\mid t\in[0,1]\}$

- 字段：`start: Point`、`end: Point`；
- `direction = end - start`；
- `length = ||end - start||`；
- 相同端点允许构造，但关系分类会把它视为退化线段。

### 5.4 `Plane`

```python
Plane(point: PointInput, normal: Iterable[float])
```

表示平面：

$\Pi(\mathbf q,\hat{\mathbf n})=\{\mathbf x\mid(\mathbf x-\mathbf q)\cdot\hat{\mathbf n}=0\}$

- 字段：`point: Point`、`normal: tuple[float, float, float]`；
- 有限非零法向在构造时归一化为 $\hat{\mathbf n}$；
- 零法向抛出 `ValueError`。

### 5.5 `Triangle`

```python
Triangle(first: PointInput, second: PointInput, third: PointInput)
```

表示有序三角形 $T=(\mathbf a,\mathbf b,\mathbf c)$。

- 字段：`first`、`second`、`third`；
- `vertices -> tuple[Point, Point, Point]`；
- `edges -> tuple[Segment, Segment, Segment]`，顺序为 $ab,bc,ca$；
- 共线或重合顶点允许构造，后续谓词负责报告退化。

### 5.6 `Cycle`

```python
Cycle(vertices: Iterable[PointInput])
```

表示有序闭合边界 $\mathcal C=(\mathbf v_0,\ldots,\mathbf v_{m-1})$，闭合边为：

$e_i=[\mathbf v_i,\mathbf v_{(i+1)\bmod m}]$

- 字段：`vertices: tuple[Point, ...]`；
- `edges -> tuple[Segment, ...]`；
- `len(cycle)` 返回顶点数 $m$；
- 迭代产生有序顶点；
- 少于三个顶点时抛出 `ValueError`；
- 它只定义边界，平面多边形或非平面候选曲面由关系函数另行构造。

## 6. 关系词汇和返回结构

### 6.1 `PlanarityKind`

| 成员 | `.value` | 含义 |
|---|---|---|
| `PLANAR` | `"planar"` | 最大面外偏差稳定落在平面阈值内 |
| `NONPLANAR` | `"nonplanar"` | 最大面外偏差稳定超过保护带 |
| `DEGENERATE` | `"degenerate"` | 点集尺度过小或近似秩小于 2 |
| `UNDETERMINED` | `"undetermined"` | 非有限输入或结果位于数值保护带 |

### 6.2 `LineRelationKind`

| 成员 | `.value` | 含义 |
|---|---|---|
| `INTERSECTING` | `"intersecting"` | 两条非平行无限直线相交 |
| `PARALLEL` | `"parallel"` | 方向平行且直线分离 |
| `COINCIDENT` | `"coincident"` | 两条无限直线重合 |
| `SKEW` | `"skew"` | 两条三维直线既不平行也不相交 |
| `DEGENERATE` | `"degenerate"` | 至少一个方向向量为零 |
| `UNDETERMINED` | `"undetermined"` | 非有限输入或关系落在保护带 |

### 6.3 `PointCycleLocation`

| 成员 | `.value` | 含义 |
|---|---|---|
| `INTERIOR` | `"interior"` | 投影点稳定位于简单多边形内部 |
| `BOUNDARY` | `"boundary"` | 投影点位于环边界容差内 |
| `EXTERIOR` | `"exterior"` | 投影点稳定位于外部 |
| `UNDETERMINED` | `"undetermined"` | 环投影不简单、输入非有限或点位于保护带 |

### 6.4 `SurfaceEmbeddingState`

公开的非平面候选曲面构造词汇；当前公开函数不直接返回该 Enum，其统计进入
`SurfaceFamilyEvidence`。

| 成员 | `.value` | 含义 |
|---|---|---|
| `EMBEDDED` | `"embedded"` | 三角化曲面无额外自交且边界与原环一致 |
| `PROVEN_NON_EMBEDDED` | `"proven_non_embedded"` | 已稳定证明候选曲面含退化三角形、额外交叠或自交 |
| `CONSTRUCTION_UNDETERMINED` | `"construction_undetermined"` | 数值证据不足以确定候选曲面是否嵌入 |

### 6.5 `SurfaceSegmentState`

公开的单个嵌入曲面—线段关系词汇；当前公开函数将它汇总到最终 `PiercingState`。

| 成员 | `.value` | 含义 |
|---|---|---|
| `INTERSECTING` | `"intersecting"` | 线段开区间确认横穿该曲面内部 |
| `NON_PIERCING` | `"non_piercing"` | 已完成所需谓词且没有严格横穿 |
| `EVALUATION_UNDETERMINED` | `"evaluation_undetermined"` | 至少一个必要求交谓词未决 |

### 6.6 `PiercingState`

| 成员 | `.value` | 含义 |
|---|---|---|
| `PIERCES` | `"pierces"` | 在声明且完整检查的曲面模型中确认有限线段严格穿过内部 |
| `DOES_NOT_PIERCE` | `"does_not_pierce"` | 在声明且完整检查的曲面模型中确认没有严格穿过内部 |
| `UNDETERMINED` | `"undetermined"` | 退化、容差保护带、曲面构造或候选曲面共识不足 |

该状态不是化学合理性结论。

### 6.7 `SegmentCycleFeature`

`SegmentCycleRelation.features` 是可共存的事实集合；即使最终状态为 `PIERCES`，也可能同时存在
端点或边界接触。

| 成员 | `.value` | 含义 |
|---|---|---|
| `TRANSVERSE_INTERIOR` | `"transverse_interior"` | 有限线段开区间横穿曲面内部 |
| `LINE_EXTENSION_INTERIOR` | `"line_extension_interior"` | 至少一个已评估曲面或三角形的内部交点仅位于线段延长线上；聚合结果中可与其他 feature 共存 |
| `CYCLE_EDGE_CONTACT` | `"cycle_edge_contact"` | 接触环边内部 |
| `CYCLE_VERTEX_CONTACT` | `"cycle_vertex_contact"` | 接触环顶点 |
| `SEGMENT_ENDPOINT_CONTACT` | `"segment_endpoint_contact"` | 交点位于目标线段端点 |
| `COPLANAR_CONTACT` | `"coplanar_contact"` | 有限线段与环区域或三角面发生共面接触 |

### 6.8 `SegmentCycleIndeterminacy`

| 成员 | `.value` | 含义 |
|---|---|---|
| `NONFINITE_INPUT` | `"nonfinite_input"` | 输入含 `NaN` 或无穷值 |
| `NUMERIC_BAND` | `"numeric_band"` | 必要谓词落入数值保护带 |
| `TOLERANCE_DOMAIN` | `"tolerance_domain"` | 派生参数容差不能稳定划分 $[0,1]$ |
| `DEGENERATE_CYCLE` | `"degenerate_cycle"` | 环的局部尺度或秩退化 |
| `DEGENERATE_SEGMENT` | `"degenerate_segment"` | 目标线段长度退化 |
| `DEGENERATE_TRIANGLE` | `"degenerate_triangle"` | 候选曲面含退化三角形 |
| `SELF_INTERSECTION` | `"self_intersection"` | 平面环投影确认自交 |
| `SURFACE_DISAGREEMENT` | `"surface_disagreement"` | 合法候选曲面对穿越结论不一致，或没有任何可形成共识的嵌入曲面且无更具体原因 |
| `INCOMPLETE_SURFACE_FAMILY` | `"incomplete_surface_family"` | 顶点数或计算预算导致枚举/求交不完整 |
| `SURFACE_CONSTRUCTION` | `"surface_construction"` | 至少一个候选曲面无法确定是否嵌入 |

### 6.9 `CycleSurfaceModel`

| 成员 | `.value` | 含义 |
|---|---|---|
| `PLANAR_POLYGON` | `"planar_polygon"` | 最佳拟合平面上的简单多边形内部 |
| `VERTEX_TRIANGULATION_FAMILY` | `"vertex_triangulation_family"` | 非平面环全部合法顶点三角剖分的共识 |

### 6.10 `PlanarityMeasurement`

```python
PlanarityMeasurement(
    kind: PlanarityKind,
    centroid: Point,
    normal: tuple[float, float, float] | None,
    singular_values: tuple[float, float, float],
    maximum_deviation: float,
    rms_deviation: float,
    length_scale: float,
    length_tolerance: float,
)
```

| 字段 | 含义 |
|---|---|
| `kind` | 平面度分类 |
| `centroid` | 环顶点质心；非有限输入时坐标为 `NaN` |
| `normal` | 最佳拟合平面单位法向；退化/未决时可为 `None`，正负方向不保证固定 |
| `singular_values` | 降序的三个奇异值 $\sigma_1,\sigma_2,\sigma_3$ |
| `maximum_deviation` | 顶点最大绝对面外偏差；不可定义时为 `NaN` |
| `rms_deviation` | 顶点面外偏差 RMS；不可定义时为 `NaN` |
| `length_scale` | 当前环的局部尺度 $L$ |
| `length_tolerance` | 当前环的派生 $\epsilon_L$ |

### 6.11 `LineRelation`

```python
LineRelation(
    kind: LineRelationKind,
    distance: float | None,
    parallel_measure: float,
)
```

| 字段 | 含义 |
|---|---|
| `kind` | 两条无限直线的分类 |
| `distance` | 可确定时的最短距离；退化或未决时为 `None` |
| `parallel_measure` | $q_{\parallel}=\lVert\hat{\mathbf d}_1\times\hat{\mathbf d}_2\rVert$；不可计算时为 `NaN` |

### 6.12 `PointPairDistance`

```python
PointPairDistance(first_index: int, second_index: int, distance: float)
```

`first_index` 和 `second_index` 是输入 `Sequence[Point]` 的位置，始终满足
`first_index < second_index`，并不是化学 Atom 的 `idx`。非有限点对的 `distance` 为 `NaN`。

### 6.13 `ClosestCycleEdge`

```python
ClosestCycleEdge(edge_index: int, edge: Segment, distance: float)
```

记录最近环边的零基索引、对应 `Segment` 和它到目标线段的最短距离。

### 6.14 `SurfaceFamilyEvidence`

```python
SurfaceFamilyEvidence(
    enumeration_complete: bool,
    enumerated_surface_count: int,
    embedded_surface_count: int,
    proven_non_embedded_surface_count: int,
    construction_undetermined_count: int,
    intersecting_surface_count: int,
    non_piercing_surface_count: int,
    evaluation_undetermined_count: int,
    segment_triangle_tests_used: int,
    triangle_pair_tests_used: int,
)
```

| 字段 | 含义 |
|---|---|
| `enumeration_complete` | 候选曲面族构造/枚举及本次所需线段评估是否均未因预算中断 |
| `enumerated_surface_count` | 已进入构造分类的候选曲面数 |
| `embedded_surface_count` | 确认嵌入的候选曲面数 |
| `proven_non_embedded_surface_count` | 确认非嵌入、从共识中排除的曲面数 |
| `construction_undetermined_count` | 嵌入状态未决的候选曲面数 |
| `intersecting_surface_count` | 线段确认穿过的嵌入曲面数 |
| `non_piercing_surface_count` | 线段确认不穿的嵌入曲面数 |
| `evaluation_undetermined_count` | 线段关系未决的嵌入曲面数 |
| `segment_triangle_tests_used` | 已使用的 segment–triangle 谓词次数 |
| `triangle_pair_tests_used` | 已使用的 triangle-pair 嵌入检查次数 |

完整非平面枚举满足：

$n_{\mathrm{enum}}=n_{\mathrm{emb}}+n_{\mathrm{nonemb}}+n_{\mathrm{construct\_undetermined}}$

$n_{\mathrm{emb}}=n_{\mathrm{hit}}+n_{\mathrm{miss}}+n_{\mathrm{eval\_undetermined}}$

### 6.15 `SegmentCycleRelation`

```python
SegmentCycleRelation(
    state: PiercingState,
    features: frozenset[SegmentCycleFeature],
    indeterminacy_causes: frozenset[SegmentCycleIndeterminacy],
    surface_model: CycleSurfaceModel | None,
    intersection_points: tuple[Point, ...],
    closest_boundary_edge: ClosestCycleEdge | None,
    surface_evidence: SurfaceFamilyEvidence,
    settings: GeometrySettings,
)
```

| 字段 | 含义 |
|---|---|
| `state` | 最终三态关系 |
| `features` | 所有已确认的穿越、延长线命中和接触事实 |
| `indeterminacy_causes` | 判定过程中观察到的未决或不完整证据；最终 `state` 已有确定共识时也可能非空 |
| `surface_model` | 已选曲面模型；模型无法选择时为 `None` |
| `intersection_points` | kernel 显式返回并经 $\epsilon_{\mathrm{merge}}$ 合并的离散有限线段交点/接触点；不含共面接触区域或延长线交点 |
| `closest_boundary_edge` | 最近环边；输入非有限或尺度不可比较时为 `None` |
| `surface_evidence` | 候选曲面和求交预算证据；始终存在 |
| `settings` | 本次判定实际使用的配置对象 |

## 7. 关系函数

### 7.1 `measure_planarity`

```python
def measure_planarity(
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PlanarityMeasurement
```

对环顶点坐标做最佳拟合平面 SVD。质心和中心化矩阵为：

$\mathbf c=\frac1m\sum_{i=0}^{m-1}\mathbf v_i$

$X=[\mathbf v_i-\mathbf c]$

$X=U\Sigma V^{\mathsf T},\qquad \sigma_1\ge\sigma_2\ge\sigma_3$

最佳拟合平面法向取 $\sigma_3$ 对应的右奇异向量 $\hat{\mathbf n}$。面外偏差为：

$h_i=|(\mathbf v_i-\mathbf c)\cdot\hat{\mathbf n}|$

$h_{\max}=\max_i h_i$

$h_{\mathrm{rms}}=\sqrt{\frac1m\sum_i h_i^2}$

定义秩比 $q_{\mathrm{rank}}=\sigma_2/\sigma_1$。关键分类为：

- $L\le\epsilon_{\mathrm{abs}}$：`DEGENERATE`；
- $\sigma_1\le\epsilon_L$：`DEGENERATE`；
- $\epsilon_L<\sigma_1\le g\epsilon_L$：`UNDETERMINED`；
- $q_{\mathrm{rank}}\le\epsilon_u$：`DEGENERATE`；
- $\epsilon_u<q_{\mathrm{rank}}\le g\epsilon_u$：`UNDETERMINED`；
- $h_{\max}\le k_{\mathrm{plane}}\epsilon_L$：`PLANAR`；
- $h_{\max}>gk_{\mathrm{plane}}\epsilon_L$：`NONPLANAR`；
- $k_{\mathrm{plane}}\epsilon_L<h_{\max}\le gk_{\mathrm{plane}}\epsilon_L$：`UNDETERMINED`。

返回 `PlanarityMeasurement`，不作芳香性或环合理性判断。

### 7.2 `determine_line_relation`

```python
def determine_line_relation(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> LineRelation
```

设 $\mathcal L_i=\mathbf o_i+t\mathbf d_i$，先用尺度安全方法得到单位方向
$\hat{\mathbf d}_i$，再计算：

$q_{\parallel}=\lVert\hat{\mathbf d}_1\times\hat{\mathbf d}_2\rVert$

平行方向的直线距离：

$d_{\parallel}=\lVert(\mathbf o_2-\mathbf o_1)\times\hat{\mathbf d}_1\rVert$

非平行方向的三维最短距离：

$d_{\mathrm{skew}}=\frac{|(\mathbf o_2-\mathbf o_1)\cdot(\hat{\mathbf d}_1\times\hat{\mathbf d}_2)|}{\lVert\hat{\mathbf d}_1\times\hat{\mathbf d}_2\rVert}$

令角度容差和本次距离容差为：

$\epsilon_\theta=\max(\epsilon_{\mathrm{param}},k_{\mathrm{mach}}\epsilon_{\mathrm{machine}})$

$\epsilon_d=\epsilon_{\mathrm{abs}}+\epsilon_r d$

实际分类边界为：

- 任一方向为零：`DEGENERATE`；
- $q_{\parallel}=0$ 且 $d_{\parallel}\le\epsilon_d$：`COINCIDENT`；
- $q_{\parallel}=0$ 且 $d_{\parallel}>g\epsilon_d$：`PARALLEL`；
- $0<q_{\parallel}\le g\epsilon_\theta$：`UNDETERMINED`；
- $q_{\parallel}>g\epsilon_\theta$ 且 $d_{\mathrm{skew}}\le\epsilon_d$：`INTERSECTING`；
- $q_{\parallel}>g\epsilon_\theta$ 且 $d_{\mathrm{skew}}>g\epsilon_d$：`SKEW`；
- 其余距离保护带或非有限输入：`UNDETERMINED`。

返回 `LineRelation`。

### 7.3 `line_distance`

```python
def line_distance(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float
```

是 `determine_line_relation()` 的标量投影：可确定时返回 $d_{\parallel}$ 或
$d_{\mathrm{skew}}$，退化或未决时返回 `NaN`。

### 7.4 `point_segment_distance`

```python
def point_segment_distance(
    point: Point,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float
```

对点 $\mathbf p$ 和线段 $S=[\mathbf a,\mathbf b]$：

$t=\operatorname{clip}_{[0,1]}\frac{(\mathbf p-\mathbf a)\cdot(\mathbf b-\mathbf a)}{\lVert\mathbf b-\mathbf a\rVert^2}$

$d(\mathbf p,S)=\lVert\mathbf p-[\mathbf a+t(\mathbf b-\mathbf a)]\rVert$

若线段长度不超过当前 $\epsilon_L$，函数返回 $\lVert\mathbf p-\mathbf a\rVert$；非有限输入返回
`NaN`。

### 7.5 `segment_segment_distance`

```python
def segment_segment_distance(
    first: Segment,
    second: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float
```

对 $S_1=[\mathbf a,\mathbf b]$、$S_2=[\mathbf c,\mathbf d]$：

$d(S_1,S_2)=\min_{s,t\in[0,1]}\lVert\mathbf a+s(\mathbf b-\mathbf a)-\mathbf c-t(\mathbf d-\mathbf c)\rVert$

退化线段按点—线段或点—点距离处理；非有限输入返回 `NaN`。

### 7.6 `point_pair_distances`

```python
def point_pair_distances(
    points: Sequence[Point],
) -> tuple[PointPairDistance, ...]
```

按稳定的 $i<j$ 顺序返回全部无序点对：

$d_{ij}=\lVert\mathbf p_i-\mathbf p_j\rVert$

共有 $n(n-1)/2$ 条记录。该函数不读取 `GeometrySettings`，非有限点对距离为 `NaN`。

### 7.7 `find_point_pairs_below_distance`

```python
def find_point_pairs_below_distance(
    points: Sequence[Point],
    threshold: float,
) -> tuple[PointPairDistance, ...]
```

先调用 `point_pair_distances()`，再保留严格满足 $d_{ij}<d_{\mathrm{caller}}$ 的记录。
`threshold` 完全由调用方解释，geometry 不把它视为化学阈值。

### 7.8 `locate_point_in_planar_cycle`

```python
def locate_point_in_planar_cycle(
    point: Point,
    cycle: Cycle,
    plane: Plane,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PointCycleLocation
```

在 `plane` 上建立正交基 $(\mathbf u,\mathbf v)$，将三维点投影为：

$\pi(\mathbf p)=((\mathbf p-\mathbf q)\cdot\mathbf u,(\mathbf p-\mathbf q)\cdot\mathbf v)$

实现先检查所有非相邻投影边是否存在稳定自交或数值未决，再计算点到边界的距离和 winding
number。该检查不是通用 simple-polygon 证明；调用方仍须保证环顶点按一条简单边界有序，相邻边
重叠、回折等输入不在该接口的完整验证范围内。

$d_{\partial\mathcal C}(\mathbf p)=\min_i d(\pi(\mathbf p),\pi(e_i))$

$w(\mathbf p)=\frac1{2\pi}\sum_i\operatorname{atan2}\left((\mathbf z_i\times\mathbf z_{i+1})_z,\mathbf z_i\cdot\mathbf z_{i+1}\right)$

其中 $\mathbf z_i=\pi(\mathbf v_i)-\pi(\mathbf p)$。

- $d_{\partial\mathcal C}\le\epsilon_L$：`BOUNDARY`；
- 边界距离位于保护带：`UNDETERMINED`；
- $||w|-1|\le\epsilon_{\mathrm{winding}}$：`INTERIOR`；
- $|w|\le\epsilon_{\mathrm{winding}}$：`EXTERIOR`；
- 其他：`UNDETERMINED`。

该接口判断的是点在给定平面上的投影位置；它不会验证 `point` 本身位于该平面，也不会验证
`plane` 是环的最佳拟合平面。数值尺度 $L$ 在投影前由三维 `point` 和环顶点共同计算，因此把
同一投影点沿平面法向远移可能改变派生容差。

### 7.9 线段—环共同数学模型

平面环使用 `PLANAR_POLYGON`。设线段 $S=[\mathbf s_0,\mathbf s_1]$，最佳拟合平面为
$\Pi(\mathbf c,\hat{\mathbf n})$：

$h_0=\hat{\mathbf n}\cdot(\mathbf s_0-\mathbf c)$

$h_1=\hat{\mathbf n}\cdot(\mathbf s_1-\mathbf c)$

当线段稳定横跨平面时：

$t=\frac{h_0}{h_0-h_1}$

$\mathbf x_\cap=\mathbf s_0+t(\mathbf s_1-\mathbf s_0)$

只有 $t$ 稳定位于 $(0,1)$ 且 $\mathbf x_\cap$ 稳定位于简单多边形内部，才确认
`TRANSVERSE_INTERIOR` 和 `PIERCES`。端点、环边、环顶点、共面接触以及仅无限延长线命中都不算
严格穿越，而是写入 `features`。

非平面环使用 `VERTEX_TRIANGULATION_FAMILY`。含 $m$ 个顶点的简单边界有：

$n_{\mathrm{surface}}=\operatorname{Cat}_{m-2}=\frac1{m-1}\binom{2m-4}{m-2}$

个组合顶点三角剖分。候选曲面只有在三角形拓扑正确、唯一边界等于原环且任意三角形对没有
组合允许部分以外的几何交集时，才计为 `EMBEDDED`。对三角形 $T_i,T_j$，嵌入条件为：

$f(T_i)\cap f(T_j)=f(T_i\cap T_j)$

三角形法向和交点重心坐标使用：

$\mathbf n_\triangle=(\mathbf b-\mathbf a)\times(\mathbf c-\mathbf a)$

$\lambda_a=\frac{((\mathbf b-\mathbf x_\cap)\times(\mathbf c-\mathbf x_\cap))\cdot\mathbf n_\triangle}{\lVert\mathbf n_\triangle\rVert^2}$

$\lambda_b=\frac{((\mathbf c-\mathbf x_\cap)\times(\mathbf a-\mathbf x_\cap))\cdot\mathbf n_\triangle}{\lVert\mathbf n_\triangle\rVert^2}$

$\lambda_c=1-\lambda_a-\lambda_b$

所有 $\lambda_i>g\epsilon_u$ 表示三角形严格内部；边界和保护带分别记录接触或未决。

令 $K$ 表示曲面构造/枚举以及本次所需线段评估均未因预算中断，并记：

- $n_{\mathrm{emb}}$：确认嵌入的候选曲面数；
- $n_{\mathrm{construct\_undetermined}}$：曲面构造未决数；
- $n_{\mathrm{hit}}$：线段确认穿过的嵌入曲面数；
- $n_{\mathrm{miss}}$：线段确认不穿的嵌入曲面数；
- $n_{\mathrm{eval\_undetermined}}$：线段求交未决的嵌入曲面数。

最终共识为：

$\mathrm{PIERCES}\iff K\land n_{\mathrm{construct\_undetermined}}=0\land n_{\mathrm{eval\_undetermined}}=0\land n_{\mathrm{emb}}>0\land n_{\mathrm{hit}}=n_{\mathrm{emb}}$

$\mathrm{DOES\_NOT\_PIERCE}\iff K\land n_{\mathrm{construct\_undetermined}}=0\land n_{\mathrm{eval\_undetermined}}=0\land n_{\mathrm{emb}}>0\land n_{\mathrm{miss}}=n_{\mathrm{emb}}$

其他情况返回 `UNDETERMINED`。结论只相对于该顶点三角剖分族，不宣称覆盖所有连续跨越曲面。

### 7.10 `iter_segment_cycle_relations`

```python
def iter_segment_cycle_relations(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[SegmentCycleRelation]
```

批量惰性接口。它只为 `cycle` 执行一次平面度测量或非平面候选曲面准备，再按输入顺序逐条产生
`SegmentCycleRelation`。每条结果与单独调用 `determine_segment_cycle_relation()` 等价；同一环有
多条候选键时优先使用本接口。

### 7.11 `determine_segment_cycle_relation`

```python
def determine_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> SegmentCycleRelation
```

单线段入口，等价于对只含一个 `segment` 的序列调用 `iter_segment_cycle_relations()` 并取第一项。
返回完整状态、接触特征、未决原因、曲面模型、交点、最近环边和枚举证据。

### 7.12 `closest_cycle_edge`

```python
def closest_cycle_edge(
    cycle: Cycle,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> ClosestCycleEdge | None
```

对每条环边计算：

$d_i=d(e_i,S)$

$d_{\min}=\min_i d_i$

$i^*=\min\{i\mid d_i\le d_{\min}+\epsilon_L\}$

容差内并列时取最小 edge index，保证稳定输出。输入非有限、局部尺度退化或没有有限距离时返回
`None`。

## 8. 化学对象转换 API

### 8.1 结构协议与泛型来源

转换层使用结构协议而非运行时导入 Core，因此既接受 Hotpot 原生对象，也接受具有相同最小属性的
自定义对象：

| 来源对象 | 最小结构要求 |
|---|---|
| Atom-like | `coordinates: Sequence[float]`、`idx: int` |
| Bond-like | `atom1`、`atom2`，两端均为 Atom-like |
| Ring-like | 有序 `atoms: Sequence[Atom-like]` |
| Structure-like | 可迭代 `atoms` 和 `bonds` |
| Molecule-like | Structure-like 要求及 `rings_for_scope(ring_scope)` |

泛型结果保留具体来源类型和原对象引用。转换会把当前坐标复制进不可变几何值对象，但不会复制、
冻结或修改来源化学对象。

### 8.2 `PairScope`

```python
PairScope = Literal["all", "bonded", "nonbonded"]
```

- `"all"`：所有无序原子对；
- `"bonded"`：仅显式成键原子对；
- `"nonbonded"`：仅无显式键的原子对。

### 8.3 `RingScope`

```python
RingScope = Literal["full_graph", "ligand_skeleton"]
```

实际选环行为由来源对象的 `rings_for_scope()` 实现。Hotpot 中 `full_graph` 使用完整分子图，
`ligand_skeleton` 用于忽略金属配位连接后观察配体骨架环。

### 8.4 `RingFamily`

| 成员 | `.value` | 含义 |
|---|---|---|
| `NETWORKX_CYCLE_BASIS` | `"networkx_cycle_basis"` | Core 使用 NetworkX cycle basis 选择环 |

`BondRingScanReport.ring_family` 当前固定写入这一成员。对于自定义 Molecule-like 对象，这是调用方
遵守 `rings_for_scope()` 协议的约定标签，不是转换层对其内部算法的动态识别结果。

### 8.5 `AtomGeometry`

```python
AtomGeometry[AtomT](source: AtomT, point: Point, key: int)
```

`source` 是原 Atom-like 对象，`point` 是坐标快照，`key=int(source.idx)`。

### 8.6 `AtomPairTarget`

```python
AtomPairTarget[AtomT](
    first: AtomGeometry[AtomT],
    second: AtomGeometry[AtomT],
    bonded: bool,
)
```

记录一个稳定顺序的原子对，以及来源图中是否存在显式键。

### 8.7 `BondGeometry`

```python
BondGeometry[BondT](
    source: BondT,
    segment: Segment,
    key: tuple[int, int],
)
```

bond key 为两个端点 atom key 的升序元组：

$k_{\mathrm{bond}}=\operatorname{sort}(k_{a_1},k_{a_2})$

### 8.8 `RingGeometry`

```python
RingGeometry[RingT](
    source: RingT,
    cycle: Cycle,
    key: tuple[int, ...],
)
```

ring key 是 atom key 正向和反向序列全部循环移位中字典序最小的元组，因此对起点和绕行方向不变。

### 8.9 `BondRingTarget`

```python
BondRingTarget[RingT, BondT](
    ring: RingGeometry[RingT],
    bond: BondGeometry[BondT],
)
```

表示一个待做线段—环判定的来源组合。

### 8.10 `AtomPairDistance`

```python
AtomPairDistance[AtomT](
    target: AtomPairTarget[AtomT],
    measurement: PointPairDistance,
)
```

把来源原子对和纯几何距离记录组合在一起。

### 8.11 `BondRingFinding`

```python
BondRingFinding[RingT, BondT](
    target: BondRingTarget[RingT, BondT],
    relation: SegmentCycleRelation,
)
```

把来源 Ring × Bond 对和完整线段—环关系组合在一起。

### 8.12 `RingEdgeDistance`

```python
RingEdgeDistance[BondT](
    source_bond: BondT,
    measurement: ClosestCycleEdge,
)
```

用于把来源环键与最近边度量组合。当前 package 没有公开函数直接构造该记录，调用方可在需要把
几何 edge index 映射回化学 Bond 时使用。

### 8.13 `BondRingScanReport`

```python
BondRingScanReport[RingT, BondT](
    findings: tuple[BondRingFinding[RingT, BondT], ...],
    ring_scope: RingScope,
    ring_family: RingFamily,
    max_ring_size: int,
    selected_ring_count: int,
    excluded_ring_count: int,
    candidate_pair_count: int,
    evaluated_pair_count: int,
    piercing_pair_count: int,
    does_not_pierce_pair_count: int,
    undetermined_pair_count: int,
    scan_complete: bool,
)
```

| 字段/属性 | 含义 |
|---|---|
| `findings` | 每个已评估候选对的完整记录 |
| `ring_scope` | 本次请求的环范围 |
| `ring_family` | 当前转换层写入的 Hotpot Core 环族标签；目前固定为 `NETWORKX_CYCLE_BASIS`，并非从自定义对象动态探测 |
| `max_ring_size` | 纳入扫描的最大环原子数 |
| `selected_ring_count` | 满足大小限制的环数 |
| `excluded_ring_count` | 因大小限制排除的环数 |
| `candidate_pair_count` | 排除环自身边后的 Ring × Bond 候选数 |
| `evaluated_pair_count` | 实际得到 relation 的候选数 |
| 三个 `*_pair_count` | 三种最终状态各自的数量 |
| `scan_complete` | 已选择范围内所有候选均评估，且曲面构造/枚举及所需线段评估均未因预算中断 |
| `piercings` | 只含 `PIERCES` 的只读派生 tuple |
| `undetermined` | 只含 `UNDETERMINED` 的只读派生 tuple |

`scan_complete=True` 不代表扫描覆盖全部可能环；必须同时检查 `ring_scope`、`ring_family`、
`max_ring_size` 和 `excluded_ring_count`。空选择也可能产生完整的空报告。

### 8.14 `point_from_atom`

```python
def point_from_atom(atom: AtomT) -> Point
```

读取 `atom.coordinates` 并返回不可变坐标快照；不修改 `atom`。

### 8.15 `segment_from_bond`

```python
def segment_from_bond(bond: BondT) -> Segment
```

把 `bond.atom1` 和 `bond.atom2` 的当前坐标转换为有限线段；不读取键级或化学类型。

### 8.16 `cycle_from_ring`

```python
def cycle_from_ring(ring: RingT) -> Cycle
```

按 `ring.atoms` 的现有顺序创建闭合 `Cycle`。输入顺序决定边界连接关系。

### 8.17 `iter_atom_geometries`

```python
def iter_atom_geometries(
    structure: _StructureLike[AtomT, BondT],
) -> Iterator[AtomGeometry[AtomT]]
```

按 `structure.atoms` 的来源顺序惰性产生 `AtomGeometry`。

### 8.18 `iter_atom_pair_targets`

```python
def iter_atom_pair_targets(
    structure: _StructureLike[AtomT, BondT],
    pair_scope: PairScope,
) -> Iterator[AtomPairTarget[AtomT]]
```

按原子来源顺序的组合 $i<j$ 产生原子对，并从 `structure.bonds` 判断 `bonded`。不支持的
`pair_scope` 抛出 `ValueError`。

### 8.19 `iter_ring_geometries`

```python
def iter_ring_geometries(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
) -> Iterator[RingGeometry[RingT]]
```

调用 `mol.rings_for_scope(ring_scope)`，保留 `len(ring.atoms) <= max_ring_size` 的环，并按
canonical ring key 排序后惰性返回。

### 8.20 `iter_bond_ring_targets`

```python
def iter_bond_ring_targets(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
) -> Iterator[BondRingTarget[RingT, BondT]]
```

对每个选中环和按 bond key 排序的化学键产生候选组合，但排除该环自身的边。与之不同，直接调用
`determine_bond_ring_relation(ring, bond)` 不执行该排除。

### 8.21 `measure_atom_pair_distances`

```python
def measure_atom_pair_distances(
    structure: _StructureLike[AtomT, BondT],
    pair_scope: PairScope,
) -> tuple[AtomPairDistance[AtomT], ...]
```

计算所选原子对的欧氏距离，同时保留来源 Atom 引用、atom key 和成键事实。返回稠密 tuple。

### 8.22 `determine_bond_ring_relation`

```python
def determine_bond_ring_relation(
    ring: RingT,
    bond: BondT,
    *,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingFinding[RingT, BondT]
```

单个化学对象适配器：把 `ring`、`bond` 转换为 `Cycle`、`Segment`，调用
`determine_segment_cycle_relation()`，并把来源和结果封装为 `BondRingFinding`。

### 8.23 `iter_bond_ring_findings`

```python
def iter_bond_ring_findings(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[BondRingFinding[RingT, BondT]]
```

按 canonical ring key 和 bond key 的顺序惰性产生结果。每个环调用一次
`iter_segment_cycle_relations()`，因此同一环的候选键共享平面度或曲面准备。

### 8.24 `scan_bond_ring_relations`

```python
def scan_bond_ring_relations(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScanReport[RingT, BondT]
```

消费完整惰性流并返回稠密报告。该接口适合诊断、审计和需要逐候选证据的业务逻辑。

### 8.25 `determine_bond_ring_piercing_state`

```python
def determine_bond_ring_piercing_state(
    mol: _MoleculeLike[AtomT, BondT, RingT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PiercingState
```

面向只需要聚合三态的快速入口：

1. 遇到第一个 `PIERCES` 立即返回；
2. 若没有穿越但至少一个候选为 `UNDETERMINED`，返回 `UNDETERMINED`；
3. 其余情况返回 `DOES_NOT_PIERCE`。

返回值只覆盖声明的 `ring_scope` 和 `max_ring_size`；该标量接口不携带
`excluded_ring_count`。需要审计覆盖范围时使用 `scan_bond_ring_relations()`。
