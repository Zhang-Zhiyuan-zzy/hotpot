# 环几何有效性与键—环关系统一评审

本文合并原 FF-Q004“非平面环穿越能否严格判定”和 FF-Q005“环自身质量与键—环互穿是否
应分离”。本文是该问题唯一的详细结论；讨论范围严格限定为纯几何，不调用力场、不使用真实
能量，也不依赖元素、杂化或经验构象数据库。

面向化学应用的公开结论统一只有三类：`REASONABLE`（明确合理）、`UNREASONABLE`（明确
不合理）和 `UNCERTAIN`（尚不能明确判定）。自交、接触、曲面分歧等名称只作为数学原因和
证据，不再扩张成一组平行的业务状态。

## 1. 当前现状

### 1.1 当前实现及调用语义

审查基线为分支 `fix/complexes-build-pipeline`、提交 `4f78048`。当前实现位于
`hotpot/cheminfo/geometry.py`：

| 位置 | 当前行为 | 主要问题 |
|---|---|---|
| `_segment_intersects_triangle()`，约 318-349 行 | Möller–Trumbore 浮点求交，直接返回 `bool` | 平行、共面、擦边和确定不相交没有结构化区分 |
| `_planar_polygon_normal()`，约 352-368 行 | 用绝对/尺度容差判定环是否共面 | 该布尔值会选择两套不同曲面语义 |
| `_line_intersects_planar_polygon()`，约 409-433 行 | 平面交点投影后执行 2D point-in-polygon | 能正确处理 simple concave polygon |
| `_line_intersects_polygon()`，约 436-476 行 | 平面环走真实边界；非平面环走 arithmetic-centroid fan | 外部中心产生虚构面积，且在平面阈值处不连续 |
| `bond_intersects_ring()`，约 1411-1430 行 | 共享环原子的键直接返回 `False`，其余返回一个 `bool` | `False` 同时承担不相交、非法输入和数值不可判定 |
| `find_bond_ring_intersections()` / `has_bond_ring_intersection()` | 只收集布尔真值 | 无法向质量门控传播接触、曲面分歧和不可判定原因 |

`tests/test_cheminfo/test_geometry.py:144-154` 的
`test_nonplanar_ring_keeps_center_fan_surface_semantics` 仍明确锁定历史 center-fan 行为。

### 1.2 已动态复现的缺陷

使用当前 `geometry.py`、Python 3.11.16 和 NumPy 2.4.6，构造 L 形凹六边界：

```python
ring = (
    (0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (3.0, 1.0, dz),
    (1.0, 1.0, 0.0),
    (1.0, 3.0, 0.0),
    (0.0, 3.0, 0.0),
)
probe = Line((1.2, 1.2, -2.0), (1.2, 1.2, 2.0))
```

`(1.2, 1.2)` 位于 L 形凹口外部，正确结果应为不穿环。当前结果为：

```text
dz=0       branch=planar  hit=False
dz=2e-8    branch=planar  hit=False
dz=5e-8    branch=fan     hit=True
dz=1e-7    branch=fan     hit=True
dz=1e-3    branch=fan     hit=True
dz=0.2     branch=fan     hit=True
```

只增加 `5e-8 Å` 的无意义坐标扰动就翻转结果。原因不是环发生了实际几何改变，而是程序从
真实凹多边形切换到 centroid fan；该 centroid 位于凹口外，额外三角形覆盖了本不属于环孔
的区域。严重程度为 **高**，因为误报会直接拒绝力场构筑候选。

当前非法输入还会在 `_line_intersects_polygon()` 中直接返回 `False`。这会把“无法判定”
静默解释为“确认没有穿环”，同样不适合作为质量门控。

### 1.3 两个问题应合并审议，但保持三层计算

“环自身几何”和“键—环关系”属于同一个几何质量问题，但不能写成两个相互替代的布尔函数。
正确的数据流是：

```text
ring boundary geometry
        ↓
ring surface ensemble
        ↓
bond–ring geometric relation
        ↓
forcefield/complex-builder policy
```

- 第一层只检查闭合折线自身的几何状态；
- 第二层回答能否根据该边界构造可审查的跨越曲面族；
- 第三层判断给定有限键段与整个曲面族的关系；
- 最后一层才决定接受、拒绝或重试。

环被标为 `UNCERTAIN` 时仍可继续收集键—环证据，但结果不能被升级为明确合理或明确不合理；
反之，一个环自身没有硬碰撞，也可能因为孔洞塌缩或跨越面不稳定而无法得到明确关系。任何
不可判定情况都不能转换为 `REASONABLE`。

### 1.4 必须承认的数学边界

一个非平面闭合空间折线只有一维边界，没有天然唯一的二维内部曲面。同一边界的不同合法
三角剖分可以产生不同三维跨越面。例如：

```text
p0=(-1,-1,0)  p1=( 1,-1,1)
p3=(-1, 1,1)  p2=( 1, 1,0)
```

采用对角线 `p0-p2` 时中心面高度为 `z=0`；采用 `p1-p3` 时中心面高度为 `z=1`。短键段
`[(0,0,-0.1), (0,0,0.1)]` 只穿过前一个面。这不是浮点误差，而是曲面选择本身会影响答案。

因此：

- 对指定且非退化的曲面，segment–triangle intersection 可以明确判断；
- 对非平面环边界本身，只有先声明曲面模型或对一族曲面取共识，才能给出结论；
- 单个 Earcut 面是可复现的工程约定，不是唯一几何事实；
- 形式精确 predicates 只能消除数值舍入不确定性，不能替项目选择环面语义。

### 1.5 已验证的第三方库结论

本地已隔离安装并测试 `mapbox-earcut==2.1.0`：

- 对上述 L 形六边形生成 `n-2=4` 个三角形；
- 三角形面积和与多边形面积均为 `5.0`；
- 凹口外 probe 不被覆盖，真实内部 probe 被覆盖；
- 本机 200,000 次六边形三角化约 `0.530 s`，约 `2.65 µs/call`。

但实际测试也确认，仅改变同一边界的循环起点，Earcut 可以选择另一组同样合法的对角线。
它适合作为快速单面生成器或测试参照，不足以独自承担非平面环的最终共识判定。

## 2. 环几何状态分级

### 2.1 纯几何输入与尺度

把 `n` 元环表示为有序闭合空间折线：

\[
\Gamma=(\mathbf p_0,\mathbf p_1,\ldots,\mathbf p_{n-1},\mathbf p_0),
\qquad e_i=[\mathbf p_i,\mathbf p_{i+1\bmod n}].
\]

用环自身中位边长定义尺度：

\[
s=\operatorname{median}(\|e_i\|),
\qquad \epsilon=\epsilon_{abs}+\epsilon_{rel}s.
\]

除原始距离外，状态判断尽量使用以 `s` 归一化的无量纲量，保证整体平移、旋转和均匀缩放
不会改变分类。不调用力场，不引入真实能量，也不使用元素专属参考参数。

### 2.2 必须计算的几何证据

1. **输入与边界退化**
   - 坐标是否有限；
   - 是否至少有 3 个有效顶点；
   - 是否存在重复连续顶点、零长边或重复边。
2. **非相邻边中心线距离**

   \[
   d_{ij}=\min_{u,v\in[0,1]}
   \|(\mathbf p_i+u\mathbf e_i)-(\mathbf p_j+v\mathbf e_j)\|.
   \]

   `d_ij <= epsilon` 表示确定自交或数值上无法与擦过区分。
3. **有限厚度的几何间隙**
   - 非相邻边—边最小距离 `q_ee=d_ee/s`；
   - 顶点—非关联边最小距离 `q_pe=d_pe/s`；
   - 非局部顶点—顶点最小距离 `q_pp=d_pp/s`。

   这些量等价于检查以 `alpha*s` 为半径的几何 tube/capsule 是否自碰撞；`alpha` 是公开的
   纯几何门控参数，不称为范德华或物理半径。
4. **局部曲率和全局厚度**

   \[
   \tau=\min\left(\min_iR_{\mathrm{curv},i},\frac12d_{self}\right),
   \qquad q_\tau=\tau/s.
   \]

   该 polygonal thickness/reach 同时描述局部急折和远端环段贴近。
5. **塌缩与起伏**
   - 中心化坐标 SVD 的 `sigma_2/sigma_1`：是否接近线状塌缩；
   - `sigma_3/sigma_2`：非平面程度，只作描述，不单独判失败；
   - best-fit 投影的 `4*pi*A/P^2`：投影孔洞是否塌缩；
   - Cremer–Pople 或一般离散 Fourier puckering modes：描述起伏类型。

非平面、明显 puckering 或非零扭转本身均不是失败条件。

### 2.3 环结构的三态结论

环报告使用全局统一的 `GeometryVerdict`：

| 公开结论 | 环结构的明确含义 | 后续力场行为 |
|---|---|---|
| `REASONABLE` | 没有硬几何冲突，且全部关键裕量明确远离阈值 | 可作为正常候选继续判断和优化 |
| `UNREASONABLE` | 确认存在零长/重复边、非相邻中心线自交、严重几何自碰撞或明确塌缩 | 拒绝当前候选并重新构筑 |
| `UNCERTAIN` | 没有证据确认不合理，但某项落入容差/警戒区，或环面无法稳定构造 | 保留并排序；必要时优化后重新评估 |

几何模块不再公开 `VALID/SUSPECT/INVALID/UNRESOLVED` 四套平行状态。相关数学差异保存在
`RingGeometryReason` 中，例如：

```text
NONFINITE_COORDINATES
DEGENERATE_EDGE
SELF_INTERSECTION
LOW_EDGE_CLEARANCE
LOW_POLYGONAL_THICKNESS
COLLAPSED_APERTURE
NUMERICAL_BOUNDARY
SURFACE_UNSTABLE
SURFACE_UNDEFINED
```

每个 reason 必须携带涉及顶点/边索引、原始距离、归一化距离、阈值和判定裕量。这样力场只
消费三态，诊断和测试仍能看到具体数学原因。

“自打结”在本轮按可直接检测的中心线自交、非相邻环段近接触和孔洞塌缩处理。非平凡 knot
的拓扑分类不属于首轮质量门控，避免把罕见拓扑结构直接等同于几何错误。

### 2.4 可选合理度排序

硬状态不依赖任何能量。如果候选构象需要连续排序，可选计算：

\[
P_{geo}=w_c[\delta_c-q_{clear}]_+^2+
w_\tau[\delta_\tau-q_\tau]_+^2+
w_l\sum_i[|\ell_i/s-1|-\delta_l]_+^2+
w_a[\delta_a-4\pi A/P^2]_+^2.
\]

`P_geo` 仅由坐标、拓扑和无量纲门槛组成，复杂度为 `O(n^2)`。可用单调映射，例如
`reasonableness_score = exp(-P_geo)`，对 `UNCERTAIN` 候选排序。

未经有标签样本校准前，该值只能称为 `reasonableness_score`，不能声称是统计概率。后续若
用已知合理/不合理结构校准，才可另外提供 `reasonable_probability`。无论是否有评分，明确
合理和明确不合理的硬判定优先级都高于该分数。

## 3. 环—键几何状态分级

### 3.1 判定对象和有限键段语义

`BondRingPairReport` **只描述一个确定的 ring–bond pair**：一个环和一条目标键，不代表整个
分子。分子级判断由 `MoleculeGeometryReport` 聚合全部环报告和全部有效 ring–bond pair
报告。

目标键严格定义为两个原子坐标之间的有限闭线段：

\[
\mathbf b(t)=\mathbf a+t(\mathbf c-\mathbf a),\qquad 0\le t\le1.
\]

只有交点满足 `epsilon_t < t < 1-epsilon_t`，即位于两个键端点之间且不落入端点容差区，才
可能构成穿环。无限直线延长部分的交点 `t < 0` 或 `t > 1` 必须判为无关，绝不能视为
`PIERCES`。原文“键在端点之外穿过”的表述错误，现明确改为“键段内部穿过”。

属于该环或与环共享原子的键沿用现有业务规则，不进入 ring–bond pair 判定；批量报告记录
`skipped_reason=shared_ring_atom`，而不是把它伪装成一次经过计算的合理结论。

每个有效三角面上的数学事件先分为：

- `NO_HIT`：有限键段没有交点；
- `TRANSVERSE_INTERIOR_HIT`：键段内部横向穿过三角形严格内部；
- `BOUNDARY_CONTACT`：目标键段命中三角形边或顶点；
- `COPLANAR_CONTACT`：目标键段与三角面共面接触或重叠；
- `NUMERICALLY_UNRESOLVED`：交角、参数或重心坐标落入数值容差带。

明确横穿要求 segment parameter 严格位于 `(epsilon, 1-epsilon)`，重心坐标严格位于三角形
内部，且键方向与三角面法向的夹角不落入近平行容差带。内部对角线上的重复交点必须按空间
位置去重。

### 3.2 环—键对的三态结论

对一个 ring–bond pair 的全部有效环面聚合后，仍只返回 `GeometryVerdict`：

| 公开结论 | ring–bond pair 的明确含义 |
|---|---|
| `REASONABLE` | 环自身明确合理，且所有有效环面都确认有限键段没有命中，并与环面边界保持明确裕量 |
| `UNREASONABLE` | 环自身可判定，且所有有效环面都确认有限键段内部横向穿过环面内部 |
| `UNCERTAIN` | 曲面间结论不一致、只发生边/顶点/共面接触、落入数值容差，或环面不能稳定定义 |

此前的 `CLEAR/PIERCES/TOUCHES/AMBIGUOUS_SURFACE/UNDEFINED_RING` 不再作为五个公开状态。
它们被收束为 `BondRingRelationDetail` 中的证据代码：

```text
ALL_SURFACES_CLEAR
ALL_SURFACES_PIERCED
BOUNDARY_CONTACT
COPLANAR_CONTACT
SURFACE_DISAGREEMENT
SURFACE_UNSTABLE
SURFACE_UNDEFINED
NUMERICAL_BOUNDARY
```

原文“没有任何明确横穿，但至少一个有效环面出现擦边、顶点或共面接触”的准确含义是：
**目标有限键段**与至少一个候选环面的三角形边、顶点或三角面发生接触，但没有得到对全部
有效环面都成立的、位于键段内部和环面内部的横向穿越证据。该情形现在统一归入
`UNCERTAIN`，而不是另立一个业务状态。

如果环自身已经 `UNREASONABLE`，无需再用 ring–bond pair 结论重复判坏；该 pair 标记为
`not_evaluated`，分子级结果直接继承环的不合理结论。如果环自身为 `UNCERTAIN`，pair 不能
升级为明确合理或明确不合理。

### 3.3 分子级聚合与力场参考线

`MoleculeGeometryReport` 对所有环和 ring–bond pair 采用保守聚合：

```text
任一组件 UNREASONABLE -> molecule UNREASONABLE
否则任一组件 UNCERTAIN -> molecule UNCERTAIN
全部组件 REASONABLE    -> molecule REASONABLE
```

后续力场流程获得清晰参考线：

| 分子结论 | 默认行为 |
|---|---|
| `REASONABLE` | 作为正常候选进入优化 |
| `UNREASONABLE` | 拒绝并重新构筑，不浪费优化资源 |
| `UNCERTAIN` | 按 `reasonableness_score` 排序，优先优化最高分候选，优化后再次执行同一几何判定 |

分子级不把多个未经校准的分数相乘成伪概率；首轮可取最差组件分数作为保守排序值。只有经过
标注数据校准后，才将其命名为 `reasonable_probability`。

## 4. 具体实施方案

### 4.1 数据类型和公开接口

在 `geometry.py` 顶部的数据类型区增加明确类型，不使用 `Any`：

```python
class GeometryVerdict(Enum):
    REASONABLE = "reasonable"
    UNREASONABLE = "unreasonable"
    UNCERTAIN = "uncertain"
```

内部的 `RingGeometryReason` 和 `BondRingRelationDetail` 只保存原因，不作为第四、第五种公开
结论。配套不可变 dataclass：

```text
RingGeometryThresholds
RingGeometryMetrics
RingGeometryIssue
RingGeometryReport
RingSurface
RingSurfaceEnsemble
BondRingPairReport
MoleculeGeometryReport
```

候选公开接口放在 `geometry.py` 底部并加入 `__all__`：

```python
assess_ring_geometry(ring: Ring, *, thresholds: RingGeometryThresholds) -> RingGeometryReport
assess_bond_ring_pair(
    ring: Ring,
    bond: Bond,
    *,
    thresholds: RingGeometryThresholds,
) -> BondRingPairReport
assess_molecule_geometry(
    mol: Molecule,
    *,
    ring_scope: RingScope = "full_graph",
    max_ring_size: int = 8,
) -> MoleculeGeometryReport
```

`Molecule`、`Ring` 和 `Bond` 使用 `TYPE_CHECKING` 或既有无循环导入方式声明，禁止为方便继续
使用 `Any`。

### 4.2 环自身检查算法

`assess_ring_geometry()` 依次执行：

1. 提取按拓扑顺序排列的环顶点并计算尺度 `s`；
2. 检查有限坐标、顶点数、重复顶点和零长边；
3. 复用并收紧现有 `_point_segment_distance()`、`_segment_distance()` 计算所有非相邻边及
   point–segment 距离，避免建立第二套距离 kernel；
4. 计算非局部 point–point、point–segment 和 segment–segment clearance；
5. 计算局部曲率半径、polygonal thickness、SVD 比率及投影孔洞指标；
6. 根据硬阈值、警戒阈值和数值容差生成三态 verdict 与逐项 reason；
7. 可选计算 `reasonableness_score`，但不覆盖明确 verdict。

时间复杂度 `O(n^2)`；在 `n <= 8` 下可忽略。

### 4.3 环面族构造算法

首轮实施采用统一路径，不再按“平面/非平面”切换两套覆盖语义：

1. 用 SVD best-fit plane 只建立二维参数化坐标系；
2. 验证投影边界为非退化 simple polygon；
3. 递归枚举所有位于该边界内部的合法非交叉三角剖分；
4. 将三角形索引提升回原始三维环顶点；
5. 验证每个候选恰有 `n-2` 个三角形、索引合法、三角形非退化、二维面积守恒；
6. 检查提升后的非邻接三角形是否自交，删除非嵌入曲面；
7. 规范化三角形/曲面排序，使循环移位、遍历反向和原子重编号不改变曲面集合；
8. 若投影自交、近零面积或不存在有效面，记录 `SURFACE_UNSTABLE/SURFACE_UNDEFINED`，
   最终 verdict 为 `UNCERTAIN`，禁止回退到 centroid fan。

凸八边形最多有 Catalan 数 `C_6=132` 个三角剖分，每个曲面只有 6 个三角形。最坏情况下
每个 bond–ring pair 仅需约 792 次小型 triangle test；在 AABB broad phase 后不是性能瓶颈。

首轮不在投影失败后静默枚举任意抽象三角盘，因为这会引入尚未批准的另一套语义。未来若需
支持高度折叠但无 simple projection 的环，应作为显式第二阶段扩展，并仍返回独立证据。

### 4.4 键—环关系算法

1. 使用 ring/bond AABB 排除明显分离的候选；
2. 对每个有效曲面的三角形执行 finite segment–triangle test；
3. 将现有布尔 kernel 改为能区分 `NO_HIT`、`TRANSVERSE_INTERIOR_HIT`、边界接触、共面
   接触和数值不确定的结构化私有结果；
4. 对内部对角线、公共顶点和容差内同一点的重复命中去重；
5. 汇总单曲面的交点计数、横穿方向和最小边界裕量；
6. 按第 3.2 节规则聚合整个 surface ensemble；
7. 将数学事件映射为统一三态，并把 surface disagreement/undefined 原因完整传递到报告。

现有 `bond_intersects_ring()` 和 `has_bond_ring_intersection()` 若继续保留为便利布尔接口，只能
在确定状态下压缩结果；`REASONABLE` 返回 `False`，`UNREASONABLE` 返回 `True`，遇到
`UNCERTAIN` 必须抛出明确异常，不能返回 `False`，也不能做 try/except fallback。

### 4.5 库选择

- **NumPy**：继续承担 SVD、向量运算和小规模距离计算；
- **自有穷举器**：`n <= 8` 时最多 132 个剖分，递归 ear decomposition 足够清晰且可测试；
- **mapbox-earcut 2.1.0**：支持 Python >=3.9、仅依赖 NumPy、采用 ISC 许可；保留为单面
  生成/测试 oracle，不作为最终 ensemble 裁决器；
- **CGAL exact predicates**：只有形式精确性成为硬需求时再作为独立编译后端评估；其构建和
  Python 3.9-3.14 分发成本不适合作为当前默认依赖；
- Shapely、Trimesh、SciPy Delaunay 均不能同时解决支持矩阵、依赖重量和曲面语义问题，当前
  不引入。

### 4.6 分步迁移

1. 增加统一三态 enum、reason/detail、dataclass、异常和尺度相关阈值对象，不改变现有调用结果；
2. 实现环自身纯几何检查及其独立测试；
3. 实现全部合法三角剖分枚举、三维提升和 surface validation；
4. 实现结构化 segment–triangle 结果和 ensemble 共识聚合；
5. 将 `find_bond_ring_intersections()`、综合 geometry quality gate 和 forcefield 调用者迁移到
   结构化报告；
6. 删除 center-fan 分支及锁定 historical 行为的测试；
7. 最后决定是否保留严格抛错的布尔便利接口，不保留旧静默语义。

每一步独立提交，先建立测试围栏再切换生产调用。

### 4.7 必须通过的测试矩阵

1. 当前 L 形反例在 `dz=0`、`2e-8`、`5e-8`、`1e-3` 下保持相同关系；
2. 平面凸/凹环、轻微 puckered ring、chair ring 和显著折叠环；
3. 本文四顶点双对角线反例必须得到 `UNCERTAIN`，detail 为 `SURFACE_DISAGREEMENT`；
4. 零长边、重复顶点、近共线、中心线自交和低 clearance 环状态；
5. 明确横穿、明确分离、擦边、顶点命中、共面重叠和共享端点；
6. 循环起点、遍历反向、原子重编号、整体平移、任意刚体旋转和均匀缩放不变性；
7. `n=3..8` 的剖分计数、无重复曲面、面积守恒和三维自交过滤；
8. 容差两侧及随机微扰的稳定性；
9. `full_graph` 与 `ligand_skeleton` ring scope；
10. ring–bond pair 与 molecule 两个作用域明确，public assessor、geometry quality gate 和
    forcefield 调用结果一致；
11. 性能回归：记录每个环、每个 bond–ring pair 和整分子的时间，不以单一微基准替代集成测量。

验收标准是：业务层只出现三个 verdict；不存在非法输入静默返回 `REASONABLE`；不存在
center-fan fallback；所有 `UNCERTAIN` 均携带原因和排序分数；所有几何分类对表示顺序和
刚体变换保持不变；有限直线延长部分的交点绝不判为键段穿环。
