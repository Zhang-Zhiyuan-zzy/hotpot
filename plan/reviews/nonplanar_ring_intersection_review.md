# 非平面环穿越判定问题与整改设计

## 1. 当前语义

`geometry._line_intersects_polygon()`（当前 `geometry.py:436-476`）把环分成两类：

1. `_planar_polygon_normal()` 判定为平面环时，将交点和环投影到二维，用真正的多边形边界
   执行 point-in-polygon；该路径能够正确处理凹多边形。
2. 判定为非平面环时，计算全部顶点的算术平均坐标 `center`，再用
   `(center, vertex[i], vertex[i+1])` 建立三角扇面；线段与任意一个三角形相交即判为穿环。

第二条路径由注释明确描述为 “Preserve the historical center-fan surface”。
`tests/test_cheminfo/test_geometry.py:144-154` 也以
`test_nonplanar_ring_keeps_center_fan_surface_semantics` 锁定旧行为。

## 2. 历史 center-fan 的问题

### 2.1 顶点均值不保证位于环内部

凸环的顶点均值通常位于内部，中心扇面可以形成合理的铺面；凹环、折叠大环和某些螯合环
则不满足这一条件。中心一旦落在环外部，连接中心与各条边会生成跨越真实环边界的虚构
三角形。当前实现使用 `any(...)`，重叠三角形不会以方向或奇偶规则相互抵消，因此这些
额外区域都会被当成环内部。

结果是：一根实际从环凹口/外部空间经过的键会被报告为穿环，随后络合物候选会在
`_build_ligand_proxies()` 或 `evaluate_geometry_quality()` 中被拒绝。

### 2.2 平面与非平面路径在阈值处不连续

两条路径代表不同的几何区域：平面路径使用真实凹多边形，非平面路径使用中心扇面。极小的
坐标噪声只要越过 `_planar_polygon_normal()` 的阈值，就会切换算法并改变布尔结果。

这意味着：

- 同一化学构象在文件格式舍入、后端坐标精度或刚体变换误差后可能得到相反判断；
- `tolerance` 不再只是数值容差，而会间接选择两套不同拓扑语义；
- 候选接受/拒绝可能依赖不可见的浮点尾数，降低可复现性。

### 2.3 非平面闭合边界不存在唯一“内部平面”

对于 puckered ring，只有有序闭合边界，没有数学上唯一的内部曲面。center-fan 实际选择了
一种曲面，却没有在 API 或报告中声明这种选择，也没有判断中心是否位于合理投影内部。
因此它会把一个任意建模选择伪装成确定的化学事实。

### 2.4 对凹环和大环风险最高

当前质量门控默认只检查不超过 8 元的环，但 5-8 元非平面环、桥环和螯合环仍可能出现
凹投影。问题不是只存在于极端宏环；金属络合物构筑中的扭曲配体恰好可能触发这一分支。

## 3. 已动态复现的显著缺陷

使用当前 `geometry.py`、Python 3.11.16 和 NumPy 2.4.6，构造以下按边界顺序排列的凹六边形：

```python
planar = (
    (0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (3.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 3.0, 0.0),
    (0.0, 3.0, 0.0),
)
probe = Line((1.2, 1.2, -2.0), (1.2, 1.2, 2.0))
```

`(1.2, 1.2)` 位于 L 形多边形的凹口外部，正确结果是不相交。只将第三个顶点改为
`(3.0, 1.0, 5e-8)`，其余拓扑和坐标保持不变，当前实现得到：

```text
dz=0       max_plane_distance=0.0e+00  branch=planar  hit=False
dz=2e-08   max_plane_distance=1.4e-08  branch=planar  hit=False
dz=5e-08   max_plane_distance=3.5e-08  branch=fan     hit=True
dz=1e-07   max_plane_distance=7.0e-08  branch=fan     hit=True
dz=1e-03   max_plane_distance=7.0e-04  branch=fan     hit=True
dz=2e-01   max_plane_distance=1.399e-01 branch=fan    hit=True
```

本例尺度为 3 Å，平面阈值为 `3e-8 Å`。也就是说，只需 `5e-8 Å` 的无化学意义扰动，结果
就从 `False` 翻转为 `True`。原因是顶点均值约为 `(1.3333, 1.3333, z)`，本身落在 L 形
凹口外部；从这个“外部中心”建立的扇面把 `(1.2, 1.2)` 错误覆盖。

严重程度评为 **高**：它会把本应合格的构象误判为 bond-ring intersection，并进入候选
拒绝与重试流程；候选不足时可进一步导致整个络合物构筑失败。

## 4. 推荐的新语义

不再区分“平面多边形算法”和“非平面中心扇面算法”。对全部环采用同一套有明确边界的
分片线性曲面：

1. 按环的拓扑顺序取得顶点，拒绝重复边、非闭合或少于 3 个有效顶点的输入；
2. 通过 SVD/PCA 建立 best-fit plane，只将它用作二维参数化坐标系；
3. 将有序边界投影到该二维平面，检查投影边界是否为 simple polygon；
4. 使用尊重凹边界的确定性 constrained triangulation；推荐调用
   `mapbox-earcut`，禁止引入算术中心作为新顶点；
5. 将每个二维三角形的顶点索引映射回原始三维坐标，得到共享原环边界的 piecewise-linear
   surface；
6. 使用现有 `_segment_intersects_triangle()` 检测有限键段与这些三维三角形；
7. ear 候选相同时按稳定 atom index 决定，保证循环起点变化和环顺序反转不改变结论；
8. 若 best-fit 投影退化或自交，不返回“没有穿环”；应产生明确的
   `undefined_ring_surface` 几何检查失败，使质量门控 fail closed。

该定义承认非平面环的内部曲面需要约定，但约定满足三个关键性质：

- 三角形只使用真实环顶点，曲面边界严格等于环边界；
- 凹环不会因外部 centroid 产生虚构面积；
- 平面到轻微非平面的变化使用同一算法，不会在 planarity threshold 处切换语义。

不建议直接使用无约束 Delaunay triangulation，因为它可能跨越凹多边形外部；也不建议仅
增大平面 tolerance，这只能移动结果翻转阈值，不能消除 center-fan 的错误区域。

## 5. 接口调整建议

- 新增私有 `_triangulate_ring_surface(points, atom_indices, tolerance)`，返回仅由现有顶点
  构成的三角形索引；
- `_line_intersects_polygon()` 只负责调用统一 triangulation 和 segment-triangle kernel；
- `CyclePlanes.is_line_intersect_the_cycle()` 与 `bond_intersects_ring()` 继续共享同一底层
  kernel，避免对象式接口和分子接口再次分叉；
- 增加一个明确的 ring-surface failure 类型或结构化检查结果，让综合质量门控区分
  “确认未相交”和“无法定义可靠曲面”；
- 删除代码与测试中的 `historical` / `keeps` 命名，测试改为描述当前数学语义。

上述修改不改变“键若与环共享端点则不算穿环”的现有规则。

## 6. 必须新增的测试

1. 本报告的 L 形凹环在 `dz=0`、`2e-8`、`5e-8`、`1e-3` 下均判定 probe 不穿环；
2. 穿过 L 形真实内部区域的 probe 在相同扰动序列下均判定穿环；
3. 凸平面环、凹平面环、轻微 puckered ring 和典型 chair 六元环；
4. 环顶点循环移位、顺序反转、整体平移和任意刚体旋转的不变性；
5. probe 穿过顶点/边界、与环共享原子、与环面近平行等容差边界；
6. best-fit 投影自交、顶点重复和近共线退化时显式报告 undefined，而不是返回 False；
7. `find_bond_ring_intersections()`、`has_bond_ring_intersection()` 和综合质量门控对同一输入
   给出一致结论；
8. 真实配位络合物中的 5-8 元螯合环回归，确认不会因配位造成的合理 puckering 被误拒绝。

默认最多检查 8 元环，ear clipping 的 `O(n²)` 成本相对于 Open Babel 构筑和优化可以忽略。

## 7. 能否给出确定的“是否穿环”

答案分成两个层次：

1. **对合法、非退化输入，并且先明确环面定义后，可以给出确定且可复现的布尔结论。**
   本方案将“环面”定义为：把有序环边界投影到 best-fit plane，在二维边界内完成约束三角化，
   再把三角形索引映射回原始三维环顶点所形成的分片线性曲面。键段与该曲面相交即为穿环。
   调用三角化库前必须把循环起点和遍历方向规范化；固定该规则后，同一输入只会得到一个
   结果。
2. **对任意非平面闭合边界，不存在脱离约定的、天然唯一的物理“内部曲面”。** 同一条翘曲的
   空间闭合折线可以张成多个不同曲面，一根线段可能穿过其中一个而不穿过另一个。任何算法，
   包括 CGAL，都必须先接受某种 spanning-surface 定义，才能回答“穿过哪个面”。因此本方案
   给出的是严格遵守 Hotpot 已声明几何语义的确定判定，不应表述为测得了唯一的化学事实。

对以下边界情况，不应伪造普通 `True/False`：投影自交、重复或近重合顶点、面积近零、键段
与三角面共面重叠、以及只擦过边/顶点且落入数值容差带。底层建议返回三态结果：

```text
CLEAR | INTERSECTS | UNDEFINED
```

其中 `UNDEFINED` 携带原因。络合物质量门控应将其视为不通过（fail closed），但不能把它记录
成“已经证明穿环”。现有布尔便利接口可在收到 `UNDEFINED` 时抛出明确的几何定义异常；不应
静默返回 `False`，也不应回退到历史 center-fan。

## 8. 严格性和成熟度

需要区分三种“严格”：

| 层级 | 本方案能够达到的程度 | 限制 |
|---|---|---|
| 语义确定性 | 严格 | 必须先接受上述分片线性环面定义 |
| 正常浮点输入的工程判定 | 成熟且确定 | 需使用尺度相关容差并显式处理边界事件 |
| 计算几何的形式精确性 | 默认不达到 | NumPy、Earcut 和现有线段-三角形 kernel 都使用浮点数，不是 exact predicates |
| 化学结构真实性 | 不是该判定能够证明的内容 | 输入坐标本身包含构象、力场和文件精度误差 |

Ear clipping / constrained polygon triangulation 和有限线段-三角形相交都是成熟的计算几何
方法。这里不成熟的是当前 Hotpot 的 **外部 centroid fan + 两套分支语义**，不是三角化方法
本身。不过，“成熟方法”不等于对任意非法多边形具有数学保证；输入合法性检查和输出后验
检查仍是 Hotpot 的责任。

如果项目要求对已经存储的二进制浮点坐标执行形式精确的 orientation/intersection predicates，
可以使用 CGAL 的 exact-predicate/exact-construction kernel。但这会引入 C++ 扩展、复杂构建
和明显更重的分发成本，而且仍不能替代 Hotpot 对非平面环面语义的选择。对于 3-8 元环，
坐标化学误差远大于机器舍入误差，默认采用经过验证的浮点方案更合适。

## 9. 可复用的高性能几何库

截至 2026-09-17，候选对比如下：

| 库 | 能解决的部分 | Python 范围/集成代价 | 判断 |
|---|---|---|---|
| [`mapbox-earcut`](https://github.com/skogler/mapbox_earcut_python) 2.1.0 | C++ 实现的快速二维凹多边形 triangulation | `Python >=3.9`，仅依赖 NumPy，ISC | **推荐** |
| [CGAL](https://www.cgal.org/) | exact predicates、约束三角化和网格相交 | 需额外 C++ 绑定和复杂打包 | 只在形式严格性成为硬需求时采用 |
| Shapely/GEOS | 二维合法性检查和约束三角化 | 当前版要求 Python >=3.10，依赖较重 | 与 3.9 支持目标冲突 |
| Trimesh | 网格构造和 ray/mesh 相交封装 | 当前版要求 Python >=3.10，并间接调用 Earcut/Triangle | 对最多 8 个顶点明显过重 |
| `triangle` | Shewchuk Triangle 的二维约束三角化 | LGPL；所查发行版没有完整 3.14 wheel | 分发和许可成本高于收益 |
| SciPy Delaunay | 无约束 Delaunay | 不能保证保留凹多边形边界 | 不适用 |

没有一个库能在不知道 Hotpot 环面定义的前提下，直接接收非平面环和键并给出具有唯一化学
含义的答案。建议组合是：

```text
NumPy best-fit projection
  -> Hotpot validates a simple ordered boundary
  -> mapbox-earcut triangulates the 2D boundary
  -> Hotpot verifies triangulation invariants
  -> existing finite segment/triangle kernel tests original 3D triangles
```

必须验证的三角化后置条件包括：单个无孔环得到恰好 `n-2` 个三角形、索引合法、每个三角形
非退化、三角形二维面积总和与多边形面积在尺度相关容差内一致。Earcut 上游明确以速度和正常
输入鲁棒性为目标，不保证任意无效输入都产生正确网格，因此不能省略这些门控。

还必须在调用前规范化环的循环起点和遍历方向。实际测试表明，同一个 L 形边界仅改变列表的
循环起点，原始 Earcut 调用即可选择另一组同样合法的二维对角线。二维覆盖没有变化，但映射
到非平面三维顶点后，两组三角面可能不同。库本身不负责 Hotpot 所需的“环表示无关性”；
该性质必须由规范化及旋转/反向回归测试保证。若未来要求连原子重编号也不能影响结果，则需
进一步定义与 atom index 无关的几何选面准则；不能用“稳定 atom index”冒充化学不变量。

本次已在隔离目录实际安装并运行 `mapbox-earcut==2.1.0`。对第 3 节的 L 形凹六边形，其输出
4 个三角形，多边形面积和三角形面积总和均为 `5.0`；位于凹口外的 `(1.2, 1.2)` 不被任何
三角形覆盖，位于真实内部的 `(0.5, 1.2)` 被覆盖。该结果证明它能修复报告中的具体
center-fan 缺陷。在当前环境对该六边形执行 200,000 次三角化约耗时 `0.530 s`，即约
`2.65 µs/call`；对最多 8 元环而言不会成为力场流程瓶颈。该数字只是本机微基准，不是跨机器
性能承诺。完整三维实现仍需按第 6 节加入不变性、退化和容差测试。

### 最终选择

采用 **`mapbox-earcut` + Hotpot 前置合法性检查 + 三角化后验验证 + 三态内部结果**。这是当前
Python 3.9-3.14 支持目标下，确定性、成熟度、性能、依赖体积和可维护性最均衡的路线。
不采用静默备用算法；依赖缺失应在安装或导入时明确失败，避免不同机器得到不同几何语义。
