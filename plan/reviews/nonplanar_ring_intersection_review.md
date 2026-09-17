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
4. 使用尊重凹边界的确定性 ear-clipping triangulation；禁止引入算术中心作为新顶点；
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
