# 环自身质量与键—环互穿的数学拆分

## 1. 结论

将问题拆成“环自身”和“环与外部键的关系”是成立且必要的。不过，不建议直接实现为：

```text
is_ring_physically_real(ring) -> bool
bond_intersects_ring(bond, ring) -> bool
```

更准确的分层是：

```text
I. 环边界的硬几何有效性
II. 环形状的连续几何质量
III. 环孔或跨越面是否可定义
IV. 键与环孔/跨越面的空间关系
V. 业务策略：这种拓扑关系是否应被拒绝
```

其中 I 和 III 是 IV 的计算前提；II 是连续诊断或候选排序信息。一个形状异常但边界清楚的环
仍然可以执行穿环判断，一个局部距离正常但高度折叠、无法稳定定义环孔的环反而可能无法
得到唯一穿环结论。

同样，“检测到穿环”不天然等于“不物理”。轮烷、索烃、分子结和其他机械互锁结构可能正是
目标产物。几何模块应报告事实和不确定性，最终由络合物构筑质量策略决定是否拒绝。

## 2. 数学对象

把一个 `n` 元环表示为有序闭合空间折线：

\[
\Gamma=(\mathbf p_0,\mathbf p_1,\ldots,\mathbf p_{n-1},\mathbf p_0),
\qquad
e_i=[\mathbf p_i,\mathbf p_{i+1\bmod n}].
\]

这首先是一个一维闭曲线边界，而不是天然附带一个唯一的二维内部曲面。该区别解释了为什么
“环自身是否合理”与“键穿过哪一个环面”必须分开。

## 3. 第一部分：环自身的硬几何有效性

### 3.1 中心线自交

对所有不共享端点的环边计算有限线段最短距离：

\[
d_{ij}=\min_{s,t\in[0,1]}
\left\|
(\mathbf p_i+s(\mathbf p_{i+1}-\mathbf p_i))-
(\mathbf p_j+t(\mathbf p_{j+1}-\mathbf p_j))
\right\|.
\]

- `d_ij = 0`：零厚度中心线严格自交；
- `d_ij` 落入数值容差带：不能可靠区分相交和擦过；
- `d_ij > tolerance`：中心线在这一对边上分离。

这是成熟、明确的计算几何问题。它必须检查线段中部，不能只检查环原子之间的距离。

### 3.2 有限厚度的自碰撞

真实原子和键不是零厚度。可把环键近似为 capsule，定义有效半径 `rho_i`，检查：

\[
Q_{ij}^{\mathrm{edge}}=
\frac{d(e_i,e_j)}{\rho_i+\rho_j}.
\]

`Q < 1` 表示几何管发生碰撞，`Q ≈ 1` 表示临界接触。为保持纯几何语义，`rho` 不取自力场
或原子类型，而由环自身尺度

\[
s=\operatorname{median}(\|e_i\|)
\]

和公开的无量纲厚度系数构造，例如 `rho = alpha*s`。该系数表达质量门控希望保留的几何
间隙，不应被称为范德华半径或物理排斥半径。

还应分别计算：

- 非键原子—原子距离；
- 原子—非关联环键距离；
- 非相邻环键—环键距离。

这三类检查互不替代。共享顶点、相邻边以及指定图距离内的局部关系通过分子图拓扑排除；
不引入元素类型、氢键或力场参数。

### 3.3 用 polygonal thickness/reach 描述急折和塌缩

一个很有启发性的统一指标是离散曲线厚度：

\[
\tau = \min\left(
\min_i R_{\mathrm{curv},i},
\frac{1}{2}d_{\mathrm{self}}
\right),
\]

其中 `R_curv,i` 是连续三个顶点的局部曲率半径，`d_self` 是非局部环段的最小距离。再以
中位环键长归一化：

\[
T=\frac{\tau}{\operatorname{median}(\ell_i)}.
\]

小 `T` 同时捕获“局部折得过急”和“相隔很远的两段环链彼此压近”。它与 knot theory 中
的 thickness/ropelength 思想一致，可作为连续质量指标；化学阈值仍需按环类型标定。

参考：Gonzalez and Maddocks, *Global curvature, thickness, and the ideal shapes of knots*,
PNAS 1999, DOI `10.1073/pnas.96.9.4769`。

### 3.4 退化和孔洞塌缩指标

对中心化环坐标执行 SVD，令 `sigma_1 >= sigma_2 >= sigma_3`：

- `sigma_2 / sigma_1` 很小：环接近塌成一条线；
- `sigma_3 / sigma_2`：描述非平面程度，但不能单独作为失败条件；
- best-fit 投影面积与周长平方之比
  `4*pi*A/P^2`：描述投影孔洞是否显著闭合；
- 最小非相邻边间距、最小三角高度：描述数值条件是否恶化。

椅式环己烷天然非平面，因此“非平面”本身不是不合理；这些量应作为诊断和环面可定义性
指标。

## 4. 第一部分的连续几何评分

本判定不调用力场，不使用真实/近似势能，也不依赖按元素和杂化建立的参考构象数据库。
除第 3 节的硬判据外，可以保留以下纯几何无量纲描述量：

- 边长相对自身中位边长的离散程度；
- 非相邻边最小距离与 `s` 的比值；
- 局部曲率半径与 `s` 的比值；
- `sigma_2/sigma_1` 和 `sigma_3/sigma_2`；
- 投影孔洞的 `4*pi*A/P^2`；
- Cremer–Pople puckering coordinates 或一般化的离散 Fourier puckering modes。

这些量只描述当前形状，不假装预测热力学稳定性。Cremer–Pople 坐标尤其适合稳定地区分
chair、boat、twist 等起伏模式，但不自动给出合格/不合格结论。参考：Cremer and Pople,
JACS 1975, DOI `10.1021/ja00839a011`。

若构象生成阶段需要用单个标量排序，可以构造廉价的纯几何伪能量：

\[
E_{\mathrm{geo}}=
w_l\sum_i[|\ell_i/s-1|-\delta_l]_+^2+
w_c\sum_{i,j}[\delta_c-d(e_i,e_j)/s]_+^2+
w_k\sum_i[s/R_{\mathrm{curv},i}-\kappa_{\max}]_+^2+
w_a[\delta_a-4\pi A/P^2]_+^2,
\]

其中 `[x]_+=max(x,0)`。它完全由坐标、环拓扑和无量纲阈值组成，计算复杂度至多 `O(n^2)`；
对于 `n <= 8` 可以忽略。该值只用于排序或把边界案例标成 `SUSPECT`，不能命名为物理能量，
也不能覆盖明确的自交、退化和碰撞硬失败。

### 自交不等于打结

- **自交/自碰撞**：不相邻环段占据相同或冲突空间，通常是硬几何失败；
- **非平凡 knot**：闭曲线没有自交，但拓扑结型不是 unknot，可能是真实目标结构。

因此 knot class 应作为独立拓扑标签。更有业务意义的问题是：在没有断键的优化过程中，
预期 knot/link 类型是否被错误改变，而不是“只要打结就拒绝”。

## 5. 中间层：环面是否可定义

第二部分真正依赖的不是“环能量是否低”，而是能否构造明确、稳定的 `RingSurface`。建议报告：

```text
DEFINED
AMBIGUOUS
UNDEFINED
```

候选方案如下。

### 5.1 Best-fit aperture plane

用 SVD 建立最佳拟合平面，把环边界和键—平面交点投影到二维，再执行 point-in-polygon。

优点是简单、稳定，直接表达“是否穿过平均环孔”；适合近似平面的小环。缺点是高度折叠时
它只代表投影孔，不代表实际三维跨越面。它可作为快速筛查或一种明确命名的
`aperture_plane` 语义，不应伪装成普适拓扑事实。

### 5.2 单个规范三角面

将 simple 2D boundary 三角化，再把顶点提升回原始三维坐标。固定三角化后，有限键段—
三角形相交可以明确判定；但不同合法对角线会生成不同非平面曲面。单独采用 Earcut 返回的
一组结果仍包含选面任意性。

一个四顶点反例即可说明这种任意性。令：

```text
p0=(-1,-1,0)  p1=( 1,-1,1)
p3=(-1, 1,1)  p2=( 1, 1,0)
```

该空间四边形边界相同。使用对角线 `p0-p2` 时，中心处三角面高度为 `z=0`；使用对角线
`p1-p3` 时，中心处高度为 `z=1`。短键段 `[(0,0,-0.1), (0,0,0.1)]` 穿过前一个面，
却不穿过后一个面。这不是浮点误差，而是非平面边界确实允许不同跨越面。

### 5.3 全部合法三角面的共识判定（首选研究方向）

当前默认最多检查 8 元环。凸 `n` 边形的三角剖分数量为 Catalan 数 `C_(n-2)`，八边形最多
仅 `C_6 = 132` 种。因此可以：

1. 枚举所有尊重有序边界的合法三角剖分；
2. 把每个剖分提升为只使用原始环顶点的三维分片曲面；
3. 拒绝退化、三角面彼此自交或数值条件不合格的曲面；
4. 对每一个剩余曲面执行有限键段—三角形相交；
5. 对结果取共识：

```text
全部曲面均不相交 -> CLEAR
全部曲面均横穿   -> PIERCES
不同曲面结论不同 -> AMBIGUOUS_SURFACE
只有擦边/共面接触 -> TOUCHES
不存在有效曲面   -> UNDEFINED_RING
```

该方案不再把某条任意对角线冒充唯一物理环面。平面环的全部三角剖分覆盖同一个区域，结论
自然一致；轻微翘曲时通常也会取得共识；只有强折叠到选面确实影响答案时才报告模糊。

它仍然只对“由原始边界顶点形成的嵌入三角盘”这一明确曲面族给出严格共识，不代表枚举了
数学上所有包含 Steiner 点的连续 spanning surfaces。但相较单一 Earcut 面，它的任意性显著
更小，而且规模完全可控。

“合法剖分”需要明确：存在稳定 simple projection 时，只接受投影边界内部的非交叉对角线；
投影本身不稳定时，可枚举抽象循环多边形的组合剖分并在提升到三维后过滤自交面，但此时应
提高模糊等级。一个无自交的空间环也未必能在不增加内部 Steiner 点的前提下，由边界顶点
构成嵌入三角盘；因此“没有有效 vertex-only surface”只表示当前环面模型无法解析，不证明
该环在物理上不存在。

非平凡 knot 更是一个明确例子：它可以物理存在，却不可能以嵌入拓扑圆盘作为 spanning
surface。它需要更高亏格的 Seifert surface。这再次说明 `RingSurface` 的可定义性不能与
`RingGeometryReport` 的物理有效性合并成同一个布尔值。

### 5.4 离散最小面积面

若某些调用方必须取得一个规范显示面，可在合法剖分集合中选择：

\[
T^*=\arg\min_T\sum_{\Delta\in T}\operatorname{Area}_{3D}(\Delta).
\]

这近似离散 Plateau minimal surface，数学动机强于按 atom index 任意选对角线。也可以加入
相邻三角面的 bending penalty，但其权重会引入新的物理模型参数。

建议把最小面积面用于可视化、排序或次级诊断，不要用它覆盖
`AMBIGUOUS_SURFACE`。若两个近简并最小面给出不同穿环结果，正确答案仍是不确定。

## 6. 第二部分：键与环的关系

对于一个已经定义的三角环面，执行：

1. AABB broad phase；
2. 有限 segment–triangle narrow phase；
3. 对共享端点、三角内部对角线上的重复命中去重；
4. 用重心坐标和交角区分横穿、擦边和共面接触；
5. 保留交点、涉及三角形、最小边界距离和容差证据。

建议关系状态为：

```text
CLEAR
PIERCES
TOUCHES
AMBIGUOUS_SURFACE
UNDEFINED_RING
```

`TOUCHES` 不应强行并入 `PIERCES` 或 `CLEAR`。质量门控可以对后三种状态 fail closed，但报告
必须保留真实原因。

## 7. 更严格的拓扑方法：闭环—闭环 linking

单根键是开放线段，与闭环之间没有天然定义的 linking number。人为在远处闭合会使结果依赖
闭合路径。

如果目标键属于另一条完整闭合 cycle，则可对两条互不相交闭曲线计算 Gauss linking number：

\[
Lk(C_1,C_2)=\frac{1}{4\pi}
\oint_{C_1}\oint_{C_2}
\frac{(d\mathbf r_1\times d\mathbf r_2)\cdot(\mathbf r_1-\mathbf r_2)}
{\|\mathbf r_1-\mathbf r_2\|^3}.
\]

- `Lk != 0`：可明确证明存在链接；
- `Lk = 0`：不能证明一定没有链接，例如存在 linking number 为零的非平凡 link；
- 对更复杂 link 可进一步使用 Alexander/Jones polynomial 等不变量。

该方法非常适合 catenane、宏环穿链和金属有机拓扑分析，但不能替代当前单键—环的局部几何
检查。

## 8. 推荐的数据流和策略边界

```text
assess_ring_geometry(ring)
  -> RingGeometryReport

construct_ring_surfaces(ring, geometry_report)
  -> RingSurfaceEnsemble / surface status

classify_bond_ring_relation(bond, surface_ensemble)
  -> BondRingRelationReport

apply_geometry_policy(reports, level)
  -> accept / reject / retry
```

关键规则：

- `INVALID/UNDEFINED` 绝不转换成“没有穿环”；
- 环为 `SUSPECT` 但曲面可定义时，仍可执行关系判定；
- 几何模块输出证据，forcefield/complex builder 决定哪些状态需要拒绝或重试；
- 当前布尔便利接口只能对确定状态压缩结果，遇到 ambiguous/undefined 应抛出明确异常；
- 默认业务策略可以拒绝意外穿环，但必须允许机械互锁分子选择保留预期拓扑。

## 9. 分级门控建议

### Basic

- 坐标有限、边长非零、无重复连续顶点；
- 非相邻环边中心线无严格自交；
- 无严重原子重叠。

### Standard

- capsule/atom–bond/atom–atom clearance；
- polygonal thickness 和局部曲率；
- 环面 ensemble 可定义性；
- 键—环关系共识。

### Strict

- 全部合法三角剖分的共识判定；
- exact/robust predicates 或明确的数值不确定区间；
- 坐标微扰后的分类稳定性；
- knot/link 类型及几何拓扑保持。

## 10. 对现有非平面环方案的影响

`mapbox-earcut` 仍适合生成一个快速合法三角剖分，也适合平面/近似平面的常规路径。但本轮
分析说明：**单个规范化 Earcut 面是可复现的工程定义，却不是可以达到的最强判定。**

建议下一轮设计时比较两条路线：

1. 默认单规范面，只有临界案例才枚举全部剖分；
2. 对 `n <= 8` 始终枚举全部合法剖分并取共识。

鉴于最大候选数只有 132，当前更倾向第 2 条。实际实现前还需用真实小环、puckered ring、
折叠宏环和络合物螯合环测量“共识/模糊”比例，确认该曲面族不会过度产生
`AMBIGUOUS_SURFACE`。
