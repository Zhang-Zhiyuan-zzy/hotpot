# `geometry.relation` 数学实施契约

> 状态：规范性设计文档。必须先修改本文，再修改 `relation.py` 和对应测试。  
> 边界：只报告空间事实、数值接触和数学未决；不判断合理性、物理性或化学正确性。

## 1. 对象、单位与符号

- `Point` 是必须保留的基础公开值对象，即使当前仓库没有直接消费者。
- `Line` 表示无限直线 $p+t d$；`Segment` 表示 $a+t(b-a),\ t\in[0,1]$。
- $Cycle=(p_0,\ldots,p_{n-1})$ 只表示有序闭合边界，不隐含唯一内部曲面。
- 坐标长度单位记为 `CLU`。Hotpot 转换对象默认 `CLU = Å`；直接构造几何对象时必须使用一致单位。
- 数学常数 $0,1,2,\pi$ 不进入配置；所有经验阈值、数值容差和计算预算只能来自 `settings.py`。

## 2. `settings.py` 唯一参数表

```python
@dataclass(frozen=True)
class NumericToleranceSettings:
    absolute_length: float = 1.0e-8
    relative_length: float = 1.0e-10
    parameter: float = 1.0e-10
    machine_epsilon_factor: float = 64.0
    predicate_guard_factor: float = 4.0
    planarity_factor: float = 1.0
    winding_residual: float = 1.0e-10
    intersection_merge_factor: float = 4.0
    aabb_padding_factor: float = 4.0


@dataclass(frozen=True)
class SurfaceEnumerationSettings:
    maximum_cycle_vertices: int = 8
    maximum_surface_count: int = 132
    maximum_segment_triangle_tests: int = 792
    maximum_triangle_pair_tests: int = 1980


@dataclass(frozen=True)
class GeometrySettings:
    tolerance: NumericToleranceSettings = field(default_factory=NumericToleranceSettings)
    surface: SurfaceEnumerationSettings = field(default_factory=SurfaceEnumerationSettings)


DEFAULT_GEOMETRY_SETTINGS = GeometrySettings()
```

| 参数 | 默认值 | 单位/范围 | 唯一用途 |
|---|---:|---|---|
| `absolute_length` | $10^{-8}$ | `CLU`, $>0$ | 坐标绝对分辨率 |
| `relative_length` | $10^{-10}$ | 无量纲, $\ge0$ | 随局部尺度变化的相对分辨率 |
| `parameter` | $10^{-10}$ | 无量纲, $0<\epsilon_{\mathrm{param}}<1/(2g)$ | $t$、重心坐标等参数判定 |
| `machine_epsilon_factor` | $64$ | 无量纲, $\ge1$ | float64 舍入误差下限 |
| `predicate_guard_factor` | $4$ | 无量纲, $>1$ | 接触带之外的未决保护带 |
| `planarity_factor` | $1$ | 无量纲, $>0$ | 平面残差相对 $\epsilon_L$ 的倍率 |
| `winding_residual` | $10^{-10}$ | 无量纲, $0<\epsilon_{winding}<1/2$ | winding number 的整数残差 |
| `intersection_merge_factor` | $4$ | 无量纲, $\ge1$ | 合并同一几何交点 |
| `aabb_padding_factor` | $4$ | 无量纲, $\ge$ `predicate_guard_factor` | AABB 安全扩张 |
| `maximum_cycle_vertices` | $8$ | 整数, $\ge3$ | 完整顶点三角剖分的计算上限 |
| `maximum_surface_count` | $132$ | 整数, $>0$ | 八元环 Catalan 上限 $C_6$ |
| `maximum_segment_triangle_tests` | $792$ | 整数, $>0$ | $132\times(8-2)$ 次 segment–triangle 判定上限 |
| `maximum_triangle_pair_tests` | $1980$ | 整数, $>0$ | $132\times\binom{6}{2}$ 次 surface embedding triangle-pair 判定上限 |

表中列出的是全部预定义数值。`max_ring_size`、原子距离阈值等调用方业务选择不是 geometry
全局标准，必须作为显式参数传入；它们不得在 `relation.py` 中另设默认常量。
四个 surface budget 只允许阻止超预算计算；一旦截断枚举，必须记录
`enumeration_complete=False` 并返回 `UNDETERMINED`，不得用已枚举子集形成确定结论。
`maximum_segment_triangle_tests` 只计曲面与目标 segment 的求交；
`maximum_triangle_pair_tests` 每检查一对 surface triangles 计 1，其内部固定 6 次 edge–triangle
primitive 不重复计入前者。两类计数必须分别报告。
配置必须满足 $k_{\mathrm{AABB}}\ge g$；否则 settings 无效，不得执行 relation 判定。
本层不定义 weighted score、合理性评分或置信概率。归一化残差仅用于数值诊断。

## 3. 局部尺度与分量纲容差

任何 classifier 必须先检查其全部坐标、方向和 settings 派生量。存在 `NaN` 或 $\pm\infty$ 时，
不计算 $L$，直接返回 `UNDETERMINED` 并记录 `NONFINITE_INPUT`；纯距离 measurement 返回
`NaN`。非有限值不得通过比较运算偶然落入 `DOES_NOT_PIERCE`。

对本次谓词涉及的点集 $X$、线段集 $E$ 和环边界 $\partial C$，定义：

$L=\max\left(\operatorname{diam}(X),\max_{e\in E}\lVert e\rVert,\operatorname{median}_{e\in\partial C}\lVert e\rVert\right)$

只纳入当前谓词实际存在的项：无环时省略 $\partial C$，无线段集时省略 $E$；至少使用全部输入点
的直径。局部尺度不得读取分子全局包围盒，以免无关远端原子改变当前 pair 的判定。

若 $L\le\epsilon_{\mathrm{abs}}$，有限几何对象进入退化/未决分支；禁止用 $\max(1,L)$ 改变量纲。
无限 `Line` 不使用本条退化判据：其方向按第 5 节无尺度规范化，长度容差只读取直线间的
几何不变量距离。

$\epsilon_r=\max\left(\epsilon_{\mathrm{rel}},k_{\mathrm{mach}}\epsilon_{\mathrm{machine}}\right)$

其中 $\epsilon_{\mathrm{machine}}=\operatorname{finfo}(\mathrm{float64}).\mathrm{eps}$；数值 kernel
统一使用 `float64`，不得因输入 dtype 改变判定标准。

$\epsilon_L=\epsilon_{\mathrm{abs}}+\epsilon_rL$

$\epsilon_u=\epsilon_{\mathrm{param}}+\epsilon_L/L$

$\epsilon_A=\epsilon_LL$

$\epsilon_V=\epsilon_LL^2$

$\epsilon_{\mathrm{AABB}}=k_{\mathrm{AABB}}\epsilon_L$

$\epsilon_{\mathrm{merge}}=k_{\mathrm{merge}}\epsilon_L$

其中 $\epsilon_L$、$\epsilon_A$、$\epsilon_V$ 分别具有长度、面积和体积量纲，$\epsilon_u$ 无量纲。
禁止用同一 float 直接比较 determinant、cross product、距离和参数 $t$。

凡需把 $[0,1]$ 参数分成端点/内部/外部的谓词，必须先满足
$g\epsilon_u<\frac12$；否则直接返回 `UNDETERMINED` 并记录 `TOLERANCE_DOMAIN`。

令保护倍率为 $g$。对残差 $r$ 及其同量纲容差 $\epsilon$：

- $|r|\le\epsilon$：接触或位于定义边界；
- $\epsilon<|r|\le g\epsilon$：`UNDETERMINED`；
- $|r|>g\epsilon$：可稳定判断符号或分离关系。

同比缩放 $x'=sx$ 时必须同时设置 $\epsilon'_{\mathrm{abs}}=s\epsilon_{\mathrm{abs}}$。长度、面积、体积谓词分别按 $s$、$s^2$、$s^3$ 缩放，最终 state 必须不变。

本层不产生综合评分。允许输出的归一化诊断量及范围只有：

| 诊断量 | 范围 | 判定用途 |
|---|---|---|
| $\sigma_2/\sigma_1$、$\sigma_3/\sigma_2$ | $[0,1]$ | rank 与非平面程度 |
| $q_{\parallel}$ | $[0,1]$ | 直线方向平行程度 |
| $q_{\det}$ | $[0,1]$ | segment–triangle determinant 稳定性 |
| $\lambda_a,\lambda_b,\lambda_c$ | 单项可在 $(-\infty,\infty)$，总和为 $1$ | 三角形内部/边界/外部 |
| $w$ | 理想简单多边形非边界点为 $0$ 或 $\pm1$ | 平面多边形内部/外部 |

这些量不得加权为“合理性分数”或“穿环概率”。

## 4. `measure_planarity()`

环顶点质心：

$c=\frac1n\sum_{i=0}^{n-1}p_i$

对中心化矩阵 $X=[p_i-c]$ 做：

$X=U\Sigma V^\mathsf{T}$

令 $\sigma_1\ge\sigma_2\ge\sigma_3$，最佳拟合平面法向为 $V$ 中对应 $\sigma_3$ 的单位向量 $n$。

$h_i=|(p_i-c)\cdot n|$

$h_{\max}=\max_i h_i$

$h_{\mathrm{rms}}=\sqrt{\frac1n\sum_i h_i^2}$

$\epsilon_{\mathrm{plane}}=k_{\mathrm{plane}}\epsilon_L$

分类标准：

- $\sigma_1\le\epsilon_L$：`DEGENERATE`；
- $\epsilon_L<\sigma_1\le g\epsilon_L$：`UNDETERMINED`；
- $\sigma_1>g\epsilon_L$ 且 $\sigma_2/\sigma_1\le\epsilon_u$：`DEGENERATE`；
- $\sigma_1>g\epsilon_L$ 且 $\epsilon_u<\sigma_2/\sigma_1\le g\epsilon_u$：`UNDETERMINED`；
- $h_{\max}\le\epsilon_{\mathrm{plane}}$：`PLANAR`；
- $\epsilon_{\mathrm{plane}}<h_{\max}\le g\epsilon_{\mathrm{plane}}$：`UNDETERMINED`；
- $h_{\max}>g\epsilon_{\mathrm{plane}}$：`NONPLANAR`。

后三项只在 $\sigma_1>g\epsilon_L$ 且 $\sigma_2/\sigma_1>g\epsilon_u$ 时求值。

`PlanarityMeasurement.normal` 为 `Optional[Tuple[float, float, float]]`。有限但退化时保留可计算的奇异值，
`normal=None`、$h_{\max}=h_{\mathrm{rms}}=\mathrm{NaN}$；非有限输入时 normal 为 `None` 且全部
标量为 `NaN`。输出不包含芳香性判断。

## 5. `determine_line_relation()` 与 `line_distance()`

两直线为 $L_1:p_1+t d_1$、$L_2:p_2+s d_2$。

$m_i=\max_k|d_{ik}|$

$\widehat d_i=\frac{d_i/m_i}{\lVert d_i/m_i\rVert}$

$\epsilon_{\theta}=\max(\epsilon_{\mathrm{param}},k_{\mathrm{mach}}\epsilon_{\mathrm{machine}})$

$q_{\parallel}=\lVert\widehat d_1\times\widehat d_2\rVert$

$d_{\parallel}=\lVert(p_2-p_1)\times\widehat d_1\rVert$

$d_{\mathrm{skew}}=\frac{|(p_2-p_1)\cdot(\widehat d_1\times\widehat d_2)|}{\lVert\widehat d_1\times\widehat d_2\rVert}$

$\epsilon_{L,\parallel}=\epsilon_{\mathrm{abs}}+\epsilon_r d_{\parallel}$

$\epsilon_{L,\mathrm{skew}}=\epsilon_{\mathrm{abs}}+\epsilon_r d_{\mathrm{skew}}$

- 任一 $m_i=0$：`DEGENERATE`，距离未定义；任一 $m_i$ 非有限已由第 3 节处理；
- $q_{\parallel}=0$ 且 $d_{\parallel}\le\epsilon_{L,\parallel}$：`COINCIDENT`；
- $q_{\parallel}=0$ 且 $d_{\parallel}>g\epsilon_{L,\parallel}$：`PARALLEL`；
- $0<q_{\parallel}\le g\epsilon_{\theta}$：`UNDETERMINED`；
- $q_{\parallel}>g\epsilon_{\theta}$ 且 $d_{\mathrm{skew}}\le\epsilon_{L,\mathrm{skew}}$：`INTERSECTING`；
- $q_{\parallel}>g\epsilon_{\theta}$ 且 $d_{\mathrm{skew}}>g\epsilon_{L,\mathrm{skew}}$：`SKEW`；
- 落入任一保护带：`UNDETERMINED`。

距离分段为：

$d(L_1,L_2)=d_{\parallel}$，当 $q_{\parallel}=0$

$d(L_1,L_2)=d_{\mathrm{skew}}$，当 $q_{\parallel}>g\epsilon_{\theta}$

保护带或退化方向下距离为未定义。`determine_line_relation()` 返回
`LineRelation(kind, distance)`，其中 `distance: Optional[float]`；`line_distance()` 对未定义情形
返回 `NaN`。容差尺度只读取几何不变量 $d_{\parallel}$ 或 $d_{\mathrm{skew}}$，不读取任意
origin separation；因此对 $d_i'=c_i d_i$ 以及 $p_i'=p_i+a_i d_i$ 均保持不变。
`LineRelationKind` 必须包含 `UNDETERMINED`。

## 6. 距离函数

点 $p$ 到线段 $ab$：

$t=\operatorname{clip}_{[0,1]}\frac{(p-a)\cdot(b-a)}{\lVert b-a\rVert^2}$

$d(p,ab)=\lVert p-(a+t(b-a))\rVert$

若 $\lVert b-a\rVert\le\epsilon_L$，该段在关系分类中是 `DEGENERATE`；纯距离函数仍返回
$d(p,a)$，不得除以近零平方长度。

两线段距离：

$d(ab,cd)=\min_{s,t\in[0,1]}\lVert a+s(b-a)-c-t(d-c)\rVert$

点对距离：

$d_{ij}=\lVert p_i-p_j\rVert$

退化判断必须使用 $\epsilon_L$ 或 $\epsilon_L^2$；禁止将平方长度与长度容差比较。
`find_point_pairs_below_distance(points, threshold)` 只对已计算的 $d_{ij}$ 执行严格
$d_{ij}<d_{\mathrm{caller}}$ 筛选，不引入预定义阈值，也不解释 caller threshold 是否合理；
需要检查阈值边界稳定性的调用方必须读取原始 $d_{ij}$。

## 7. `locate_point_in_planar_cycle()`

在拟合平面建立正交基 $(u,v)$：

$\pi(p)=((p-c)\cdot u,(p-c)\cdot v)$

点到环边界距离：

$d_{\partial C}(p)=\min_i d(p,p_ip_{i+1})$

二维 winding number：

$w(p)=\frac1{2\pi}\sum_i\operatorname{atan2}\left((v_i-p)\times(v_{i+1}-p),(v_i-p)\cdot(v_{i+1}-p)\right)$

- $d_{\partial C}\le\epsilon_L$：`BOUNDARY`；
- $\epsilon_L<d_{\partial C}\le g\epsilon_L$：`UNDETERMINED`；
- $d_{\partial C}>g\epsilon_L$ 且 $||w|-1|\le\epsilon_{\mathrm{winding}}$：`INTERIOR`；
- $d_{\partial C}>g\epsilon_L$ 且 $|w|\le\epsilon_{\mathrm{winding}}$：`EXTERIOR`；
- 其他：`UNDETERMINED`。

必须先证明 projected cycle 是简单闭合折线；否则返回 `UNDETERMINED`。

二维 orientation：

$\operatorname{orient}(a,b,p)=(b_x-a_x)(p_y-a_y)-(b_y-a_y)(p_x-a_x)$

该值具有面积量纲，只能与 $\epsilon_A$ 比较。

对任意无共同端点的投影边 $ab$ 与 $cd$，令
$o_1=\operatorname{orient}(a,b,c)$、$o_2=\operatorname{orient}(a,b,d)$、
$o_3=\operatorname{orient}(c,d,a)$、$o_4=\operatorname{orient}(c,d,b)$。
先检查保护带：任一 $|o_i|\le g\epsilon_A$ 时，该 edge pair 为 `UNDETERMINED`，除非其
AABB 已以超过 $\epsilon_{\mathrm{AABB}}$ 的轴间距稳定分离。只有四个 orientation 均离开保护带后，
$\operatorname{sign}(o_1)\ne\operatorname{sign}(o_2)$ 且
$\operatorname{sign}(o_3)\ne\operatorname{sign}(o_4)$ 才证明边界自交；否则该 pair 稳定分离。
全部非相邻边对稳定分离后才证明边界简单。

## 8. 线段—平面与线段—三角形

线段为 $s(t)=s_0+t(s_1-s_0)$，平面由点 $a$ 和单位法向 $n$ 定义：

$\ell_s=\lVert s_1-s_0\rVert$

- $\ell_s\le\epsilon_L$：`DEGENERATE_SEGMENT`，最终关系为 `UNDETERMINED`；
- $\epsilon_L<\ell_s\le g\epsilon_L$：`NUMERIC_BAND`，最终关系为 `UNDETERMINED`；
- 只有 $\ell_s>g\epsilon_L$ 才继续下列 plane/triangle 判定。

$h_0=n\cdot(s_0-a)$

$h_1=n\cdot(s_1-a)$

按下列互斥顺序判定：

1. $|h_0|\le\epsilon_L$ 且 $|h_1|\le\epsilon_L$：`COPLANAR_CONTACT`；
2. 任一 $|h_i|\in(\epsilon_L,g\epsilon_L]$：`UNDETERMINED`；
3. 恰有一个 $|h_i|\le\epsilon_L$，另一个 $|h_j|>g\epsilon_L$：端点接触；
4. $|h_0-h_1|\le\epsilon_L$ 且两端同号稳定离面：平行且分离；
5. $|h_0-h_1|\in(\epsilon_L,g\epsilon_L]$：`UNDETERMINED`；
6. $|h_0-h_1|>g\epsilon_L$：计算 $t$。

第 6 项中：

$t=\frac{h_0}{h_0-h_1}$

$q=s_0+t(s_1-s_0)$

- $g\epsilon_u<t<1-g\epsilon_u$：线段开区间交点；
- $|t|\le\epsilon_u$ 或 $|1-t|\le\epsilon_u$：端点接触；
- $t<-g\epsilon_u$ 或 $t>1+g\epsilon_u$：只有无限延长线命中；
- $t$ 落入 $[-g\epsilon_u,-\epsilon_u)$、$(\epsilon_u,g\epsilon_u]$、
  $[1-g\epsilon_u,1-\epsilon_u)$ 或 $(1+\epsilon_u,1+g\epsilon_u]$：`UNDETERMINED`。

三角形 $abc$ 的法向为：

$N=(b-a)\times(c-a)$

- $\lVert N\rVert\le\epsilon_A$：三角形退化；
- $\epsilon_A<\lVert N\rVert\le g\epsilon_A$：`UNDETERMINED`；
- $\lVert N\rVert>g\epsilon_A$：才计算交点 $q$ 的重心坐标：

$\lambda_a=\frac{((b-q)\times(c-q))\cdot N}{\lVert N\rVert^2}$

$\lambda_b=\frac{((c-q)\times(a-q))\cdot N}{\lVert N\rVert^2}$

$\lambda_c=1-\lambda_a-\lambda_b$

- 所有 $\lambda_i>g\epsilon_u$：三角形严格内部；
- 所有 $\lambda_i\ge-\epsilon_u$ 且至少一个 $|\lambda_i|\le\epsilon_u$：边或顶点接触；
- 存在 $\lambda_i<-g\epsilon_u$：三角形外部；
- 其他：`UNDETERMINED`。

若使用 Möller–Trumbore，必须比较无量纲量：

$q_{\det}=\frac{|(b-a)\cdot((s_1-s_0)\times(c-a))|}{\lVert b-a\rVert\lVert s_1-s_0\rVert\lVert c-a\rVert}$

- $q_{\det}>g\epsilon_u$：允许使用 Möller–Trumbore 快速路径；
- $q_{\det}\le g\epsilon_u$：不得除以该 determinant，改用本节已经完成的 signed-distance
  $h_0,h_1$、参数 $t$ 和重心坐标路径；只要 fallback 的各项均稳定，仍可得到确定结论。

因此长而近乎平行、但两个端点稳定分居平面两侧的 segment 仍可确认横穿。共面重叠只记录
`COPLANAR_CONTACT`，不构成严格横穿。

## 9. 顶点三角剖分的 `EMBEDDED` 判据

对 $n$ 个有序环顶点，候选曲面必须满足：

$N_{\mathrm{triangulations}}=C_{n-2}=\frac{1}{n-1}\binom{2n-4}{n-2}$

因此 $n=8$ 时 $C_6=132$；完整枚举数必须与该值一致，否则
`enumeration_complete=False`。

1. 三角形数 $F=n-2$，内部对角线数为 $n-3$；
2. 每条环边恰属于一个三角形，每条内部对角线恰属于两个三角形；
3. 三角形邻接图连通，且 $V-E+F=1$；
4. 唯一边界严格等于原始有序 `Cycle`；
5. 每个三角形满足 $\lVert N_i\rVert>g\epsilon_A$；
6. 共享边在相邻三角形中的方向相反。

设组合单纯复形为 $K$，分片仿射映射为 $f:|K|\rightarrow\mathbb R^3$。几何嵌入要求 $f$ 单射；有限验证条件为：

$f(T_i)\cap f(T_j)=f(T_i\cap T_j)$

对任意不同三角形 $T_i,T_j$ 均成立：

| 组合共享部分 | 唯一允许的几何交集 |
|---|---|
| 无共享顶点 | 空集 |
| 共享一个顶点 | 仅该顶点 |
| 共享一条边 | 仅该完整共享边 |

三维侧向谓词统一使用：

$\operatorname{orient3d}(a,b,c,d)=((b-a)\times(c-a))\cdot(d-a)$

其结果具有体积量纲，只能与 $\epsilon_V$ 比较。非共面 triangle–triangle 关系由各顶点的
`orient3d` 符号和有限 segment–triangle 谓词共同确定；共面情形投影到主平面后只使用
`orient2d` 与 $\epsilon_A$。不得以无限直线相交代替有限三角形相交。

对每一 triangle pair，令组合上允许的共享单形为 $S_{ij}=f(T_i\cap T_j)$。数值判定固定为：

组合共享顶点代入 `orient3d` 必然为 0；这些已知零值先归入 $S_{ij}$，不进入 numeric guard。
只检查非共享顶点：其到对方平面的残差均 $\le\epsilon_V$ 时进入共面二维分支；任一残差位于
$(\epsilon_V,g\epsilon_V]$ 时为 `CONSTRUCTION_UNDETERMINED`；否则进入非共面分支。

| 顺序 | 条件 | 结果 |
|---:|---|---|
| 1 | 无共享顶点且 AABB 在任一轴以 $>\epsilon_{\mathrm{AABB}}$ 分离 | pair 通过 |
| 2 | 非共面；六条 triangle edge 的 segment–triangle 判定均完成，且实际交点全部属于 $S_{ij}$ | pair 通过 |
| 3 | 非共面；存在距 $S_{ij}$ 超过 $g\epsilon_L$ 的额外交点 | `PROVEN_NON_EMBEDDED` |
| 4 | 共面；二维 edge–edge 与 vertex–triangle 判定证明交集严格等于 $S_{ij}$ | pair 通过 |
| 5 | 共面；证明存在 $S_{ij}$ 外的线段或正面积重叠 | `PROVEN_NON_EMBEDDED` |
| 6 | 任一必要 `orient3d`、`orient2d`、距离或有限求交落入保护带 | `CONSTRUCTION_UNDETERMINED` |

第 2–5 项必须检查有限三角形的 6 条边；共享顶点/共享边只作为 $S_{ij}$ 排除，不得把其他
接触一并忽略。点到 $S_{ij}$ 的距离 $\le\epsilon_L$ 才归入允许共享单形，位于
$(\epsilon_L,g\epsilon_L]$ 时未决，$>g\epsilon_L$ 时是额外交点；共面额外重叠面积分别以
$\epsilon_A$ 和 $g\epsilon_A$ 作同样分区。所有 triangle pair 通过后，候选曲面才是
`EMBEDDED`。

- 存在稳定的额外交点、面积重叠或非共享边重合：`PROVEN_NON_EMBEDDED`；
- 谓词落入保护带、共面重叠无法分类或交集维数不确定：`CONSTRUCTION_UNDETERMINED`；
- 所有组合和几何检查明确通过：`EMBEDDED`。

折叠但不自交的 PL 曲面仍可为 `EMBEDDED`；Geometry 不判断该折叠是否物理合理。
私有入口命名为 `_determine_surface_embedding()` 并返回上述三态，不使用暗示裸 `bool` 的
`_is_embedded_surface()`。

## 10. 单个曲面的线段关系

对一个 `EMBEDDED` 曲面：

- 存在稳定的线段开区间—某一三角形相对内部横穿，且交点不位于任何三角剖分内部边/顶点：
  `INTERSECTING`；
- 无严格横穿且所有必要谓词稳定完成：`NON_PIERCING`；
- 没有确认横穿，但至少一个必要谓词未决：`EVALUATION_UNDETERMINED`。

三角剖分内部对角线不是环边界。首轮不实现 local-star 穿越证明：交点落在内部对角线或内部
顶点的 $g\epsilon_L$ 邻域时，一律为 `EVALUATION_UNDETERMINED`，不得标记成环边接触或确认
横穿。该规则明确牺牲少量确定率以换取无歧义实现；以后新增 local-star 算法前必须先扩充本文。

交点合并阈值：

$\epsilon_{\mathrm{merge}}=k_{\mathrm{merge}}\epsilon_L$

$\lVert q_i-q_j\rVert\le\epsilon_{\mathrm{merge}}$

环真实边界接触、环顶点接触、线段端点接触和共面接触属于 `NON_PIERCING` 的细分事实；若同一线段另有确认横穿，整体仍为 `INTERSECTING`，并同时保留 contact features。

## 11. `SurfaceFamilyEvidence` 全字段赋值

记 evidence 为 $(K,Q,E,P,C,I,N,U)$，依次表示 complete、enumerated、embedded、
proven-non-embedded、construction-undetermined、intersecting、non-piercing 和
evaluation-undetermined。

- 简单平面环：$K=1,\ Q=1,\ E=1,\ P=C=0$，且严格横穿时
  $(I,N,U)=(1,0,0)$，明确不穿/只有接触时 $(I,N,U)=(0,1,0)$，求交未决时
  $(I,N,U)=(0,0,1)$；
- 已证明自交的平面环：$(K,Q,E,P,C,I,N,U)=(1,1,0,1,0,0,0,0)$；
- 非有限输入、退化对象或平面/非平面模型选择未决：
  $(K,Q,E,P,C,I,N,U)=(0,0,0,0,0,0,0,0)$，并由 `indeterminacy_causes` 记录原因；
- 完整非平面枚举：$K=1$，且满足 $Q=E+P+C$、$E=I+N+U$；
- 预算中止：$K=0$；已经完成分类的 $Q$ 个候选仍满足上述两个计数恒等式，尚未开始的候选
  不计入 $Q$；若中止发生在某个候选的构造判定中，该候选计入 $C$。

`SurfaceFamilyEvidence` 始终存在。`disjoint_surface_count` 禁止使用，因为边界接触并非几何
disjoint。

## 12. 非平面环三态共识

记 embedded、intersecting、non-piercing、evaluation-undetermined、construction-undetermined 数分别为 $E,I,N,U,C$。

$\mathrm{enumerated}=E+\mathrm{proven\_non\_embedded}+C$

$E=I+N+U$

$\mathrm{PIERCES}\iff\mathrm{complete}\land C=0\land U=0\land E>0\land I=E$

$\mathrm{DOES\_NOT\_PIERCE}\iff\mathrm{complete}\land C=0\land U=0\land E>0\land N=E$

其他情况均为 `UNDETERMINED`。

`PROVEN_NON_EMBEDDED` 可以从共识中排除；`CONSTRUCTION_UNDETERMINED` 不得静默排除。结论只对 `VERTEX_TRIANGULATION_FAMILY` 成立，不宣称覆盖全部连续跨越曲面。

同一个 `Cycle` 与多个 `Segment` 判定时，必须使用
`iter_segment_cycle_relations()`。该入口只对环执行一次平面度、三角剖分和
surface embedding 构造，然后逐段惰性产生与单段入口完全相同的关系记录。
surface construction 的容差尺度只读取 `Cycle` 本身；segment–surface 谓词的容差尺度
读取当前 `Segment + Cycle`。因此无关的远端 segment 不得改变环面是否 embedded。
只允许按环顶点数全局缓存纯组合的三角剖分索引；坐标派生的平面、三角面、
embedding 结论只在当前 iterator/scan 生命周期内复用。

## 13. AABB 与最近环边

AABB 仅用于安全跳过 segment–surface 精确求交，每个轴按以下距离扩张：

$\epsilon_{\mathrm{AABB}}=k_{\mathrm{AABB}}\epsilon_L$

只有先完成 cycle/surface 构造且得到 $K=1,C=0,E>0$ 后，任一轴区间稳定分离才可将
$I=U=0,N=E$ 并判为 `DOES_NOT_PIERCE`。在模型选择、退化、自交、surface construction 或
枚举完整性仍未决时，AABB 不得把关系短路成明确不穿。落入 padding 带必须进入精确谓词。

`SegmentCycleRelation.features` 是完整事实证据；AABB 只能加速有限段是否
横穿的 state-only 内部路径，不得使完整 relation 丢失
`LINE_EXTENSION_INTERIOR` 或接触 feature。公开 relation 在旋转前后的 state、features 和
indeterminacy causes 均必须一致。

对环边 $e_i=p_ip_{i+1}$：

$d_i=d(e_i,s)$

$d_{\min}=\min_i d_i$

$I_{\min}=\{i\mid d_i\le d_{\min}+\epsilon_L\}$

$i^*=\min I_{\min}$

距离在 $\epsilon_L$ 内并列时，以最小稳定 edge index 决定，保证输出可复现。存在有限最小值时
返回 `ClosestCycleEdge(edge_index, edge, distance)`；输入非有限或没有可比较边时返回 `None`。

## 14. 公开接口与参数来源

公开枚举的 `.value` 固定为成员名的小写形式。公开记录的字段和顺序固定为：

```python
PlanarityMeasurement(
    kind, centroid, normal, singular_values,
    maximum_deviation, rms_deviation, length_scale, length_tolerance,
)
LineRelation(kind, distance, parallel_measure)
PointPairDistance(first_index, second_index, distance)
ClosestCycleEdge(edge_index, edge, distance)
SurfaceFamilyEvidence(
    enumeration_complete,
    enumerated_surface_count,
    embedded_surface_count,
    proven_non_embedded_surface_count,
    construction_undetermined_count,
    intersecting_surface_count,
    non_piercing_surface_count,
    evaluation_undetermined_count,
    segment_triangle_tests_used,
    triangle_pair_tests_used,
)
SegmentCycleRelation(
    state, features, indeterminacy_causes, surface_model,
    intersection_points, closest_boundary_edge, surface_evidence, settings,
)
```

`segment_triangle_tests_used` 每调用一次目标 segment–surface triangle 谓词计 1；
`triangle_pair_tests_used` 每检查一对 surface triangles 计 1。二者均为
$[0,\infty)$ 整数，必须不超过 `settings.py` 对应预算。平面 polygon 路径不调用
triangle family kernel，两者固定为 $0$。

| 接口 | 数学输出 | 读取的 settings |
|---|---|---|
| `measure_planarity()` | 平面度量与 `PLANAR/NONPLANAR/UNDETERMINED/DEGENERATE` | numeric tolerance、planarity factor |
| `determine_line_relation()` | `LineRelation(kind, Optional[distance])` | numeric/parameter tolerance、guard factor |
| `line_distance()` | $[0,\infty)$；未定义时 `NaN` | direction normalization、invariant distance tolerance |
| `point_segment_distance()` | $[0,\infty)$；非有限输入为 `NaN` | degeneracy tolerance |
| `segment_segment_distance()` | $[0,\infty)$；非有限输入为 `NaN` | degeneracy tolerance |
| `point_pair_distances()` | 全部 $d_{ij}$ | 无判定阈值 |
| `find_point_pairs_below_distance()` | caller threshold 下的严格事实筛选 | 无预定义参数；只读取 caller threshold |
| `locate_point_in_planar_cycle()` | `INTERIOR/BOUNDARY/EXTERIOR/UNDETERMINED` | numeric/parameter tolerance、winding residual |
| `determine_segment_cycle_relation()` | `PIERCES/DOES_NOT_PIERCE/UNDETERMINED` 与证据 | 全部 numeric 与 surface settings |
| `iter_segment_cycle_relations()` | 同一环对多个线段的惰性关系流；每条与单段入口等价 | 全部 numeric 与 surface settings |
| `closest_cycle_edge()` | `Optional[ClosestCycleEdge]` | numeric tolerance |

所有带判定的公开接口只接受 `settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS`。`relation.py` 不声明数值默认值，不出现未命名 magic threshold。
