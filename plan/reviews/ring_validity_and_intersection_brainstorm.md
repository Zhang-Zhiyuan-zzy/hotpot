# 力场最小化就绪性与全分子几何分析整改方案

本报告合并原 FF-Q004 与 FF-Q005。核心业务问题不是“当前坐标是否已经漂亮”，而是：

> 当前结构能否直接交给普通局部力场最小化，还是应先重新构筑/定向解结？

原子近接、异常键长或奇怪环构象通常属于可优化应变；稳定的环—键互穿则是当前
`complexes_build` 已知不能可靠靠普通局部最小化解除的高风险结构。完整几何分析仍有价值，
但必须与“优化就绪性”分层，不能让所有诊断异常自动拒绝优化。

本文是待实施规范，不是当前代码行为的说明；本轮只修改规划文档，不修改业务代码。

## 最关键结论

1. 默认 force-field 前置门控只回答“能否可靠直接最小化”。其核心是有限键段—环内部互穿，
   另加优化器启动所必需的坐标有效性；不能用综合几何分数代替这个业务判断。
2. 完整几何分析作为可选能力，保留四类证据：
   - 原子距离、成键距离与独立的原子拥挤；
   - 任意两根显式键的相交、重合和过近；
   - 环自身的退化、折叠与环面可定义性；
   - 有限键段是否真正穿过环内部。
3. 四类证据在同一次完整分析中共享坐标快照、按需生成的原子距离表、键段表和键对事件注册表。
   环边自碰撞与目标键撞到环边都属于“键—键碰撞”；环报告和键—环报告只引用事件编号，
   不得重新计算、重复记录或重复计分。
4. “键—环互穿”只描述有限键段稳定横穿环内部面。只有延长线穿过、碰到环边、端点落在
   环面上、共面接触，都不是明确互穿。
5. 几何分析仍使用 `REASONABLE / UNREASONABLE / UNCERTAIN`；优化工作流另用
   `READY_WITHIN_SCOPE / REBUILD_RECOMMENDED / INDETERMINATE`。两套状态禁止混用。
6. 当前构筑策略把明确、稳定的环—键互穿映射为 `REBUILD_RECOMMENDED`。这不是普适拓扑
   定理；机械互锁分子等反例必须保留为显式 TODO。
7. 本分析从几何出发，不使用力场能量、键角或扭转角。所谓“距离矩阵能量”只是
   `[0,1]` 内的无量纲伪能量，用来发现明显坐标紊乱。

本文的独立展示公式统一使用 `$$ ... $$`。表格中的判定点刻意使用等宽纯文本表达，例如
`10 * epsilon_L`，以保证不支持表格内 LaTeX 的 Markdown renderer 也能完整显示；它与正文中的
数学符号 `10\epsilon_L` 含义完全相同。

完整几何分析层保持两个步骤：

~~~python
analysis = analyze_geometry(mol)
verdict = is_geo_reasonable(analysis)
~~~

局部对象（atom pair、bond pair、ring 或 bond–ring pair）仍由 `determine_geo_status()` 返回
对应的详细状态。`analyze_geometry()` / `determine_geo_status()` 只记录几何事实与证据；
`is_geo_reasonable()` 根据显式策略把事实
翻译为三态。这里的“合理”仅表示通过几何质量门，不证明结构处于低能量、动力学稳定或可合成。
它们不直接决定是否进入力场。force-field 主流程只读取独立的
`assess_optimization_readiness()` 报告。

### 本轮业务裁决：哪些检查进入热路径

用户真正需要的不是一个包罗万象的“坏几何探测器”，而是一个对局部力场最小化有直接操作意义的
门控。基于复杂度与可修复性，首版采用以下裁决：

> **结论：完整检查并非统一低成本，因此默认实现采用用户所述的第二种方案——收缩到
> ring–bond piercing；第一种方案只作为懒计算、显式调用的分析能力预留。**

| 检查 | 优化相关性 | 有空间索引/缓存后的典型成本 | 首版位置 |
|---|---|---|---|
| 坐标有限性、零长度 physical bond 等启动前提 | 优化器可能根本无法可靠启动 | `O(N+B)` | 默认门控，必须执行 |
| 有界 ring family 内的有限键段—环内部互穿 | 普通局部最小化通常不能可靠解结 | ring perception + AABB broad phase + 候选 pair 的有界窄相 | 默认门控的核心 |
| 精确的非邻接 bond–bond crossing/overlap | 原子中心力场通常缺少 bond-tube 排斥，可能不能解除 | 通常 `O(B log B+K)`，最坏 `O(B^2)` | 可选 `bond_embedding` 分析；首版不属于 readiness 门控 |
| 稀疏 atom proximity、bond length、轻量 ring shape | 多数只是可被最小化释放的应变，但分析价值高 | 通常 `O(N log N+C_A+B+sum k)` | `diagnostic`，按需调用或优化后复检 |
| 稠密距离矩阵、全键对 oracle、全部 simple cycles、全部大环曲面 | 不应为一次普通优化付出该成本 | 二次；cycle 与 triangulation 还可能指数增长 | `exhaustive`，仅离线验证 |

因此，这些几何指标不会被删除，但“预留”指保留清晰的数据结构、算法入口与按需计算路径，
并不表示首轮必须把全部能力实现完，更不表示每个候选构型都执行。对 force-field 调用方提供两个
很薄的核心入口：

~~~python
report = find_bond_ring_piercings(mol)
has_piercing = has_confirmed_bond_ring_piercing(mol)
ready = assess_optimization_readiness(mol)
~~~

第一个入口返回确认互穿、未决 pair 及覆盖信息；中间的布尔薄包装只回答“是否已有 confirmed
piercing”，其 `False` 不能代表完整 clear；最后一个入口把优化器前提和互穿证据翻译为工作流动作。
完整诊断不能反向改变默认门控，除非某一类缺陷以后通过恢复率和成本实验被明确升级。

首版业务决策树因此只有四条：

~~~text
optimizer input 无效             -> 先修输入，不启动 optimizer
confirmed ligand-skeleton piercing -> 先 rebuild/定向解结
可能存在 piercing 但数值未决    -> 最多试一次 minimization，并强制终态复检
scope 内完整 clear               -> 直接 minimization
~~~

atom proximity、异常键长、ring puckering/rank/aperture 等可选 finding 不出现在这棵树中。它们可用于
日志、候选排序和优化后质量报告，但不能借由统一分数间接改变上述动作。

## 1. 当前代码存在的问题

| 类别 | 当前或历史语义 | 后果 |
|---|---|---|
| 可选分析覆盖过窄 | 现有公开几何 API 主要围绕环—键布尔相交展开 | 无法诊断多原子拥挤或无环体系键交叉；这不等于它们都应进入默认 FF gate |
| 结果只有 `bool` | 退化输入、数值不稳定、真正不相交都可能成为 `False` | 力场流程无法区分“通过”和“无法判断” |
| 平面与非平面环语义不同 | 平面环用 point-in-polygon，非平面环用 arithmetic-centroid fan | 极小坐标扰动可以翻转结论 |
| centroid fan 会虚构面积 | 凹环的 centroid 可能在投影多边形外 | 实际位于环外的键被误报为穿环 |
| 无限直线与有限键段混淆 | 延长线命中曾参与相交语义 | 实际键段未到达环面仍可能被误报 |
| 环边碰撞归属不清 | 目标键撞环边同时可能被记作 bond–ring 和 ring-boundary 事件 | 重复计算、重复计分、状态含义不清 |
| 环边自交重复实现 | 环自身距离和线段相交与全局键对检查本质相同 | 算法与阈值容易分叉 |
| 共享环原子的键被整体跳过 | 只要一个端点属于环便认为不相交 | 外接键从共享端点离开后再次穿过环会漏报 |
| 缺少元素尺度 | 仅用整体键长尺度或纯坐标容差 | 整体缩放后的荒谬结构仍可能通过 |
| 环集合不规范 | 依赖 `networkx.cycle_basis()` 的遍历结果 | 原子编号或边插入顺序可能改变检查对象 |
| 热路径重复工作 | `find_bond_ring_intersections()` 每次重新感知 ring 并执行 rings × bonds 扫描，candidate/refinement 还会重复调用 | 构筑尝试数增加时检查成本被成倍放大 |
| 单个三角剖分具有任意性 | 非平面闭环可以对应多个离散 spanning surface | 换一条内部对角线可能改变结论 |
| 边界与内部对角线混淆 | 三角形局部命中直接代表环面事件 | 命中剖分内部边可能被误报为撞环边 |
| 现有运行时半径来源尚未冻结 | 当前元素数据与拟采用的规范文献表并不完全等价，单位也需核对 | 实施第 3.1 节的版本化半径表前，阈值不可复现，Å/pm 混用会造成灾难性错误 |
| 测试锁定历史行为 | 旧测试保留 centroid-fan 语义 | 缺陷被误当成兼容目标 |

新实现不得用兜底分支把 `INVALID_INPUT` 或 `UNCERTAIN` 静默改写成“没有问题”。

## 2. 两层产品架构

### 2.1 默认优化就绪性门控

force-field 热路径不得为了判断环—键互穿而预先构造全原子 `N x N` 矩阵或扫描全部键对。
默认流程只懒加载必要数据：

~~~text
Molecule
   └── minimal coordinate/topology snapshot
          ├── optimizer preconditions
          ├── ligand-skeleton ring perception
          ├── bond/ring AABB broad phase
          ├── one cached bounded surface representation per candidate ring
          └── finite bond–ring narrow phase
                    └── OptimizationReadinessReport
~~~

优化就绪性分为：

~~~python
class OptimizationReadiness(Enum):
    READY_WITHIN_SCOPE = "ready_within_scope"
    REBUILD_RECOMMENDED = "rebuild_recommended"
    INDETERMINATE = "indeterminate"

class ReadinessFailureClass(Enum):
    INPUT_PRECONDITION = "input_precondition"
    BOND_RING_PIERCING = "bond_ring_piercing"
    RING_EDGE_COLLISION = "ring_edge_collision"
    UNRESOLVED_RING_SURFACE = "unresolved_ring_surface"
    UNRESOLVED_RING_ORIGIN = "unresolved_ring_origin"
    INCOMPLETE_COVERAGE = "incomplete_coverage"

class RecommendedAction(Enum):
    PROCEED = "proceed"
    PROCEED_AND_REASSESS = "proceed_and_reassess"
    REJECT_PRECONDITION = "reject_precondition"
    REBUILD = "rebuild"
    REASSESS = "reassess"

class RingOrigin(Enum):
    LIGAND_SKELETON = "ligand_skeleton"
    COORDINATION_GRAPH = "coordination_graph"
    UNRESOLVED = "unresolved"

class UnresolvedAction(Enum):
    PROCEED_AND_REASSESS = "proceed_and_reassess"
    REASSESS = "reassess"
~~~

| 证据 | Readiness | 原因与动作 |
|---|---|---|
| 坐标非有限、两个不同原子坐标在输入分辨率内完全重合、physical bond 零长度 | `disposition=INVALID_INPUT`；`readiness=INDETERMINATE` | `action=REJECT_PRECONDITION`；先修输入，不称作拓扑打结 |
| 稳定、环面集合一致的有限键段–ligand-skeleton 环内部穿越 | `REBUILD_RECOMMENDED` | 当前构筑业务中，普通局部最小化不能可靠解开；先重新构筑或定向开环修复 |
| confirmed piercing 的 ring origin 因 edge semantic 不明而无法归类 | `INDETERMINATE` | `REASSESS`：先解析拓扑语义；不得默认为 ligand-skeleton 并直接拒绝或放行 |
| 环折叠导致 surface undefined/disagreement，或数值擦边，但输入满足 optimizer 前提 | `INDETERMINATE` | 默认 `PROCEED_AND_REASSESS`：允许一次局部最小化释放形变，终态必须重新检查；strict policy 可改为 `REASSESS` |
| ring perception、相关 ring–bond pair 或 surface/triangle 评价因预算而未完成 | `INDETERMINATE` | `REASSESS`：提高预算或改用离线检查；不能把已检查部分当作 clear |
| 所有相关 bond–ring pair 明确 clear，且 `coverage_complete_within_scope=True` | `READY_WITHIN_SCOPE` | 可以进入普通力场最小化；不保证声明范围外无问题，也不保证最终几何合理 |
| 原子近接、异常键长、普通环形变 | 不改变 core readiness | 记录为可选诊断；通常先最小化，再复检 |
| 非环的精确 bond–bond crossing/overlap | 默认 core 不计算；可选分析只报告，不改变 readiness | 它可能难以由原子中心力场解除，但尚未满足升级为默认拒绝条件的实证要求 |

这里使用 `REBUILD_RECOMMENDED` 而不是 `CANNOT_OPTIMIZE`：一根开放链理论上可能从环孔退出，
单次 bond–ring surface hit 也不是 knot/link invariant；它只是针对当前构筑流程非常强的
“局部最小化不可靠”证据。`READY_WITHIN_SCOPE` 也只表示“在声明的 ring scope、ring-family、
ring-size 与 surface budget
内没有确认阻塞”，不是完整拓扑证明。

`OptimizationReadinessReport` 始终同时携带 disposition、readiness 与 action。优先级固定为：
`INVALID_INPUT` disposition 的 `REJECT_PRECONDITION` 高于 readiness；只有 disposition 为
`EVALUATED` 时才按 readiness 执行动作。

`READY_WITHIN_SCOPE` 的可执行不变量固定为：

~~~text
disposition == EVALUATED
and scan_complete
and coverage_complete_within_scope
and readiness_exception_pair_keys is empty
and confirmed_piercings is empty
~~~

只有全部条件同时成立才能返回 `READY_WITHIN_SCOPE / PROCEED`。它允许
`global_coverage_complete=False`，因为后者只声明所选 ring family 之外没有保证；report 必须把该
限制暴露给调用方。任一字段组合违反上述不变量都应由构造器拒绝，而不是留给下游猜测。
这里的 `readiness_exception_pair_keys` 收录可能遮蔽 interior piercing 的 surface/numeric failure，
以及虽已解析为非 piercing、但要求受监督最小化的 exact ring-edge collision。稳定的
`NONPIERCING_ENDPOINT_CONTACT` / `NONPIERCING_COPLANAR_CONTACT` 只进入 diagnostics，不破坏上述
ready 不变量；`BOUNDARY_NUMERIC_BAND` 仍属于 readiness exception。

“环形状奇怪”本身不进入 core blocker。只有当某个外部 physical bond 的 AABB 与该环仍可能
相交、而该环的 surface 又无法稳定构造时，才因为“不能排除核心互穿风险”得到
`INDETERMINATE`；若 broad phase 已严格证明不存在相关 bond–ring candidate，该环可直接留给
最小化处理，不因 rank、puckering 或 aperture warning 阻断。

### 2.2 可选的全分子几何分析

`analyze_geometry(mol)` 建立短生命周期、按需计算的 `_MoleculeGeometryWorkspace`；
`determine_geo_status()` 只接受局部 typed target，不接受 `Molecule`：

~~~text
Molecule
   │
   └── coordinate/topology snapshot
          ├── atom indices, elements, coordinates, radii
          ├── adjacency and graph-distance classes
          ├── lazy atom spatial index and pair metrics
          ├── lazy explicit bond segment table and AABBs
          ├── canonical ring set
          ├── memoized segment-pair relations
          └── memoized ring-surface ensembles
                    │
                    ├── AtomDistanceTopologyGate
                    ├── BondBondCollisionGate
                    ├── RingGeometryGate
                    └── BondRingPiercingGate
                              │
                              └── MoleculeGeoStatus
~~~

这里的“并列”只指完整分析中的证据语义，不代表四项都进入默认 force-field gate，也不强制
多线程执行。计算顺序可为缓存复用而调整，但不能让一个门控的业务 verdict 改写另一个门控的
客观事实。未请求的表和 cache 不得构造。

四个通道的唯一职责如下：

| 通道 | 负责的事实 | 不负责的事实 |
|---|---|---|
| Atom distance/topology | 成键过短/过长、非键原子中心异常接近、多原子拥挤、近而未成键的嫌疑 | 键段之间是否穿过 |
| Bond–bond | 任意两根有限显式键段是否相交、正长度重合、异常靠近 | 是否穿过某个环的内部面 |
| Ring | 环本身是否退化、塌缩；环面集合能否稳定构造 | 重新计算环边之间的碰撞 |
| Bond–ring | 有限目标键段是否稳定横穿环内部面 | 目标键撞到环边；只有延长线命中 |

### 2.3 评价处置状态与几何分析三态

“是否能评价”和“评价后是否合理”是两个正交维度：

~~~python
class EvaluationDisposition(Enum):
    EVALUATED = "evaluated"
    NOT_APPLICABLE = "not_applicable"
    INVALID_INPUT = "invalid_input"

class GeometryVerdict(Enum):
    REASONABLE = "reasonable"
    UNREASONABLE = "unreasonable"
    UNCERTAIN = "uncertain"
~~~

| Verdict | 含义 | 在可选分析中的用途 |
|---|---|---|
| `REASONABLE` | 所有已启用诊断都有明确安全裕量 | 记录为 clean geometry |
| `UNREASONABLE` | 至少一个稳定几何缺陷越过该诊断的 hard boundary | 标记缺陷；是否阻止优化仍由 readiness policy 决定 |
| `UNCERTAIN` | 没有 hard failure，但存在 warning、数值不稳定或模型未覆盖 | 保留证据并用于排序，不能静默改写为通过 |

`NOT_APPLICABLE` 不是第四种 verdict。例如同一根键和自身不构成 bond pair，环自身的边也
不构成外部 bond–ring pair。standalone N/A status 传给 `is_geo_reasonable()` 时抛出明确的
`NonApplicableGeometryTargetError`；`INVALID_INPUT` 映射为 `UNCERTAIN` 证据，并由 workflow
adapter 作操作性拒绝。`GeometryVerdict` 不得隐式转成 `bool`；调用方必须显式比较枚举值。

### 2.4 分析级别与计算预算

readiness 只有一条固定的 `ring_piercing` 核心路径；不存在会因一个字符串参数而悄悄扩大拒绝
范围的第二个 readiness level。可选几何分析按行向下累积能力，但不反向改变 core readiness。

| Level | 内容 | 典型复杂度 | 是否进入默认 FF 热路径 |
|---|---|---|---|
| `ring_piercing` | 输入前提 + ligand-skeleton bond–ring piercing；AABB、lazy surface、工作量上限 | ring perception + `O((R+B) log(R+B))` broad phase + “各候选 ring–bond pair 的 surface 数 × 每 surface 三角形数”之和 | 是 |
| `bond_embedding` | physical bond 的精确 crossing/overlap，忽略纯 near-clearance | broad phase 通常 `O(B log B + K)`；最坏 `O(B²)` | 否；是独立、显式请求的分析通道 |
| `diagnostic` | 累积 `bond_embedding`，再加空间索引原子近邻/键长、候选 bond-pair near-clearance、轻量 ring metrics，并可复用 core piercing 报告 | 通常 `O(N log N + C_A + B log B + K)`；最坏二次 | 否，显式请求或优化后复检 |
| `exhaustive` | 累积诊断，再加稠密原子矩阵、全 pair oracle、扩展 ring family、全部允许环面 ensemble | `O(N²+B²)`；cycle enumeration 与 triangulation 还可能指数增长 | 仅离线审查与测试 |

其中 `N/B/R/k/K/C_A` 分别为原子数、键数、环数、环大小、候选键对数和空间索引返回的候选
atom pair 数。默认窄相位的实际次数必须直接记录为 `triangle_tests_used`，不能只由渐进复杂度推断。
凸环的 vertex triangulation 数为 Catalan 数 `C_(k-2)`：8 元环 132 个，10 元环 1430 个，
12 元环 16796 个。假设 20 个八元环、300 根键，朴素全曲面扫描约需 475 万次
segment–triangle 判断；若构筑循环重复 100 次即约 4.75 亿次，因此 exhaustive 不能进入默认路径。

默认 `edge_shortest_cycle_family` 若按每条边执行一次无权 BFS，粗略上界为
`O(B * (N+B) + Z)`，其中 `Z` 是实际输出的并列最短路径总量；典型稀疏分子远低于全部 simple
cycles，但高度对称图的 `Z` 仍可能膨胀，所以 `max_ring_count` 和 incomplete coverage 语义不可省略。

首版性能决策必须以实际 benchmark 为依据：分别记录 `ring_piercing` gate、一次 Open Babel/UFF
最小化和整个 stochastic build epoch 的 wall time。全键对的 `bond_embedding` 不是严格拓扑判定，
只是 atom-centered force field 的 graph-embedding blind spot；即使未来性能、oracle 一致性和构筑
成功率实验全部达标，也必须通过一次单独的 API/策略评审，才能从诊断通道升级为 readiness
条件。首版默认严格保持 `ring_piercing`。

### 2.5 证据唯一所有权

分子级维护一个按稳定索引规范化的事件注册表：

~~~text
atom pair key = (min(atom_i, atom_j), max(atom_i, atom_j))
bond pair key = (min(bond_a, bond_b), max(bond_a, bond_b))
ring key      = canonical cyclic atom-index tuple
bond-ring key = (bond_id, ring_key)
~~~

canonical bond-pair event registry 是所有键段—键段事实的唯一 owner，但“唯一 owner”不等于
默认路径必须先运行一次全局 Bond–bond gate：

- `ring_piercing` 只为 broad phase 保留的 `(target_bond, ring_edge)` 懒计算并注册相应关系；
- `bond_embedding/diagnostic` 才把候选扩展到全体适用 physical bond pairs；
- 后续通道请求同一个规范化 pair key 时只读取缓存，不再次做窄相计算。

其他通道只附加来源标签或引用 `event_id`：

- 非相邻环边互相穿过：Bond–bond finding 所有；`RingGeoStatus` 引用它；
- 目标键撞到环边：Bond–bond finding 所有；`BondRingGeoStatus` 引用它；
- 目标键穿过环面内部：Bond–ring finding 独立所有；
- 一个原子同时出现在多组近接中：每个 atom-pair finding 只创建一次，原子拥挤状态聚合这些
  finding，不复制它们。

因此“同一症状出现于多个视图”不会增加额外罚分。统一总分使用最坏项而不是求和。

## 3. 统一符号、容差与评分

第 3–8 节同时保存核心 piercing kernel 所需定义和可选完整诊断的设计细节。首版实现时，只有
坐标/线段/环面容差与第 7 节进入默认门控；原子拥挤、全键对、环形状评分和统一伪能量可以后置，
不得因为文档已经定义便在热路径中被隐式执行。

### 3.1 原子与拓扑符号

原子 `i` 的位置、共价半径和 van der Waals 半径分别为
`\mathbf r_i`、`r_i^{\mathrm{cov}}` 和 `r_i^{\mathrm{vdW}}`。后文键和环公式中的
`\mathbf p_i` 与这里是同一个坐标，即 `\mathbf p_i=\mathbf r_i`。对任意原子对：

$$
d_{ij}=\left\|\mathbf r_i-\mathbf r_j\right\|,
\qquad
c_{ij}=r_i^{\mathrm{cov}}+r_j^{\mathrm{cov}},
\qquad
v_{ij}=r_i^{\mathrm{vdW}}+r_j^{\mathrm{vdW}}.
$$

定义两种无量纲距离：

$$
x_{ij}=\frac{d_{ij}}{c_{ij}},
\qquad
y_{ij}=\frac{d_{ij}}{v_{ij}}.
$$

- `A_{ij}=1`：拓扑中存在显式键，否则为 0；
- `M_{ij}=1`：该显式键是金属—配体键；
- `g_{ij}`：去除金属配位边后的配体共价图最短路；不连通为 `+\infty`；
- `g=1`、`g=2`、`g=3` 分别表示 1–2、1–3、1–4 原子关系；
- `g\ge4` 或 `+\infty` 才称为 remote atom pair。
- `component_relation` 区分同一共价组分、不同组分、同一配位组装体和周期像；它必须进入
  finding，不能把 `g=+infinity` 一律解释为错误。

`x_{ij}` 主要发现原子中心异常接近和成键尺度错误；`y_{ij}` 主要描述非键空间拥挤。
共价半径是经验性成键尺度，不是原子核半径，因此本文不再使用“核碰撞”术语。二者不可混用阈值。

坐标快照的输入契约固定为 Å。若 reader 的源格式声明其他长度单位，必须在创建 snapshot 前进行
一次显式换算，并在 report 中保存 `coordinate_unit="angstrom"` 与源单位；若源单位未知，依赖
绝对长度或元素半径的结果必须为 `INVALID_INPUT`，不能猜测 Å、nm 或 bohr。

v0 的规范数据源必须冻结为：共价半径取 Cordero et al. (2008)，vdW 半径取 Alvarez (2013)。
构建脚本把源表一次性换算为 Å，并把文献 DOI、源表版本/校验和、换算因子和缺失元素清单写入
随包 metadata；运行时只读取这一份版本化表。不得把 Open Babel、`periodictable` 或其他库中同名
但来源不明的半径静默混用。元素缺值时保留精确、纯坐标的相交判定，但依赖 `x_ij/y_ij/q_bb`
的结果必须为 `RADIUS_UNAVAILABLE` 或 `UNCERTAIN`。

### 3.2 有限键段符号

两根显式键 `\alpha=(a_0,a_1)` 和 `\beta=(b_0,b_1)` 的有限中心线为：

$$
\mathbf b_\alpha(t)
=
\mathbf p_{a_0}+t\mathbf d_\alpha,
\qquad
\mathbf d_\alpha=\mathbf p_{a_1}-\mathbf p_{a_0},
\qquad 0\le t\le1,
$$

$$
\mathbf b_\beta(u)
=
\mathbf p_{b_0}+u\mathbf d_\beta,
\qquad
\mathbf d_\beta=\mathbf p_{b_1}-\mathbf p_{b_0},
\qquad 0\le u\le1.
$$

最近点参数与最近距离分别为：

$$
(t^*,u^*)
=
\underset{0\le t,u\le1}{\operatorname{argmin}}
\left\|
\mathbf b_\alpha(t)-\mathbf b_\beta(u)
\right\|,
$$

$$
d_{bb}
=
\left\|
\mathbf b_\alpha(t^*)-\mathbf b_\beta(u^*)
\right\|.
$$
方向横向程度定义为：

$$
\eta_{bb}
=
\frac{
\left\|\mathbf d_\alpha\times\mathbf d_\beta\right\|
}{
\left\|\mathbf d_\alpha\right\|
\left\|\mathbf d_\beta\right\|
}.
$$

在最近点处沿键线性插值元素共价半径：

$$
r_\alpha(t^*)
=
(1-t^*)r_{a_0}^{\mathrm{cov}}
+t^*r_{a_1}^{\mathrm{cov}},
$$

$$
r_\beta(u^*)
=
(1-u^*)r_{b_0}^{\mathrm{cov}}
+u^*r_{b_1}^{\mathrm{cov}},
$$

$$
\rho_{bb}=r_\alpha(t^*)+r_\beta(u^*),
\qquad
q_{bb}=\frac{d_{bb}}{\rho_{bb}}.
$$

`d_{bb}` 决定是否发生精确中心线事件；`q_{bb}` 只为“尚未相交但异常靠近”提供元素尺度。
若半径缺失，精确相交仍可判定，`q_{bb}` 相关结论为 `UNCERTAIN`。

先定义两根键中心线所在的无限直线 `\mathcal L_\alpha`、`\mathcal L_\beta`，以及对称的
共线残差：

$$
\delta_{\mathrm{col}}
=
\max\left[
\max_{k\in\{a_0,a_1\}}
\operatorname{dist}(\mathbf p_k,\mathcal L_\beta),\quad
\max_{k\in\{b_0,b_1\}}
\operatorname{dist}(\mathbf p_k,\mathcal L_\alpha)
\right].
$$

只有 `eta_bb<=eta_parallel` 且 `delta_col<=epsilon_L` 才是稳定共线候选；
`epsilon_L<delta_col<=10 * epsilon_L` 属于不稳定带。

随后将两段投影到
`\mathbf e=\mathbf d_\alpha/\|\mathbf d_\alpha\|` 上。两个闭区间的正交投影重叠长度为：

$$
L_{\mathrm{overlap}}
=
\max\left(
0,
\min(I_\alpha^{\max},I_\beta^{\max})
-
\max(I_\alpha^{\min},I_\beta^{\min})
\right).
$$

### 3.3 环符号

环按拓扑顺序写为：

$$
R=(\mathbf p_0,\mathbf p_1,\ldots,\mathbf p_{n-1}),
\qquad
\mathbf e_i=[\mathbf p_i,\mathbf p_{(i+1)\bmod n}],
$$

$$
\ell_i=\|\mathbf p_{(i+1)\bmod n}-\mathbf p_i\|,
\qquad
s_R=\operatorname{median}_i(\ell_i).
$$

中心化坐标矩阵的奇异值满足：

$$
\sigma_1\ge\sigma_2\ge\sigma_3\ge0,
\qquad
q_{\mathrm{rank}}=\frac{\sigma_2}{\sigma_1},
\qquad
q_{\mathrm{nonplanar}}=\frac{\sigma_3}{\sigma_2}.
$$

只有 `\sigma_1`、`\sigma_2` 都离开数值零区后才计算比值。将环投影到 best-fit plane；
若投影边界是 simple polygon，定义：

$$
q_{\mathrm{aperture}}
=
\frac{
4n\tan(\pi/n)A_{\mathrm{proj}}
}{
P_{\mathrm{proj}}^2
}.
$$

对 simple projected polygon，`0<q_{\mathrm{aperture}}\le1`，规则平面 `n` 元环取 1。
它只是投影开孔指标，不是三维孔径或能量。

### 3.4 符号的定义域与取值范围

后续状态表中的“关键判别要点”只描述可观察的几何直觉；“明确判定公式/判定点”才是程序必须
执行的规则。公式不得另造同义符号。各核心符号的定义域固定如下：

| 符号 | 取值范围 | 量纲/含义 |
|---|---|---|
| `d_ij, c_ij, v_ij, d_bb, rho_bb, d_boundary` | `[0,+infinity)` | Å；距离或元素半径和 |
| `x_ij=d_ij/c_ij`, `y_ij=d_ij/v_ij`, `q_bb=d_bb/rho_bb` | `[0,+infinity)` | 无量纲元素归一化距离 |
| `t,u` | 有限线段上为 `[0,1]`；无限直线交点参数可为任意实数 | 无量纲参数 |
| `lambda_0,lambda_1,lambda_2` | 和为 1；点在闭三角形内时各自在 `[0,1]` | 重心坐标 |
| `eta_bb, eta_bs` | `[0,1]` | 无量纲横向程度；0 为平行，1 为正交 |
| `q_rank, q_nonplanar` | 分母有效时为 `[0,1]` | 无量纲 SVD 比值 |
| `q_aperture` | simple projected polygon 时为 `(0,1]` | 无量纲投影开孔指标 |
| `C_i`, 任一 `E_*`, 任一可量化 risk | `[0,1]` | 无量纲伪能量/风险；越大越异常 |
| `S_atom, S_ring, S_geo` | `[0,1]` 或 `None` | 无量纲质量分；越大越干净，未解析为 `None` |
| `mu_num` | `[0,+infinity]` | 无量纲数值裕量；越大越远离判定边界 |

若分母缺失、非正或落入数值零区，相应 ratio 不定义，必须进入明确的 unavailable/unresolved
状态，不能以 0 代替。

### 3.5 数值容差与稳定带

为保证同一个键对无论从 molecule、ring 还是 bond–ring 入口请求都得到完全相同的缓存结果，
显式键对统一使用全分子的非零 physical bond 长度中位数 `s_mol`；环面构造和
segment–triangle 谓词使用该环的 `s_R`。二者统称当前上下文尺度 `s_context`：

$$
\epsilon_L
=
\max\left(
\epsilon_{\mathrm{abs}}+\epsilon_{\mathrm{rel}}s_{\mathrm{context}},
\epsilon_{\mathrm{coord}}
\right).
$$

| 符号/配置名 | 默认值 | 范围/量纲 | 明确用途 |
|---|---:|---|---|
| `epsilon_abs` / `absolute_length_tolerance` | `1.0e-8` | Å | 长度底噪 |
| `epsilon_rel` / `relative_length_tolerance` | `1.0e-7` | 无量纲 | 相对长度底噪 |
| `epsilon_coord` / `coordinate_resolution` | `0.0` | Å | 输入坐标的已知分辨率；文件 reader 应按实际小数位设置 |
| `epsilon_L` | 动态 | Å | 点、线段和环边界距离 |
| `epsilon_A=max(epsilon_abs²+epsilon_rel*s_context², epsilon_coord²)` | 动态 | Å² | 投影与三角面积 |
| `epsilon_t=epsilon_L/max(\|\mathbf d_\alpha\|,epsilon_L)` | 动态 | 无量纲 | 第一线段参数 |
| `epsilon_u=epsilon_L/max(\|\mathbf d_\beta\|,epsilon_L)` | 动态 | 无量纲 | 第二线段参数 |
| `epsilon_b` / `barycentric_tolerance` | `1.0e-8` | 无量纲 | 三角形重心坐标 |
| `eta_parallel` / `parallel_sine_maximum` | `1.0e-6` | `[0,1]` | 稳定近乎平行的上界 |
| `eta_transverse` / `transverse_sine_minimum` | `1.0e-4` | `[0,1]` | 稳定横向关系的下界 |
| `numeric_warning_multiplier` | `10` | 正数 | 从容差带到稳定带的倍数 |
| `delta_planar` / `planarity_ratio` | `0.05` | `[0,1]` | 仅区分 planar/puckered |

方向谓词必须使用同一组符号：

- `eta <= eta_parallel`：稳定近乎平行；
- `eta_parallel < eta < eta_transverse`：方向无法稳定归类，`UNCERTAIN`；
- `eta >= eta_transverse`：稳定横向。

线段参数必须按三段处理。以 `t` 为例：

- `min(t,1-t) <= epsilon_t`：端点带；
- `epsilon_t < min(t,1-t) <= 10 * epsilon_t`：端点不稳定带；
- `min(t,1-t) > 10 * epsilon_t`：稳定内部。

共线重叠同样分三段：

- `L_overlap <= epsilon_L`：只有端点接触或无正长度重叠；
- `epsilon_L < L_overlap <= 10 * epsilon_L`：数值不稳定；
- `L_overlap > 10 * epsilon_L`：稳定正长度重叠。

`epsilon_coord=0` 时，“稳定”只表示相对于浮点算法容差稳定；它不代表输入实验坐标具有
`1e-8 Å` 精度。读取保留三位小数的 MOL2/PDB 类文件时，应由 reader 或调用方提供约
`5e-4 Å` 的量化半宽，而不能把文件舍入误差当作真实几何差异。

这些数值是 v0 工程默认值，不是化学文献常数。等号归属必须按本文逐项测试。

### 3.6 有界风险函数

对“越小越坏”的量，hard boundary 为 `h`、warning-free boundary 为 `w`，`h<w`：

$$
r_{\downarrow}(x;h,w)
=
\begin{cases}
1, & x\le h,\\
\left(\dfrac{w-x}{w-h}\right)^2, & h<x<w,\\
0, & x\ge w.
\end{cases}
$$

对“越大越坏”的量，warning boundary 为 `w`、hard boundary 为 `h`，`w<h`：

$$
r_{\uparrow}(x;w,h)
=
\begin{cases}
0, & x\le w,\\
\left(\dfrac{x-w}{h-w}\right)^2, & w<x<h,\\
1, & x\ge h.
\end{cases}
$$

两者取值均严格在 `[0,1]`。等于 hard boundary 时风险为 1；等于 warning-free boundary
时风险为 0。

对当前状态实际使用的数值退化谓词 `f_k`，令 `b_k` 为分类边界、`epsilon_k>0` 为同量纲
分辨率：

$$
\mu_{\mathrm{num}}
=
\min_k
\frac{|f_k-b_k|}{\epsilon_k}.
$$

- `mu_num <= 1`：数值未解析；
- `1 < mu_num <= 10`：结果可算但不稳定；
- `mu_num > 10`：具有稳定裕量。

谓词集合为空时定义 `mu_num=+infinity`。`mu_num` 只用于浮点符号、退化性、端点归属和方向
归属等数值边界；经验性的 hard/warning 阈值按表格的闭区间直接判定，不因恰好等于阈值而改写为
数值未解析。不得把不同量纲的原始数值直接放在同一个 `min` 中。

## 4. 原子距离、成键距离与原子拥挤

### 4.1 检查范围

该门控只使用坐标、元素半径和图距离，不使用键角、扭转角或物理势能。它同时提供三种证据：

1. `BondLengthGeometryState`：显式键是否过短或过长；
2. `AtomCrowdingGeometryState`：单个非键原子中心严重重叠与多个原子围绕同一原子挤压；
3. `TopologyDistanceAdvisory`：近距离却未成键的拓扑嫌疑。

本节的 hard/`UNREASONABLE` 只表示“完整几何分析确认存在严重距离异常”，不自动映射为
`REBUILD_RECOMMENDED`。除优化器前提失效外，初始 atom-distance finding 默认允许先做最小化。

公式上可把 `d_ij/x_ij/y_ij` 看成距离矩阵；实现上 `diagnostic` 应用 cell list/KD-tree 或分块算法
只生成阈值内 atom-pair table，避免持有多个稠密 `N x N` 数组。只有 `exhaustive` 测试/oracle
才构造完整矩阵。每个原子对先按拓扑关系分流，不得在多个互斥主状态中重复计罚。

### 4.2 v0 距离阈值

#### 显式键

| 通道 | severe-short `h_-` | warning-short `w_-` | warning-long `w_+` | severe-long `h_+` |
|---|---:|---:|---:|---:|
| 普通显式共价键，使用 `x_ij` | `0.60` | `0.75` | `1.35` | `1.60` |
| 显式金属—配体键，使用 `x_ij` | `0.50` | `0.65` | `1.55` | `1.90` |

普通共价键的两个 severe boundary 暂作 v0 hard gate。金属—配体行只提供异常程度排序，不提供
通用 hard verdict；元素、氧化/自旋态、配位数、桥联、hapticity 与键表示都会改变合理距离。
只有注册了 pair/coordination-specific profile 后，金属—配体距离才可升级为 hard gate。

#### 非键原子与局部拥挤

| 通道 | hard `h` | warning-free `w` | 适用范围 |
|---|---:|---:|---|
| 非键原子中心严重重叠，使用 `x_ij` | `0.60` | `0.80` | 所有 `A_ij=0` 原子对 |
| same-component remote 非键拥挤，使用 `y_ij` | `0.45` | `0.65` | `g_ij>=4`；排除 metal–donor 特例 |
| same-component 1–4 非键拥挤，使用 `y_ij` | `0.35` | `0.55` | `g_ij=3`；风险上限 `0.50` |
| 含 H 的 remote/1–4 非键拥挤，使用 `y_ij` | `0.35` | `0.50` | 这是 H-containing profile，不等同于氢键识别 |
| different-component 非键拥挤，使用 `y_ij` | `0.35` | `0.55` | 离子对、溶剂化和 host–guest；风险上限 `0.50` |
| 多原子拥挤负载 `C_i` | ranking-high `0.75` | ranking-low `0.40` | 至少两个 warning-range 邻居；首版始终只作 `UNCERTAIN` |

`C_i` 的两个界限含义与普通 low-is-bad 指标相反：`C_i<0.40` clear，
`0.40<=C_i<0.75` 为较低拥挤排序，`C_i>=0.75` 为较高拥挤排序。多体聚合本身首版不产生
hard failure；明确 hard 仍由单个严重 atom-pair overlap 所有。

#### 近而未成键的嫌疑

| 通道 | full-suspicion `m_full` | warning-free `m_clear` | 最大风险 |
|---|---:|---:|---:|
| remote 普通重原子 missing-edge，使用 `x_ij` | `1.10` | `1.25` | `0.50` |
| 未成键 metal–donor 接触，使用 `x_ij` | `1.20` | `1.50` | diagnostic only |

以上均是待真实有机物、离子对、氢键和金属配合物数据校准的保守工程值。

### 4.3 成键距离状态

| `BondLengthGeometryState` | 关键判别要点 | 明确判定公式/判定点 | Verdict |
|---|---|---|---|
| `DEGENERATE_BOND` | 一根 physical atom–atom bond 的两个端点重合，有限键段本身不存在 | `d_ij<=epsilon_L`，优先于所有元素尺度状态 | `UNREASONABLE`；readiness 另标 input precondition |
| `REGULAR` | 普通共价键的两个原子间距与元素尺寸相符 | `A_ij=1, M_ij=0` 且 `0.75<=x_ij<=1.35` | `REASONABLE` |
| `SUSPICIOUSLY_SHORT` | 普通共价键明显压缩，但未越过 severe boundary | `A_ij=1, M_ij=0` 且 `0.60<x_ij<0.75` | `UNCERTAIN` |
| `SUSPICIOUSLY_LONG` | 普通共价键明显拉长，但未越过 severe boundary | `A_ij=1, M_ij=0` 且 `1.35<x_ij<1.60` | `UNCERTAIN` |
| `TOO_SHORT` | 普通共价键的两个原子中心严重挤压 | `A_ij=1, M_ij=0` 且 `x_ij<=0.60` | `UNREASONABLE` |
| `TOO_LONG` | 拓扑声称存在普通共价键，但两个原子在空间上相距过远 | `A_ij=1, M_ij=0` 且 `x_ij>=1.60` | `UNREASONABLE` |
| `COORDINATION_DISTANCE_DIAGNOSTIC` | 金属—配体距离位于或接近通用经验带，但上下文尚未标定 | `M_ij=1`；记录 `x_ij` 及 `0.50/0.65/1.55/1.90` 相对位置 | 带内不改变 verdict；带外 `UNCERTAIN` |
| `METAL_METAL_UNCALIBRATED` | 金属—金属显式键缺乏统一可辩护距离区间 | `A_ij=1` 且两个原子均为金属 | `UNCERTAIN` |
| `RADIUS_UNAVAILABLE` | 元素尺度缺失，无法归一化 | `c_ij` 缺失或非正 | `UNCERTAIN` |

“显式键距离过远”和“与远距离原子成键”是同一 `TOO_LONG` 事实，不创建两个 finding。

### 4.4 独立的原子拥挤状态

对可纳入拥挤的非键原子对，依据元素、图距离和 component relation 选择
`(h_y,w_y,a_ij)`。`a_ij` 是该 profile 的最大风险权重：same-component remote 为 1，
1–4 与 different-component 为 0.5，diagnostic-only pair 为 0。定义：

$$
r_{ij}^{\mathrm{crowd}}
=
a_{ij}\,
r_{\downarrow}(y_{ij};h_{y,ij},w_{y,ij}).
$$

原子 `i` 的 warning-range 拥挤邻居集合为：

$$
\mathcal J_i
=
\left\{
j\mid
A_{ij}=0,\quad
a_{ij}>0,\quad
y_{ij}<w_{y,ij}
\right\}.
$$

局部多原子拥挤负载定义为：

$$
C_i
=
1-
\prod_{j\in\mathcal J_i}
\left(1-r_{ij}^{\mathrm{crowd}}\right),
\qquad
0\le C_i\le1.
$$

这个有界聚合能让多个中等近接共同提高风险，又不会随分子原子数无界增长。它是伪能量，
不是碰撞概率。

| `AtomCrowdingGeometryState` | 关键判别要点 | 明确判定公式/判定点 | Verdict |
|---|---|---|---|
| `EXACT_COORDINATE_OVERLAP` | 两个不同原子在当前输入分辨率内占据同一点，优化方向可能奇异 | `A_ij=0` 且 `d_ij<=epsilon_L` | `UNREASONABLE`；readiness 另标 input precondition |
| `CLEAR` | 所有可检查的非键原子都有明确间隙；任一原子周围也没有近接簇 | 所有 `x_ij>=0.80`，且所有适用 `y_ij>=w_y(i,j)`，并且 `C_i<0.40` | `REASONABLE` |
| `SINGLE_NEAR_CONTACT` | 一对非键原子的中心异常接近，但尚未越过 severe boundary | `A_ij=0` 且 `0.60<x_ij<0.80`，或适用 profile 中 `h_y(i,j)<y_ij<w_y(i,j)` | `UNCERTAIN` |
| `SEVERE_NONBONDED_ATOM_OVERLAP` | 两个没有拓扑键的原子中心已异常接近 | `A_ij=0` 且 `x_ij<=0.60` | `UNREASONABLE` |
| `SEVERE_NONBONDED_CROWDING` | 一对 same-component remote 非键原子虽未达到上一行阈值，但空间占据已严重互侵 | `A_ij=0`、`g_ij>=4`、`x_ij>0.60`、不属于 metal–donor 特例，且适用 `y_ij<=h_y(i,j)` | `UNREASONABLE` |
| `CONTEXT_DEPENDENT_CLOSE_CONTACT` | 1–4 或不同组分原子进入 severe profile，但离子对、host–guest 或局部构象可能解释该距离 | `a_ij=0.50` 且 `y_ij<=h_y(i,j)`，同时 `x_ij>0.60` | `UNCERTAIN` |
| `MULTI_ATOM_CROWDING_WARNING` | 至少三个原子形成局部挤压簇，多个中等近接共同出现 | `|\mathcal J_i|>=2` 且 `0.40<=C_i<0.75` | `UNCERTAIN` |
| `MULTI_ATOM_CROWDING` | 多个非键原子同时挤向同一原子；这是需优先优化的高风险簇，但方向分布尚未证明它物理冲突 | `|\mathcal J_i|>=2` 且 `C_i>=0.75` | `UNCERTAIN`，排序高于上一行 |
| `RADIUS_UNAVAILABLE` | 至少一个相关元素缺少 vdW/covalent 半径 | 应评价 pair 的 `c_ij` 或 `v_ij` 不可用 | `UNCERTAIN` |

补充规则：

- `g=2` 的 1–3 pair 不进入 vdW 拥挤聚合，但仍接受 `x_ij` 的 near/severe 原子中心距离检查；
  这只评价距离结果，不反推出键角。
- `g=3` 的 1–4 pair 使用单独的 relaxed vdW profile，并把该通道风险限制为 `<=0.50`；
  这只检查实际空间接近，不引入扭转势。
- 普通已成键 pair 只进入成键距离通道，不再进入非键拥挤。
- 未显式成键的 metal–donor 近接默认只给 advisory；但 `x_ij<=0.60` 的原子中心严重重叠仍是 hard。
- `MULTI_ATOM_CROWDING` 必须返回中心原子、全部参与原子和每个 pair 的 `d_ij/x_ij/y_ij`，
  而不是只返回一个聚合分数。
- 单对状态与多原子聚合状态可以同时存在；二者引用相同 atom-pair IDs，总风险取最大值，
  不因同一近接被重复相加。`C_i` 不是概率；在引入包含邻居—邻居距离的 compact-cluster
  证据并完成数据校准前，多体聚合不得单独升级为 hard。
- 非键 atom pair 的唯一主状态优先级为：exact coordinate overlap → 严重原子中心重叠 → profile-specific severe contact →
  near contact → clear；missing-edge/coordination 只作为正交 advisory，不替换主状态。

### 4.5 距离—拓扑不一致提示

| `TopologyDistanceAdvisory` | 关键判别要点 | 明确判定公式/判定点 | Verdict |
|---|---|---|---|
| `MISSING_EDGE_SUSPECTED` | 共价图中很远或不连通的普通重原子靠近到近似成键尺度 | `A_ij=0`、`g_ij>=4` 或 `+\infty`、两原子均为非金属非 H、`0.60<x_ij<1.25` | `UNCERTAIN` |
| `POSSIBLE_COORDINATION_CONTACT` | 金属和潜在 donor 很近，但距离不能证明漏了一根配位键 | `A_ij=0`、恰有一个金属、非金属满足既有 donor 规则、`x_ij<1.50` | diagnostic only |
| `NO_TOPOLOGY_DISTANCE_ISSUE` | 没有近而未连的 remote pair | 所有适用 pair `x_ij>=1.25` | 不改变 verdict |

`MISSING_EDGE_SUSPECTED` 不得自动加键，也不能单独产生 hard failure。离子对、氢键、多中心键和
金属第二配位层都是反例。已显式成键的 H 不推断第二根键；degree 0 的游离 H 才能进入
missing-edge 候选。

### 4.6 由归一化 pair-distance matrix 导出的无量纲伪能量

先定义互不含糊的候选集合：

- `mathcal B`：半径可用的普通显式共价键，以及具有已注册 context-specific profile 的金属—配体键；
- `mathcal C`：半径可用的全部非键原子对，用于严重原子中心重叠；
- `mathcal V`：通过 4.2/4.4 拓扑、component、元素 mask 的 vdW 拥挤原子对；其 `a_ij` 已包含
  1–4 与 different-component 的风险上限；
- `mathcal L`：通过 4.5 条件的 remote 普通重原子 missing-edge 候选。

这里的“matrix”描述完整数学关系，不要求运行时分配稠密矩阵；`diagnostic` 只对空间索引和拓扑
筛出的 pair 求值，`exhaustive`/测试 oracle 才物化全矩阵。该量只汇总“成键过短、成键过长、
非键过近、近而未成键与局部拥挤”，不包含键角、扭转角、力场项或任何真实能量。

定义六类 pair/channel 风险：

$$
E_{\mathrm{bond,short}}
=
\max_{(i,j)\in\mathcal B}
r_{\downarrow}(x_{ij};h_{-,ij},w_{-,ij}),
$$

$$
E_{\mathrm{bond,long}}
=
\max_{(i,j)\in\mathcal B}
r_{\uparrow}(x_{ij};w_{+,ij},h_{+,ij}),
$$

$$
E_{\mathrm{atom,overlap}}
=
\max_{(i,j)\in\mathcal C}
r_{\downarrow}(x_{ij};0.60,0.80),
$$

$$
E_{\mathrm{pair,crowd}}
=
\max_{(i,j)\in\mathcal V}
r_{ij}^{\mathrm{crowd}},
$$

$$
E_{\mathrm{cluster,crowd}}
=
0.5\,
\max_{i:\,|\mathcal J_i|\ge2}
r_{\uparrow}(C_i;0.40,0.75),
$$

$$
E_{\mathrm{missing}}
=
\max_{(i,j)\in\mathcal L}
r_{\downarrow}(x_{ij};1.10,1.25).
$$

空集合的 `max` 定义为 0。距离门控伪能量和分数为：

$$
E_{\mathrm{atom}}
=
\max\left(
E_{\mathrm{bond,short}},
E_{\mathrm{bond,long}},
E_{\mathrm{atom,overlap}},
E_{\mathrm{pair,crowd}},
E_{\mathrm{cluster,crowd}},
0.5E_{\mathrm{missing}}
\right),
$$

$$
S_{\mathrm{atom}}=1-E_{\mathrm{atom}}\in[0,1].
$$

`0.5E_missing` 保证漏键嫌疑本身最多为 `UNCERTAIN`。半径缺失且没有其他 hard failure 时，
`S_atom=None`；若已有明确 hard failure，总分仍为 0。

## 5. 全分子键—键碰撞

### 5.1 检查对象与拓扑关系

本节属于 `bond_embedding/diagnostic`。它不在首版 `ring_piercing` 默认路径中；精确 crossing/overlap
是否在未来提议升级为默认重构信号，取决于第 11.5 节的性能与恢复率实证以及单独的策略评审。

所有可解释为两个真实原子中心之间有限 physical edge 的显式键都进入检查，不因
`single/double/aromatic/covalent/coordinate` 类型而跳过。zero-order display edge、dummy edge、
haptic/pseudo edge 和未经 cell-aware unwrapping 的 periodic-image edge 必须记录
`bond_semantic`；不满足有限 physical edge 模型时为 `UNCERTAIN` 或 `NOT_APPLICABLE`，不能静默套用
普通键段 hard verdict。每个适用无序键对只评价一次：

| `BondPairTopologyRelation` | 定义 | 默认处理 |
|---|---|---|
| `SAME_BOND` | 同一根键与自身 | `NOT_APPLICABLE` |
| `SHARED_ENDPOINT` | 两根不同键共享一个真实 Atom | 公共端点接触本身合理；仍检查同射线正长度重合 |
| `ONE_BOND_SEPARATED` | 两键不共享原子，但某对端点在图中直接成键 | 精确交叉仍 hard；纯近接保守为 `UNCERTAIN` |
| `REMOTE` | 其余不共享端点键对，包括不同连通分量 | 完整检查 |

不能用“都是共价键”作为豁免。真正需要豁免的只有由拓扑明确解释的共同端点。

### 5.2 共享的窄相位几何原语

唯一的 `_segment_pair_relation()` 计算并缓存：

~~~text
d_bb, t_star, u_star
closest_point_alpha, closest_point_beta
eta_bb
q_bb
collinearity residual
L_overlap
numeric margins
~~~

它同时供以下场景复用：

1. 全分子 bond–bond 检查；
2. 环边之间的自交/重合检查；
3. 目标键与真实环边的碰撞检查；
4. 环面三角片边界一致性验证所需的纯 segment kernel。

只有“两根显式 physical bonds”的关系进入全局 bond-pair registry。三角片内部对角线可以复用
同一纯几何 kernel，但只能进入 surface-local cache，不能生成伪造的 `BondPairFinding`。

在精确计算前可用 expanded AABB、sweep-and-prune 或空间索引产生候选，但 broad phase
只能排除“确定足够远”的键对，不能决定最终状态。测试必须将 broad-phase 结果与
brute-force 全键对结果对齐。

为不漏掉 `q_bb<w_bb` 的 pair，对键 `alpha` 定义
`R_alpha=max(r_cov(a0), r_cov(a1))`，其 AABB 每个方向至少扩张
`w_bb * R_alpha + epsilon_L`。两根扩张 AABB 不相交才可排除；任一半径缺失时不得以该规则排除。

### 5.3 v0 键对阈值

| 配置名 | 符号 | 默认值 | 取值/用途 |
|---|---|---:|---|
| `hard_bond_clearance_ratio` | `h_bb` | `0.10` | remote 键管严重靠近 |
| `warning_bond_clearance_ratio` | `w_bb` | `0.25` | 键管 warning-free 边界 |
| `parallel_sine_maximum` | `eta_parallel` | `1.0e-6` | 近共线/近平行 |
| `transverse_sine_minimum` | `eta_transverse` | `1.0e-4` | 稳定横向 |
| `stable_overlap_multiplier` | — | `10` | `L_overlap>10 * epsilon_L` 才是稳定正长度重合 |

`h_bb/w_bb` 是基于插值共价半径的严重缺陷筛选值，不是“真实键半径”，也没有文献直接支持
这两个数字。精确中心线相交和稳定正长度重合不依赖半径，始终是 hard evidence。

### 5.4 键对状态

| `BondPairGeometryState` | 关键判别要点 | 明确判定公式/判定点 | Verdict |
|---|---|---|---|
| `CLEAR` | 两根有限键段明确分离，中心线间隙具有元素尺度裕量 | `q_bb>=w_bb` 且不存在任何 contact/overlap 事件 | `REASONABLE` |
| `EXPECTED_SHARED_ENDPOINT` | 两根键只在它们共同的真实原子处汇合 | `SHARED_ENDPOINT`，交集仅为公共端点，且 `L_overlap<=epsilon_L` | `REASONABLE`；不进入 collision list |
| `TRANSVERSE_INTERIOR_CROSSING` | 两根有限键段从彼此内部横穿 | `d_bb<=epsilon_L`；`t^*`、`u^*` 都在稳定内部；`eta_bb>=eta_transverse` | `UNREASONABLE` |
| `NONTOPOLOGICAL_ENDPOINT_CONTACT` | 一个未连接原子的键端点落到另一根键的内部 | 无共享原子；`d_bb<=epsilon_L`；一个参数在稳定端点带，另一个在稳定内部 | `UNREASONABLE` |
| `COLLINEAR_POSITIVE_OVERLAP` | 两根近共线键沿同一空间路径重合一段正长度 | 共线残差 `<=epsilon_L`、`eta_bb<=eta_parallel`、`L_overlap>10 * epsilon_L` | `UNREASONABLE` |
| `ATOM_OVERLAP_REFERENCED` | 两个不共享拓扑身份的键端点落在同一位置；根因是 atom-pair overlap | 两个最近点参数都在稳定端点带，且已有对应 atom-pair hard finding | 继承 Atom finding；不重复计分 |
| `NEAR_CENTERLINE_COLLISION` | 两根键的内部中心线未严格相交，但已挤得极近 | 无共享端点；`t^*`、`u^*` 均在稳定内部；`q_bb<=h_bb`；无精确事件 | `UNCERTAIN`；高风险排序，不作默认 hard gate |
| `NEAR_CENTERLINE_GRAZE` | 两根键内部中心线处于碰撞警戒带 | 无共享端点；`t^*`、`u^*` 均在稳定内部；`h_bb<q_bb<w_bb` | `UNCERTAIN` |
| `NEAR_ENDPOINT_SEGMENT_CONTACT` | 一个键端点非常接近另一根键的内部，但尚未精确接触 | 无共享端点；仅一个最近参数在端点带，另一个在稳定内部，且 `q_bb<w_bb` | `UNCERTAIN` |
| `NUMERICALLY_UNRESOLVED` | 最近点、端点、平行性或重叠长度落在容差过渡带 | `eta_parallel<eta_bb<eta_transverse`，或参数/`L_overlap` 位于 1–10 倍容差带 | `UNCERTAIN` |
| `RADIUS_UNAVAILABLE` | 可判断是否精确相交，但不能评价元素尺度间隙 | 半径缺失且没有精确事件 | `UNCERTAIN` |
| `UNSUPPORTED_BOND_SEMANTIC` | 图中的 edge 不能解释为两个原子中心之间的一根普通有限物理键 | pseudo/dummy/未展开 periodic edge 等 | `UNCERTAIN` 或 `NOT_APPLICABLE`，由显式 semantic policy 决定 |

两根共享端点的键通常 `EXPECTED_SHARED_ENDPOINT`，不以 `d_bb=0` 误报碰撞。但若两键从
公共原子沿同一射线重合，`COLLINEAR_POSITIVE_OVERLAP` 优先级更高，必须判为
`UNREASONABLE`。共享端点也不得掩盖零长度键；零长度键由成键距离和输入有效性通道处理。

若两个不共享拓扑原子的键端点彼此重合，原子中心重叠由 Atom gate 唯一所有；Bond–bond gate
只引用该 atom-pair event。若其中一个端点落在另一键的内部，则仍由
`NONTOPOLOGICAL_ENDPOINT_CONTACT` 唯一所有，因为这是原子—键而非单纯原子—原子缺陷。

键对主状态必须先分流拓扑关系，不能先用 `q_bb`。正确优先级为：

~~~text
SAME_BOND                              -> NOT_APPLICABLE
zero-length physical edge              -> INVALID_INPUT
stable positive collinear overlap       -> COLLINEAR_POSITIVE_OVERLAP
SHARED_ENDPOINT                         -> EXPECTED_SHARED_ENDPOINT
disjoint stable interior crossing       -> TRANSVERSE_INTERIOR_CROSSING
disjoint endpoint-on-interior contact   -> NONTOPOLOGICAL_ENDPOINT_CONTACT
disjoint endpoint-on-endpoint overlap   -> ATOM_OVERLAP_REFERENCED
numeric tolerance band                  -> NUMERICALLY_UNRESOLVED
disjoint interior near-clearance         -> NEAR_CENTERLINE_COLLISION/GRAZE
disjoint endpoint–segment near-clearance -> NEAR_ENDPOINT_SEGMENT_CONTACT
otherwise                               -> CLEAR
~~~

因此普通 V 形共享端点键即使 `d_bb=q_bb=0`，也不会进入 near-collision 分支。同射线正长度重合
在 shared-endpoint 早退之前检查，所以不会被豁免。

### 5.5 键对风险和输出

对一个 physical bond pair 的风险采用分段定义：

$$
E_{bb,\alpha\beta}
=
\begin{cases}
1,
& \text{stable crossing, endpoint-on-interior, or positive overlap},\\
0,
& \text{expected shared endpoint},\\
\min\left(0.5,r_{\downarrow}(q_{bb};h_{bb},w_{bb})\right),
& \text{disjoint near-clearance},\\
\mathrm{None},
& \text{unsupported radius/semantic or numerically unresolved}.
\end{cases}
$$

- `0.10/0.25` 的“键管”只是排序启发式，不代表真实物理键半径；在 benchmark 校准前，
  所有纯 near-clearance 都只为 `UNCERTAIN`，风险最高 0.5；
- 稳定中心线相交与正长度重合是图嵌入缺陷，独立于半径阈值；但只有 supported physical edge
  才按 hard 映射；
pair 到 channel 的归约顺序必须显式处理 `None`：

1. 任一适用 pair 是 hard finding：`E_bb=1`、`S_bb=0`；
2. 否则，任一已请求评价的 pair 风险为 `None`：`E_bb=None`、`S_bb=None`；
3. 否则，`E_bb=max_(alpha<beta) E_bb(alpha,beta)`、`S_bb=1-E_bb`，范围 `[0,1]`；
4. 若确实不存在适用 pair，channel disposition 为 `NOT_APPLICABLE`；仅在分子级聚合时把该空通道
   当作中性项 0，不能把“本应评价但未完成”当作空集合。

`BondPairFinding` 至少保存：

~~~text
event_id
bond_indices = ((a0, a1), (b0, b1))
topology_relation
state
distance
normalized_clearance
closest_parameters
closest_points
crossing_sine
overlap_length
source_contexts
ring_ids
risk
~~~

分子报告必须提供：

- `collision_bond_pairs`：supported physical edges 的明确横穿、非拓扑端点接触和正长度重合；
- `uncertain_bond_pairs`：所有 near-clearance、unsupported semantics 或数值不稳定；
- `all_bond_pair_findings`：完整非 clear 证据。

这样调用方可以直接定位发生碰撞的 bond pairs，而不是只得到一个全局布尔值。

## 6. 环自身的几何状态

### 6.1 环门控只保留环特有问题

环边之间的相交、重合或过近已经由 Bond–bond gate 计算。`RingGeoStatus` 只做两件事：

1. 计算环特有的 rank、puckering 分类、投影开孔和环面可构造性；
2. 引用与该环边界有关的 atom/bond finding IDs。

因此一个环边自交在全分子注册表中只有一个 `BondPairFinding`。环报告通过
`boundary_collision_event_ids` / `boundary_graze_event_ids` 字段显示关联，但这些引用不是
`RingGeometryState`，也不创建第二个事件、第二份风险或第二次计算。

### 6.2 v0 环阈值

| 配置名 | 符号 | severe/ranking-high | warning-free | 用途 |
|---|---|---:|---:|---|
| `severe_rank_ratio` / `warning_rank_ratio` | `s_rank/w_rank` | `0.05` | `0.12` | 环是否接近一条线；只用于排序 |
| `warning_aperture_ratio` | `w_aperture` | 不设 hard | `0.15` | 投影孔过窄，只产生 warning |
| `planarity_ratio` | `delta_planar` | — | `0.05` | 只分 planar/puckered，不决定合理性 |

`q_nonplanar` 没有 hard 阈值。正常 chair、boat、macrocycle puckering 不应仅因非平面而失败。

### 6.3 环状态

| `RingGeometryState` | 关键判别要点 | 明确判定公式/判定点 | Verdict |
|---|---|---|---|
| `REGULAR_PLANAR` | 环边界有清楚开孔，未塌成窄条，且各原子近似位于同一平面 | `surface_set_complete=True`、`surface_construction_resolved=True`、`N_admissible>0`、`q_rank>=0.12`、`q_aperture>=0.15` 且 `q_nonplanar<=0.05` | `REASONABLE` |
| `REGULAR_PUCKERED` | 环边界有清楚开孔且未塌缩，但像 chair/boat 一样正常离开最佳拟合平面 | `surface_set_complete=True`、`surface_construction_resolved=True`、`N_admissible>0`、`q_rank>=0.12`、`q_aperture>=0.15` 且 `q_nonplanar>0.05` | `REASONABLE` |
| `NEAR_COLLAPSE` | 环整体接近线状，或投影孔显著收窄 | `s_rank<q_rank<w_rank`，或 `0<q_aperture<w_aperture` | `UNCERTAIN` |
| `SEVERELY_LINEARIZED` | 环顶点几乎排成一条线；该形状很差，但长窄宏环仍可能合法 | `q_rank<=s_rank` | `UNCERTAIN`；不得单独阻止最小化 |
| `DEGENERATE_BOUNDARY` | 环缺点、连续顶点重合或有零长度边，几何原语不能可靠建立 | `n<3` 或 `min_i ell_i<=epsilon_L`；非有限坐标由全局 precondition 提前拦截 | `UNREASONABLE`；readiness 另标 input precondition |
| `PROJECTION_DEGENERATE` | 最佳平面投影没有稳定的 simple polygon 内部 | 投影非 simple，或 `A_proj<=epsilon_A` | `UNCERTAIN` |
| `SURFACE_UNSTABLE` | 已枚举候选中至少一个组合剖分在三维退化/相交谓词上无法稳定归类，所以不能用其余候选制造“无环面”结论 | `surface_set_complete=True` 且 `N_construction_unresolved>0`；记录相关 `mu_num` | `UNCERTAIN` |
| `SURFACE_UNDEFINED` | 完整枚举且所有构造谓词均已解析后，仍不存在符合规则的候选环面 | `surface_set_complete=True`、`surface_construction_resolved=True` 且 `N_admissible=0` | `UNCERTAIN` |
| `SURFACE_SET_INCOMPLETE` | 存在候选环面，但 per-ring surface budget 阻止了完整枚举 | `surface_set_complete=False` | `UNCERTAIN`；相关 pair 的 readiness 为 `INDETERMINATE` |
| `UNSUPPORTED_RING_SIZE` | 环超过首版完整枚举上限 | `n>max_ring_size` | `UNCERTAIN` |

环原子发生拥挤时，`RingGeoStatus.atom_crowding_event_ids` 引用 Atom gate 的 finding。
它不会再造一个 `RING_ATOM_COLLISION` 状态。

环 intrinsic 主状态优先级为：invalid boundary → unsupported size → surface set incomplete →
surface unstable → surface undefined → projection degenerate → severely linearized → near collapse →
regular planar/puckered。边界 bond collisions
和 atom crowding 只通过引用字段附着，由其 owner 决定几何 verdict。

### 6.4 环面的统一三维模型

非平面闭合空间折线没有天然唯一的二维内部。本方案定义的候选集合为：

> 按环顶点的循环次序枚举全部抽象组合三角剖分，再直接映射到原始三维坐标；只保留
> 仅使用原环顶点、拓扑和几何上均为嵌入三角盘的离散 spanning surfaces。

best-fit projection 只服务于 aperture/rank 等可选形状诊断，不参与核心候选曲面集合的定义。
否则 PCA 方向的不稳定或二维投影自交会漏掉合法的三维曲面，并把历史上的分支语义跳变重新引入。

每个候选环面必须满足：

1. 恰有 `n-2` 个非退化三角形；
2. 原始环边恰好出现一次，内部对角线恰好被两个三角形共享；
3. 三角形定向一致：任意两个共享内部边的三角形沿该边的遍历方向相反；这不要求相邻三角形
   法向彼此接近，也不把非平面折叠误判为定向失败；
4. 非邻接三角形不相交；
5. 邻接三角形只在公共边或公共顶点接触；
6. 曲面边界与输入环的循环边界完全一致，Euler characteristic 与连通性满足一个三角盘；
7. 所有三维相交/退化谓词均达到第 3.5 节规定的数值裕量。

每个抽象组合剖分都必须落入且只能落入一个集合：

$$
N_{mathrm{all,combinatorial}}
=
N_{mathrm{admissible}}
+N_{mathrm{proven,invalid}}
+N_{mathrm{construction,unresolved}}
+N_{mathrm{not,enumerated}}.
$$

明确违反 1–6 的候选才可进入 `proven_invalid`；谓词落入数值容差带的候选必须进入
`construction_unresolved`，不得当作 invalid 丢弃。组合枚举未被截断时
`N_not_enumerated=0` 且 `surface_set_complete=True`；`N_construction_unresolved=0` 时
`surface_construction_resolved=True`。只有两者都为 true，才允许由剩余 admissible surfaces
形成确定共识。

report 构造器必须强制以下双向不变量，禁止调用方手工拼出互相矛盾的 flag/count：

~~~text
surface_set_complete <=> N_not_enumerated == 0
surface_construction_resolved <=> N_construction_unresolved == 0
pair_evaluation_complete => surface_set_complete and surface_construction_resolved
pair_evaluation_complete => N_valid == N_admissible
not pair_evaluation_complete => 0 <= N_valid <= N_admissible
~~~

这里 `N_valid` 是已完成 pair relation 分类的 admissible surfaces 数，不是“几何上有效曲面总数”的
另一个副本。

该集合不是所有连续 spanning surfaces，不包含 Steiner 点，也不等价于“真实化学环面”。
因此类型和文档必须使用 `ADMISSIBLE_VERTEX_SURFACES`，不能声称数学完备。

默认 `ring_piercing` 路径必须按以下顺序控制成本：

1. 所有环使用同一候选曲面语义；只有顶点在数值意义上严格共面、边界是 simple polygon，且已
   证明所有 admissible triangulations 覆盖同一平面区域时，才可把等价曲面合并为一次计算优化；
   “近似平面”不得触发另一套判定语义；
2. ring AABB 与 bond AABB 不相交时不构造该环的 surface ensemble；
3. 除了第 1 项已证明几何等价的严格共面环，其余通过 broad phase 的环都懒加载同一套 surface
   ensemble；不得按 `REGULAR_PLANAR/REGULAR_PUCKERED` 的经验阈值切换核心算法；
4. `max_ring_size=8` 时，每环最多 132 个 vertex triangulations；surface ensemble 每环只缓存一次；
5. 只有通过 ring–bond AABB broad phase 的 pair 才消耗 surface/triangle budget；与任何外部
   physical bond 都不可能相交的环无需构造 surface ensemble；
6. `max_ring_size=8` 时，`max_surface_candidates_per_ring` 不得小于 132；全调用的
   `max_surface_triangle_tests_per_call=250_000` 是待基准确认的 v0 候选默认值；若它不能同时满足
   第 11.5 节的延迟门槛，发布前必须下调并重新记录适用范围，不得以无上限实现上线；
7. 达到 `max_surface_candidates_per_ring` 时，该环 `surface_set_complete=False`；达到全调用 triangle
   budget 时，尚未完成的 pair `pair_evaluation_complete=False`；两者都令全局
   `scan_complete=False`，但必须保存不同 reason，不得静默退回 centroid fan；
8. 只有某个 pair 的 admissible surface ensemble 已完整枚举并一致 pierced，才是 confirmed
   piercing；此后可停止扫描其他 pair 并返回 `REBUILD_RECOMMENDED`，但全局 `scan_complete=False`
   必须如实记录；只枚举了部分 surface 时，即使目前全部命中也不能提前确认；
9. `has_confirmed_bond_ring_piercing()` 的 `False` 只表示当前没有 confirmed finding；返回
   `READY_WITHIN_SCOPE` 前则必须完成声明 scope 内所有相关 pair。任何 unresolved 或预算截断均为
   `INDETERMINATE`。

### 6.5 环风险

仅对环特有、可量化的指标计分：

$$
E_{\mathrm{ring,intrinsic}}
=
0.5\,
\max\left[
r_{\downarrow}(q_{\mathrm{rank}};0.05,0.12),\quad
r_{\downarrow}(q_{\mathrm{aperture}};0,0.15)
\right].
$$

该公式只在两个 ratio 都已定义时评价；否则 `S_ring=None`。SVD aspect ratio 和投影 aperture 都是
形状诊断，不足以证明环物理不可能，因此整个 intrinsic 通道最大风险为 0.5，不能单独制造
hard failure。
环边碰撞和原子拥挤风险从共享 finding 引用，不再加入第二次。环面 undefined 或数值不稳定且
没有其他 hard failure 时，`S_ring=None`；否则
`S_ring=1-E_ring,intrinsic`，范围 `[0,1]`。

## 7. 环面集合一致的有限键段—环穿越

### 7.1 语义边界

`BondRingGeometryState` 针对“一个有限目标键段 + 一个规范环”：

- 目标键是环自身边：`NOT_APPLICABLE`；
- 目标键连接该环的两个原子但不是该 ring key 的边：把它标记为 `RING_CHORD_OR_BRIDGE`，不作为
  普通外部穿环键；它属于 fused/bridged graph embedding，只有 `bond_embedding` 或专门 ring-topology
  分析处理；
- 目标键撞到真实环边：属于 Bond–bond collision，不属于“穿环”；
- 只有无限延长线穿过环面：不是穿环；
- 键端点落在环面上：不是明确穿环；
- 键与环面共面：不是明确穿环；
- 有限键段与声明的候选 spanning surface 内部形成稳定横向交点：才是本模块所称的 piercing。

“piercing”始终带有本节声明的环面模型限定。对一般非平面环，它不是天然唯一的化学内部，
也不是开放键与闭环之间严格的拓扑不变量。

### 7.2 线段—环面判据

目标键段为：

$$
\mathbf b(t)=\mathbf a+t\mathbf d,
\qquad 0\le t\le1.
$$

对候选环面三角片法向 `\mathbf n`，线—面的横向程度为：

$$
\eta_{bs}
=
\frac{|\mathbf d\cdot\mathbf n|}
{\|\mathbf d\|\|\mathbf n\|}.
$$

对非平行的三角片平面，以该片任一顶点 `\mathbf v_0` 计算无限直线交点参数：

$$
t^*
=
\frac{(\mathbf v_0-\mathbf a)\cdot\mathbf n}
{\mathbf d\cdot\mathbf n},
\qquad
\mathbf x=\mathbf b(t^*).
$$

交点到原始环边界的最短距离，以及有限键段到整个候选环面的最短距离分别为：

$$
d_{\mathrm{boundary}}(\mathbf x,R)
=
\min_i
\operatorname{dist}(\mathbf x,\mathbf e_i).
$$

$$
d_{bS}
=
\min_{\substack{0\le t\le1\\\mathbf y\in S}}
\left\|\mathbf b(t)-\mathbf y\right\|.
$$

为避免用一个任意最近点误分 endpoint/coplanar contact，定义整个容差接触集合：

$$
\mathcal C_{bS}
=
\left\{
(t,\mathbf y)\mid
0\le t\le1,\ \mathbf y\in S,\
\|\mathbf b(t)-\mathbf y\|\le\epsilon_L
\right\},
$$

$$
\mathcal T_{bS}
=
\left\{t\mid(t,\mathbf y)\in\mathcal C_{bS}\right\}.
$$

若实现为了报告单个代表接触而保存 `(t_c, y_c)`，它必须来自
`argmin_{0<=t<=1, y in S} ||b(t)-y||`；状态归类仍使用整个 `C_bS/T_bS`，不能由任意一个
argmin 决定。

对某一三角片 `T`，再定义两端点到该片平面的最大法向距离：

$$
\delta_{\mathrm{plane}}(\mathbf b,T)
=
\max_{q\in\{\mathbf b(0),\mathbf b(1)\}}
\left|(\mathbf q-\mathbf v_0)\cdot\widehat{\mathbf n}\right|.
$$

`\Pi_T` 表示到三角片 `T` 所在平面的正交投影，`\overline T` 表示包含边界的闭三角形。

一个稳定的 triangle face-interior crossing 必须同时满足：

$$
10\epsilon_t<t^*<1-10\epsilon_t,
\qquad
\eta_{bs}\ge\eta_{\mathrm{transverse}},
$$

$$
\min(\lambda_0,\lambda_1,\lambda_2)>10\epsilon_b,
\qquad
d_{\mathrm{boundary}}(\mathbf x,R)>10\epsilon_L.
$$

其中 `lambda_k` 是三角形重心坐标。命中内部对角线时，triangle-level 事件不满足上述
face-interior 重心坐标条件，但必须在整个 surface 层合并相邻三角形的成对命中；合并后的
交点只要满足相同的线段内部、横向和原始边界裕量，仍是稳定 surface-interior crossing。
内部对角线不是环边界。`d_boundary<=epsilon_L` 的事件交由共享 Bond–bond finding 解释；
`epsilon_L<d_boundary<=10 * epsilon_L` 为边界不稳定带。

本节所有判定只使用以下冻结阈值；“稳定”统一表示离数值边界至少
`m_stable=10` 个基础容差：

| 判定量 | 基础阈值 | 稳定判定点 | 取值范围 |
|---|---:|---:|---|
| 点/线/面距离 | `epsilon_L=max(epsilon_abs+epsilon_rel*s_R, epsilon_coord)` | clear/interior margin `>10*epsilon_L` | Å，`[0,+infinity)` |
| 线段端点参数 | `epsilon_t=epsilon_L/max(norm(d),epsilon_L)` | 内部交点 `10*epsilon_t<t*<1-10*epsilon_t` | 无量纲；有限段为 `[0,1]` |
| 三角形重心坐标 | `epsilon_b=1.0e-8` | face interior `min(lambda_k)>10*epsilon_b` | 和为 1 |
| 近乎平行上界 | `eta_parallel=1.0e-6` | `eta_bs<=eta_parallel` | 无量纲，`[0,1]` |
| 稳定横向下界 | `eta_transverse=1.0e-4` | `eta_bs>=eta_transverse` | 无量纲，`[0,1]` |
| 数值过渡带 | `m_stable=10` | 基础容差与 `10x` 容差之间均为 unresolved | 无量纲倍数 |

对一个 surface，先按坐标容差合并相邻三角片在公共内部对角线产生的重复命中，再保存全部唯一
交点。对一致定向的三角片，额外记录每个交点的方向符号与 mod-2 parity：

$$
s_m
=
\operatorname{sign}(\mathbf d\cdot\mathbf n_m),
\qquad
p_S=N_{\mathrm{hit},S}\bmod2.
$$

这些是诊断证据；首版 surface relation 采用“至少一个稳定 interior hit 即 pierced”，不会让
两个交点在计数上相互抵消。零个 hit 为 clear；稳定 endpoint/coplanar contact 是“已解析的
non-piercing diagnostic”，而落在端点、方向或边界的 1–10 倍容差过渡带才是可能遮蔽 interior
hit 的 ambiguous event。随后再对环面集合取共识：

$$
N_{\mathrm{valid}}
=
N_{\mathrm{pierced}}+N_{\mathrm{clear}}+N_{\mathrm{contact}}+N_{\mathrm{ambiguous}}.
$$

这里 `N_contact` 只统计已稳定分类的 endpoint/coplanar/ring-edge 非穿越接触；它不属于
`N_ambiguous`。

必须显式检查 `N_valid>0`，禁止用 `all([])` 把空集合判为 clear 或 pierced。

### 7.3 键—环状态

| `BondRingGeometryState` | 关键判别要点 | 明确判定公式/判定点 | Verdict |
|---|---|---|---|
| `RING_BOUNDARY_BOND` | 目标键本身就是该 ring 的一条边 | `(a_0,a_1) in E_boundary(R)` | `NOT_APPLICABLE` |
| `RING_CHORD_OR_BRIDGE` | 目标键两端都是该 ring 的原子，但该键不是 ring boundary；它是环图自身的 chord/bridge，不是外部穿线 | `{a_0,a_1} subset of V(R)` 且 `(a_0,a_1) not in E_boundary(R)` | `NOT_APPLICABLE` 于 bond–ring piercing；可由 `bond_embedding` 分析 |
| `CLEAR` | 实际有限键段与所有可接受环面明确分离 | `surface_set_complete=True`、`surface_construction_resolved=True`、`pair_evaluation_complete=True`、`N_valid>0`、`N_clear=N_valid` | `REASONABLE` |
| `PIERCED` | 有限键段与每个可接受环面的内部都至少形成一个稳定横向交点 | `surface_set_complete=True`、`surface_construction_resolved=True`、`pair_evaluation_complete=True`、`N_valid>0`、`N_pierced=N_valid`，且每个 surface 至少一个命中满足 7.2 的 face-interior 条件或内部对角线合并后的等价条件 | `UNREASONABLE` |
| `SURFACE_SET_INCOMPLETE` | per-ring surface cap 耗尽，尚有候选环面未生成 | `surface_set_complete=False`；保存已生成数量和截断原因 | `UNCERTAIN`；readiness 必须为 `INDETERMINATE` |
| `PAIR_EVALUATION_INCOMPLETE` | ring 的 surface set 已知，但全调用 triangle budget 在该 pair 完成前耗尽 | `surface_set_complete=True` 且 `pair_evaluation_complete=False` | `UNCERTAIN`；readiness 必须为 `INDETERMINATE` |
| `SURFACE_DISAGREEMENT` | 至少一个可接受环面稳定 pierced，而另一个没有确认 piercing | `0<N_pierced<N_valid` | `UNCERTAIN`；进入 core-unresolved |
| `NONPIERCING_ENDPOINT_CONTACT` | 键只有一个真实端点接触环面的稳定内部区域；有限线段没有横穿环面，也没有擦到真实环边 | `C_bS` 非空；`sup_{t in T_bS} min(t,1-t)<=epsilon_t`；接触点均满足 `d_boundary>10*epsilon_L`；且所有 surface 均无 stable interior hit、boundary event 或 ambiguity | `UNCERTAIN` 几何诊断；piercing 关系已解析为非穿越，不影响默认 readiness |
| `NONPIERCING_COPLANAR_CONTACT` | 键段确实有一段贴在/滑过三角面稳定内部；仅与平面平行但相距很远不算接触 | 存在三角片 `T` 和区间 `[t0,t1] subset T_bS`：`eta_bs<=eta_parallel`、`delta_plane(b,T)<=epsilon_L`、`(t1-t0)*norm(d)>10*epsilon_L`，且该区间投影位于 `T_bar` 并始终满足 `d_boundary>10*epsilon_L`；同时无 stable interior hit、boundary event 或 ambiguity | `UNCERTAIN` 几何诊断；piercing 关系已解析为非穿越，不影响默认 readiness |
| `RING_EDGE_COLLISION_REFERENCED` | 目标键中心线明确击中真实 ring edge；这是 bond-embedding collision，不是 interior piercing | `d_boundary<=epsilon_L` 且存在 canonical exact `BondPairFinding` | piercing 关系已解析为非穿越；几何 verdict 继承 Bond–bond finding |
| `BOUNDARY_NUMERIC_BAND` | 候选交点非常靠近真实 ring edge，尚不能稳定区分 interior hit 与 edge contact | `epsilon_L<d_boundary<=10 * epsilon_L` | `UNCERTAIN`；进入 core-unresolved |
| `SURFACE_UNSTABLE` | 至少一个组合剖分的构造无法稳定归类，不能用剩余曲面制造共识 | `surface_set_complete=True` 且 `surface_construction_resolved=False` | `UNCERTAIN` |
| `SURFACE_UNDEFINED` | 完整且数值已解析的构造结果中没有有效候选环面 | `surface_set_complete=True`、`surface_construction_resolved=True` 且 `N_admissible=N_valid=0` | `UNCERTAIN` |
| `NUMERICALLY_UNRESOLVED` | 参数、方向、面积或重心坐标落在数值容差带 | 相关 `mu_num<=1`，或 `eta_parallel<eta_bs<eta_transverse` | `UNCERTAIN` |

三个 stable non-piercing contact 状态都要求所有已完整评价的 surface 均无 piercing/ambiguity，且至少
一个 surface 出现相应 contact；若多种 contact 并存，主状态优先级为 ring-edge collision →
endpoint → coplanar，完整 subtype 列表保存在 diagnostics。稳定 endpoint/coplanar contact 只进入
diagnostics，不进入 `readiness_exception_pair_keys`。exact ring-edge collision 虽不是 interior
piercing，仍进入 `readiness_exception_pair_keys`，并按第 7.4 节生成独立 workflow action。

`LINE_EXTENSION_HIT` 只保留为 `BondRingDiagnostics.line_extension_hit=True`，主状态仍是
`CLEAR`。其明确判定点为：无限直线存在满足
`eta_bs>=eta_transverse`、`min(lambda_k)>10*epsilon_b`、
`d_boundary>10*epsilon_L` 的稳定内部面交点，但
`t^*<-10*epsilon_t` 或 `t^*>1+10*epsilon_t`，且有限段本身满足 `d_bS>10*epsilon_L`。它不参与风险、
verdict 或碰撞列表。

pair 主状态按以下顺序归约，防止同一事件因分支顺序得到不同结果：

~~~text
target is ring boundary/chord/bridge -> NOT_APPLICABLE
surface set 未完成                -> SURFACE_SET_INCOMPLETE
surface construction 未解析       -> SURFACE_UNSTABLE
pair evaluation 未完成            -> PAIR_EVALUATION_INCOMPLETE
N_admissible == N_valid == 0   -> SURFACE_UNDEFINED
完整枚举且全部有效环面稳定 pierced -> PIERCED
完整枚举且全部有效环面稳定 clear   -> CLEAR
存在 pierced 与任一 non-piercing/ambiguous 混合 -> SURFACE_DISAGREEMENT
没有 pierced/ambiguous，存在稳定非穿越接触：
  exact real-ring-edge hit       -> RING_EDGE_COLLISION_REFERENCED
  otherwise stable endpoint      -> NONPIERCING_ENDPOINT_CONTACT
  otherwise stable coplanar      -> NONPIERCING_COPLANAR_CONTACT
没有 pierced，但存在 ambiguity：
  any boundary tolerance band     -> BOUNDARY_NUMERIC_BAND
  otherwise numerical transition  -> NUMERICALLY_UNRESOLVED
~~~

某个有效 surface 已存在稳定 interior hit 时，该 surface 仍归为 pierced；附带的其他 near-boundary
或 numerical diagnostics 不抹去已确认交点。只有在没有稳定 hit 时，ambiguity 才决定该 surface
的 uncertain relation。

诊断量 `surface_nonpiercing_fraction=(N_clear+N_contact)/N_valid` 的范围为 `[0,1]`，但不同三角剖分没有
等概率物理意义，所以它不是概率，只能用作报告和同类 uncertain 结果的次级排序。

### 7.4 与键—键碰撞的严格分工

目标键靠近或命中某条环边时，计算过程如下。这里的 `registry` 是事件所有者；默认
`ring_piercing` 只对当前候选 pair 懒计算，并不启动全键对扫描：

~~~text
canonical registry.get_or_compute(target_bond, ring_edge)
        │
        ├── creates one BondPairFinding
        ├── RingGeoStatus references event_id
        └── BondRingGeoStatus references event_id
                but does not create another collision
~~~

- 中心线击中环边：core report 把 resolved non-piercing `BondRingGeoStatus` 放入
  `nonpiercing_contacts`，并把 pair key 放入 `readiness_exception_pair_keys`，绝不放入
  `unresolved_pairs`；只有显式物化可选 `MoleculeGeoStatus` 时，canonical `BondPairFinding` 才在
  `collision_bond_pairs` 中出现一次；
- 中心线横穿环内部，且离所有环边有稳定裕量：`piercing_bond_ring_pairs` 中出现一次；
- 同一构型可能同时有一个真实 interior piercing 和另一个独立的环边近接；两者是不同事实，
  可以同时保留，但总风险仍取最大值而不相加。
- 若在同一个 `_MoleculeGeometryWorkspace` 内继续执行 `bond_embedding/diagnostic`，相同 pair key
  直接复用该 finding；公开 API 的第二次独立调用必须建立新 workspace，只复用 canonical key
  规则而不能复用旧 finding/cache。

默认 `ring_piercing` 对“目标 bond 与真实 ring edge 精确 crossing”只返回
`INDETERMINATE / PROCEED_AND_REASSESS` 并强制终态复检；它不是 interior piercing。
显式 `bond_embedding` 会把它报告为确定的键段碰撞，但首版仍不改写 readiness；是否将这类
碰撞升级为重构条件，留给独立的恢复率实验和后续策略评审。

### 7.5 共享环原子的外接键

只共享一个环原子的外接键不能整体跳过：

1. 它与两根 incident ring edges 的共同端点接触是预期事件；
2. 它与其余环边仍接受正常 Bond–bond 检查；
3. 去除共享端点容差邻域后，剩余开线段仍接受 Bond–ring piercing 检查；
4. 若剩余线段从远端再次穿环，必须报告 `PIERCED`。

### 7.6 当前互穿策略与明确 TODO

当前 de-novo build/repair 的几何评价与业务动作必须分成两条独立映射：

~~~text
BondRingGeometryState.PIERCED
-> GeometryVerdict.UNREASONABLE

BondRingGeometryState.PIERCED
+ ring_origin = LIGAND_SKELETON
+ v0 fixed readiness mapping
-> OptimizationReadiness.REBUILD_RECOMMENDED
~~~

理由是普通有机分子与配位络合物自动构筑中，稳定横穿通常是初始构型打结或优化失败的信号，
普通局部最小化不能可靠越过所需的协同重排路径。但它不是普适化学定理，也不应默认用于拒绝
导入的实验机械互锁结构。

> **TODO — topology-aware piercing policy**
>
> 当前 v0 fixed piercing mapping 会把所有 confirmed ligand-skeleton piercing 都建议重构，因此会
> 错误处理有意机械互锁的结构。已知反例包括：
>
> - rotaxane：轴分子有意穿过大环；
> - catenane：另一个闭环必然穿过某些 spanning surface；
> - threaded host–guest complex：用户明确希望保留穿孔客体；
> - periodic structure：跨周期边界的有限段表示可能产生假穿越。
>
> 后续需增加显式的 target-topology/intent policy，使用户可把已声明的 expected interlock 映射为
> `PROCEED`，并要求用户或上游构筑器提供预期互锁关系；不能只删除 finding。
> 周期结构必须先进行 cell-aware unwrapping/minimum-image 预处理，否则返回 `INVALID_INPUT` 或
> `INDETERMINATE`。对显式请求的 `full_graph` coordination ring/cage，piercing 默认先映射为
> `INDETERMINATE`，直到该图语义经数据校准。
>
> molecular knot 不能只靠修改 `is_geo_reasonable()` 的策略解决：非平凡结的闭合边界不存在
> 普通嵌入盘，其识别需要单独的 knot/link invariant 与闭合曲线模型。该能力应作为独立模块，
> 不能伪装成当前 bond–ring pair 的简单例外。

在 TODO 完成前，ligand-skeleton 的 `PIERCED` 保留客观证据，在 de-novo build/repair 中映射为
`REBUILD_RECOMMENDED`；调用方不得删除 finding。

## 8. 分子级聚合、评分与决策

### 8.1 可选几何分析的 Verdict 聚合

~~~text
任一已评价 finding 为明确 hard failure  -> UNREASONABLE
否则存在 warning / unsupported / unknown -> UNCERTAIN
否则所有适用门控均有安全裕量           -> REASONABLE
~~~

`INVALID_INPUT` 在 workflow 层操作性拒绝；`NOT_APPLICABLE` 从聚合候选中移除。
一个门控为空并不自动表示通过：只有“没有适用 pair”才是 N/A；“本应评价但无法构造”是
`UNCERTAIN`。

### 8.2 去重后的统一伪能量

当四个通道全部返回数值风险时：

$$
E_{\mathrm{geo}}
=
\max\left(
E_{\mathrm{atom}},
E_{\mathrm{bb}},
E_{\mathrm{ring,intrinsic}},
E_{\mathrm{br}}
\right),
$$

$$
S_{\mathrm{geo}}=1-E_{\mathrm{geo}}\in[0,1].
$$

若有 `None`，不能直接执行数学上的 `max(..., None, ...)`。完整规则为：

- `E_br=0` 对应稳定 `CLEAR`，`E_br=1` 对应当前策略下的稳定 `PIERCED`；其余
  surface/endpoint/numerical ambiguity 首版不可量化，记为 `None`；
- 同一 BondPair finding 被环报告引用时不再次进入 `max`；
- 任一 owner finding 为 hard：无论其他通道是否为 `None`，固定 `E_geo=1`、`S_geo=0`；
- 没有 hard，但任一已启用通道为 `None`：`E_geo=None`、`S_geo=None`；
- 否则才对所有数值通道执行 `max`，可量化 warning 得到 `0<S_geo<1`；
- 所有检查均 warning-free：`S_geo=1`。

| `S_geo` | 含义 |
|---|---|
| `1` | 无 warning，`REASONABLE` |
| `[0.67,1)` | 在同为 `UNCERTAIN` 的候选中优先保留（几何风险较低） |
| `[0.33,0.67)` | 在同为 `UNCERTAIN` 的候选中居中保留 |
| `(0,0.33)` | 在同为 `UNCERTAIN` 的候选中最后保留（几何风险较高） |
| `0` | 至少一个 hard failure，`UNREASONABLE` |
| `None` | 存在不可量化不确定性；在 uncertain 候选中最后排序 |

这里的“优先”只表示构型候选的选择顺序，不表示风险更高；`S_geo` 越大，几何风险越低。
这些分区只是候选排序约定，不是化学阈值。score 不能把 `UNREASONABLE` 候选提升到
`UNCERTAIN`，也不能把 `UNCERTAIN` 自动改成 `REASONABLE`。

### 8.3 Workflow 决策与证据报告分离

`MoleculeGeoStatus` 是不可变证据快照；`is_geo_reasonable()` 产生 verdict；force-field workflow
独立生成 `OptimizationReadinessReport`。不得在证据对象中保存一个随阶段变化的 `passed: bool`。

同一 verdict 内候选建议按以下可执行的稳定键排序：

~~~text
(verdict_rank,
 score is None,
 0.0 if score is None else -score,
 energy,
 attempt_index)
~~~

能量只在几何 verdict 和 score 相同或相近的候选之间用于排序，不参与本几何门控公式。

### 8.4 优化就绪性不读取统一几何分数

`assess_optimization_readiness()` 使用显式 allow/block 表，不使用 `E_geo` 或“任一几何 hard 即拒绝”：

| Finding / disposition | 默认 pre-minimization action | 默认 post-minimization action |
|---|---|---|
| 非有限坐标、不同原子完全重合、零长度 physical bond | 拒绝启动，先做 preconditioning | 失败 |
| confirmed ligand-skeleton `PIERCED` | `REBUILD_RECOMMENDED` | 失败并保留交点证据 |
| 环折叠造成的 surface undefined/disagreement，或可能掩盖穿越的 numeric unresolved | `INDETERMINATE / PROCEED_AND_REASSESS`；默认允许一次最小化，strict policy 可要求先提高判定精度 | 终态仍未解析则失败关闭 |
| ring/surface/triangle budget 或 ring perception 未完成 | `INDETERMINATE / REASSESS`；先提高预算或转离线检查 | 失败关闭 |
| 只有稳定的 non-piercing endpoint/coplanar contact | `READY_WITHIN_SCOPE` + diagnostic；没有要求预先重构的确认性证据 | 优化后重新检查 |
| 普通 atom near-contact / multi-atom crowding | `READY_WITHIN_SCOPE` + diagnostic；没有要求预先重构的确认性证据 | 若仍 severe，再按 post-check policy 处理 |
| 普通异常键长 | `READY_WITHIN_SCOPE` + diagnostic；极端长度可由 backend precondition 单独限制 | 结合收敛、拓扑保持与终态距离判断 |
| ring rank/aperture/puckering 异常 | `READY_WITHIN_SCOPE` + diagnostic；没有要求预先重构的确认性证据 | 重新计算；仅形状异常仍不等于拓扑失败 |
| 精确 disjoint bond crossing/positive overlap | `ring_piercing` 路径不计算；可选 `bond_embedding` 只报告 | 不改变首版 readiness；可按独立 post-check policy 标记失败 |
| 目标 bond 与真实 ring edge 精确 crossing | 核心路径记录为 `INDETERMINATE / PROCEED_AND_REASSESS`；可选 `bond_embedding` 补充碰撞证据 | 必须复检；不能把边界 crossing 误称为 interior piercing |
| bond near-clearance | `READY_WITHIN_SCOPE` + diagnostic | 终态仍只作为诊断，除非校准后升级 |

这里的“允许最小化”是经验预期，不是保证成功。局部 optimizer 的一次 minimization 与
`complexes_build` 的多 epoch 构筑—扰动—优化循环是不同能力，测试必须分别统计。

## 9. API、数据类与代码组织

### 9.1 拟议结构树

~~~text
hotpot/cheminfo/geometry.py
├── __all__
├── Exceptions
├── Enums
│   ├── EvaluationDisposition
│   ├── GeometryVerdict
│   ├── OptimizationReadiness
│   ├── RecommendedAction
│   ├── ReadinessFailureClass
│   ├── RingOrigin
│   ├── UnresolvedAction
│   ├── BondLengthGeometryState
│   ├── AtomCrowdingGeometryState
│   ├── TopologyDistanceAdvisory
│   ├── BondPairTopologyRelation
│   ├── BondPairGeometryState
│   ├── RingGeometryState
│   └── BondRingGeometryState
├── Public frozen data classes
│   ├── GeometryQualityThresholds
│   ├── GeometryPolicy
│   ├── BondRingSearchConfig / OptimizationReadinessConfig
│   ├── OptimizationReadinessPolicy
│   ├── GeometryAnalysisConfig / ReadinessCoverage
│   ├── GeometryAnalysisCoverage
│   ├── AtomPair / BondPair / BondRingPair
│   ├── AtomPairFinding / AtomCrowdingFinding
│   ├── AtomPairGeoStatus / BondPairGeoStatus
│   ├── BondPairFinding / BondRingPiercing
│   ├── RingGeoStatus / BondRingGeoStatus
│   ├── MoleculeGeoStatus / GeometryAnalysisReport
│   └── BondRingPiercingReport / OptimizationReadinessReport
├── Private immutable snapshots and workspace
│   ├── _AtomGeometrySnapshot
│   ├── _BondSegmentSnapshot
│   ├── _SegmentPairRelation
│   ├── _RingSurface / _RingSurfaceEnsemble
│   └── _MoleculeGeometryWorkspace
├── Shared scalar/vector helpers
├── Atom distance/topology helpers
├── Shared segment-pair helpers
├── Ring-intrinsic and surface helpers
├── Bond-ring surface helpers
├── Per-target status builders
└── Public interfaces at file bottom
    ├── find_bond_ring_piercings()
    ├── has_confirmed_bond_ring_piercing()
    ├── assess_optimization_readiness()
    ├── analyze_geometry()
    ├── determine_geo_status()
    ├── is_geo_reasonable()
    └── evaluate_geometry_quality()
~~~

如果实现阶段决定拆文件，也必须保持上述依赖方向：atom/segment 基础层不依赖 ring；
ring 和 bond–ring 只能调用基础层，不能反向调用 forcefields。

### 9.2 公开签名

Python 3.9 兼容代码使用 `typing.Union`，不使用 `X | Y`。运行时不得使用 `Any`。
首版签名按以下结构收束；所有影响结论的 scope、容差和预算都必须进入 frozen config，并原样写入
report，禁止依赖不可见的模块全局状态：

~~~python
GeometryTarget = Union[AtomPair, BondPair, Ring, BondRingPair]
GeometryStatus = Union[
    AtomPairGeoStatus,
    BondPairGeoStatus,
    RingGeoStatus,
    BondRingGeoStatus,
]

RingScope = Literal["ligand_skeleton", "full_graph"]
RingFamily = Literal[
    "edge_shortest_cycle_family",
    "bounded_chordless_cycles",
    "all_simple_cycles",
]
GeometryAnalysisLevel = Literal["bond_embedding", "diagnostic", "exhaustive"]

@dataclass(frozen=True)
class BondRingSearchConfig:
    ring_scope: RingScope = "ligand_skeleton"
    ring_family: RingFamily = "edge_shortest_cycle_family"
    max_ring_size: int = 8
    max_ring_count: int = 256
    max_surface_candidates_per_ring: int = 132
    max_surface_triangle_tests_per_call: int = 250_000
    absolute_length_tolerance: float = 1.0e-8
    relative_length_tolerance: float = 1.0e-7
    coordinate_resolution: float = 0.0
    barycentric_tolerance: float = 1.0e-8
    parallel_sine_maximum: float = 1.0e-6
    transverse_sine_minimum: float = 1.0e-4
    numeric_warning_multiplier: float = 10.0

@dataclass(frozen=True)
class OptimizationReadinessConfig:
    bond_ring: BondRingSearchConfig = DEFAULT_BOND_RING_SEARCH_CONFIG

@dataclass(frozen=True)
class OptimizationReadinessPolicy:
    unresolved_geometry_action: UnresolvedAction = UnresolvedAction.PROCEED_AND_REASSESS
    ring_edge_collision_action: UnresolvedAction = UnresolvedAction.PROCEED_AND_REASSESS

@dataclass(frozen=True)
class GeometryAnalysisConfig:
    level: GeometryAnalysisLevel = "diagnostic"
    thresholds: GeometryQualityThresholds = DEFAULT_GEOMETRY_THRESHOLDS
    bond_ring: BondRingSearchConfig = DEFAULT_BOND_RING_SEARCH_CONFIG

def find_bond_ring_piercings(
    mol: Molecule,
    *,
    config: BondRingSearchConfig = DEFAULT_BOND_RING_SEARCH_CONFIG,
) -> BondRingPiercingReport: ...

def has_confirmed_bond_ring_piercing(
    mol: Molecule,
    *,
    config: BondRingSearchConfig = DEFAULT_BOND_RING_SEARCH_CONFIG,
) -> bool: ...

def assess_optimization_readiness(
    mol: Molecule,
    *,
    config: OptimizationReadinessConfig = DEFAULT_READINESS_CONFIG,
    policy: OptimizationReadinessPolicy = DEFAULT_READINESS_POLICY,
) -> OptimizationReadinessReport: ...

def analyze_geometry(
    mol: Molecule,
    *,
    config: GeometryAnalysisConfig = DEFAULT_GEOMETRY_ANALYSIS_CONFIG,
) -> GeometryAnalysisReport: ...

@overload
def determine_geo_status(
    target: AtomPair,
    *,
    config: GeometryAnalysisConfig = DEFAULT_GEOMETRY_ANALYSIS_CONFIG,
) -> AtomPairGeoStatus: ...

@overload
def determine_geo_status(
    target: BondPair,
    *,
    config: GeometryAnalysisConfig = DEFAULT_GEOMETRY_ANALYSIS_CONFIG,
) -> BondPairGeoStatus: ...

@overload
def determine_geo_status(
    target: Ring,
    *,
    config: GeometryAnalysisConfig = DEFAULT_GEOMETRY_ANALYSIS_CONFIG,
) -> RingGeoStatus: ...

@overload
def determine_geo_status(
    target: BondRingPair,
    *,
    config: GeometryAnalysisConfig = DEFAULT_GEOMETRY_ANALYSIS_CONFIG,
) -> BondRingGeoStatus: ...

def determine_geo_status(
    target: GeometryTarget,
    *,
    config: GeometryAnalysisConfig = DEFAULT_GEOMETRY_ANALYSIS_CONFIG,
) -> GeometryStatus: ...

def is_geo_reasonable(
    evidence: Union[GeometryStatus, GeometryAnalysisReport],
    *,
    policy: GeometryPolicy = DEFAULT_GEOMETRY_POLICY,
) -> GeometryVerdict: ...

def evaluate_geometry_quality(
    mol: Molecule,
    *,
    config: GeometryAnalysisConfig = DEFAULT_GEOMETRY_ANALYSIS_CONFIG,
    policy: GeometryPolicy = DEFAULT_GEOMETRY_POLICY,
) -> GeometryQualityReport: ...
~~~

`BondRingSearchConfig` 中这四个数值谓词字段分别对应第 3.5/7.2 节的 `epsilon_b`、
`eta_parallel`、`eta_transverse` 和 `m_stable`；它们与长度容差一起原样写入 report。
配置构造时必须满足 `0<=eta_parallel<eta_transverse<=1`、所有基础容差非负且
`numeric_warning_multiplier>1`。
`250_000` 是需要在固定机器基准中校准的 v0 安全上限，不是化学常数；在基准完成前允许调低，
不得删除全调用预算。`max_ring_size=8` 与每环 132 个 vertex triangulations 的理论上限一致。
若实际 admissibility 规则产生更多候选，或任一全局预算不足，必须显式报告未完成。

配置不变量必须在创建 config 时明确验证：`bond_embedding` 是独立的全键 crossing/overlap
分析；`diagnostic` 累积执行 `bond_embedding +` 稀疏距离/环指标，并可组合但不得改写一份
`ring_piercing` core report；`exhaustive` 累积 `diagnostic`，并要求
`bond_ring.ring_family="all_simple_cycles"`。如果调用方希望 dense atom/bond oracle 但仍只分析默认
ring family，应为各 channel 提供显式开关或另取 `dense_oracle` 名称，不能把这种组合仍称为
`exhaustive`。模块应分别导出可直接使用的 `DEFAULT_GEOMETRY_ANALYSIS_CONFIG`（diagnostic +
edge-shortest family）和 `EXHAUSTIVE_GEOMETRY_ANALYSIS_CONFIG`（exhaustive + all-simple-cycles），
避免调用方只改 level 而得到名不副实的配置。

v0 policy 只允许在 `PROCEED_AND_REASSESS` 与 `REASSESS` 之间选择如何处理几何未决和 ring-edge
collision。其余映射固定：ligand-skeleton confirmed piercing → `REBUILD`；coordination-graph
或 unresolved-origin piercing → `REASSESS`；incomplete coverage → `REASSESS`。可选
`bond_embedding` finding 不进入本 policy。在 topology-aware TODO 实施前，不能把这些字段开放成任意
`RecommendedAction`，以免构造出与 ready 不变量冲突的 policy。

对局部 `GeometryStatus`，`is_geo_reasonable()` 只映射该对象；对 `GeometryAnalysisReport`，它必须
先验证 `completed_channels == enabled_channels` 且没有 truncated/incomplete channel，否则直接
返回 `UNCERTAIN`，再对 `analysis.status` 聚合。不得绕过 coverage 直接评价
`MoleculeGeoStatus`。

`find_bond_ring_piercings()` 是核心薄包装；其 report 直接暴露
`confirmed_piercings`、`unresolved_pairs`、`nonpiercing_contacts` 和 `scan_complete`，不触发
atom matrix 或全键对
诊断。`has_confirmed_bond_ring_piercing()` 的 `False` 只表示“没有 confirmed piercing”，不表示
`OptimizationReadiness.READY_WITHIN_SCOPE`。需要区分 surface undefined/disagreement 的调用方必须使用
`assess_optimization_readiness()`。该命名避免旧 `has_*` 布尔接口把未知状态伪装成安全。

为避免 `core ↔ geometry` 循环导入，类型只在 `TYPE_CHECKING` 下导入，运行时 facade 在函数内部
执行明确的局部导入或使用无循环的基础协议；不能用 `Any` 掩盖循环依赖。对象、实例和参数命名
遵守开发规范：`mol`、`bond_pair`、`ring`、`clone_mol` 或 `working_mol`，不使用没有化学
含义的 `working`。

### 9.3 主要数据结构

| 名称 | 职责 |
|---|---|
| `GeometryQualityThresholds` | 可选几何质量分析的经验阈值来源；记录版本和单位 |
| `GeometryPolicy` | 只把可选分析的客观状态映射为三态；不包含构筑/互锁策略 |
| `BondRingSearchConfig` | 固定 ring scope/family/size、surface budget 和 piercing-kernel tolerance |
| `OptimizationReadinessConfig` | 组合一份 `BondRingSearchConfig`；其固定业务语义就是 core ring–bond piercing |
| `OptimizationReadinessPolicy` | v0 只配置几何未决和 ring-edge collision 是“先优化再复检”还是“先重评”；其他安全关键映射固定 |
| `ReadinessCoverage` | 保存声明范围、检测/评价/遗漏环数与 pair 数、预算使用量和完整性 |
| `GeometryAnalysisCoverage` | 保存完整分析各 channel 的启用/完成状态、候选计数与截断原因，并引用 readiness coverage |
| `OptimizationReadinessReport` | 保存 disposition、readiness、failure class、piercing report 与推荐动作 |
| `BondRingPiercingReport` | 分别保存 confirmed、unresolved 与 resolved non-piercing contact，及逐 pair/全局完整性 |
| `AtomPairGeoStatus` / `BondPairGeoStatus` | standalone pair 分析结果；输入 pair 必须携带同一 molecule snapshot/topology context |
| `AtomPairFinding` | 保存 atom pair、`d/x/y`、图距离、状态和风险 |
| `AtomCrowdingFinding` | 保存中心原子、参与原子、`C_i` 和所引用 pair IDs |
| `BondPairFinding` | 保存发生碰撞/近擦的确切 bond pair 与最近点证据 |
| `RingGeoStatus` | 保存环特有指标、surface 状态及引用的 atom/bond event IDs |
| `BondRingGeoStatus` | 保存 surface 共识、交点、边界引用和 piercing 证据 |
| `MoleculeGeoStatus` | 聚合四个通道；提供 collision pairs、piercing pairs 和不确定项 |
| `GeometryAnalysisReport` | `{status, level, coverage}`；完整可选诊断，不直接决定是否启动 optimizer |
| `GeometryQualityReport` | `{analysis, verdict, score, policy_id}` 的质量评价适配层 |
| `_MoleculeGeometryWorkspace` | 单次评价缓存；不得持久挂在可变 `Molecule` 或全局缓存上 |

报告保存稳定 atom/bond indices、坐标快照和只读数值。后续优化会改变对象坐标，因此不能只保存
`Atom/Bond/Ring` 活引用。

关键报告字段至少固定为：

~~~text
BondRingGeoStatus
  snapshot_id / coordinate_revision / coordinate_unit / config_fingerprint
  disposition
  state
  bond_key / ring_key / ring_origin
  surface_set_complete / surface_construction_resolved / pair_evaluation_complete
  combinatorial/admissible/proven_invalid/construction_unresolved/not_enumerated surface counts
  incomplete_reason
  surfaces_examined / triangle_tests_used
  confirmed_hits / diagnostics

BondRingPiercingReport
  snapshot_id / coordinate_revision / coordinate_unit / config_fingerprint
  disposition
  confirmed_piercings: Tuple[BondRingPiercing, ...]
  unresolved_pairs: Tuple[BondRingGeoStatus, ...]
  nonpiercing_contacts: Tuple[BondRingGeoStatus, ...]
  scan_complete
  coverage: ReadinessCoverage

ReadinessCoverage
  ring_scope / ring_family / max_ring_size
  max_ring_count / max_surface_candidates_per_ring
  max_surface_triangle_tests_per_call
  absolute_length_tolerance / relative_length_tolerance / coordinate_resolution
  barycentric_tolerance / parallel_sine_maximum / transverse_sine_minimum
  numeric_warning_multiplier
  detected_ring_keys / evaluated_ring_keys / omitted_ring_keys
  candidate_pair_keys / evaluated_pair_keys / readiness_exception_pair_keys / unvisited_pair_keys
  surface_candidates_used / triangle_tests_used
  ring_perception_complete / coverage_complete_within_scope
  global_coverage_complete / budget_exhausted

OptimizationReadinessReport
  disposition
  readiness
  action: RecommendedAction
  failure_classes: Tuple[ReadinessFailureClass, ...]
  policy_id / policy_basis
  piercing_report: BondRingPiercingReport
  limitations
  snapshot_id / coordinate_revision

GeometryAnalysisReport
  status: MoleculeGeoStatus
  level
  coverage: GeometryAnalysisCoverage

GeometryAnalysisCoverage
  enabled_channels / completed_channels
  atom_candidate_pair_keys / bond_candidate_pair_keys / ring_keys
  truncated_channels / incomplete_reasons
  bond_ring: ReadinessCoverage

GeometryQualityReport
  analysis: GeometryAnalysisReport
  verdict
  score
  policy_id
~~~

所有 key collection 均为按 canonical key 去重、稳定排序的 tuple；`rings_detected`、
`rings_evaluated`、`candidate_pairs` 等数量只作为由这些 tuple 计算的只读 property，不另存一份可能
失配的整数。`failure_classes` 同样是稳定去重的 tuple，ready 时为空。因 confirmed piercing 提前
结束而尚未访问的 pair 放入 `unvisited_pair_keys`，不能混入“阻止无条件 `PROCEED`”的
`readiness_exception_pair_keys`。`NONPIERCING_ENDPOINT_CONTACT`、
`NONPIERCING_COPLANAR_CONTACT` 等已明确不构成稳定 interior piercing 的诊断进入
`nonpiercing_contacts`，而不是 `unresolved_pairs`；`RING_EDGE_COLLISION_REFERENCED` 也进入
`nonpiercing_contacts`，但其 pair key 同时进入 `readiness_exception_pair_keys`；
`BOUNDARY_NUMERIC_BAND` 必须进入 `unresolved_pairs`。所有 child
report/finding/event 均继承同一个 snapshot identity。

`OptimizationReadinessReport.scan_complete`、`coverage_complete_within_scope`、
`confirmed_piercings`、`unresolved_pairs` 和 `nonpiercing_contacts` 都是转发到
`piercing_report` 及其 coverage 的只读
property，不重复存储。这样第 2.1 节的 ready 不变量可以直接执行，也不会产生两份互相冲突的数据。

其中 `RecommendedAction` 至少包含 `PROCEED`、`PROCEED_AND_REASSESS`、
`REJECT_PRECONDITION`、`REBUILD` 和 `REASSESS`。优先级是：invalid input →
`REJECT_PRECONDITION`；已完整确认 ligand-skeleton piercing → `REBUILD`；没有确认 piercing 但因
计算预算导致相关项未评价 → `REASSESS`；只因可优化的环形变/接触导致几何未决 → 默认
`PROCEED_AND_REASSESS`；只有 scope 内完整 clear 才 `PROCEED`。strict policy 可以把
`PROCEED_AND_REASSESS` 改成 `REASSESS`，但必须显式配置。若某个完整 pair 已确认 piercing，
即使为尽早重构而没有继续扫描其余 pair，动作仍为 `REBUILD`；此时 `scan_complete=False` 必须
保留，不能假称全分子证据完整。

### 9.4 函数职责与复用关系

| 函数 | 输入 → 输出 | 唯一职责 |
|---|---|---|
| `_snapshot_molecule_geometry()` | `Molecule → workspace` | 冻结一次坐标、拓扑、元素和半径 |
| `_build_atom_spatial_index()` | `workspace → neighbor index` | diagnostic 按距离阈值生成稀疏 atom-pair 候选 |
| `_atom_pair_metrics()` | `atom pair + workspace → d/x/y` | 按需计算一个 pair 的距离与元素归一化量 |
| `_build_dense_atom_distance_oracle()` | `workspace → matrices` | 仅 exhaustive/test 一次建立完整 `d/x/y` 与图距离表 |
| `_classify_atom_pair()` | `atom pair + workspace → finding` | 成键距离、严重原子中心重叠与 topology advisory |
| `_classify_atom_crowding()` | `atom findings → crowding findings` | 聚合同一原子周围的多原子拥挤 |
| `_build_bond_segment_table()` | `workspace → segment snapshots` | 一次建立端点、方向、长度与 AABB |
| `_candidate_bond_ring_pairs()` | `segment/ring AABBs → bond-ring keys` | 默认核心 broad phase，只产生可能接触环面的 pair |
| `_candidate_bond_pairs()` | `segment table → pair keys` | bond_embedding/diagnostic 全键 broad phase，仅排除确定太远的键对 |
| `_segment_pair_relation()` | `two segment snapshots → relation` | 唯一的最近距离、交叉、共线重叠原语 |
| `_classify_bond_pair()` | `relation + topology → finding` | 形成唯一 BondPair 状态 |
| `_canonical_rings()` | `workspace + scope + family → ring keys + coverage` | 稳定枚举、规范化和去重，并报告截断/遗漏 |
| `_ring_intrinsic_metrics()` | `ring + workspace → metrics` | 计算 SVD、planarity 和 aperture |
| `_enumerate_ring_surfaces()` | `ring metrics → surface ensemble` | 枚举 admissible vertex surfaces 并缓存 |
| `_segment_triangle_event()` | `segment + triangle → event` | 只产生 triangle-level 几何事件 |
| `_segment_surface_relation()` | `events + original boundary → relation` | 合并内部对角线并区分 interior/boundary |
| `_determine_ring_geo_status()` | `ring + workspace → status` | 形成环特有状态并引用共享 findings |
| `_determine_bond_ring_geo_status()` | `bond-ring pair + workspace → status` | 对 surface ensemble 取共识 |
| `_determine_molecule_geo_status()` | `workspace → molecule status` | 聚合四通道证据，不做业务映射 |
| `_find_bond_ring_piercings()` | `minimal workspace → piercing report` | 默认热路径，只计算确认互穿所需证据 |
| `find_bond_ring_piercings()` | `Molecule → BondRingPiercingReport` | 小型公开包装，分别返回 confirmed/unresolved/non-piercing-contact pairs |
| `assess_optimization_readiness()` | `Molecule + readiness policy → report` | 唯一的 force-field 前置决策入口 |
| `analyze_geometry()` | `Molecule + analysis level → report` | 显式请求完整或离线几何诊断 |
| `determine_geo_status()` | `typed target → typed status` | 唯一公开详细判定入口 |
| `is_geo_reasonable()` | `local status or analysis report + policy → verdict` | 先验证 coverage，再做规则映射；不重新计算几何 |
| `evaluate_geometry_quality()` | `Molecule → quality report` | 接入现有完整质量门 |

异常、Enum、dataclass 放在文件顶部；helper 按从基础到高级的顺序分区；公开接口全部放在底部并
列入 `__all__`。

### 9.5 环感知范围

默认路径不能把当前近线性的 cycle-basis 类算法无条件替换成最坏指数级的全部 simple cycles：

1. 默认 `ring_scope="ligand_skeleton"`，先移除金属—配体边；
2. `ring_piercing` 使用显式声明的 `edge_shortest_cycle_family`；其严格定义见本列表后的说明，
   不能依赖 NetworkX 遍历或边插入顺序；
3. `diagnostic` 可显式选择 bounded chordless cycles；`exhaustive` 才可选择 all simple cycles，
   两者都必须有 ring-count/size budget；
4. `3<=n<=max_ring_size` 进入 surface 检查；更大环记录 `UNSUPPORTED_RING_SIZE`；
5. `ring_scope="full_graph"` 仅由用户显式请求，并标注金属配位形成的图环；
6. edge-shortest family 可能遗漏不属于该 family 的复合大环孔，这是 fast mode 的明确适用范围
   限制，不得宣称完整 simple-cycle coverage；`READY_WITHIN_SCOPE` 的 scope 必须包含
   `ring_family`；
7. 对“已经属于所选 ring family”但因 `max_ring_size`、ring-count cap、surface budget 或算法中断
   未完成的环：若其 AABB 与任一非环 physical bond 仍可能相交，必须记录 omitted/unresolved 并
   返回 `INDETERMINATE`；只有 broad phase 能严格排除所有候选 pair 时才可安全跳过；
8. `coverage_complete_within_scope=True` 只表示所声明 family 中所有相关 pair 均已评价；
   `global_coverage_complete` 只有目标图的全部 simple cycles 已完整枚举，且每个 cycle 要么完整
   评价、要么由 broad phase 严格证明不存在候选，同时不存在 max-ring-size、ring-count、surface
   或 triangle-test omission 时才能为 true。普通默认报告通常应明确为 false。

`RingOrigin` 的分类不得靠元素猜测：先构造移除全部已标记 metal–ligand coordination edges 的
ligand-skeleton graph；若 cycle 的每条 boundary edge 都仍存在于该图中，则为
`LIGAND_SKELETON`，否则为 `COORDINATION_GRAPH`。无金属体系中的 cycle 自然落入前者。任何 edge
semantic 未解析时，该 origin 必须为 `UNRESOLVED`，并映射为
`INDETERMINATE / REASSESS / UNRESOLVED_RING_ORIGIN`，不能默认当作 ligand ring。

这里必须区分两个阶段：若未知 edge semantic 已使所请求 `ring_scope` 的 target graph 本身无法
确定，则 `ring_perception_complete=False`、`coverage_complete_within_scope=False`，即使尚未发现
任何 piercing 也按 `INCOMPLETE_COVERAGE / REASSESS` 处理；只有 target graph 已经可以枚举且已取得
confirmed piercing、但该 ring 的 ligand/coordination 来源仍无法归类时，才使用
`UNRESOLVED_RING_ORIGIN`。

`edge_shortest_cycle_family` 的数学定义如下。先由 `ring_scope` 选择 target graph：默认
`ligand_skeleton` 使用移除已标记配位边后的图，显式 `full_graph` 使用完整图。对 target graph
中的每条边 `e=(u,v)`，临时移除 `e`，求 `u` 到 `v` 的最短路径长度 `d_e`，并收集长度恰为
`d_e` 的全部最短路径；每条路径加回
`e` 形成一个 cycle。对所有 edge 的结果取并集，再按循环移位/反向规范化去重。只有
`d_e+1<=max_ring_size` 的 cycle 进入 surface 检查；若并列最短路径枚举超过 ring-count budget，
`ring_perception_complete=False`。因为所有并列最短路径都必须收集，该集合的定义与 edge 插入顺序
和任意 tie-break 无关；原子重编号后只需在显式图同构映射下比较等价 cycle keys。它仍不等于全部
simple cycles，因此默认结果只能称为 `READY_WITHIN_SCOPE`。

这不妨碍显式请求的可选全局 Bond–bond 分析检查全部显式键；ring scope 只影响“哪些闭合边界
被当作环面”，可选全键分析仍不得反向改变首版 readiness。

## 10. Force-field 接入规则

默认 `complexes_build` 热路径只调用
`assess_optimization_readiness(config=DEFAULT_READINESS_CONFIG)`；该接口的固定语义就是
`ring_piercing`，没有可扩大拒绝范围的 level 参数：

~~~text
candidate coordinates
    -> optimizer preconditions
       ├── invalid -> precondition repair / reject
       └── valid
            -> bond–ring piercing preflight
               ├── REBUILD_RECOMMENDED -> opening/rebuild branch
               ├── INDETERMINATE
               │     ├── coverage incomplete -> raise budget / offline reassess
               │     └── relaxable geometry  -> one minimization + mandatory recheck
               └── READY_WITHIN_SCOPE  -> ordinary local minimization
                                            -> convergence checks
                                            -> post-minimization piercing recheck
~~~

| 阶段 | Readiness 处理 | 可选 geometry diagnostics |
|---|---|---|
| candidate | confirmed piercing 不进入普通 minimizer；unresolved 不得当成 clear | atom crowding、键长和 ring shape 只记录或用于候选排序 |
| refinement | 重构/定向解结后重新创建坐标 snapshot，再次判定 | 可以按 finding 做定向扰动，不把 warning 当作拓扑失败 |
| final | 必须再次确认无 piercing，并检查 backend 收敛与拓扑保持 | severe contact 若仍存在可按独立 post-check policy 报错；不能反向宣称初始结构不可优化 |

可选 `bond_embedding` 报告 supported physical bonds 的精确 crossing/positive overlap，但首版
不把它接入 rebuild branch。若未来性能与恢复率 benchmark 支持升级，必须另行设计并评审策略；
纯 near-clearance、SVD rank、aperture 和多原子拥挤分数始终不作为默认 pre-minimization blocker。

建议的定向修复输入：

- `AtomCrowdingFinding`：提供拥挤中心和参与原子；
- `BondPairFinding`：提供碰撞键、最近点和线段参数；
- `BondRingGeoStatus`：提供 piercing bond、ring 和交点；
- `RingGeoStatus`：提供塌缩指标或 surface 构造失败原因。

稳定键交叉、正长度重合和环互穿通常是构筑问题，不得假设普通局部优化一定能自动解开；其中只有
后者是首版必须实现的核心路径。

迁移时需要覆盖：

- `geometry.bond_intersects_ring()`；
- `geometry.find_bond_ring_intersections()`；
- `geometry.has_bond_ring_intersection()`；
- `geometry.is_geometry_reasonable()`；
- `CyclePlanes.is_line_intersect_the_cycle()`；
- `Molecule.has_bond_ring_intersection`；
- `Molecule.intersection_bonds_rings`；
- `Ring.is_bond_intersect_the_ring`；
- force-field candidate rejection、perturbation、refinement 和 final gate；
- `closest_ring_edge_to_bond()` 与 `closest_ring_opening_edge()`。

现有 `is_geometry_reasonable()` 不能继续同时承担“完整几何是否干净”和“能否启动 optimizer”两种
含义。迁移完成后，前者连接 `analyze_geometry()`，后者只连接
`assess_optimization_readiness()`。不保留把 `UNCERTAIN/INDETERMINATE` 静默压缩成 `False`
或“没有互穿”的兼容路径。

## 11. 验收测试矩阵

验收顺序必须与业务优先级一致：先证明默认门控只检查必要内容且不会错误放行，再验证可选的
完整几何分析。后者失败不得用兜底分支污染或放宽前者。

### 11.1 Core readiness contract

最小状态矩阵必须逐项断言 report 的 disposition、readiness、action 和 failure class：

| 输入证据 | disposition / readiness / action | `failure_classes` |
|---|---|---|
| 有效无环分子，ring perception 完整 | `EVALUATED / READY_WITHIN_SCOPE / PROCEED` | `()` |
| 声明范围内所有相关 ring–bond pair 均完整 clear | `EVALUATED / READY_WITHIN_SCOPE / PROCEED` | `()` |
| 至少一个完整确认的 ligand-skeleton piercing | `EVALUATED / REBUILD_RECOMMENDED / REBUILD` | `(BOND_RING_PIERCING,)` |
| 显式 `full_graph` 中只有 confirmed coordination-graph piercing | `EVALUATED / INDETERMINATE / REASSESS` | `(BOND_RING_PIERCING,)`，ring origin 必须随 finding 保存 |
| confirmed piercing 但 ring origin 未解析 | `EVALUATED / INDETERMINATE / REASSESS` | `(UNRESOLVED_RING_ORIGIN,)` |
| 只有 surface undefined/disagreement 或可能遮蔽 interior hit 的 numeric ambiguity | 默认 `EVALUATED / INDETERMINATE / PROCEED_AND_REASSESS`；strict policy 为 `REASSESS` | `(UNRESOLVED_RING_SURFACE,)` |
| 只有稳定的 non-piercing endpoint/coplanar contact | `EVALUATED / READY_WITHIN_SCOPE / PROCEED`，并保留 diagnostic | `()` |
| target bond 与真实 ring edge 精确 crossing | `EVALUATED / INDETERMINATE / PROCEED_AND_REASSESS`；可选 `bond_embedding` 只补充碰撞证据 | `(RING_EDGE_COLLISION,)` |
| 相关 ring/pair 因 ring/surface/triangle budget 未评价 | `EVALUATED / INDETERMINATE / REASSESS` | `(INCOMPLETE_COVERAGE,)` |
| 非有限坐标、不同原子坐标重合或零长度 physical bond | `INVALID_INPUT / INDETERMINATE / REJECT_PRECONDITION` | `(INPUT_PRECONDITION,)` |

组合证据的优先级必须固定测试：

1. invalid input + apparent piercing：input precondition 优先，不把无效几何称为 confirmed；
2. 一个完整 confirmed piercing + 其他 unresolved pair：动作仍为 `REBUILD`，但
   `scan_complete=False`；
3. 已检查 pair 全 clear + 一个相关 omitted/超预算 ring：`INDETERMINATE`；
4. `has_confirmed_bond_ring_piercing()==False` + unresolved surface：布尔薄包装为 false，但 readiness
   必须为 `INDETERMINATE`；
5. 只有 atom crowding、普通异常键长、ring rank/aperture warning 或非环 bond crossing 时，
   默认 `ring_piercing` readiness 不变；显式 `analyze_geometry()` 仍报告对应异常。

### 11.2 热路径隔离、缓存与生命周期

使用 spy/counter 证明一次默认 `ring_piercing` 调用：

- `_build_dense_atom_distance_oracle()` 调用 0 次；
- `_build_atom_spatial_index()` 调用 0 次；
- 全局 `_candidate_bond_pairs()` 调用 0 次；
- atom crowding classifier、ring intrinsic scorer、`analyze_geometry()` 调用 0 次；
- 只允许一次坐标/拓扑 snapshot、一次声明 ring-family 感知、bond–ring AABB、必要的 surface 与
  segment–surface narrow phase；
- 每个 candidate bond–ring pair 最多执行一次窄相；同一 ring 被多根 bond 查询时只构造一次
  surface ensemble；目标 bond–ring edge 关系经 canonical registry 至多计算一次。

第二次独立调用必须建立新 workspace。optimizer 修改坐标后，final gate 必须读取新的
`coordinate_revision/snapshot_id`，旧 report 和旧 surface cache 不得复用；旧 report 自身仍保持
不可变。

另加一条隔离回归：对同一 immutable snapshot 与同一 `BondRingSearchConfig`，依次执行
`assess_optimization_readiness()` → 任意 level 的 `analyze_geometry()` →
`assess_optimization_readiness()`，前后两个 readiness report 必须逐字段相同。若 optional analysis
使用不同的 bond-ring config，其 report/cache key 必须包含完整 config fingerprint，既不得覆盖也
不得被 core report 复用。

### 11.3 Bond–ring 几何、覆盖率与 oracle

1. cyclopropane、cyclobutane、chair/boat cyclohexane、cyclooctane 都有 clear 正例；
2. 稳定有限段穿过环内部为 `PIERCED`；只有延长线命中仍为 `CLEAR`；稳定 endpoint/coplanar
   contact 是 resolved non-piercing diagnostic；exact ring-edge hit 是独立 collision finding；只有
   boundary tolerance band 和数值过渡带为 unresolved；
3. L 形凹六边界外部 probe 在微小 z 扰动下始终 clear；四顶点双对角线反例得到
   `SURFACE_DISAGREEMENT`；
4. 共享一个环原子的外接键在去除端点邻域后再次穿环，必须被发现；
5. 只有 surface set 完整、construction 已解析且 `N_admissible=N_valid=0` 才为
   `SURFACE_UNDEFINED`；surface cap 前后得到 `SURFACE_SET_INCOMPLETE`，
   triangle-test cap 前后得到 `PAIR_EVALUATION_INCOMPLETE`，ring-count cap 则令
   `ring_perception_complete=False`；另构造一个三维谓词落在容差带的 triangulation，必须得到
   `SURFACE_UNSTABLE` 而不是把它丢弃后三角面共识；逐项断言 6.4 节四条 flag/count 不变量，
   各 reason 和计数不能混用；
6. broad-phase 正确性测试使用足以完成两条路径的高预算，以“关闭 AABB 的全部 ring–bond pair
   扫描”为 oracle；optimized path 的 confirmed/unresolved/non-piercing-contact 集合必须完全一致。预算 cap 行为另按
   第 5 项测试，不能因 oracle 检查更多 pair 而制造假差异；
7. 无环、单环、fused、spiro、bridged、恰好 `max_ring_size`、`max_ring_size+1` 均验证 coverage
   字段；相关大环或 omitted ring 不得返回 ready；
8. `ligand_skeleton` 不包含金属配位图环；`full_graph` 只在显式请求时包含并标记来源；任何
   未解析 edge semantic 必须产生 `RingOrigin.UNRESOLVED` 和
   `INDETERMINATE / REASSESS / UNRESOLVED_RING_ORIGIN`；
9. canonical ring set 不依赖边插入顺序；原子重编号后，在显式图同构映射下得到等价 ring keys，
   不能只对 `networkx.cycle_basis()` 的偶然输出排序；
10. 刚体平移、旋转和 ring 循环移位/反向不改变关系；同一坐标分别以 Å 与可明确换算的其他
    单位输入时结果一致；未知单位为 `INVALID_INPUT`；输入坐标分辨率变化只按声明容差影响临界状态；
11. 一个 surface 上多个命中必须保存唯一 hit count、方向和 parity；内部对角线命中要合并，
    不能误作真实边界；
12. 每个 report 的 detected/evaluated/omitted rings、candidate/evaluated/core-unresolved pairs、预算
    使用量、`coverage_complete_within_scope` 和 `global_coverage_complete` 必须与实际计数一致。

### 11.4 Force-field 生命周期集成

1. mock optimizer 按 action 而非只按 readiness 分派：`REJECT_PRECONDITION/REBUILD/REASSESS` 不进入
   ordinary minimizer；`PROCEED` 调用一次；`PROCEED_AND_REASSESS` 调用一次并强制执行终态 gate；
2. candidate 初检 clear，optimizer 改坐标后 final gate 必须重新 snapshot 和评价；若终态产生
   piercing，最终失败并保留新的交点证据；
3. rebuild/定向修复后必须重新评价，不能复用旧状态；每个 epoch 记录 readiness、action 与
   snapshot ID；
4. `bond_embedding` 只有显式启用时才扫描 disjoint bond pairs；其实现存在与否不能改变默认
   `ring_piercing` 结果；
5. candidate、refinement、final 三阶段均使用同一份显式 config/policy，或在日志中记录变更，
   不允许阶段间静默改变 scope 和阈值。
6. `PROCEED_AND_REASSESS` 的终态转移必须封闭：终态 clear → 成功；终态 confirmed piercing →
   失败并转 `REBUILD`；终态仍 core-unresolved → 失败关闭或返回 `REASSESS`。同一 candidate 不得
   再次进入 ordinary minimizer，从而形成无限“优化—未决—再优化”循环。

### 11.5 性能门槛与可恢复性实证

固定 benchmark corpus 的文件哈希、每组样本数、随机种子、线程数、Python/NumPy/Open Babel
版本和 CPU 信息。至少覆盖 `N≈50/200/500/1000`，并按 ring 数、最大 ring size、candidate
ring–bond pair 数与 triangle-test 数分层；同时报告 `INDETERMINATE` 比率，防止靠低预算快速退出
“刷过”耗时门。每组先做 5 次进程/库预热，再正式重复 20 次并报告 median/p95/max；所谓 warm
不允许跨调用复用 geometry workspace 或结果 cache。CI 只跑操作计数与复杂度围栏；wall-time
门槛在固定基准机运行。

默认 `ring_piercing` 的首版通过线为：

- 相对同一分子的一次固定步数短 UFF 最小化，median wall time 不超过 5%，p95 不超过 10%；
- `N<=200` 的参考 CPU corpus 上 p95 绝对耗时不超过 50 ms；此值须随基准机器记录，不能跨机器
  伪装成通用常数；
- 任意输入都不超过配置的 ring/surface/triangle 计数预算；预算不足返回 `INDETERMINATE`；
- 内存门使用可复现的对象存活与斜率判据：预热 50 次后运行 10 个、每个 100 次调用的窗口，
  `weakref` 断言 workspace/surface ensemble/report 在释放引用后均无意外存活，`tracemalloc` 拟合的
  retained-heap 斜率不超过 `1 KiB/call`；RSS 仅记录，不作为受 allocator 高水位影响的硬门槛。

`bond_embedding` 只有同时满足以下条件才可提议在后续版本进入 readiness 设计评审：与 brute-force pair oracle 零差异；
预先人工标注的正常配合物不少于 500 个且零误拒；p95 增量不超过短 UFF 的 10%；在配对固定失败
集上，build 成功率绝对提高至少 5 个百分点且 molecule-level cluster bootstrap 的 95% CI 下限
大于 0。否则保留为显式选项。

“局部最小化能否修复”不能只凭直觉。每类缺陷至少使用 20 个不同结构、每个 5 个扰动/seed，
分别测试一次普通 UFF 最小化与完整 `complexes_build` epoch。一次成功要求同时满足：optimizer
收敛、目标缺陷消失、键拓扑不变、指定立体化学不变、配位关系符合该用例 policy。报告释放率及
molecule-level cluster bootstrap 95% 置信区间；同一 molecule 的 5 个 seed 是相关重复，不能当作
100 个独立 Bernoulli 样本。终态 piercing 必须用与输入完全相同的 search config 复检：

- 若某类普通应变的释放率至少 95%，且所有成功样本均保持拓扑/立体化学，支持其作为默认
  non-blocker；低于该线不自动升级为 blocker，只说明需保留 post-check；
- 若 confirmed piercing 在 one-shot local minimization 中的释放率 95% 置信区间上限低于 5%，
  为 `REBUILD_RECOMMENDED` 提供直接实证；若未达到该证据线，de-novo build 仍可遵循用户批准的
  保守 `REBUILD` policy，但 report/documentation 必须标记
  `policy_basis=conservative_build_policy`，且不得宣称普适不可修复；
- 完整 build epoch 的成功不能算作 local minimizer 自己解结，二者结果必须分栏。

### 11.6 可选几何分析：数学与阈值

1. 每个 low-is-bad 指标测试 `h-delta, h, h+delta, w-delta, w, w+delta`；
2. 每个 high-is-bad 指标测试 `w-delta, w, w+delta, h-delta, h, h+delta`；
3. 所有数值风险位于 `[0,1]` 并随缺陷严重度单调；hard finding 得到 `S_geo=0`，warning-free
   得到 `S_geo=1`，不可量化不确定性得到 `S_geo=None`；
4. 零长度键、零面积三角形和 SVD 数值零点进入明确状态；
5. 刚体平移、旋转不改变结论；原子重编号后，在图同构映射下得到等价 finding 集合。

### 11.7 可选原子距离与拥挤分析

正例至少包括 methane、ethene、ethyne、benzene、常见氢键、离子对与正常 Fe–N/Cu–N/Eu–O
配位结构。反例与边界至少包括：不同非键原子完全重合；多原子共同拥挤；显式键压缩/拉长；
missing-edge advisory；整体 `0.1x/10x` 缩放；半径缺失；普通/H-containing profile 的
`y=h_y,w_y`；`x=0.60,0.80`；`C_i=0.40,0.75`；`g=2/g=3/remote` 分流；不同 component；
metal–metal 与未标定 metal–ligand。多体拥挤始终不得单独改变 readiness。

### 11.8 可选全键对分析

必须覆盖：remote 键内部横穿、正长度共线重合、未连接端点落到另一键内部、普通共享端点 V 形、
共享端点同射线重合、`ONE_BOND_SEPARATED`、多根 M–L 键共享金属、非同一原子的端点重合、
`q_bb=h_bb/w_bb`、方向与共线容差边界、半径缺失和 unsupported edge semantic。每个 broad-phase
candidate 至多一次 narrow phase；结果必须与 brute-force oracle 一致并可从稳定 bond indices
反序列化。

### 11.9 可选环形状分析

测试 planar/puckered、rank/aperture warning、projection degenerate、surface unstable、边界自交与
原子拥挤引用。`q_rank=0.05/0.12`、`q_aperture=0/0.15` 及 SVD 零点的等号归属必须固定。
正常 chair/boat 与合理宏环不能仅因非平面而 hard fail；ring finding 只能引用 canonical atom/bond
event，不复制风险。

### 11.10 全分子可选分析与去重

分别构造 atom-only、bond-only、ring-only 与 bond–ring-only 异常，再构造四者并存的分子。
所有 evidence 均保留，但每个 canonical event 只有一个 owner/`event_id`，跨视图引用均可解析且
不携带第二份 risk；`E_geo` 取最大而不求和。每类 lazy resource 至多构造一次，坐标变化使 cache
失效，Python 3.9–3.14 与支持的 NumPy 版本得到相同分类。

## 附录 A：历史缺陷的动态证据

审查基线中，旧实现的重要位置包括：

- `_segment_intersects_triangle()`；
- `_line_intersects_polygon()`；
- `bond_intersects_ring()`；
- 锁定 historical center-fan 行为的 geometry test。

L 形凹六边界和位于凹口外部的有限 probe 曾得到：

~~~text
dz=0       branch=planar  hit=False
dz=2e-8    branch=planar  hit=False
dz=5e-8    branch=fan     hit=True
dz=1e-7    branch=fan     hit=True
dz=1e-3    branch=fan     hit=True
dz=0.2     branch=fan     hit=True
~~~

仅增加 `5e-8 Å` 的 z 扰动就翻转布尔结果。原因不是化学结构改变，而是代码从 planar branch
切换到 centroid fan，后者在凹环中生成虚构面积。新验收目标应是完整 surface 语义稳定，而不是
保留旧分支输出。

## 附录 B：为什么必须区分三类几何事件

### B.1 原子拥挤不等于键碰撞

两个原子可以严重接近而相关键段并未相交；反之，两根长键可以在中点横穿，而四个端点都相距
很远。因此在显式完整分析中 atom-pair 和 bond-pair 必须独立检查。二者可以共享坐标、半径和
距离基础数据，但不能互相替代；默认 `ring_piercing` 门控无需为了这项分析主动计算它们。

### B.2 环边碰撞不等于穿环

目标键撞到环边时发生的是两条有限键段相交。目标键从环孔中央穿过时，可能离每根环边都很远，
但它横穿了环的 spanning surface。二者的修复方向不同：

- 撞环边：移动碰撞键或局部边界；
- 穿环：改变目标键或整个组分相对环孔的拓扑路径。

因此底层 segment relation 可以复用，finding 所有权必须分开。

### B.3 非平面环面不唯一

非平面闭合空间折线没有天然唯一的二维内部。例如：

~~~text
p0=(-1,-1,0)  p1=( 1,-1,1)
p3=(-1, 1,1)  p2=( 1, 1,0)
~~~

两条不同对角线形成不同离散曲面，短键段可能只穿过其中一个。按循环顶点顺序枚举组合三角剖分、
提升到三维、剔除非嵌入曲面后再取共识，可以减少任意选择一条对角线造成的误判，但仍是明确声明的
vertex-only 工程模型。

对 `n<=8`，凸 `n` 边形最多有 Catalan 数 `C_(n-2)` 个三角剖分，八边形为 132 个。
Earcut 只能提供一个剖分，不能证明“所有剖分”都一致。完整性测试需要使用已知 Catalan 计数、
凹多边形计数以及循环移位/反向后的集合等价。

## 附录 C：外部几何库与实现边界

不存在一个可直接替代本方案全部化学语义的单一库：

- NumPy 适合批量距离、向量和 SVD；
- SciPy spatial structure 可选用于 broad phase，但不应成为正确性的唯一来源；
- CGAL 提供成熟、鲁棒的计算几何谓词和 AABB tree，但引入 C++ 绑定与部署成本；
- libigl/trimesh 更适合 mesh 查询，可用作交叉验证 oracle，不能直接定义化学拓扑豁免；
- Shapely/GEOS 主要处理二维投影，不能独立解决三维键段与非平面环面语义。

首版可以用小而明确的 segment–segment 和 segment–triangle 原语，配合高覆盖率边界测试；
若真实规模显示性能或鲁棒性不足，再把窄相位替换为 CGAL 等实现。上层状态、阈值与证据数据类
不应依赖具体几何 backend。

## 附录 D：数据来源、局限与实施顺序

可用于尺度或方法论依据的资料包括：

- Cordero et al., Dalton Transactions 2008, DOI: `10.1039/B801115J`：共价半径；
- Alvarez, Dalton Transactions 2013, DOI: `10.1039/C3DT50599E`：vdW 半径；
- MolProbity/Chen et al., Acta Cryst. D 2010, DOI: `10.1107/S0907444909042073`：
  拓扑感知的 clash 思路；
- Cremer and Pople, JACS 1975, DOI: `10.1021/JA00839A011`：环 puckering 描述；
- Bruns and Stoddart, Chem. Soc. Rev. 2009, DOI: `10.1039/B819333A`：机械互锁分子；
- Ayme et al., Chem. Soc. Rev. 2013, DOI: `10.1039/C2CS35229J`：分子结与互锁拓扑。

这些文献不直接给出本文的 `0.10/0.25` 等 hard/warning 数字。实施前必须：

1. 固定共价与 vdW 半径表的确切来源、版本和许可证；
2. 把源单位明确转换为 Å，并用已知元素值做单元测试；
3. 不得宣称当前 `periodictable`/JSON 数据已经等同于 Cordero/Alvarez 表；
4. 用正常和人工畸变的有机物、离子对、金属配合物校准所有 v0 阈值。

首轮明确不处理：

- 键角和扭转角合理性；
- 力场能量、梯度与真正势能；
- 自动成键或断键；
- knot/link invariant；
- 机械互锁结构的自动意图识别；
- 周期边界条件；
- 隐式氢的空间占据；只有具有显式坐标的原子进入本轮距离与拥挤门控；
- 大于 `max_ring_size` 的精确全曲面枚举。

建议实施顺序：

1. 先实现最小坐标前提、bounded ring perception、AABB 与有限 segment–surface piercing；
2. 提供 `find_bond_ring_piercings()` 和 `assess_optimization_readiness()`，接入
   candidate/final 两端的核心闭环；
3. 对 current one-shot Open Babel/UFF、完整 build epoch 和 `ring_piercing` gate 分开做时间与恢复率基准；
4. 若基准达标，再把精确 physical bond crossing/overlap 作为 `bond_embedding` 可选分析；
5. 随后实现 sparse atom proximity、bond-pair registry、ring intrinsic metrics 等
   `diagnostic` 能力；
6. 最后实现 dense/full-pair/all-surface oracle 的 `exhaustive` 模式，并用正负样本校准 v0 阈值；
7. 每个节点单独提交并保留可回退性。
