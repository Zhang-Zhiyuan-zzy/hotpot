# 化学类型公开接口一致协议与事实存储设计

状态：Plan20 实施设计；本文件定义所有 Hotpot 化学类型公开接口的共同协议，
与[主规划](fact_rule_architecture_implementation_plan.md)共同规范实施。
所有新增、删除、改名、读写权限、返回类型和公开路径变化均须逐项审阅；
本文记录已指定的规划目标及待审细节，没有创建生产接口。Atom 的列式事实表是
该协议的一项具体实现；MetalCluster 在此登记类型关联，具体实施归 Plan21。

## 1. 适用范围与类型角色

协议覆盖化学类型的公开构造器、属性、方法、集合访问、返回对象和转换边界。
Molecule、Atom、Bond、Crystal 适用，构象与分子集合、环/角/二面角/原子对
视图、立体化学值、科学结果及公开运输合约同样适用。未来 MetalCluster 按注册
聚合类型遵守同一协议。独立 wrapper 保留已批准的组件接口；第三方类型由
Hotpot adapter 在边界遵守协议，不改写其 API。

| 类型角色 | 代表类型/接口 | 权威、权限与生命周期 |
|---|---|---|
| 事实载体及领域句柄 | Molecule、Atom、Bond | 每项事实有唯一 owner；句柄/属性投影同一事实，变更走完整操作 |
| 独立领域值 | Crystal、坐标帧、立体化学值 | 拥有自己的值或引用明确 owner；是否可变按已审合约，不默认增加 setter |
| 注册聚合类型（Plan21） | MetalCluster | 关联 Molecule 及其 Atom/Bond/坐标，内部图或点云、外部多连接位点超原子；具体存储与权限后定 |
| 集合及选择入口 | MolBundle、构象集合/投影 | 拥有成员关系或选择规则；访问不维护第二份成员事实，显式复制按生命周期创建独立值 |
| 派生视图 | Ring、JointRing、Angle、Torsion、AtomPair、pair table | 固定成员/来源，只读；不成为事实 owner，不借读取反写源对象 |
| 科学输出与报告 | MCA prediction、partial-charge set、优化/状态结果 | 值、单位、来源和保证明确；返回的历史结果不随源对象变更而改写 |
| 快照与运输值 | StereoPayload、ConformerBatch、native buffers | 明确字段、映射、缓冲所有权与有效寿命；不成为第二份持久事实源 |

一致协议按角色确定行为，不要求共同 Python 基类、公开 `Protocol`、万能 DTO、
统一物理布局或相同可变性。Molecule UID、parent、dirty 状态只在确有来源依赖
时使用，不强加给独立 Crystal 或一次独立调用返回的构象值。

MetalCluster 的当前登记仅记录类型骨架和未来
`Molecule.metal_clusters -> list[MetalCluster]` 关联。注册结构定义与分子中的实例
相区分，成员事实继续引用原 owner，图/点云及超原子连接表达不成为第二份持久
拓扑。结构、匹配通式与其他公开成员留待用户规定，见
[Plan21 草稿](../21_metal_cluster/metal_cluster_implementation_plan.md)。

## 2. 唯一事实与统一声明

每个类型将唯一事实、公开表达和派生值分开。一个事实只保存一次：
atomic_number 与 symbol 是同一事实的两种表达；Atom/Molecule 坐标入口指向
同一当前坐标；Bond 的 kind/order 表达一套受约束的键表示；Crystal 的六参数
是约定的晶胞值，矩阵由数学规则计算；结果的对象投影不另存一份预测数组。
来源、presence 和有效性是解释值所需的元数据，不能演变成第二份当前事实。

私有声明按类型和领域组织，共用定义结构，不建立一个携带所有可选字段的总表：

| 私有模块 | 职责 |
|---|---|
| `structure/_fields.py` | 各化学类型的事实字段、公开映射、单位/编码、权限、结果或视图来源 |
| `structure/_dependencies.py` | 跨类型事实组依赖、影响范围、按需重算和读取策略 |
| `structure/_state.py` | 引用唯一声明并组织载体状态，不复制字段目录 |
| `structure/_commit.py` | 完整写集的安装、依赖标记和回滚 |
| `structure/_atom_table.py` | Atom 的类型化数值存储，不包含通用接口政策或化学规则 |

每条接口声明使用相同信息顺序：`type/interface, role, owner, fact_or_dependencies,
shape/dtype/unit, presence/source, read_policy, write_policy, encode/decode,
operation, invalidation_scope`。字段可以按角色省略，无需为每个属性新增记录类。

已获批准的字段声明应使相应类型的存储访问、属性映射、快照选择和依赖维护
协同更新。Atom 可据此生成描述符；其他类型可用已有显式 property、构造器或
组合值，均引用同一语义声明。内部增列或新增私有字段不自动发布同名属性；
不开放运行时任意字段注入，不覆盖已有方法。`new_attrs` 仅为扩展示例。

## 3. 接口映射、读取权限与变更权限

每个公开入口都要回答：它读哪个事实或计算；事实由谁拥有；输入能否完整确定
变更；读取是否需要当前性；写入影响哪些依赖；未知、失效、冲突如何表达。
构造器、方法和集合操作也需要该映射，不能只盘点 property。

属性可写须同时满足：对象角色允许变更、输入能唯一确定目标事实/完整表示、
写入不绕过完整性要求，并且该 setter 已获批准。输入唯一确定约定的目标事实
或完整表示，并不要求编码是数学双射，也不自动授权 setter。只读视图和不可变
结果中的数值即使可独立赋值，仍保持只读。多输入共同决定事实时，使用完整
method，由 chemistry/physics/mathematics
规则计算提案、operation 提交；不能依赖 setter 的调用顺序补全表示。

| 类型/已有入口 | 事实或计算映射 | 协议下的权限与处理 |
|---|---|---|
| `Atom.atomic_number` / `Atom.symbol` | 同一 atomic_number | 保留已存在读写入口；编码/翻译后走同一路径，不即时重算化学 |
| Atom x/y/z、`Atom.coordinates`、`Molecule.coordinates` | 同一 Cartesian 坐标块 | 现有标量/批量赋值共用提交与失效；读取不泄露权威可变别名 |
| `Atom.formal_charge`、`implicit_hydrogens` | 明确形式电荷/隐式H计数声明 | 建议保留唯一事实的赋值，来源与依赖同步更新；冲突不靠默认值覆盖 |
| `Bond.bond_kind` / `Bond.bond_order` | 一套耦合键表示 | 保留经批准入口；不能因可映射就给当前只读 kind 新增 setter |
| `Molecule.charge` | 分子净电荷声明 | 与原子形式电荷分开；覆盖完整时重新确认总量一致性，不自动分摊差额 |
| `Crystal(a,b,c,alpha,beta,gamma)`、`Crystal.from_matrix()` | 同一六参数晶胞值 | 构造/转换完整确定值；参数可写性按值对象合约审阅，不自动增设矩阵 setter |
| `Bond.length`、`Angle.degrees`、`Torsion.degrees` | 当前几何的派生计算 | 只读；不能由单个读数唯一反推整套原子坐标 |
| `Ring.is_aromatic` | 完整芳香赋值的视图 | 建议只读；移除现有 setter 的变更须审阅，写入从 Molecule 完整操作发起 |
| `conformer_add/get/load/clear` | 源帧/生成结果、当前坐标及集合选择 | 添加不隐式选中；load 完整更新当前坐标；clear 的范围显式记录 |
| `MolBundle` 的集合接口 | 有序分子成员 | 只改变该集合承诺的关系，不复制或重新拥有每个 Molecule 的事实 |
| `Atom.mca`、`Molecule.mca_sites`、`Atom.partial_charge` | 当前选定科学结果 | 只读投影；有效性归选定结果 owner，已返回历史值保持自身内容 |
| stereo 与 wrapper 的公开值/报告 | 已审输入输出 schema | 维持字段、单位、顺序及缓冲保证；不因为统一协议改变现有 ABI |

`Crystal.matrix` 仍是主规划中的候选接口，不能在上述现状表中冒充现有 property。
`atomic_numbr` 不作为新增别名，现有拼写为 `atomic_number`。需要新增 method
时先列出完整签名和权限，未列入审阅表的公开入口不设立。

## 4. 跨类型依赖与按需重算

事实变更只作廉价标记，不立即运行昂贵规则。状态属于权威事实或选定结果的
owner；视图、快照、历史输出不各自保存一套可变 dirty 真相。

| 私有状态 | 含义 | 读取策略 |
|---|---|---|
| `CURRENT` | 值与当前依赖匹配 | 直接读 |
| `NEEDS_RECALC` | 已有计算值的依赖改变 | 需要时调用已声明纯重算操作；不返回旧值冒充当前值 |
| `ABSENT` | 没有选定值 | 采用已审无结果语义，不自动启动计算后端 |
| `UNRESOLVED` | 缺少输入或选择 | 报告未知/未解析，不补零 |
| `CONFLICT` | 显式事实与约束不一致 | 报告冲突，完整操作处理，不静默覆盖显式来源 |

对需要来源有效性管理的事实/结果，固定组位图与有界 dirty 种子是有效性
基础设施，不是 revision counter、新图或通用缓存。标记先按组登记受影响实体；
首次需要时按声明范围处理。只有需拓扑传播的组才沿唯一 topology index 展开
相应连通域并在任务内复用；晶胞、纯几何等组按各自依赖处理，独立值不强加
拓扑状态。读取必须先检查目标组及上游是否有未展开种子，再判断目标 CURRENT，
避免“改 A、先读远端 B”时漏掉芳香/共轭影响。无关组不承担展开成本，不为
每次标量赋值扫描或复制全分子。

| 事实变更 | 同步安装/维护 | 标记的依赖与保留范围 |
|---|---|---|
| 原子/键成员、Bond 端点 | 成员、表行、唯一 topology index、stereo/约束/帧同步 remap 或原子拒绝 | 受影响连通域的表示和拓扑相关结果；保留可归属的显式来源 |
| atomic_number | 唯一字段 | 受影响 P2 与 representation 依赖；保留身份、坐标、无关组件 |
| isotope/radical | 对应事实/presence | 规则声明的 isotope/radical 敏感组，不盲目标全部价态 |
| formal/net charge、spin、H | 完整声明或赋值 | 依赖电荷/自旋/H 的表示与结果；保留独立拓扑/几何 |
| 键表示/芳香赋值 | 完整耦合表示 | 下游依赖；本次完整提案已经更新的字段不相互标脏 |
| 当前坐标 | 同一坐标块 | 几何相关结果；保留源帧、拓扑和显式 stereo |
| 晶胞、约束、源帧 | 相应唯一 owner | 仅标各自依赖；独立 Crystal 的旧值不随宿主替换而改写 |
| 结果附着或构象选择 | 来源/依赖检查及完整选择 | 其他受改变事实影响的结果；本次最终输出不自失效 |

来源参与重算决定：推断 H 随元素变化可标记后重算；明确声明的形式电荷不被
默认价态替换。断开芳香环时，推断表示标记重算，显式源表示保留 ledger 并
重新验证/报告冲突。改变总电荷不自动把差额塞给某个原子。

按需重算限于输入充分、有已声明纯规则和执行上下文的组。属性 getter 不下载
模型、启动 GPU、运行外部程序或完整优化；已有能量失效和从未计算 MCA 等
情况按其已审语义报告，等待显式操作。P4 环/路径/角/二面角仍按任务构造，
不恢复载体缓存。持久昂贵输出采用明确的 P2/P3 owner 和依赖策略。

## 5. 读取、提交与生命周期

可变入口执行“编码输入 → 验证完整写集 → 安装事实 → 标记依赖”。需要派生值
时，由 operation 准备上游输入、构造一次快照、调用纯规则，并在获准保留结果
时完整提交。视图需要派生值时委托 owner 对应的准备操作；重算与安装由 owner
的操作完成，视图自身不写事实、不保有第二份 dirty 状态。历史结果/独立值只
解释自身内容，不向活跃分子发起隐式变更。

输入数组默认隔离外部可变别名，独占所有权转移仅用于明确的内部构建流程。
公开数值缓冲区及声明为值、快照或不可变结果的返回对象，按各自合约提供独立
副本或真正不可变内容；不能仅凭 `frozen=True` 或可重新开启写入的 ndarray view
宣称不可变。活跃载体句柄和集合成员可按已审接口返回同一对象，修改须经过
该对象自己的受控入口，不要求复制或冻结载体。后台计算用独立快照；native
可写任务使用任务局部缓冲区。持久事实表与 `NativeMoleculeBuffers` 分别承担
权威与运输。

批量和标量写共用声明及提交逻辑。批量向量化后一次安装；标量只处理触及的
字段/依赖，不复制整表。编码、校验和可能失败的用户调用在安装前完成；跨块
事务保留必要切片以回滚。跨字段快照复制、源身份捕获和写提交共享一致性
边界，不能依赖 GIL 假定 NumPy 不会并发。

有来源的视图/更新同时绑定对象归属与所需内容身份；分子内 UID 的相等不能
证明两个对象属于同一分子。独立值按自身值和寿命标识，不附加无意义的 owner。
长操作在提交前重新核对必要输入；源已变化时不能用旧提案覆盖新事实。

copy/components/merge 依据各自生命周期合约 remap 值、来源、presence、有效性
与 dirty 范围，不把 NEEDS_RECALC/CONFLICT 复制成 CURRENT。结果是否携带、
集合是否共享成员、脱离对象如何检查均明确记录，不用一条深复制规则代替。

优化事务统一安装最终坐标、ensemble、最终能量/力与依赖状态。input provenance
和 output validity 分开；能量绑定最终帧，选取 ensemble member 不使 ensemble
自失效。已返回的科学结果仍是原次执行的历史值；“当前选定结果”由载体投影
进行有效性检查。失败只保留原提交与真实诊断，不暴露半更新状态。

## 6. 实施与跨类型验收

Stage1 为所有公开化学类型建立“类型—角色—入口—owner—事实/依赖—权限—
操作—生命周期—审阅状态”的清单，同时冻结现有签名与效率基线。该清单是
实现和验收依据，不新增每种类型都要继承的运行时接口。

Stage3 建立共用私有声明与提交规则，按类型迁移；Atom 表原型只是其中一个
工作包。Stage5 落地依赖组和已审惰性重算；Stage7 收敛结果、视图、约束与
坐标。单项迁移同时删除旧写者/冗余值，不保留双重生产路径。

验收按角色抽取技术契约：载体别名同源、单次完整写入与权限；Crystal 等值
的构造/单位/生命周期；集合成员及构象选择；视图只读与来源失效；结果历史
内容稳定、当前性投影；运输值的 schema/缓冲/ABI。验证包含 dirty 正反例、
远端先读、一次重算、源变化拒绝、失败回滚、copy/remap/脱离。

性能覆盖各类型的标量/批量路径、矩阵导出、集合操作、几何视图、晶胞转换、
结果投影与运输。记录首次 dirty 读取、重复 clean 读取、分配/转换次数和 RSS，
保证不会因统一协议增加全表复制、重复摘要或无用包装对象。科学观测交付
版本化报告；全局效率遵守主规划的至少持平、总体期望更快要求。

## 7. Atom 的具体事实存储方案

### 7.1 权威与布局

Molecule 私有持有 `_AtomFactTable`，Atom 以稳定 `_uid` 及既有
`_topology_index` 定位成员行。元素、同位素、形式电荷、H、芳香与坐标
不再分别存在表和 Atom.attrs 中。保留 UID/id/idx 的区别，不增第二份行索引。

推荐 NumPy 类型化列/同类型连续块（SoA），用重构前基准和 AoS 原型比较
后确定。混合浮点 N×F 仅作为任务级特征矩阵；每原子 Python dict/dataclass
不作为数值主存储。Bond、Crystal、视图和结果按其数据形态采用适当表示，
无需复制 Atom 的表布局。

| 唯一私有事实 | 存储候选 | 限制 |
|---|---|---|
| atomic_number | uint8 | 不另存 symbol/period/group |
| isotope | uint16 + presence | 未指定与明确值分开 |
| formal_charge | int16 + presence/source | 与净电荷/partial charge 分开 |
| radical electron count | 无符号整型 + presence | 与 spin 声明分开 |
| implicit H / atom aromatic flag | 整型/bool + source/validity | 保留显式来源及耦合赋值语义 |
| current coordinates | float64 N×3 + presence | x/y/z 和 Molecule.coordinates 共用该值 |
| source/display id | 经基线确认的类型化列或边表 | 不替换 UID，不把 idx 作为持久身份 |

最终 dtype 按合法输入域、ABI 与基准确定。symbol、电子组态、valence、
hybridization、邻接、环、距离为派生值；partial charge/MCA 从选定结果投影；
x/y/z_constraint 从 Molecule 的 ConstraintSet 投影，均不新增重复事实列。

### 7.2 行访问、扩展和导出

已批准字段从统一声明接入列分配、Atom 描述符、快照和依赖；不手写多份
getter/setter，也不自动公开私有列。n/s/p/d/f/g 与 electron_configuration
建议改为只读派生；implicit_hydrogens setter 建议保留为明确计数声明；
is_aromatic 的单原子写入建议改由完整芳香操作，具体权限仍待审。

批量编辑只准备触及列/行的候选缓冲区，单次提交；插删同步 remap 坐标、
presence/source/status。脱离 Atom 的行转移至同 schema 只读单行表，活动表
移除该行，保留检查语义且不留下两个可变权威。句柄不因压缩改指另一原子。

公开 `Molecule.atom_matrix` 为完整19列 float64 独立数组的只读 property，
不接收 fields/dtype 参数。它与 Atom.attrs 共用唯一事实投影；既有
atom_attr_matrix 的改名/别名去向单独记录，不能增加第二份矩阵或生成实现。
内部矩阵导出仍明确字段顺序、dtype、单位和未知值策略，不保存持久 one-hot 或
归一化特征，不把 UID 转成有损浮点。字段选择式 native transport 可在有效
寿命和只读保证内借用缓冲区，否则显式打包，不另造通用运输系统。

Atom 专项基准覆盖逐原子读取/赋值、列批量读写、N×3 变换、构建/插删、
N×F 导出、原生打包和逐原子结果投影，确保高效批量访问与标量接口同时成立。

## 8. 公开接口审阅清单

### 8.1 跨类型审阅范围

统一协议不自动批准新增符号、setter 或返回语义。每项变化提交确切类型、
名字/签名、读写权限、输入输出/单位、事实 owner、依赖与生命周期、旧入口
去向及验证依据。用户已指定的未来 MetalCluster 类型及 Molecule 属性登记如下；
尚未规定的细节继续留白。其余未获批准的接口不设立，不从示例推导新 API。

| 类型/范围 | 规划目标与待审项 | 审阅位置 |
|---|---|---|
| Molecule/Atom | 事实投影、attrs、批量与矩阵入口、结果当前性 | 下列 A1–A13 与主规划第16节 |
| Bond | 耦合 kind/order、现有入口权限、隐藏键接口迁移 | 主规划第7/16节；不自动增加 kind setter |
| Crystal | 六参数值/回指、matrix 候选、顶层导出和占位删除 | 主规划 Decision9；不自动新增参数/矩阵 setter |
| MetalCluster / Molecule | 用户指定未来类型及 `Molecule.metal_clusters -> list[MetalCluster]`；构造器、成员、权限、注册/匹配方式、返回语义和导出路径后定 | 主规划 Decision24 与 Plan21；当前只作设计登记 |
| 视图/集合 | AtomPairs 改名、Ring/JointRing 写接口、约束投影、构象/copy/member语义 | 主规划 Decision17/19 与第7节 |
| 结果/运输/wrapper | schema 合并、返回类型、单位/字段顺序、缓冲语义及 ABI 变化 | 数据对象审计 D01–D10、主规划与 Plan17 接口清单 |

### 8.2 Atom 存储落地的具体候选

以下编号用于引用具体提案，不构成另一套 Atom 专属协议。

| 编号 | 公开接口或行为 | 提案 | 当前状态 |
|---|---|---|---|
| A1 | `Atom.atomic_number`, `Atom.symbol` | 保留现有名字与读写；同一 atomic_number 事实，移除立即重算副作用 | 保留签名；副作用变更待审 |
| A2 | `Atom.attrs`, `Atom(attrs_array=..., update_electron_config=...)` | 移除长期可写事实数组暴露；完整导入使用私有builder，Stage5删除两个旧构造参数；保留独立19列快照，Stage7C缺失结果用NaN且保持模型输入编码，详见[具体决定](stage1_contract_decisions.md) | 2026-10-28用户已确认；按Stage3/5/7C分步验证 |
| A3 | `Atom.coordinates`, x/y/z, `Molecule.coordinates` | 保留现有赋值入口、共享唯一坐标事实；返回数组副本/不可变行为按基线明确 | 返回别名行为变更待审 |
| A4 | `Molecule.updata_atoms(*, atoms=None, **columns) -> None` | 按用户指定拼写；Atom句柄选行，None为全部；只接受已审事实/别名，广播、重叠一致、一次校验/提交；不公开UID | 2026-10-28用户确认能力与改名；Stage3实施验证 |
| A5 | `Molecule.atom_matrix` property | 无fields/dtype参数，返回完整19列float64独立数组；沿既有schema/单位，缺失可选结果按A2在Stage7C编码NaN；不自动增加MCA或新私有字段 | 2026-10-28用户确认；复用既有atom_attr_matrix投影，其旧名去向单独确认 |
| A6 | 例子 `Atom.new_attrs` | 仅说明schema联动；具体名字、dtype、权限、依赖另列审阅 | 不添加示例属性 |
| A7 | 现有 `Atom.valence`、`n/s/p/d/f/g`、`electron_configuration`、`implicit_hydrogens`、`is_aromatic` 的赋值权限 | valence/电子组态和芳香投影改只读；隐式H setter保留为明确计数声明，完整推断走既有method；不新增hybridization property | 确切签名/消费者影响在Stage1提交审批，权限改变待审 |
| A8 | `Atom.partial_charge` | 只读 Optional[float]，None为无选定值；全量assignment是唯一写入口（原Decision22提案） | 返回类型/setter变化待审 |
| A9 | `Atom.mca`, `Molecule.mca_sites` | 保留现有名字与无结果异常；从唯一 typed record 投影，状态失效时不返回旧值 | 有效性行为待审 |
| A10 | 公开 readonly result/iterator、dirty状态API、schema registry | 本设计不新增；批量读取先用现有接口或私有operation | 不设立 |
| A11 | 多输入才能决定的新增method | 优先复用现有赋值method；确实缺失时先提交完整签名、映射与权限供审阅 | 尚无获批新增method |
| A12 | 新异常类、领域结果/DTO合并改名 | 与主规划及数据对象审计逐项列明；不能因私有设计需要自动导出 | 待审；不新增公开异常 |
| A13 | `Atom.isotope: Optional[int]` | 若公开：读写同位素质量数，None表示未指定；唯一映射私有isotope值/presence；当前先沿既有stereo transport访问 | 新属性候选，待审，不设立；不附带radical/hybridization新属性 |

A4/A5 已按用户指定的名字和完整矩阵语义确认，不为其他类型自动生成对应方法，也不另建
atom_table/fact_store/edit_attrs 等平行公开入口。保留高效批量提交，
不能退化为必须逐个 setter 才能写入。

A4 只接受已批准可写字段，按 Atom 句柄选行，规范化为私有列更新合约，
不接受任意 setattr/FactPatch。A5复用既有19列顺序，空分子为(0,19)，
不借完整导出引入新科学特征或外部模型；按需选择字段保留为内部任务能力。
