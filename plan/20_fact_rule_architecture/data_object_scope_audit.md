# Plan20：数据对象作用域、重复与精简审计

状态：2026-10-28 文档审计及实施提案；没有修改生产类、函数签名或导出。
代码基线：`d9eb80e`，生产实现基线与 Plan20 的 `0a4b0ee` 相同。
本报告是 [主实施计划](fact_rule_architecture_implementation_plan.md) 的数据合约依据，
按[化学类型公开接口一致协议](chemical_type_interface_contract.md)组织事实、视图、
结果与运输值。公开类、字段、构造器、方法、读写权限、返回类型和导出路径的变更，
均需按本报告第 8 节逐项审阅；本次批准修订规划不等于批准这些公开变化。

## 1. 审计口径与主要结论

扫描当前 `hotpot/**/*.py` 的 **403 个 Python 文件**，全部可被 AST 解析。
90 个文件定义了 **347 个 `@dataclass`、3 个 `TypedDict`、13 个显式 `Protocol`**，
共 363 个候选定义；没有发现 `NamedTuple`、`namedtuple()` 或 Pydantic 数据模型。
计数包含私有类、嵌套类和无字段 dataclass，但不包含 re-export、类型别名、枚举、
异常、测试夹具、第三方源码、C++ structs、JSON/SQL schema 或规划中的拟新增类。

**363 是候选合约库存，不是“363 个纯数据对象”的断言。** 其中包括：

- 科学输入、结果、提案、过程消息等真正的数据记录；
- 有数学行为的 `Point`、`Cycle`、几何关系结果等值对象；
- 采用 dataclass 写法的可执行 stage、adapter、recorder；
- 行为接口 Protocol，以及可变的 GPU 搜索/调度工作状态。

本报告按语义、寿命、生产者和消费者判断是否重复，不按装饰器或字段数量直接删除。
仅凭“当前没有直接名称引用”不能证明无用：公开导出、动态注册、pickle/消息协议、
文档和已有用户预留承诺也属于消费者。

已证实的主要精简机会集中在科学状态结果/诊断、力场停止参数/键快照，以及重复的
CLI 选项定义。已有真实作用的过程消息、独立 wrapper 合约和数据治理快照不宜强行
合成一个万能 DTO。对于 `MolecularFacts`、`TopologySelection`，本报告没有预设其有问题：
前者是当前真实科学边界；后者只是当前 Plan20 提案，生产代码尚无该类。

## 2. 按作用域和功能分类

下表的 D/T/P 分别为 dataclass / TypedDict / Protocol；详细定义索引见附录。
每个域只拥有其语义事实，跨域通过明确投影或序列化交流。

| 作用域 | D/T/P | 主要对象与具体功能 | 寿命、权威和处理方向 |
|---|---:|---|---|
| 科学状态 `calculator` | 105/0/2 | `MolecularFacts`、身份、provider 输入/结果、候选、选择、backend projection | 事实快照、科学推断和执行选择分层；优先消除同域同构定义 |
| 立体化学 `stereo.py` | 20/0/0 | transport records、anchor/carrier/group、payload、验证/赋值/对齐报告 | 运输契约和解释结果分开；Plan17 公共 schema 身份/字段序受保护 |
| 数学几何 `geometry` | 29/0/1 | 点/线/面/环等数学值、relation findings、screening reports | 数学值对象不是 Atom 权威；保留独立数学语义 |
| 搜索 `search` | 5/0/0 | query IR、编译条件、native prepared inputs、匹配配置 | query 值/执行工作状态分开；不另造图存储 |
| 通用原生运输 `_native` | 5/0/0 | 分字段数组/立体化学/晶胞缓冲 | 一次调用的 ABI 投影；不能直接充当持久分子事实 |
| Kekule `kekulize` | 0/0/5 | 只读 Atom/Bond/Ring 等规则协议 | 行为/读取协议，无记录实例；避免与载体类重复定义事实 |
| 构象 `conformers` | 4/0/1 | sampling request/batch、两种 adapter、sampler protocol | request/result 保留；adapter 是行为对象，应归 adapters |
| 力场 `forcefields` | 70/3/1 | options、拓扑快照、native results、轨迹、过程报告、CLI/stage DTO | 区分 native transport、领域结果和报告投影；最高密度精简域之一 |
| 独立 `obWrappers` | 10/0/0 | 规则变更、build/optimization/check/frame reports | 独立组件边界；保留稳定契约，由 Hotpot adapter 投影 |
| 独立 `rdWrappers` | 5/0/0 | ETKDG options、constraints、conformer records/batch、runtime info | Plan17 组件边界；不与中立 sampling batch 盲目合并 |
| AI artifact / CBond / MCA | 37/0/0 | artifact identity、CBond path/structure、MCA DTO/任务/消息/预测/resource plan | model/input/result/IPC/workspace 分域；同域命名可统一 |
| 通用 `pipeline` | 10/0/2 | spec、payload、context、artifact、stage/run result、recovery 配置 | 外部工作流契约；不得依赖 chemistry 内部可变状态 |
| 数据治理 `ds` | 10/0/0 | 证书快照、文件绑定、治理键、入库回执 | 独立信任/一致性边界；不能因带 digest 就与模型结果身份合并 |
| xTB plugin | 18/0/0 | backend info、request/parsed result/run report、state projection、stream/CLI | 独立程序证据、科学投影、流式序列化分层 |
| 通用 process harness | 4/0/0 | request、process result、artifact/process provenance | 可复用进程事实；领域报告优先组合，减少扁平重复字段 |
| ComplexFormer | 7/0/0 | 训练/数据/模型/优化器配置、任务组件和统计 profile | 可选训练域；列出规划问题，独立整理，不混入化学内核迁移 |
| 参数优化 `opti` | 4/0/0 | categorical/discrete/float/int 参数域 | 4 种约束不同；可共享 schema 约定，暂不合成大量可选字段类 |
| `works` | 3/0/0 | 高斯转换/报告中的临时记录 | 随外部 commands/plugin 归属决策处理 |
| CLI 入口与 utils | 1/0/1 | 顶层 command descriptor、`ModelLike` protocol | 行为协议/命令元数据，非化学事实 |
| **合计** | **347/3/13** | **363 个候选定义** | **不能把候选总数视为必须全部压缩的目标** |

`core.py`、`bundle.py`、`graph` 没有以上四类语法定义；并非没有数据对象。
其 carrier、`Conformers`、views、`AtomPairs`、`AtomPairKey` 仍在人工审计范围。
`ds` 的普通 Mapping/slots 对象及许多版本化 dict/JSON 合约也不计入 363；
此次没有为提高“类型覆盖率”而要求把每个 dict 再包成新 dataclass。

未来 `MetalCluster` 归入“注册聚合类型”：关联 Molecule 的成员事实，向外提供多连接
位点超原子表达。当前生产代码没有该类型，不计入上述库存。已有
`plugins/ccdc_api/miner.py::_identify_metal_cluster` 返回金属原子列表，是局部启发式，
不能视为已注册 MetalCluster 定义或直接采用为新匹配标准；相关复用评估归 Plan21。

## 3. 已证实的同构和高相似家族

### 3.1 科学状态结果：五个同构记录应优先收敛

当前五类拥有完全相同的字段序：
`identity, facts_identity, candidates, admissibility, resolution, confidence, findings`。
差别主要是 `candidates` 的元素类型与科学量语义：

| 当前定义 | 代码证据 | 当前用途 |
|---|---|---|
| `NetChargeResult`、`LewisFormalChargeResult` | `calculator/charge/contracts.py:55,81` | 总电荷与 Lewis 形式电荷推断结果 |
| `CoordinationElectronCountingResult`、`OxidationStateInferenceResult` | `calculator/oxidation_state/contracts.py:67,91` | 配位电子计数与氧化态推断结果 |
| `ElectronicStateInferenceResult` | `calculator/electronic_state/contracts.py:75` | 自旋/电子态推断结果 |

`calculator/state_resolver.py:600-608` 已跨结果统一读取 admissibility/resolution，
说明共同外壳确有消费者，而非纯粹形式相似。

提案：一个同域 `StateInferenceResult[T]`，用显式 quantity/tag 区分科学量，
`candidates` 保持静态泛型；空候选时也能无歧义序列化和分派。不要通过五个空子类
再加一个基类来声称“减少了五类”：若保留五个公开运行时子类，类数反而增加。
只有批准公开构造器/返回类型的统一后，5→1 才实际净减少 4。

### 3.2 科学状态诊断与开关：按同域收敛，保留角色和权限

- `ScientificFinding` (`calculator/contracts.py:149`)、`BackendProjectionFinding`
  (`execution_selection.py:95`)、`DecisionReason` (`state_selection.py:63`)
  均为 `code, message, related_identities=()`。
  `StaticEligibilityFinding` (`state_selection.py:55`) 是前两字段子集。
  可采用同域 `StateFinding`，用 code namespace 与承载位置保存规则诊断、选择理由、
  后端投影和适用域角色，4→1 的候选净减少为 3。
- `ProviderToggle` (`calculator/contracts.py:199`) 与 `RuleToggle`
  (`state_rules/contracts.py:100`) 同构；`RuleValue` 已复用 `ProviderValue`。
  可采用 `StateActivation[TId]` 统一记录结构，2→1 净减少 1。
  provider/rule ID 命名空间、默认启用与控制权限仍分别定义；不能因共用记录就把
  两种启用策略合并。

这些类型在公开 API 测试中有导出断言，均为待审变更。`ExecutionPolicySet` 与
`RecoveryPolicySet` 虽同为 `policy_ids`，位于不同执行阶段、有不同合法取值和作用域，
不应据此再建立一个跨域任意 policy 容器。

### 3.3 力场停止条件：存在纯字段搬运层

`forcefields/contracts.py:222` 的 `OptimizationStoppingCriteria` 与
`forcefields/native.py:82` 的 `OptimizationStoppingOptions` 拥有相同的 5 个字段、
字段序、默认值和单位。
`forcefields/workflows.py:280-298` 的 `_native_stopping_options()` 只逐字段复制。

提案：领域层保留一个经过验证的停止参数记录；native adapter 直接将其转为原生
绑定参数，删除中间 Python dataclass，2→1。C++ 绑定结构是 ABI 边界，另行计数且保留。
两个现有 Python 名称均可能被导入，最终名称及移除路径须审阅；不额外增加同义别名树。

### 3.4 力场拓扑记录：相同键值不必维护两份 Python schema

`forcefields/topology.py:32` 的 `BondTopologySignature` 与
`forcefields/trajectory.py:137` 的 `BondTopology` 均为
`atom_indices, bond_order, bond_kind`。
后者额外有端点规范化和 `from_bond()` 工厂，前者的生产函数同样规范化端点。

提案：同一不可变键记录服务事务参考和历史轨迹；采集工厂放 structure adapter，
轨迹只组合该记录，2→1。历史轨迹的整数 index 不能替代新 Atom UID；两种坐标系的
映射放在完整 snapshot/envelope 中，不能在所有底层记录里同时重复塞入 index/id/uid。

原子侧 `AtomTopologySignature` 与轨迹 `AtomIdentity` 也高度相似，后者多存 `symbol`。
可复用一个原子快照行并按 `atomic_number` 翻译 symbol，但应先明确轨迹导出与快照
版本约束。此项暂不计入确定的数量减少。

### 3.5 CLI 的科学选项重复，应复用已有领域 options

`forcefields/cli.py:87` 的 `_RunOptions` 复制
`forcefields/stage.py:78` 的 `ForceFieldRouteOptions` 所有科学字段，另加 `output_format`。
后者的文档本来就声明由 CLI 与 pipeline 共用。

提案：CLI work item 组合已有 `ForceFieldRouteOptions`，自身只携带输出格式/路径；
删除 `_RunOptions`，不新建另一个 options wrapper，净减少 1。
xTB 的 `_RunOptions` 与 `_XTBStageOptions` 也有类似交集，但包含执行环境、输出和
后检查语义差异，先组合现有请求/运行设置，证明消息及 CLI 协议保持后再决定类数；
不把它计入本轮已明确的减少额。

## 4. 无用、过期与“看似冗余但必须保留”的区分

| 对象/家族 | 审计证据 | 处理结论 |
|---|---|---|
| `AtomPairKey` | `core.py:4499-4517` 仅定义与 repr 字符串，未找到生产/测试构造；旧 wildcard 可能隐式暴露 | 非 dataclass 的删除候选，先确认公开删除决策，不计入 347 |
| `CoordinationGeometryCandidate` / `CoordinationGeometryResult` | `forcefields/contracts.py:427,434` 无实例化生产者；`coordination.py:93-104` 的唯一预留入口始终抛 NotImplementedError | dormant；历史 Plan03 明确曾按用户要求保留。应由最终 placeholder 决策处理，不因零调用自动删除 |
| `RingEdgeDistance` | `geometry/convert.py:199-201` 仅定义/导出；文档与 `test_native_migration_characterization.py:65` 冻结了其存在 | 无活跃生产者的已文档接口；待审删除或绑定实际 producer，不能宣称私有死代码 |
| `Triangle` | 内部调用少，但公开文档和 `geometry/test_object.py:83,98` 真正使用 | 保留；静态名称频率不能取代消费审计 |
| `BuildAndOptimizeReport` / `ComplexBuildReport` | `forcefields/contracts.py:386-403` 已继承 `ForceFieldWorkflowReport` 并收紧 build/optimization 类型 | 这是已有合理继承；保留有意义的窄类型，不为压数量改成任意可选字段 |
| 中立与 rdWrapper 的两个 `ConformerBatch` | 中立为 coordinates/ordinals；wrapper 带 status/engine/version/records/diagnostic，adapter 会选取/投影 | 生命周期和保证不同；保留两层，明确命名/导入限定，不复制 wrapper schema |
| `HybridizationChange` / `CoordinateChange` | `obWrappers/contracts.py:84,93` 字段名同为 atom_index/before/after，但值分别为离散杂化态与三维坐标 | standalone 已公开、量纲不同；不压成 before:object/after:object 的万能变更 |
| `MolecularFactsIdentity` / `ScientificStateIdentity` / `BackendIdentity` | 均可表现为 digest；分别指化学输入、选定科学状态和后端上下文 | 保留域分离，防止错误比较；共用私有 hash helper 不要求共用运行时身份类型 |
| `ProposedCandidateFactor` / `CandidateFactor` | `state_composition/contracts.py:239,217` 分别表示单来源提案与保留多来源/完整性的归并候选 | 保留不同阶段语义；可组合共享 payload，不消除溯源与完整性差别 |
| `StateRequest` / `BackendStateRequest` | 前者科学输入，后者后端电子/UHF投影 | 保留科学态与传输态边界 |
| `MolecularFacts` / `NativeMoleculeBuffers` / `_AtomFactTable` | 分别是科学快照、task-local ABI packing、Plan20 拟议事实权威 | 同一来源的不同用途投影，不是三个持久事实源；共享字段 schema，不强制合成一个类 |
| ds `_CertificationSnapshot` / `_StateToken` / `_FileStamp` | `ds/public/certificate.py:93-155` 分别固定文件状态、可信输入与已绑定证明 | 证书一致性边界；保留 bytes/parsed view 的绑定语义，不当成普通化学缓存删除 |

**本轮没有找到可在不改变公开/协议/预留合约的前提下无条件直接删除的私有 dataclass。**
这不表示全部定义都最优；已证明的同构家族值得统一，但公开变更和科学语义必须先闭合。

## 5. MCA/CBond、进程和报告的具体整理

### 5.1 MCA 的同义字段和不同表示

`AImodels/mca/serialization.py:24-59` 当前提供 `AtomDTO`、`BondDTO`、`MoleculeDTO`。
`MoleculeDTO` 同时保存 Hotpot 图和 `rdkit_binary` 是明确设计：位点识别使用 Hotpot 的
配位语义，特征构建使用 RDKit 立体化学，见该文件开头说明及 `pipeline.py:40-60`。
不能为了去重丢掉其中一路的科学语义。

可以进一步裁剪的字段包括 `atom_count`、`atom_symbols` 与原子序数的同义信息，以及
site detector 未实际消费的字段。应先形成“字段→具体消费者”表，按通用字段选择式
snapshot 构造所需投影；不能默认重新导入就执行 valence/partial-charge 计算。
`atom_count` 可从 atoms 得到，symbol 由原子序数翻译；是否保留为传输加速元数据应有
测量依据，且永不成为核心事实的第二权威。

MCA 当前 `AtomPrediction`、`SitePrediction` 共享 atom_index/element/value，site 多一个
site_type。两者分别表达每原子预测和规则选定位点，现有 public DTO、`to_dict()` 与
`is_Nuc_site` 消费稳定，优先保留并统一字段序/命名规则。可复用私有批量数值数组，
公开投影按需构造，避免逐 Atom 保存第二份预测值。
现 `calculator/mca_inference.py:20-38` 已返回 typed `MoleculePrediction`，同时又向
`Atom._mca`、`Molecule._mca_sites` 和 properties 写入三处状态；迁移目标是一个附着结果
加投影，不能把现状误记为“仅保存无类型 ndarray”。

### 5.2 MCA worker / GPU 消息：按阶段组织，避免巨型可选载荷

`workers.py:12-51` 的 SiteDetection/Feature/CPUPreprocess Task/Result 与
`gpu_consumer.py:17-59` 的 Config/Request/Ready/Result/Failure 都有真实生产者和消费者。
例如 `build_features_worker()` 真正生产 `FeatureResult`，GPU backend 与 Plan17 benchmark
均消费；低引用频次不是无用证据。

建议同域命名统一为 `MCA<Stage>Task/Result`、`MCAGPU...`（公开路径/类改名待审），
共有 molecule_index/request_id 字段放在相同位置；若组合关联信息能去掉重复消息 envelope
才引入基类或 context。不要为每条消息新增一个通用 TaskEnvelope，从 11 个具体消息变为
11 个子类再加 1 个基类；也不要合成含十几个 Optional 字段的万能消息。
`_MoleculeState`、`_PendingChunk`、`_PendingRequest` 是 task-local 可变调度状态，
应在 workspace 区域，不能归为不可变科学结果或载体事实。

CBond `CBondStep` / `CBondPathResult` / `CBondStructureResult` 的 score、路径权重和
聚合结构权重不是同一个概率量；`apply.py:374` 与 `:546` 有真实结果构造。
可以共享步骤与 donor identity，不能把 path_probability 和归并后的 structure probability
当成同义字段合并。`_SearchContext` / `_SearchState` 是内部算法上下文与累积状态。

### 5.3 外部进程与结果报告优先组合

`plugins/_harness/contracts.py:29,49` 的 `NativeProcessResult` 与 `ProcessProvenance`
重复 argv/cwd/return_code/stdout/stderr/elapsed_seconds；
`plugins/xtb/contracts.py:116-143` 的 `XTBRunReport` 又扁平保存过程字段及 provenance。
这里应以一个执行事实值被 provenance/领域报告组合；保持旧 JSON 的需要交给序列化投影，
避免三处可独立赋值的内部状态。这个改变主要减少冗余字段与搬运函数，不一定减少类数。

`pipeline.Artifact` 使用 stage-relative path，`XTBArtifact` 和 harness artifact 使用
workspace/absolute path，ds `_BoundFile` 使用证书绑定路径；不能在丢失路径基准后统一。
若共用 digest/size schema，必须同时保留 path scope/base，与模型 artifact manifest 的
model identity 区分。当前 artifacts/manifest.py 是加载 dict 的函数，**没有 `ArtifactManifest`
类**；该名字若出现在目标树，应标为拟新增概念，不能计入已存在的复用成果。

ComplexFormer 的 `RunArgs`、`DataModuleArgs`、`ConfigArgs`、`OptimizerConfigure` 等
重复了大量可选设置，且 `run/datacls.py:44-67` 按同名字段隐式复制，
`run/run_tools.py:150-162` 真正使用。建议在其独立任务内改为 data/model/training/runtime
配置组合与显式投影；Plan20 本轮只更新受影响 chemistry imports，不把训练配置重写
扩成化学内核的前置依赖。

## 6. 统一组织、命名与字段结构提案

### 6.1 所有权与复用粒度

优先顺序是：复用现有同域定义 → 泛型同域记录 → 组合共享值 → 有真实替代关系的继承。
继承仅在子类型改变约束/保证时使用；新增“CommonBase”本身不是规模缩减。
通用 hash、字段选择、单位转换和序列化 helper 可以复用，不必各自成为数据对象。

| 功能 | 统一 suffix / 命名规则 | 规定内容 |
|---|---|---|
| 权威数据存储 | 私有 `...FactTable` / `...Store` | 每项核心事实有唯一 owner；经审阅接口投影同一事实，不另存同义表达或派生副本 |
| 输入截面 | `...Snapshot` | owner/内容身份、行身份、明确字段集合、不可变视图或独立副本 |
| 用户意图/计算输入 | `...Request` | 输入与选择参数；不能夹带执行后的测量值 |
| 数值/算法参数 | `...Options` | 同域公共选项只有一个 schema；保留已批准的旧公开名称直到审阅 |
| 任务消息 | `...Task` / `...Request` | 可序列化的工作说明与关联键 |
| 科学输出 | `...Result` | 明确 quantity、输入来源、值与单位、证据，不等同 accepted/rejected 判定 |
| 观察和诊断 | `...Finding` / `...Observation` | code、测量、阈值/不确定性、消息与关联身份 |
| 执行汇总 | `...Report` | 执行事实、来源、结果/观察组合、错误及资源；科学监控数值动态变化应报告 |
| 变更建议 | `...Assignment` / `...Update` | 来源身份、被影响行/字段和值；由 operation 验证并提交 |
| 来源和身份 | `...Provenance` / `...Identity` | 身份与来源分开；owner/内容身份按需组合 |
| 算法暂态 | 私有 `_...Workspace` / `_...State` | task-local 可变状态；不可附着于载体成为第二权威 |

具体域优先使用 State、ForceField、MCA、CBond、XTB 等既有领域词；
相同域同一角色用相同前缀/后缀。模块内部私有类型不强加冗长全局前缀。
wrapper 保持现有公开类名与 namespace，adapter 用限定导入化解同名 `ConformerBatch`。

### 6.2 同系记录的字段定义与顺序

新建或经批准统一后的记录遵循以下组序，**只包含其真实需要的组**：

1. schema/version 与 quantity/variant（只有序列化或判别确有需要时）；
2. subject/source owner、必要的依赖身份和行身份；
3. request 的显式输入/选项，或 result 的数值/坐标/候选；
4. 单位及 presence/unknown 信息；
5. findings/observations 与不确定性；
6. method/backend/model version、provenance；
7. 执行状态、运行耗时和错误信息（仅执行记录）。

共有字段采用相同类型、单位和排序；单位固定在域合约或列 schema 中，不为每行重复字符串。
对同一标识统一 `atom_uid` / `atom_index` / `source_id` 的含义，不能为统一拼写而混用。
只依赖 topology 的快照不强塞 geometry；只做核电荷计算的规则不携带 MCA runtime。
已有公开 stereo/native ABI 的字段序必须保留，不能按此通则擅自重排。

### 6.3 化学类型公开接口的一致协议

[共同协议](chemical_type_interface_contract.md)覆盖所有化学类型的构造器、属性、
方法、集合访问、返回对象和转换边界。Molecule、Atom、Bond、Crystal、构象与分子
集合、派生视图、科学结果及运输值按同一结构声明角色、owner、事实/依赖、单位与
编码、presence/source、读写权限、变更操作和生命周期；不为每个入口另造记录类。
MetalCluster 以未来注册聚合类型纳入该关系图；Plan20 只登记类型与 Molecule 属性，
注册定义、分子实例与超原子映射的具体组织见
[Plan21 草稿](../21_metal_cluster/metal_cluster_implementation_plan.md)。

`structure/_fields.py` 维护各类型的字段与接口映射，`_dependencies.py` 维护跨类型
依赖、读取策略和失效范围，`_commit.py` 负责完整写集的原子安装。私有声明按领域
分组复用，不要求所有类型继承同一基类、采用同一数值布局或拥有相同可变性。
只有角色允许变更、输入能完整确定事实且接口已批准时才提供 setter；耦合事实经
完整参数方法调用 chemistry/physics/mathematics 规则，不能靠赋值顺序拼出表示。
视图和不可变结果保持只读；独立 Crystal 等值不强加 Molecule UID 或 dirty 状态。

Atom 的具体实现是 Molecule-owned typed columnar `_AtomFactTable`；Atom 通过已审
映射读取或修改唯一数值事实。已批准 schema 字段可联动存储、属性、快照和依赖，
内部增列不会自动发布同名公开属性。科学快照按字段取值，ABI 按请求打包，神经
网络矩阵按 dtype/字段顺序在内部显式导出；公开 `atom_matrix` 是完整19列
float64 property，与 Atom.attrs 共用唯一投影。三者共享声明与来源，
各自保留用途和寿命，不把公开完整矩阵的工作量替代内部字段选择基准。

依赖状态归权威事实或选定结果 owner。变更仅标 `NEEDS_RECALC`；需求读取先展开
相关上游的未处理 dirty 种子，再检查目标当前性。昂贵推断不在赋值时运行，读取也
不隐式启动模型、GPU 或外部优化。已返回的历史结果保持其内容，当前结果投影才
检查与活动事实的关系；视图/快照不复制一套可变失效状态。批量结果读取一次校验，
不因共用协议新增同义 snapshot/result/update 家族或逐行全分子摘要扫描。

## 7. 数量账本：精简潜力与净变化不能混淆

### 7.1 已有定义的可审议收敛

| 归并项 | 当前→目标定义数 | 理论净减少 | 前提 |
|---|---:|---:|---|
| 五种科学推断结果外壳 | 5→1 | 4 | quantity tag/泛型和公开返回类型审阅 |
| 四种科学状态 finding/reason | 4→1 | 3 | 保留 namespace、角色和字段 presence |
| provider/rule toggle | 2→1 | 1 | ID 域和启用权限仍分开 |
| 两种 Python 停止参数 | 2→1 | 1 | 不改独立 C++ 绑定结构 |
| 两种力场 bond snapshot | 2→1 | 1 | 统一端点规范化与历史导出 |
| FF CLI 重复 options | 1→0（复用已有 options） | 1 | 输出设置移到已有 work item，不新增 wrapper |
| **上述合计** | **16→5** | **11** | **均未实施，涉及公开项尚待审** |

故在“不考虑 Plan20 新机制新增对象”的局部条件下，347 个 dataclass 的理论数为 336；
3 TypedDict、13 Protocol 未因此减少。这不是实施后总量承诺。
另有 `CoordinationGeometryCandidate/Result` 与 `RingEdgeDistance` 共 3 个 dataclass 的
条件删除候选；若批准，可另减 3。`AtomPairKey` 不在 dataclass 口径中，单独记录。

### 7.2 新机制与后续类型的新增成本

| 新概念/机制 | 当前是否已有可复用表达 | 计数处理 |
|---|---|---|
| `_AtomFactTable`、schema/字段访问器 | Atom attrs 与通用 native 字段 schema 部分可用，目标事实表未实现 | 私有实现类单独计数；字段不默认各造一个 record/descriptor 类 |
| `MoleculeRecord` / builder 输入 | 有 MCA DTO、native buffers、scientific facts，但保证/用途不同 | 优先组合已有字段 schema；如新增 dataclass，必须 +1 |
| Topology/Representation/Geometry/Cell/Constraint/SourceFrames identities | 已有 facts/state/backend identity 可借鉴，目标六种表达尚未落地 | 可用同域值/别名/组合；每个新运行时类计入，不因“Identity”后缀免记 |
| `TopologyChange`、`AtomFactUpdate`、`CoordinateUpdate`、representation/stereo/constraint update | 现有 `_WorkingCopyCommit`、immutable assignments 可复用/替换 | 替代旧记录计净值；仅增加语义名不意味着必须加 dataclass |
| `TopologySelection` | 当前仅规划，无生产类 | 如新建运行时类明确 +1；不要把它作为现有冗余对象删一次又加一次 |
| typed P3 results / provenance / source stereo records / source frames | 有 MCA public DTO、scientific result、StereoPayload、ConformerBatch | 优先复用/组合；每个新 record 列旧类型去向及 +1/-1 |
| 依赖组/dirty 状态 | 新私有依赖机制 | Enum/bitset/schema table 可表达；不为每个属性建立状态对象 |
| MetalCluster（Plan21） | 当前只有局部金属分组列表，无目标领域类型；Plan20 仅设计登记 | 实施阶段按实际类种类记新增，不改当前库存、不抵扣 Plan20 精简收益；注册定义、实例、运输不预设各造一类 |

**实施后净减少尚未被证明。** Stage1 必须冻结同口径的
`当前定义 - 删除/归并 + 新增 = 最终定义` 清单，分别统计 dataclass、TypedDict、
Protocol、普通数据/实现类、type alias 与 C++ ABI structs。新增基类也计数；
把字段搬到另一个类、改名、重导出或把 dataclass 改成普通 class 均不算减少。

最终规模目标是关键域的记录数量与必要字段/转换层同时下降；不能用11的局部潜力
抵消未记账的新类，或把训练/插件的独立对象当作必须消除的化学内核负担。

## 8. 公开接口待审表

本表列明本审计提出的公开变化；没有列出的新公开类/属性/方法不由本报告设立。
现有名称保留到逐项决策冻结；是否硬迁移由主规划的公开路径决策控制。

| 编号 | 现有公开面 | 提案变化 | 审阅内容 |
|---|---|---|---|
| D01 | 五种 `...Result` | 统一为拟议 `StateInferenceResult[T]` | 新类名/路径、quantity tag、候选泛型、构造参数/字段序、返回类型、序列化 |
| D02 | 三种 Finding/Reason 与 StaticEligibilityFinding | 统一为拟议 `StateFinding` | code namespace、related_identities 缺省、角色投影、旧名称去向 |
| D03 | `ProviderToggle` / `RuleToggle` | 统一为拟议 `StateActivation[TId]` | 新类及 ID 类型、enabled 语义、provider/rule 权限边界 |
| D04 | `OptimizationStoppingCriteria` / `OptimizationStoppingOptions` | 保留一个 Python 参数类型 | 最终名称/路径、旧导入去向；5字段/单位/默认值保持 |
| D05 | `BondTopologySignature` / `BondTopology` | 保留一个不可变 bond record | 最终名称/路径、端点排序、构造/工厂、轨迹历史 schema |
| D06 | `AtomTopologySignature` / `AtomIdentity` | 可能统一原子快照并把 symbol 改为计算投影 | 暂不计减少；uid/index/source id、symbol 读权限与历史格式 |
| D07 | dormant geometry candidate/result、`RingEdgeDistance`、隐式导出的 `AtomPairKey` | 删除或补生产者后保留 | 历史预留承诺、文档/测试/用户数据影响；不自动删除 |
| D08 | MCA Task/Result/DTO 类名 | 候选统一 MCA prefix/suffix | 是否属于支持的高级导入、消息pickle兼容和路径；现有预测public DTO先保持 |
| D09 | harness/`XTBRunReport` 的重复过程字段 | 内部组合、公开读取/JSON保持或另行调整 | 若改公开属性/构造器，列出逐字段diff；不能以内部重组静默破坏 |
| D10 | 全部化学类型的事实映射、读写入口与生命周期 | 按[共同协议](chemical_type_interface_contract.md)列跨类型待审项；Atom attrs/批量/矩阵候选为具体实施项 | 精确签名、权限、owner、依赖、缓冲与旧入口去向；本审计不新增公开接口 |

`_RunOptions` 等私有实现可按批准的内部迁移改组，但其跨进程 payload 及 CLI 行为仍需
验证；private 不等于可以丢失过程消息语义。

## 9. Stage1 / 迁移阶段的交付和验收

1. 以附录扫描脚本冻结精确库存，为每个候选记录 role/owner/public/lifetime/producer/
   consumer/serialization/target/action；域负责人审阅。所有公开化学类型另按共同
   协议盘点构造器、属性、方法、集合及返回值的事实映射、权限、依赖与审阅状态。
2. 对同构家族冻结类型、单位、默认值、presence、数量量纲、字段序与构造/序列化
   消费者；批准后在同一阶段移除旧类型及搬运函数。每个新概念记入净变化账本，
   记录继承/组合是否实际减少类、重复事实字段和函数间转换。
3. 按角色检查语义：载体入口同源及完整写入，独立值的构造/单位，集合成员及构象
   选择，视图只读与来源有效性，结果历史内容及当前投影，运输值的 schema/缓冲
   寿命。共同协议不要求新增公共基类、setter 或统一布局。
4. 检查 ndarray 与 Mapping 的实际可变性和别名，不以 `frozen=True` 代替内容保证。
   核心事实表支持高效受控批量写；快照隔离。依赖验证覆盖正反例、远端先读、一次
   重算、源变化拒绝、回滚及 copy/remap，不让视图和历史结果成为第二状态 owner。
5. 记录各类型标量/批量路径、矩阵导出、集合操作、几何视图、晶胞转换、结果投影、
   snapshot packing、worker pickle、report encoding 和启动/测试耗时的重构前基准。
   精简类型不以增加复制/摘要扫描次数为代价；遵守全局至少持平、总体期望更快标准。
6. 科学运行输出逐例观测、能量/几何/恢复/失败与历史差异报告；187/182是来源与
   当前适用队列的记录，不是永久成功数目标。监控器触发不直接宣判科学真伪；
   类型/索引/单位/所有权/序列化等工程契约仍由确定性测试核验。
7. standalone wrapper schema、StereoPayload 与 ABI 字段序保持各自已批准合约；
   合并 DTO 不改科学语义。所有类型新增、删除、改名、权限及返回变化逐项审阅，
   私有 schema 增加字段不自动授权公开接口。
8. 每阶段报告新增/保留/合并/删除数量、剩余争议和实际生产者/消费者迁移；最后的
   零引用检查同时覆盖生产、测试、docs、stubs、CMake、消息/JSON版本和公开导出。

## 10. 静态库存的复现方法

在仓库根执行下列只读脚本。该脚本展示候选发现口径，不负责自动判断纯数据性、
公开承诺、动态消费者或合并正确性。附录定义行号对应上述基线。

```python
import ast
from collections import Counter
from pathlib import Path

def leaf(node):
    if isinstance(node, ast.Call):
        return leaf(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""

counts = Counter()
for path in sorted(Path("hotpot").rglob("*.py")):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        markers = {leaf(x) for x in node.decorator_list + node.bases}
        kinds = markers & {"dataclass", "NamedTuple", "TypedDict", "Protocol"}
        if kinds:
            counts.update(kinds)
            print(f"{path}:{node.lineno}", node.name, ",".join(sorted(kinds)))
print(dict(counts))
```

当前输出计数为 `dataclass=347, TypedDict=3, Protocol=13`。
以下附录列出全部363候选；同一行多个定义各计一个，不把 package re-export 重复计数。

## 附录：按源文件的完整定义索引

源文件链接固定到本审计的 `d9eb80e` 基线，保留当时的定义位置；当前实施位置由迁移记录索引。

| 源文件 | 类型和定义行号 | 数量 |
|---|---|---:|
| [__main__.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/__main__.py#L23) | `_CommandSpec`:23 | 1 |
| [cheminfo/AImodels/artifacts/registry.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/artifacts/registry.py#L10) | `ArtifactSpec`:10 | 1 |
| [cheminfo/AImodels/cbond/apply.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/cbond/apply.py#L24) | `CBondStep`:24；`CBondPathResult`:34；`CBondStructureResult`:44；`_SearchState`:57；`_SearchContext`:65 | 5 |
| [cheminfo/AImodels/cbond/stage.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/cbond/stage.py#L39) | `_PreparedCBondStage`:39；`_CBondStage`:114 | 2 |
| [cheminfo/AImodels/mca/batching.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/batching.py#L15) | `AtomRow`:15；`LogicalBatch`:21 | 2 |
| [cheminfo/AImodels/mca/gpu_backend.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/gpu_backend.py#L78) | `_MoleculeState`:78；`_PendingChunk`:88；`_PendingRequest`:94 | 3 |
| [cheminfo/AImodels/mca/gpu_consumer.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/gpu_consumer.py#L17) | `GpuConsumerConfig`:17；`GpuBatchRequest`:25；`GpuConsumerReady`:34；`GpuBatchResult`:41；`GpuConsumerFailure`:51 | 5 |
| [cheminfo/AImodels/mca/gpu_memory.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/gpu_memory.py#L17) | `GPUDeviceMemory`:17 | 1 |
| [cheminfo/AImodels/mca/pipeline.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/pipeline.py#L31) | `PreparedMolecule`:31 | 1 |
| [cheminfo/AImodels/mca/resources.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/resources.py#L55) | `CPUWorkerAssignment`:55；`GPUCPUAssignment`:66；`GPUResourcePlan`:76 | 3 |
| [cheminfo/AImodels/mca/result_types.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/result_types.py#L9) | `AtomPrediction`:9；`SitePrediction`:19；`MoleculePrediction`:30 | 3 |
| [cheminfo/AImodels/mca/runtime.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/runtime.py#L120) | `ResolvedModelArtifact`:120 | 1 |
| [cheminfo/AImodels/mca/serialization.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/serialization.py#L24) | `AtomDTO`:24；`BondDTO`:38；`MoleculeDTO`:49 | 3 |
| [cheminfo/AImodels/mca/site_detection.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/site_detection.py#L16) | `DetectedSite`:16 | 1 |
| [cheminfo/AImodels/mca/workers.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/AImodels/mca/workers.py#L12) | `SiteDetectionTask`:12；`SiteDetectionResult`:18；`FeatureTask`:24；`FeatureResult`:32；`CPUPreprocessTask`:38；`CPUPreprocessResult`:47 | 6 |
| [cheminfo/_native/buffers.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/_native/buffers.py#L88) | `NativeUtf8Table`:88；`NativeStereoRecordBuffers`:98；`NativeAtomTransportBuffers`:121；`NativeBondTransportBuffers`:143；`NativeMoleculeBuffers`:152 | 5 |
| [cheminfo/calculator/charge/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/charge/contracts.py#L29) | `NetChargeCandidateEvidence`:29；`AtomFormalChargeEvidence`:37；`ComponentChargeEvidence`:46；`NetChargeResult`:55；`LewisFormalChargeCandidate`:68；`LewisFormalChargeResult`:81 | 6 |
| [cheminfo/calculator/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/contracts.py#L106) | `MolecularFactsIdentity`:106；`BackendIdentity`:113；`ComponentIdentity`:120；`HydrogenInventory`:128；`SemanticBond`:139；`ScientificFinding`:149；`CandidateProvenance`:158；`MolecularFacts`:169；`ProviderToggle`:199；`ProviderSet`:208；`CandidateBudget`:215；`ScientificOperationRequirements`:240；`BackendContext`:248 [P] | 13 |
| [cheminfo/calculator/electronic_state/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/electronic_state/contracts.py#L40) | `ElectronicStateCandidateEvidence`:40；`SpinContributionCandidate`:50；`ElectronicStateCandidate`:62；`ElectronicStateInferenceResult`:75；`NeutralAtomReference`:88 | 5 |
| [cheminfo/calculator/execution_selection.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/execution_selection.py#L60) | `BackendStateValues`:60；`BackendStateRequest`:69；`BackendProjectionFinding`:95；`BackendExecutionDecision`:104；`BackendExecutionEvidence`:115；`ExecutionPolicySet`:126；`BackendStateProjection`:133；`BackendProjectionResult`:158；`ScientificStateWorkflowResult`:181 | 9 |
| [cheminfo/calculator/legality.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/legality.py#L42) | `_HardRuleAssessment`:42 | 1 |
| [cheminfo/calculator/oxidation_state/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/oxidation_state/contracts.py#L40) | `OxidationStateOverride`:40；`CoordinationFragmentAssignment`:48；`CoordinationCountingCandidate`:58；`CoordinationElectronCountingResult`:67；`OxidationStateCandidateEvidence`:80；`OxidationStateInferenceResult`:91；`ElementOxidationStateDomain`:104 | 7 |
| [cheminfo/calculator/state_composition/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_composition/contracts.py#L94) | `ScientificStateIdentity`:94；`FactorKey`:101；`CandidateConstraint`:119；`FrontierBound`:130；`CandidateDomainSummary`:139；`RejectedPathSummary`:148；`StateValueRef`:157；`DerivedStateValue`:167；`BranchAssignment`:190；`FactorCandidateValue`:198；`FactorDomainContribution`:207；`CandidateFactor`:217；`ProposedCandidateValue`:231；`ProposedCandidateFactor`:239；`ConstraintGuard`:251；`GuardedCandidateConstraint`:259；`CandidateProposal`:267；`CoupledScientificStateCandidate`:276；`ScientificStateCandidateSpace`:292；`ScientificResolutionResult`:307 | 20 |
| [cheminfo/calculator/state_composition/decision_search.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_composition/decision_search.py#L30) | `_AuthorityVector`:30；`_PreferenceKey`:37；`_DecisionEquivalenceKey`:45；`_DecisionRecord`:54；`_DecisionBound`:65；`_DecisionSearchCompiler`:75 [P] | 6 |
| [cheminfo/calculator/state_composition/expand.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_composition/expand.py#L87) | `_SelectedValues`:87；`_SearchNode`:100；`_FrontierEntry`:108 | 3 |
| [cheminfo/calculator/state_resolver.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_resolver.py#L65) | `StateRequest`:65；`ProvidedScientificResults`:80；`ScientificStateBundle`:93 | 3 |
| [cheminfo/calculator/state_rules/anionic_x_ligands.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/anionic_x_ligands.py#L68) | `_XLigandRegistry`:68 | 1 |
| [cheminfo/calculator/state_rules/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/contracts.py#L78) | `RuleDependency`:78；`RuleSpecification`:86；`RuleToggle`:100；`EvidenceItem`:109；`RankAdjustment`:120；`RuleApplicability`:131；`RuleContext`:139；`RuleOutcome`:151；`RuleTrace`:165；`ActivatedRule`:173；`RuleSet`:181 | 11 |
| [cheminfo/calculator/state_rules/controller.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/controller.py#L39) | `_RuleExecution`:39 | 1 |
| [cheminfo/calculator/state_rules/f_block_hund.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/f_block_hund.py#L42) | `_HundPriorRegistry`:42 | 1 |
| [cheminfo/calculator/state_rules/ionic_configuration_spin.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/ionic_configuration_spin.py#L65) | `_IonicConfiguration`:65；`_IonicConfigurationRegistry`:79 | 2 |
| [cheminfo/calculator/state_rules/main_group_closed_shell.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/main_group_closed_shell.py#L76) | `_ShellDomain`:76；`_ClosedShellRegistry`:82 | 2 |
| [cheminfo/calculator/state_rules/manifest.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/manifest.py#L35) | `_ScientificStateRuntimeManifest`:35 | 1 |
| [cheminfo/calculator/state_rules/monoatomic_ions.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/monoatomic_ions.py#L61) | `_IonEntry`:61；`_IonRegistry`:68 | 2 |
| [cheminfo/calculator/state_rules/neutral_l_donors.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/neutral_l_donors.py#L63) | `_AtomSignature`:63；`_ComponentSignature`:70；`_DonorRegistry`:78 | 3 |
| [cheminfo/calculator/state_rules/oxidation_priors.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/oxidation_priors.py#L49) | `_ElementOxidationPrior`:49；`_OxidationPriorArtifact`:56 | 2 |
| [cheminfo/calculator/state_rules/profile_decision.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/profile_decision.py#L65) | `_AssumptionSeverityPolicy`:65 | 1 |
| [cheminfo/calculator/state_rules/profiles.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/profiles.py#L24) | `_ScientificStateProfileArtifact`:24 | 1 |
| [cheminfo/calculator/state_rules/trusted_history.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_rules/trusted_history.py#L75) | `TrustedScientificStateRecord`:75 | 1 |
| [cheminfo/calculator/state_selection.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/calculator/state_selection.py#L55) | `StaticEligibilityFinding`:55；`DecisionReason`:63；`StaticProfileEligibility`:72；`DecisionCertifiability`:82；`ExecutionStateDecision`:90 | 5 |
| [cheminfo/conformers/adapters.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/conformers/adapters.py#L113) | `OpenBabelConformerAdapter`:113；`RDKitConformerAdapter`:139 | 2 |
| [cheminfo/conformers/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/conformers/contracts.py#L23) | `ConformerSamplingRequest`:23；`ConformerBatch`:31 | 2 |
| [cheminfo/conformers/sampler.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/conformers/sampler.py#L17) | `ConformerSampler`:17 [P] | 1 |
| [cheminfo/forcefields/acceptance.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/acceptance.py#L56) | `_CoordinationMetrics`:56 [T]；`_BondRingAcceptanceFinding`:65；`_BondRingAcceptanceFacts`:75；`_AtomPairAcceptanceIssue`:144 | 4 |
| [cheminfo/forcefields/auto.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/auto.py#L59) | `_NativeFastBuildAttempt`:59 | 1 |
| [cheminfo/forcefields/backend.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/backend.py#L33) | `_CandidateOptimizationResult`:33 | 1 |
| [cheminfo/forcefields/cli.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/cli.py#L79) | `_InputRecord`:79；`_RunOptions`:87；`_WorkItem`:105；`_Outcome`:112 | 4 |
| [cheminfo/forcefields/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/contracts.py#L122) | `AcceptanceCheck`:122；`StructureAcceptanceThresholds`:136；`ForceFieldAcceptanceEvidence`:153 [T]；`ForceFieldValidationReport`:170；`OptimizationAttemptReport`:200；`OptimizationRoutingReport`:214；`OptimizationStoppingCriteria`:222；`ForceFieldRunReport`:254；`Build3DReport`:305；`CandidateRejection`:312；`RingUntanglingReport`:320；`CoordinationBondRestorationReport`:333；`ComplexBuildDiagnostics`:356；`BuildWorkerResult`:375；`ForceFieldWorkflowReport`:386；`BuildAndOptimizeReport`:396；`ComplexBuildReport`:402；`ForceFieldSetupReport`:407；`CoordinationEnvironment`:416；`CoordinationGeometryCandidate`:427；`CoordinationGeometryResult`:434 | 21 |
| [cheminfo/forcefields/ligand.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/ligand.py#L59) | `_LigandCandidate`:59 | 1 |
| [cheminfo/forcefields/native.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/native.py#L82) | `OptimizationStoppingOptions`:82；`MetalPlacementOptions`:91；`RingScreeningOptions`:116；`CoordinationStageOptions`:125；`ComplexOptimizationOptions`:140；`StructureSnapshot`:165 | 6 |
| [cheminfo/forcefields/native_adapters.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/native_adapters.py#L109) | `_SelectedStructureResult`:109 [P]；`NativeOptimizationPerturbationStreams`:115；`NativePerturbationStreams`:123 | 3 |
| [cheminfo/forcefields/native_packing.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/native_packing.py#L39) | `ComplexSessionInput`:39 | 1 |
| [cheminfo/forcefields/native_reports.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/native_reports.py#L52) | `NativeBondRingFinding`:52；`NativeRingCheckpointReport`:63；`NativeTopologyRevision`:83；`NativeTrajectoryBatch`:89；`CoordinationStageResult`:110；`ComplexOptimizationResult`:132；`ComplexWorkflowResult`:168 | 7 |
| [cheminfo/forcefields/repair.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/repair.py#L31) | `_RingUntanglingResult`:31；`_BondRingPairKey`:38；`_WatchedRingPiercing`:44；`_RingPiercingWatchResult`:50；`_WatchedRingRepairObservation`:56；`_RingTrajectoryRecorder`:61 | 6 |
| [cheminfo/forcefields/stage.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/stage.py#L78) | `ForceFieldRouteOptions`:78；`ForceFieldRouteOperations`:97；`_PreparedForceFieldStage`:107；`_ForceFieldStage`:165 | 4 |
| [cheminfo/forcefields/topology.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/topology.py#L22) | `AtomTopologySignature`:22；`BondTopologySignature`:32；`TopologyReference`:41 | 3 |
| [cheminfo/forcefields/trajectory.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/trajectory.py#L110) | `AtomIdentity`:110；`BondTopology`:137；`BondTopologyRevision`:159；`RingFrameEvidence`:167；`CoordinationFrameEvidence`:184；`OptimizationFrameEvidence`:203；`ForceFieldFrame`:225；`ForceFieldTrajectoryArchive`:555 | 8 |
| [cheminfo/forcefields/workflows.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/workflows.py#L111) | `_PreparedComplex`:111 | 1 |
| [cheminfo/forcefields/working_copy.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/forcefields/working_copy.py#L32) | `_BondAttributePayload`:32 [T]；`_WorkingCopyCommit`:43；`_MoleculeCommitSnapshot`:53 | 3 |
| [cheminfo/geometry/convert.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/geometry/convert.py#L87) | `_AtomLike`:87 [P]；`AtomGeometry`:153；`AtomPairTarget`:160；`BondGeometry`:167；`RingGeometry`:174；`BondRingTarget`:181；`AtomPairDistance`:187；`BondRingFinding`:193；`RingEdgeDistance`:199；`BondRingScanReport`:205；`BondRingScreeningReport`:252；`BondRingScreeningPlan`:301；`_PreparedRingFrame`:322；`BondRingFrameWorkspace`:329 | 14 |
| [cheminfo/geometry/object.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/geometry/object.py#L38) | `Point`:38；`Line`:69；`Segment`:94；`Plane`:118；`Triangle`:141；`Cycle`:172 | 6 |
| [cheminfo/geometry/relation.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/geometry/relation.py#L123) | `PlanarityMeasurement`:123；`LineRelation`:135；`PointPairDistance`:142；`ClosestCycleEdge`:149；`SurfaceFamilyEvidence`:156；`SegmentCycleRelation`:170；`SegmentCycleScreening`:182 | 7 |
| [cheminfo/geometry/settings.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/geometry/settings.py#L23) | `NumericToleranceSettings`:23；`SurfaceEnumerationSettings`:76；`GeometrySettings`:98 | 3 |
| [cheminfo/kekulize/_protocols.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/kekulize/_protocols.py#L10) | `AtomLike`:10 [P]；`BondLike`:28 [P]；`RingLike`:35 [P]；`JointRingLike`:55 [P]；`MoleculeLike`:60 [P] | 5 |
| [cheminfo/obWrappers/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/obWrappers/contracts.py#L74) | `RuleDescriptor`:74；`HybridizationChange`:84；`CoordinateChange`:93；`RuleApplication`:102；`RuleExecutionReport`:113；`BuildReport`:125；`SingleOptimizationReport`:133；`OptimizationCheckReport`:145；`OptimizationFrame`:168；`OptimizationReport`:185 | 10 |
| [cheminfo/rdWrappers/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/rdWrappers/contracts.py#L32) | `ETKDGOptions`:32；`CoordinationEmbeddingConstraints`:42；`ConformerRecord`:57；`ConformerBatch`:65；`RuntimeInfo`:77 | 5 |
| [cheminfo/search/_predicate_ir.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/search/_predicate_ir.py#L43) | `_AtomExpression`:43；`_BondExpression`:101 | 2 |
| [cheminfo/search/_smarts_syntax.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/search/_smarts_syntax.py#L29) | `Token`:29 | 1 |
| [cheminfo/search/native.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/search/native.py#L22) | `_PreparedQuery`:22；`_NativeAutomorphismTask`:371 | 2 |
| [cheminfo/stereo.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/cheminfo/stereo.py#L72) | `StereoGroupRef`:72；`StereoCarrier`:124；`StereoAnchor`:132；`StereoRecord`:140；`HydrogenTransportState`:152；`AtomTransportRecord`:161；`BondTransportRecord`:173；`StereoPayload`:181；`StereoValidationIssue`:295；`StereoValidationReport`:305；`_Level2Descriptor`:313；`_Level2AlgebraResult`:321；`StereoDerivedDescriptor`:970；`StereoAssignmentReport`:981；`StereoAlignmentReport`:990；`_Gate6Graph`:1022；`_Gate7CoordinationSite`:1032；`_Gate7CoordinateCandidate`:1042；`_Gate6SemanticContext`:1058；`_AlignmentMatch`:4340 | 20 |
| [ds/gates/_checker.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/ds/gates/_checker.py#L496) | `_CheckerDefinition`:496 | 1 |
| [ds/governance.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/ds/governance.py#L189) | `PinnedRoot`:189；`GovernanceKey`:208 | 2 |
| [ds/public/certificate.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/ds/public/certificate.py#L93) | `_FileStamp`:93；`_ObservedFile`:103；`_BoundFile`:109；`_StateToken`:119；`_CertificationSnapshot`:129；`_VerifiedCertificateBoundary`:191 | 6 |
| [ds/storage.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/ds/storage.py#L284) | `TargetIngestResult`:284 | 1 |
| [pipeline/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/pipeline/contracts.py#L74) | `StageSpec`:74；`MolecularRecord`:82；`MolecularPayload`:90；`StageContext`:97；`Artifact`:107；`StageResult`:116；`PipelineRunResult`:127；`PreparedMolecularStage`:151 [P]；`MolecularStage`:163 [P] | 9 |
| [pipeline/state_recovery.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/pipeline/state_recovery.py#L34) | `RecoveryBudget`:34；`RecoveryPolicySet`:59；`RecoveryDecision`:66 | 3 |
| [plugins/ComplexFormer/config_task.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/ComplexFormer/config_task.py#L258) | `TaskComponents`:258 | 1 |
| [plugins/ComplexFormer/data_process/normalize_data.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/ComplexFormer/data_process/normalize_data.py#L54) | `DataProfile`:54 | 1 |
| [plugins/ComplexFormer/optim_config.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/ComplexFormer/optim_config.py#L22) | `OptimizerConfigure`:22 | 1 |
| [plugins/ComplexFormer/run/datacls.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/ComplexFormer/run/datacls.py#L70) | `RunArgs`:70；`DataModuleArgs`:274；`ConfigArgs`:324；`CallbackConfig`:363 | 4 |
| [plugins/_harness/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/_harness/contracts.py#L19) | `ProcessRequest`:19；`NativeProcessResult`:29；`ArtifactProvenance`:41；`ProcessProvenance`:49 | 4 |
| [plugins/opti/params_space.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/opti/params_space.py#L16) | `CategoricalParams`:16；`DiscreteUniformParams`:21；`FloatParams`:28；`IntParams`:35 | 4 |
| [plugins/xtb/cli.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/xtb/cli.py#L76) | `_InputRecord`:76；`_RunOptions`:83；`_WorkItem`:98；`_Outcome`:104 | 4 |
| [plugins/xtb/contracts.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/xtb/contracts.py#L59) | `XTBGeometry`:59；`XTBParsedResult`:67；`XTBBackendInfo`:78；`XTBArtifact`:91；`XTBRequest`:101；`XTBRunReport`:116 | 6 |
| [plugins/xtb/stage.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/xtb/stage.py#L78) | `_XTBStageOptions`:78；`_PreparedXTBStage`:90；`_XTBStage`:172 | 3 |
| [plugins/xtb/state_projection.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/xtb/state_projection.py#L56) | `XTBBackendContext`:56 | 1 |
| [plugins/xtb/state_resolution.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/xtb/state_resolution.py#L38) | `ResolvedXTBState`:38 | 1 |
| [plugins/xtb/stream.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/plugins/xtb/stream.py#L38) | `XTBStreamProvenance`:38；`XTBStreamMetadata`:63；`XTBStreamRecord`:74 | 3 |
| [utils/types.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/utils/types.py#L23) | `ModelLike`:23 [P] | 1 |
| [works/convert.py](https://github.com/Zhang-Zhiyuan-zzy/hotpot/blob/d9eb80e239f24a694a8cb23fd412eb6baf030cbb/hotpot/works/convert.py#L30) | `ConversionFailure`:30；`ConversionWorkerResult`:47；`_ActiveConversion`:55 | 3 |
