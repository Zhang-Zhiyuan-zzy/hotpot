# Hotpot 开发规范

[English version](development.en.md) · [规则暂存区](development.tmp.md)

本文档是 Hotpot 当前有效的开发契约，适用于人工开发者和自动化编码代理。它是经整理的
当前规范，不是按时间追加的讨论日志。

文中 **MUST（必须）**、**SHOULD（应当）** 和 **MAY（可以）** 是规范性用语。模块契约可增加限制，
但不得隐式放宽本文要求。

## 1. 目标与规则治理

Hotpot 的核心目标是提供统一、可检查、适合配位化学的化学对象与计算基础设施。任何改动都必须优先保证：

1. 化学和数值语义明确，不用静默猜测制造“看似成功”的结果；
2. 数据所有权、后端边界、单位、适用域和失败状态可见；
3. 新能力复用既有抽象，不建立无必要的平行实现；
4. 源码、安装包与明确支持的运行环境行为一致；
5. 逻辑可测试、可审查、可回退，实验性结论不冒充已验证契约。

### 1.1 规则暂存与并入 `[DEV-GOV-001]`

- 对话、代码审计或失败复盘中识别的新规则，先写入 `development.tmp.md`，记录作用域、理由和证据 commit。
- 规则并入时 MUST 同时重构中英文两份正式文档，合并重复内容，并删除过时或不再适用的规则。
- 正式文档不做无限追加；新并入规则 MUST 在文末记录并入日期和来源 commit hash。
- `development.tmp.md` 仅保留当前待处理批次和最近一次并入回执，不作永久追加式历史。

## 2. 修改前先确定契约

实施开始前 MUST：

1. 搜索 Core、I/O、conversion、graph、geometry、search、calculator 和模型目录中的相同或相近实现；
2. 确定公共入口、数据所有权、索引约定、单位、异常和下游调用者；
3. 区分纯整理、缺陷修复、兼容性扩展、科学语义变更和性能优化；
4. 对缺陷先建立最小回归测试；纯重构先建立行为等价基线；
5. 在改变公共行为前写清契约、适用域、迁移影响和验收方式。

禁止复制 converter、reader、parser、site detector 或模型调用链来绕开现有实现。只有在输入、输出、
不变量、所有权、生命周期和失败语义一致时，才应抽取共享 helper。不得为减少表面代码量而引入大量
`mode`、Boolean 开关、宽泛回调或联合类型。

## 3. 架构与职责边界

### 3.1 Hotpot 对象是内部化学事实源

- 外部输入在系统边界通过 `hotpot.cheminfo.convert.to_hotpot_mol()` 归一化。
- 内部化学语义、原子索引、site detection 和结果挂载使用 Hotpot `Molecule`、`Atom`、`Bond`。
- RDKit、Open Babel、Pybel 或第三方 graph 的转换只存在于共享边界层，不得在各业务模块重复实现。
- Open Babel 负责读取其支持的分子文件，并提供它能够感知的信息；其结果仍受 Hotpot 边界契约约束。
- 转换 MUST 保留原子顺序、已知键语义和可用元数据。Hotpot 内部索引为 0-based；只有面向人的输出 MAY 显示 1-based。
- 已是 `Molecule` 的输入 SHOULD NOT 被无故复制、重新解析或经 SMILES 往返，以免丢失对象身份、坐标、键元数据或配位信息。
- 第三方后端提供输入感知或数值证据，不自动成为 Core 化学语义的规范真值。

若某个可变对象工作流需要事务和提交语义，该语义 MUST 由所属模块契约单独定义；本通用规范不对所有 Hotpot 函数强制同一种事务模型。

### 3.2 数学事实、科学语义与流程职责 `[DEV-ARCH-001]`

- 数学事实层与科学语义层 MUST 严格区分，并 SHOULD 在模块和代码实施上尽量分离。
- `geometry` 等数学模块只提供数学对象、度量、关系、退化情况和不确定性等事实；不得评价结构在化学或物理上是否合理、现实、高质量或适用，也不得决定修复策略。
- 为浮点稳定性、退化识别或数学关系分类服务的容差属于数学层；表达化学或物理标准的阈值、评分和接受规则 MUST 归属于拥有该语义的 `chemistry`、`forcefields` 或其他科学模块。
- 科学语义层可以消费数学事实，并将其映射为化学、物理、模型适用域或力场质量结论；不得将这些科学判定反向放入数学模块。
- 控制器独占重试、扰动、回滚、选择和退出权；这些决策必须在主流程中显式可见。
- 记录器可记录、查询、评分、排序和保存事实，但不得决定流程跳转。
- CLI、plot 和 movie 属于展示层，serialization/persistence 属于交付层；两者都不得改变科学计算路径。
- Core 便利方法和 CLI 应为薄入口，不得复制业务实现。

## 4. 公共契约、兼容性与源码结构

### 4.1 有证据的兼容性 `[DEV-COMP-001]`

- 只为明确支持的已发布公共 API、artifact schema、Python 版本或 backend 版本保留兼容。
- 不得仅为内部实现方便改变公共返回类型、对象关系、只读属性或异常语义。
- 无历史契约的内部字段直接使用当前严格契约；禁止推测性旧字段默认值、别名或 wrapper。
- 版本或后端选择必须集中在一个明确的 composition boundary；专用实现放入隔离的 façade/adapter，共享业务保持单一事实源。
- 平行 façade 的公共名、签名、返回类型、单位和异常契约 MUST 一致，并由自动化测试核对。
- 预留但尚未实施的公共参数必须在 docstring 中明确标记当前行为，不得伪装已生效。

### 4.2 类型与化学对象命名

- 除了无法静态表达的最小动态第三方边界，禁止使用 `Any`。
- 不可避免的 `Any` MUST 在邻近注释或文档中说明原因，且 MUST NOT 沿调用链扩散到内部领域逻辑。
- 已知 Hotpot 对象使用 `Molecule`、`Atom`、`Bond`、`Crystal` 等具体类型。循环导入应通过
  `TYPE_CHECKING`、延迟注解或 forward reference 解决。
- 结构化多态使用最小 `Protocol`、类型别名、泛型或联合类型；序列化 payload 优先使用 `TypedDict`、dataclass 或递归值类型。
- 类名表达完整领域概念；局部变量和形式参数优先使用 `mol`、`atom`、`bond`、`cbond` 等约定简称。
- 对象存在来源、所有权或生命周期差异时，名称应包含对象和角色，例如 `source_mol`、`clone_mol`、`working_mol`、`target_atom`。
- 当某语义在当前 API 层是唯一默认时，使用基础名称；只有真实存在并列语义时才增加限定词。

### 4.3 模块公共面与排布 `[DEV-MOD-001]`

对新建或大幅重构的模块：

- 文件顶部通过 `__all__` 明确公共面；
- Exception、Enum 和数据契约放在实现 helper 之前；
- 私有 helper 按职责分区；
- 公开操作放在支撑实现之后，由低层到高层排列；
- 公开或持久化的有限状态集 SHOULD 使用 `Enum`；局部静态类型限制 MAY 使用 `Literal`；
- 生产模块不保留未使用 import。星号导入只允许在受 `__all__` 约束的 package 组装入口使用。

### 4.4 新实现接管后的清理 `[DEV-CLEAN-001]`

- 新事实源、生命周期或入口接通后，MUST 全仓搜索其消费者。
- 删除已取代的私有 helper、字段、wrapper、测试和无依据兼容分支，不保留两套活跃实现。
- 全仓零内部消费者只是死代码证据，不是删除已文档化公开抽象的充分条件。
- 必须保留的旧公共 API 使用明确 deprecation 契约，不使用隐藏 wrapper 无期维持。

## 5. 化学与科学语义

- 不同化学解释使用具名 profile 或枚举，不使用含混 Boolean 或隐藏分支。
- 语义视图在副本或只读视图上计算，不得通过临时删键、加键再恢复来修改原图。
- `BondKind` 是键语义的事实源。不得只根据 numeric bond order 猜测 `UNKNOWN`、`ZERO` 或 `DATIVE`。
- ring family 是算法契约的一部分；`cycle basis`、`relevant cycles` 与其他环集不得混称。下游逻辑不应依赖任意环基或输入顺序。
- 单位必须在 API、字段、表头和图例中明确。后端数值在边界层一次性转换为内部单位。
- 语义 profile 不得暗中重新计算 implicit hydrogen、芳香性或键类型。后端信息损失 MUST 以 `UNKNOWN`、异常、fixture 或文档显式表达。

Search 保持 `Query*`、`Substructure`、`Searcher`、`Hit/Hits` 公共抽象，生产 SMARTS 匹配使用 NetworkX-backed Hotpot search。
RDKit 可用于特征、构象、格式桥接和绘图，但不得隐式替代该事实源。

## 6. 失败、诊断与控制流 `[DEV-ERR-001]`

- 未知程序异常 MUST 原样向上传播，不得转换成成功、空结果、零值或 CPU 结果。
- 执行失败、不可返回结果和“可返回但未通过科学质量标准”是不同状态，API 必须明确区分。
- 禁止用宽泛 `try/except`、层叠 `if/else` 或默认值实现无条件兜底。
- fallback 只能在产品契约明确允许时存在，并且必须有名称、状态标记、文档和专门测试。
- 保留终止值、末帧、报告或诊断轨迹不等于宣告成功；失败状态必须仍显式可见。
- 当 API 承诺失败证据时，warning-return 和 exception 路径都必须保留该证据。
- 只读科学属性尚未计算时，应抛出带调用指引的 `AttributeError`，不返回伪造值。

## 7. 可选的科学历史与持久化 `[DEV-OBS-001]`

本节只适用于提供 history、trajectory 或 provenance 的工作流，不要求所有计算默认记录全部状态。

- 全量历史记录 MUST 由用户或 API 显式开启。默认记录至多保留模块契约所需的有界摘要或选中终态，并必须说明资源成本。
- 实施前必须评估帧数、状态大小、证据计算、内存、压缩和 I/O 开销；优先去重、池化或有界保留。
- 每个观测必须包含解释它所需的最小完整状态。当拓扑变化时，coordinate-only conformer 不是权威记录。
- 记录、显式选择、向业务对象物化和磁盘持久化是不同职责。
- 持久格式必须声明 schema/format version、单位和缺失值语义；JSON 不得写入 `NaN`/`Infinity`。
- 路径应可移植；读取不应依赖执行任意对象的 pickle；归档必须通过 round-trip 测试。
- 覆写已有归档时必须确定性清理旧 artifact；有损格式必须声明省略的状态。

## 8. 模型、API 与 CLI

- 模型输入转换、特征/构象、runtime、原始输出、科学筛选、Core 属性挂载和 CLI 应保持分层。
- 原始模型预测不等于经适用域验证的化学位点。训练域外的输入必须拒绝或显式标记。
- 纯推理发布不包含私有训练代码、checkpoint 或训练数据。模型 artifact 需要 manifest、hash、license、适用域和数值 parity 证据。
- 显式请求的计算后端不可用时必须报错；只有明确的 `auto` 模式允许自动选择后端。
- 压缩、量化、剪枝或新模型发布前必须完成定量 parity 验证。
- CLI 是 Python API 的薄封装；stdout 只输出稳定可重定向的数据，日志与警告写入 stderr。
- CLI 不重新实现 reader、模型、site detection 或绘图的科学逻辑，也不隐藏底层异常。

## 9. 验证与交付

- 所有测试和 test-only fixture 位于 `tests/`。Benchmark 可位于独立目录或外部 artifact，但不得混入生产 package。
- 测试深度与风险相称：纯函数/契约单测、模块集成、真实 reader/backend/CLI 闭环，以及影响包内容时的 wheel 外部 smoke test。
- 至少覆盖与改动相关的成功和失败契约。测试 double 必须满足当前数据契约，不得要求生产代码兼容过时 mock。
- 修改共享 API、安装内容或跨版本路径时，使用仓库提供的兼容 runner/CI；具体 corpus 和矩阵由相关模块测试文档维护。
- focused green 只能证明已覆盖范围，不得扩大成全仓兼容结论。
- Golden expectation 必须人工审查 diff，不得用当前实现自动覆写 golden 来隐藏回归。

## 10. 性能、依赖与运行时边界

- 性能优化不得改变化学语义、失败契约或结果顺序；性能声明需要可重复基准。
- 全量历史记录 MUST 为 opt-in；默认历史必须有界。高成本搜索 SHOULD 为 opt-in 或有明确边界。
- 缓存 key/signature 必须覆盖被计算消费的 atom、bond、connectivity、`BondKind`、aromaticity 和 semantics 状态。
- 缓存不得依赖“修改原图→计算→恢复”的过程生成。
- 可能组合爆炸的搜索应提供 existence fast path、流式迭代和显式上限，禁止静默截断。
- 测试、fuzz 和 benchmark 应使用固定 seed。生产随机算法应提供可选 seed，但不要求默认固定；非确定性来源必须记录。
- 运行时依赖、entry point 或 package data 变化时，同步检查 `pyproject.toml`、`setup.py` 和 `MANIFEST.in`。
- 可选重依赖延迟导入，不使非相关用户承担导入失败。类型标注不应引入训练框架或其他重依赖。

## 11. Git 与工作区纪律

- 开发在专用 feature branch 上进行。
- 一个逻辑节点对应一个可独立理解和回退的 commit，并使用 Conventional Commit message。
- 测试围栏、实现、文档和构建变更可分开提交，但最终序列必须能解释行为为何改变。
- 不得提交用户或其他开发者的无关修改、未跟踪目录、缓存、临时图片或本地环境文件。
- 不得用 reset、checkout 或整文件覆盖破坏他人未提交工作；重叠修改必须先合并意图。

## 12. Definition of Done

- [ ] 已确认契约、所有权、单位、适用域和失败语义。
- [ ] 已复用现有抽象，新增抽象的不变量和职责边界明确。
- [ ] 没有静默 fallback、吞异常、无条件兜底或伪成功结果。
- [ ] 风险相称的成功/失败、集成、真实路径和包装测试已通过，报告未超出覆盖范围。
- [ ] 若提供科学历史或持久化，已评估资源并验证 schema、round-trip、覆写和有损输出语义。
- [ ] 新实现接管后已清理被取代路径，公共面、lint、格式和 `git diff --check` 通过。
- [ ] 文档说明用户可见行为和限制，commit 原子化且未夹带无关文件。

## 13. 当前有效规则来源

| 规则 | 并入日期 | 暂存提交 | 实施证据提交 |
|---|---|---|---|
| `DEV-COMP-001` | 2026-09-23 | `339c53e` | `0e90c63`, `c604ef5`, `c5e53c9`, `f5a66a7`, `15cb23c` |
| `DEV-ARCH-001` | 2026-09-23 | `339c53e`, `882d9dd` | `aa72c67`, `b8dc637`, `5e344ad`, `bf67119`, `ebb4a53`, `dc3fc2b` |
| `DEV-ERR-001` | 2026-09-23 | `339c53e` | `43b83d9`, `186d7b4`, `1ea39cc`, `81fd33b` |
| `DEV-OBS-001` | 2026-09-23 | `339c53e` | `81e02de`, `3631c7c`, `5795d8f`, `ebb4a53`, `dc3fc2b` |
| `DEV-MOD-001` | 2026-09-23 | `339c53e` | `e789e7a`, `d08792a`, `f26ca0f`, `6b8755e`, `ba67c91` |
| `DEV-CLEAN-001` | 2026-09-23 | `339c53e` | `08639d7`, `87277f0`, `c1ace9f`, `f5a66a7`, `15cb23c` |
| `DEV-GOV-001` | 2026-09-23 | `339c53e` | `df78c7c`, `6e2ebac`, `a34375b`, `58a43df` |

## 14. 模块契约与参考

- `tests/README.md`
- `tests/smarts_conformance/SMARTS_CONFORMANCE.md`
- `tests/smarts_conformance/README.md`
- `hotpot/cheminfo/geometry/README.md`
- `hotpot/cheminfo/graph/README.md`
- `hotpot/cheminfo/kekulize/Kekulize.md`
- `hotpot/cheminfo/AImodels/INFERENCE_COMPATIBILITY.md`
- `hotpot/cheminfo/AImodels/mca/MODEL_CARD.md`
- `hotpot/cheminfo/AImodels/mca/README.md`
- `hotpot/cheminfo/AImodels/cbond/README.md`

当本文与更具体的模块契约表面冲突时，必须先确定作用域和冲突原因，然后更新规范或模块设计；
不得自行选择更宽松的解释。
