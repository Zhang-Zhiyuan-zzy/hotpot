# Hotpot 开发与代码改动原则

本文档定义 Hotpot 代码开发、重构、模型接入、测试和交付时必须遵守的基本原则。它适用于人工开发者与自动化编码代理。

文中的 **MUST（必须）**、**SHOULD（应当）** 和 **MAY（可以）** 是规范性用语。模块自己的公开契约和测试文档可以增加约束，但不得削弱本文档的要求。

## 1. 总体目标

Hotpot 的核心目标是提供统一、可检查、适合配位化学的化学对象与计算基础设施。代码改动必须优先保证：

1. 化学语义明确，不能用静默猜测制造“看似成功”的结果；
2. 现有抽象稳定，新增能力优先复用和扩展已有接口；
3. 科学适用域、单位、误差和后端边界对用户可见；
4. 源码运行、安装包运行和跨 Python 版本行为一致；
5. 每个逻辑改动可测试、可审查、可回退。

## 2. 修改前的工作流程

任何实现开始前都 MUST：

1. 搜索 Core、I/O、转换、search、calculator 和模型目录中是否已有相同或相近实现；
2. 确认公共入口、数据所有权、索引约定、异常类型和下游调用者；
3. 区分缺陷修复、兼容性扩展、科学语义变更和纯性能优化；
4. 对缺陷先增加最小回归测试，再修改实现；
5. 对会改变公共行为的工作先写清契约、适用域和迁移影响。

禁止复制一份 converter、parser、site detector 或模型调用链来绕开现有实现。若现有实现只需小幅调整即可通用，SHOULD 直接改进原实现并补充兼容测试。

## 3. Hotpot 原生对象是内部唯一事实源

### 3.1 输入归一化

- 外部输入 MUST 在系统边界通过 `hotpot.cheminfo.convert.to_hotpot_mol()` 归一化。
- 内部化学语义、原子索引、site detection 和结果挂载 MUST 使用 Hotpot 的 `Molecule`、`Atom`、`Bond`。
- RDKit、Open Babel、Pybel 或第三方 graph 的转换逻辑不得在各模型中重复实现。
- 转换 MUST 保持原子顺序；Hotpot 内部索引为 0-based。只有面向人的 CLI 或报告 MAY 显示为 1-based，并必须明确转换。
- 已存在的 `Molecule` 输入不应被无故复制、重新解析或经 SMILES 往返，从而丢失坐标、键元数据或配位信息。

### 3.2 后端职责边界

- Open Babel 负责受支持文件的读取及其能够提供的感知信息。
- NetworkX 是 Hotpot 子结构搜索和 SMARTS 匹配的图后端。
- RDKit MAY 用于模型特征、构象、格式桥接和绘图，但不得替代 Hotpot 的生产 search backend。
- 模型后端不得成为 Core 化学语义的隐式事实源。

## 4. 保持既有抽象和公共契约

### 4.1 Search 抽象

以下对象结构是稳定抽象，除非有经过论证且获准的架构变更，否则 MUST 保留：

- `Query`、`QueryAtom`、`QueryBond`
- `Substructure`
- `Searcher`
- `Hit`、`Hits`

活跃 SMARTS 编译入口是 `hotpot.cheminfo.search.smarts.substructure_from_smarts()`；`Molecule.search_substructure()` 是便利入口。不得创建一套平行的 search API。

搜索结果 MUST 遵守以下契约：

- query-to-target mapping 对调用者只读；
- 同一目标原子集合的 query automorphism 合并为一个 `Hit`；
- `Hit.bonds` 只表示 query 边对应的目标键；
- 额外的目标诱导边通过 `Hit.induced_bonds` 表示；
- 存在性判断使用 `has_match()`；
- 大型或高对称查询使用 `iter_mappings()` / `max_matches` 限界；
- 截断必须通过 `Hits.truncated` 显式暴露，禁止静默截断。

### 4.2 兼容性优先

- 新能力 SHOULD 通过新增明确参数、枚举、profile 或方法实现。
- 不得仅为内部方便改变公共返回类型、对象关系或属性的只读性质。
- 修复旧行为时 MUST 增加覆盖旧入口的回归测试。
- 删除疑似遗留模块前，必须确认生产引用和可能的外部深层导入影响。

## 5. 化学语义必须显式且非破坏性

### 5.1 命名语义 profile

涉及不同化学解释时，MUST 使用命名 profile，而不是含混的 Boolean 开关或隐藏分支。

当前 SMARTS 目标语义包括：

- `FULL_GRAPH`：默认语义，保持完整分子图行为；
- `LIGAND_SKELETON`：配体骨架 descriptor view，在非金属侧排除 metal–ligand 边对 `D/X/v/R/r` 的影响。

descriptor view MUST 在复制或只读视图上计算，不得临时删除、添加或恢复原分子中的键。递归 SMARTS MUST 继承父查询的 semantics。未来若需要有机金属共价语义，应新增独立命名 profile，不得偷偷改变现有 profile。

### 5.2 键的语义与数值键级分离

`BondKind` 是化学键语义的事实来源，必须保留：

- `SINGLE`
- `DOUBLE`
- `TRIPLE`
- `AROMATIC`
- `ZERO`
- `DATIVE`
- `UNKNOWN`

键方向、来源和 source metadata 在上游可提供时 MUST 保留。不得仅根据 numeric bond order 猜测 `UNKNOWN`、`ZERO` 或 `DATIVE`。不能无损表示的转换 SHOULD 明确失败，而不是悄悄改成单键。

### 5.3 尊重输入感知结果

- 语义 profile 不得暗中重新计算 implicit hydrogen、芳香性或键类型。
- Open Babel 或其他 reader 的信息损失必须通过 `UNKNOWN`、异常、fixture 和文档显式表达。
- 外部工具的结果是比较证据，不自动构成 Hotpot 的规范真值。
- Hotpot 的 `M`、`Ln`、`An`、`NP`、`NG` 等扩展不得未经翻译直接交给其他 SMARTS 引擎作为 oracle。

## 6. 错误处理与控制流

- 未知异常 MUST 向上传播；不得把失败转换为成功结果、空结果或零值。
- 禁止使用宽泛 `try/except`、层叠 `if/else` 或默认值进行无条件兜底。
- 只有产品契约明确允许的 fallback 才可存在，并且 MUST 有名称、文档、日志或结果标记以及专门测试。
- 语法错误、已识别但不支持的功能、输入感知错误和模型适用域错误必须可区分。
- SMARTS malformed input 使用 `SmartsSyntaxError`；已识别但未实现的语法使用 `UnsupportedSmartsError`。
- 只读科学属性在尚未计算时应抛出带调用指引的 `AttributeError`，不得返回 `0`、`None` 或伪造值。

## 7. AI 模型与科学结果

### 7.1 分层

模型接入 SHOULD 保持以下层次独立：

1. 输入转换与 domain checks；
2. 特征与构象构建；
3. ONNX runtime；
4. 原始模型输出；
5. site detection 或其他科学筛选；
6. Core 对象属性挂载；
7. Python API 与 CLI。

不得在 CLI、Core property 或绘图代码中重新实现模型推理及位点规则。

### 7.2 原始预测与可靠位点分离

MCA 的两个结果层次不得混淆：

- `Atom.mca`：模型对每个受支持原子的 MCA 预测；
- `Molecule.mca_sites`：经 site detection 和适用域规则筛选的重要、较可靠位点。

全原子预测不等于该原子属于模型已验证的反应位点。金属中心及直接配位原子等适用域规则必须独立、明确、可测试。

### 7.3 科学边界

- 物理量单位 MUST 写入字段、表头、文档和图例，例如 `mca_kj_mol`、`MCA(kJ/mol)`。
- 训练域之外的输入必须默认拒绝或显式标注；不得无提示外推。
- MCA 当前默认拒绝显式氢目标、分子总电荷与原子形式电荷不一致的图，以及未经验证的 charged molecule。
- 越域开关（如 `allow_charged=True`）必须由用户主动指定，且不应被描述为已验证结果。
- MCA 不得与 Mayr `N` 或 `s_N N` 混称。

### 7.4 纯推理发布

- 生产发布路径 SHOULD 使用 ONNX Runtime，不包含训练循环、优化器、私有 checkpoint 或私有训练数据。
- 模型 artifact MUST 配有 manifest、哈希、外部权重完整性检查、model card、license、适用域和数值 parity 结果。
- 明确请求 `device="cuda"` 而 CUDA provider 不可用时 MUST 报错；只有 `device="auto"` 可以自动回落到 CPU。
- 动态 shape 优先于“每种输入尺寸一个 ONNX”。模型必须有明确尺寸上限及越界异常。
- 剪枝、FP16、INT8 或其他压缩必须先与原 checkpoint 做定量 parity；达不到科学容差的候选不得发布。

## 8. CLI 原则

CLI 是现有 Python API 的薄封装：

- MUST 复用 reader、predictor、site detection 和 draw backend；
- stdout 只输出稳定、可重定向的数据，不混入 `Done`、调试文字或进度信息；
- `-o` 和 stdout 重定向必须产生相同的数据内容；
- 日志和警告写入 stderr；
- CLI 不捕获并隐藏底层解析、domain 或 runtime 异常；
- 表格、JSON 或图片中的索引、单位和 site 含义必须稳定并有测试；
- 可视化只是结果展示，不得改变预测或 site selection。

## 9. 测试与科学证据

### 9.1 测试位置

所有测试模块和 test-only fixture MUST 位于 `tests/`。禁止把测试脚本、临时数据或 benchmark 输出混入 `hotpot/` 生产包。

### 9.2 分层测试

每项改动至少包含与风险相称的测试：

1. 纯函数或对象契约单元测试；
2. 模块间集成测试；
3. 真实 reader / model / CLI 闭环测试；
4. 涉及安装内容时的 wheel-outside-source-tree smoke test。

配位化学和 SMARTS 语义 SHOULD 覆盖三层目标：

- 不依赖感知后端的纯 Hotpot 图；
- 通过 Open Babel 读取的 MOL2/SDF fixture；
- 真实 CIF 或其他代表性结构文件。

不得仅用不含金属的有机分子证明配位语义正确。

### 9.3 SMARTS conformance

SMARTS 改动 MUST 同步检查：

- parser 与 matcher 的 focused regression；
- `tests/smarts_conformance` strict contract；
- corpus schema、case ID、classification、feature tag、evidence 和 license；
- coordination fixture manifest；
- differential、fuzz 和 benchmark 中受影响的证据。

Golden expectation 必须人工审查 diff。任何工具都不得根据当前实现自动覆盖 golden，从而把回归伪装成新标准。

### 9.4 兼容性声明

修改 Core、conversion、search、MCA 或 CBond 后，合并前 MUST 运行：

```bash
bash tests/run_inference_compatibility.sh 3.9 3.10 3.11 3.12 3.13 3.14
```

该矩阵只证明其覆盖的推理、转换和搜索路径，不代表整个 legacy repository 在所有版本上均兼容。测试报告不得把 scoped green 扩大表述为全仓 green。

## 10. 性能、缓存与确定性

- 性能优化不得改变化学语义或结果顺序契约。
- 缓存 key/signature MUST 覆盖被计算逻辑消费的全部 atom、bond、connectivity、`BondKind`、aromaticity 和 semantics 状态。
- 缓存不得依赖先修改原图、计算、再恢复的流程。
- 对存在组合爆炸风险的搜索，应提供 existence fast path、流式迭代和显式上限，而不是静默丢弃结果。
- 测试、构象生成、fuzz 和 benchmark 应使用固定 seed；非确定性来源必须记录。

## 11. 包装与依赖

- 运行时依赖、entry point 或 package data 变化时，MUST 同步检查 `pyproject.toml`、`setup.py` 和 `MANIFEST.in`。
- 模型 graph、external shard、manifest、规则文件和必要资源必须实际进入 wheel。
- 发布前 MUST 构建 wheel，在源码目录之外安装，并执行真实推理或目标功能 smoke test。
- 推理包不应为了类型标注或 import side effect 引入训练框架。
- 可选重依赖应延迟导入，使不使用该功能的用户不承担无关导入失败。

## 12. Git 与工作区纪律

- 开发在专用 feature branch 上进行。
- 一个逻辑节点对应一个可独立理解和回退的 commit。
- commit message 使用 Conventional Commit 风格，例如 `test:`、`feat:`、`fix:`、`refactor:`、`docs:`、`ci:`、`build:`、`perf:`。
- 测试围栏、实现、文档和构建变更可按逻辑节点分别提交，但最终提交序列必须能解释行为为何改变。
- 不得提交用户拥有的无关修改、未跟踪目录、生成缓存、临时图片或本地环境文件。
- 不得通过 reset、checkout 或整文件覆盖破坏他人尚未提交的工作；遇到重叠修改时应先检查并合并意图。

## 13. 禁止的反模式

以下做法原则上禁止：

- 为单一模型复制 molecule converter 或文件 reader；
- 使用 RDKit SMARTS 替换 NetworkX-backed Hotpot search；
- 无必要改变 `Searcher`、`Hits`、`Hit`、`Query*` 的对象结构；
- 使用 Boolean 或隐式分支代替命名化学 semantics；
- 为计算 ligand rings 临时删除再恢复原图键；
- 根据 numeric bond order 猜测丢失的 `BondKind`；
- 宽泛捕获异常后返回空列表、默认值、CPU 结果或成功状态；
- 静默截断搜索结果；
- 把每原子模型预测直接宣称为可靠反应位点；
- 未经 parity 验证就发布量化、剪枝或新模型；
- 恢复 fixed-shape ONNX 文件矩阵；
- 将测试数据、训练代码或 checkpoint 混入 inference runtime；
- 只在源码树中测试，不验证安装后的 wheel；
- 用局部测试通过宣称全仓库兼容。

## 14. Definition of Done

代码改动只有同时满足以下条件才视为完成：

- [ ] 已确认并复用现有抽象，没有建立无必要的平行实现；
- [ ] 公共 API、化学语义、单位、索引和异常契约明确；
- [ ] 没有静默 fallback、吞异常或无条件兜底；
- [ ] 新增回归测试位于 `tests/`，并覆盖成功与失败路径；
- [ ] 真实文件、模型或 CLI 路径按风险完成闭环验证；
- [ ] 相关 focused tests、strict contracts 和兼容矩阵通过；
- [ ] 包装变化已通过 wheel 外部安装测试；
- [ ] 文档说明适用域、已知限制和用户可见行为；
- [ ] lint、格式和 `git diff --check` 通过；
- [ ] 提交原子化，且未夹带用户或其他开发者的无关文件。

## 15. 相关规范入口

- `tests/README.md`
- `tests/smarts_conformance/SMARTS_CONFORMANCE.md`
- `tests/smarts_conformance/README.md`
- `hotpot/cheminfo/AImodels/INFERENCE_COMPATIBILITY.md`
- `hotpot/cheminfo/AImodels/mca/MODEL_CARD.md`
- `hotpot/cheminfo/AImodels/mca/README.md`
- `hotpot/cheminfo/AImodels/cbond/README.md`
- `.github/workflows/inference_compatibility.yml`

当本文与更具体的模块契约发生表面冲突时，开发者必须先明确冲突原因并更新文档或设计，不得自行选择更宽松的解释。
