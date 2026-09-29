# SMARTS 解析与子结构搜索高覆盖测试实施手册

> 本文可以直接交给 AI Agent 执行。目标不是让 Agent 只提供测试建议，而是要求它检查当前仓库、编写测试代码、运行测试并交付证据充分的合规性报告。

## 0. 给执行 Agent 的总指令

你负责为当前项目建立一套高覆盖、可重复、可离线运行、可持续扩展的 SMARTS 解析与子结构搜索测试系统。

你必须实际完成以下工作：

1. 检查仓库、项目约束、现有 API、测试框架和已有测试。
2. 明确项目声称兼容的 SMARTS 方言及分子预处理规则。
3. 建立统一测试适配器和有来源记录的测试语料。
4. 分别测试词法/语法、查询图编译、匹配语义、结果枚举、批量搜索、健壮性和性能。
5. 在条件允许时使用多个独立实现做差分验证，但不得把任何单一实现直接视为标准。
6. 实际运行测试、覆盖率和短时健壮性测试。
7. 输出机器可读结果及人类可读报告。
8. 不得为了全绿而改写预期结果、吞掉异常、放宽断言或无理由跳过案例。
9. 除非任务明确授权修复生产代码，否则不要修改 SMARTS 解析器或搜索实现；发现缺陷时保留最小复现和失败证据。

遇到非关键不确定性时，根据仓库现状作出保守假设并写入文档。只有当目标方言或匹配契约完全无法推断、且不同选择会实质改变验收结果时，才向用户提出一个简短问题。

## 1. 首先理解“SMARTS 标准”的边界

SMARTS 没有一个被所有工具完整、统一实现的现代机器可读标准。Daylight SMARTS 文档通常被视为历史基线，但 RDKit、Open Babel、CDK、ChemAxon 等实现对芳香性、环、手性、递归查询和扩展语法可能有不同解释。

因此必须把测试分成三种合规配置：

- core：项目承诺支持的 Daylight-like 核心语法。
- extension：项目明确声明支持的扩展。
- compatibility：为了兼容特定工具而定义的行为，例如 rdkit-compatible。

不要将 OpenSMILES 当作 SMARTS 的规范来源；OpenSMILES 主要描述 SMILES。它可以辅助验证靶分子输入，但不能单独决定 SMARTS 查询语义。

每个测试案例必须属于以下分类之一：

- valid_core：目标核心方言中的合法输入。
- valid_extension：项目明确支持的扩展。
- invalid_syntax：在目标方言中确定存在词法或语法错误。
- unsupported_feature：语法本身合法，但项目明确不支持。
- dialect_disputed：规范不清晰或主流实现不一致。
- semantic_case：重点验证查询匹配语义。
- regression：曾经触发项目缺陷的最小复现。
- robustness_only：只要求不崩溃、不挂起，不预设其合法性。

必须区分：

- 语法非法；
- 语法合法但查询不可满足；
- 语法合法但实现不支持；
- SMARTS 合法，但靶分子 SMILES 或分子预处理失败；
- 不同方言行为不同。

“化学上不合理”不等于“SMARTS 语法非法”。不得仅因某查询无法匹配真实分子就将其放入非法语料。

## 2. 建立分层测试模型

将完整执行过程拆成以下阶段，并让失败报告指出准确阶段：

    SMARTS 文本
      -> 词法分析
      -> 语法分析
      -> 查询 AST / 查询图编译
      -> 查询校验
      -> 靶分子解析与预处理
      -> 子图同构匹配
      -> 匹配去重与结果限制
      -> 批量搜索或索引层

至少分别验证：

1. Parse acceptance：SMARTS 是否应被接受。
2. Parse diagnostics：拒绝发生在哪一阶段，错误位置是否合理。
3. Query semantics：编译后的原子、键、逻辑、递归条件是否正确。
4. Match existence：是否存在至少一个匹配。
5. Match enumeration：返回哪些查询原子到靶原子的映射。
6. Uniquification：不同自动同构映射如何去重。
7. Search behavior：批量分子搜索、缓存、索引、分页和并发是否改变结果。
8. Robustness：恶意或极端输入能否导致崩溃、挂起或资源失控。
9. Performance：正常和病理输入的时间、内存与规模增长趋势。

## 3. 仓库检查

开始写代码前，读取并遵守：

- AGENTS.md 或等价 Agent 指令；
- README、CONTRIBUTING 和开发文档；
- 构建、依赖和测试配置；
- 现有 SMARTS、SMILES、分子图及子结构搜索测试；
- 项目已有的错误类型、日志和报告规范。

定位并记录：

- SMARTS 解析入口；
- 查询编译入口；
- 单分子匹配入口；
- 批量搜索或索引入口；
- SMILES 解析及 sanitization 入口；
- 芳香性、隐式氢、立体化学配置；
- 是否可返回 AST、查询图、匹配原子索引和全部 embeddings；
- 是否有 uniquify、useChirality、maxMatches 等选项；
- 是否已依赖 RDKit、Open Babel、CDK 或其他参考实现；
- 测试、覆盖率、benchmark 和 fuzz 工具。

优先复用项目现有语言、测试框架和命令。不要为了方便而在另一种语言中复制一套无法进入项目 CI 的测试。

## 4. 先冻结项目的匹配契约

在测试目录或项目文档目录新增 SMARTS_CONFORMANCE.md，至少写清：

- 目标 SMARTS 方言及扩展；
- 芳香性模型和芳香化时机；
- 是否执行价态、环和隐式氢计算；
- 显式氢与隐式氢的处理；
- R、r、x、D、X、v、H、h 的项目定义；
- 递归 SMARTS 的支持范围和深度限制；
- 原子映射号是否参与匹配；
- 手性默认开启还是由选项开启；
- 未指定键的语义；
- disconnected query 的语义；
- component-level grouping 是否支持；
- reaction SMARTS 是否属于本模块；
- 匹配结果是否包含查询自动同构产生的重复；
- maxMatches 是在去重前还是去重后生效；
- 解析失败、unsupported 和 target preprocessing failure 如何暴露。

如果当前项目没有明确契约，依据代码和现有测试推导一个“当前行为配置”，同时单独定义“目标核心配置”。不要偷偷把当前 bug 写成目标标准。

## 5. 建议交付结构

路径应服从仓库现有惯例，但应提供以下等价内容：

    tests/smarts_conformance/
      README.md
      feature_matrix.yaml
      corpus/
        manifest.yaml
        valid_core.jsonl
        valid_extensions.jsonl
        invalid_syntax.jsonl
        disputed.jsonl
        semantic_matches.jsonl
        regressions.jsonl
        robustness.jsonl
      adapter.*
      corpus_validator.*
      test_parse.*
      test_query_graph.*
      test_semantics.*
      test_enumeration.*
      test_search.*
      test_metamorphic.*
      test_regressions.*
      differential/
        runner.*
        normalize.*
        README.md
      fuzz/
      benchmarks/

另外交付：

- SMARTS_CONFORMANCE.md；
- smarts_conformance_report.md；
- 可选的 smarts_conformance_report.json；
- smoke、core、heavy 三种运行命令；
- 差分测试、fuzz 和 benchmark 的独立命令；
- 新增回归案例和审核 golden 数据的说明。

## 6. 统一测试适配器

不要让数据驱动测试直接散落调用内部 API。建立薄适配层，尽可能实现：

    parse_smarts(text, options) -> {
      accepted,
      phase,
      error_code,
      error_position,
      diagnostic,
      ast_or_query_summary
    }

    prepare_target(smiles, options) -> {
      accepted,
      phase,
      error_code,
      atom_identity_map,
      molecule
    }

    match_smarts(smarts, smiles, options) -> {
      query_accepted,
      target_accepted,
      matched,
      raw_embedding_count,
      embeddings,
      unique_target_atom_sets,
      truncated,
      error_code
    }

    search_smarts(smarts, molecule_records, options) -> {
      matched_record_ids,
      per_record_matches,
      truncated,
      errors
    }

适配器要求：

- SMARTS 必须经 SMARTS/query parser 解析，不能误用 SMILES parser。
- 靶分子必须经项目真实使用的 SMILES/分子加载路径处理。
- 不支持的字段返回明确的 unsupported，不得伪造数据。
- 不对错误消息全文作脆弱断言；优先断言稳定 error code、阶段和位置范围。
- 将库异常转换成测试层结构化结果，但不得吞掉崩溃或未知异常。
- 测试适配器本身要有少量自测，避免适配错误被误判为解析器错误。

### 6.1 匹配结果的正确规范化

必须区分三种结果：

1. embedding：按查询原子顺序排列的靶原子身份元组。
2. target atom set：忽略查询原子对应顺序后的靶原子集合。
3. existence：是否至少存在一个匹配。

对 embeddings 只能排序外层列表；不能把每个元组内部随意排序，否则会丢失查询原子到靶原子的对应关系。只有比较 target atom set 时才可对元组内部排序。

跨工具比较时，不要直接依赖各工具内部原子索引。优先使用稳定原子身份：

- 测试输入中的 atom-map 标签；
- 或适配器维护的输入原子序号映射；
- 或经过验证的稳定图同构标识。

先验证 atom-map 标签不会意外改变被测匹配语义，再用其作跨实现身份标识。

精确匹配数量至少可能有三种定义：

- 原始子图 embeddings 数量；
- 按查询映射去重后的数量；
- 按靶原子集合去重后的数量。

只有在 API 契约明确时才断言精确数量。不得把不同实现的默认 uniquify 行为直接比较。

## 7. 测试语料格式和治理

推荐使用 JSONL，并用 manifest 固定 schema_version、方言、来源和默认选项。

解析案例示例：

    {
      "id": "atom.logic.precedence.001",
      "smarts": "[C,N;H1]",
      "classification": "valid_core",
      "expected_outcome": "accept",
      "features": ["atom_logic", "or", "low_precedence_and"],
      "dialect": "daylight_core",
      "source": "curated",
      "notes": "必须配合语义案例验证优先级"
    }

非法案例示例：

    {
      "id": "invalid.bracket.unclosed.001",
      "smarts": "[C",
      "classification": "invalid_syntax",
      "expected_outcome": "reject",
      "expected_phase": ["lex", "parse"],
      "features": ["unclosed_bracket"],
      "source": "curated"
    }

语义案例示例：

    {
      "id": "semantic.alcohol.001",
      "smarts": "[#6]-[O;H1]",
      "target": {
        "format": "smiles",
        "text": "CCO"
      },
      "options": {
        "use_chirality": false
      },
      "expected": {
        "matched": true
      },
      "features": ["atomic_number", "hydrogen_count", "single_bond"],
      "oracle": {
        "kind": "curated"
      }
    }

每条外部语料必须保存：

- 原始来源；
- 许可证；
- 版本、commit 或发布日期；
- 导入和转换方式；
- 是否经人工审核；
- 参考引擎及版本；
- 参考引擎运行选项。

编写 corpus validator，强制检查：

- ID 唯一；
- schema 合法；
- 分类和期望结果不矛盾；
- 必需 feature tag 存在；
- 无无理由重复案例；
- 外部案例具有 provenance；
- disputed 案例不会进入核心强制断言；
- known failure 具有原因和追踪 ID。

## 8. 语法覆盖要求

每一类至少包含：简单正例、组合正例、确定反例、边界例和能够区分语义的靶分子案例。

### 8.1 原子与方括号表达式

覆盖：

- 非括号原子和方括号原子；
- 目标方言允许的 organic subset；
- 多字符元素符号及大小写；
- 芳香原子符号；
- 通配符 *、芳香/脂肪族通配语义；
- 原子序数 #n；
- 同位素；
- 正负电荷及允许的等价写法；
- 显式氢、总氢数和隐式氢条件；
- atom map/class；
- 手性标记；
- 未知、截断或大小写错误的元素符号。

重点区分：

- [H] 表示氢原子；
- [C;H1] 中 H1 表示氢计数条件；
- H 与 h 在目标方言中的不同语义；
- atom map 是元数据还是匹配约束。

### 8.2 原子查询属性

按目标方言覆盖：

- D：degree；
- X：connectivity；
- v：valence；
- H、h：氢相关条件；
- R：环成员或环计数；
- r：环尺寸相关条件；
- x：环连接相关条件；
- 电荷、同位素、原子序数；
- 手性类别；
- 属性缺省值、零值、多位数字和边界值；
- 合法但不可满足的属性组合。

R、r、x 及芳香环经常受到环感知算法影响，报告中必须记录分子预处理和环模型。

### 8.3 逻辑表达式和优先级

覆盖：

- !：NOT；
- &：高优先级 AND；
- 隐式 AND；
- ,：OR；
- ;：低优先级 AND；
- 多层组合及重复否定；
- 原子逻辑和键逻辑；
- 缺少左/右操作数；
- 连续运算符；
- 空表达式。

不能只验证“解析成功”。为每个优先级规则设计能够让两种解析方式产生不同匹配结果的最小分子集合，从而验证真实语义。

### 8.4 键表达式

按方言覆盖：

- 隐式键；
- 单键 -；
- 双键 =；
- 三键 #；
- 芳香键 :；
- 任意键 ~；
- 环键 @ 与非环键 !@；
- 方向键 / 和反斜线；
- 键逻辑组合与否定；
- 环闭合位置的键声明；
- 芳香原子间默认键；
- 两端冲突的环闭合键；
- 悬空、重复或缺少操作数的键运算符。

### 8.5 图结构

覆盖：

- 单原子与线性结构；
- 分支和多层嵌套分支；
- 环闭合数字；
- %nn 形式的环编号；
- 环编号重用；
- 未闭合或只出现一次的环编号；
- 分支与环交叉组合；
- 点号分隔的 disconnected query；
- 空分支、空组分和不匹配括号；
- 大量分支、环和深层嵌套。

大于两位数的环号、%(...) 或其他实现专属形式必须按实际方言分类，不得未经确认放入核心语料。

### 8.6 递归 SMARTS

覆盖：

- $() 基础递归；
- 递归条件与其他原子条件组合；
- 递归内的分支、环和逻辑表达式；
- 多层递归；
- 空递归和未闭合递归；
- 递归锚点语义；
- 深度限制；
- 可能导致大量回溯的递归模式。

### 8.7 芳香性

至少覆盖：

- 苯及取代苯；
- 吡啶型芳香氮；
- [nH] 型芳香氮；
- 五元芳香杂环；
- 稠合和桥连环系；
- 芳香键 :；
- 芳香原子之间的隐式键；
- Kekulé 与芳香写法；
- 输入已芳香化与由项目执行芳香化；
- 不同芳香性模型下的争议案例。

芳香性差异必须优先归因于预处理/模型，再判断是否属于查询引擎错误。

### 8.8 立体化学

若项目声明支持，覆盖：

- @ 和 @@；
- 四面体中心；
- 对映体阳性和阴性对照；
- 双键方向 / 和反斜线；
- 查询指定手性与未指定手性的差异；
- useChirality 开关；
- 改变输入原子顺序后映射回原身份的结果；
- 不完整或非法手性标记；
- 项目支持的非四面体扩展。

若项目不支持，应分类为 unsupported_feature，并验证其拒绝或降级行为与文档一致，不能将其伪装成核心语法错误。

### 8.9 方言扩展

单独调查和分类：

- component-level grouping；
- reaction SMARTS；
- dative bond；
- zero-order bond；
- hybridization 查询；
- 特定工具专属原子、键或范围语法；
- 超大环编号或额外递归能力。

## 9. 非法输入与诊断测试

确定性非法语料至少覆盖：

- 空输入和纯空白；
- 未闭合或多余的 []、()、$()；
- 未闭合环号；
- 悬空键；
- 缺少逻辑操作数；
- 非法运算符组合；
- 非法或截断电荷；
- 非法同位素；
- 非法 atom map；
- 非法或截断元素符号；
- 控制字符、NUL 和异常 Unicode；
- 极长数字；
- 极长 token；
- 超深嵌套；
- 超长输入；
- 只含标点的输入。

对每个负例先判断它是 invalid_syntax、unsupported_feature 还是 dialect_disputed。不要因为 RDKit 或任一工具拒绝就自动认定为标准非法。

负例至少断言：

- 不崩溃；
- 不死循环；
- 不产生未捕获异常；
- 在配置的资源限制内返回；
- 错误阶段稳定；
- 若提供错误位置，则位置处于有效范围；
- 重复执行结果一致。

错误消息的自然语言通常不适合作精确全文断言。

## 10. 匹配语义测试

对每个核心语法功能建立：

- 明确阳性；
- 明确阴性；
- 只改变一个关键特征的 near-negative；
- 多匹配位点；
- 对称分子；
- 环/非环对照；
- 芳香/Kekulé 对照；
- 电荷、同位素、显式/隐式氢对照；
- 手性对照；
- 递归查询；
- 逻辑优先级对照；
- 多组分靶分子。

分别验证：

- 查询能否编译；
- 靶分子能否准备；
- 是否存在匹配；
- embedding 的查询原子对应关系；
- 靶原子集合；
- 去重后的数量；
- useChirality 和 uniquify 等选项；
- atom map 是否按契约影响匹配；
- maxMatches 截断标志是否正确。

不得把以下变化默认视为等价：

- 互变异构体；
- 质子化状态；
- 芳香性模型；
- 显式氢和隐式氢；
- 手性被省略与手性未知；
- 不同盐组分或 disconnected component。

## 11. 批量 search 与索引层测试

如果项目除了单分子匹配还提供批量搜索、数据库扫描或子结构索引，必须额外验证：

1. 以未经索引的逐分子 matcher 作为本项目内部 baseline。
2. 对相同固定分子集，索引搜索与 baseline 的命中 ID 集合完全一致。
3. 结果顺序只有在 API 明确保证时才断言；否则比较集合。
4. maxResults、分页、流式输出和提前终止不会引入假阳性。
5. 若允许截断，结果必须是完整结果的子集并显式标记 truncated。
6. 查询缓存不能在不同芳香性、手性或 uniquify 选项之间泄漏。
7. 重复 molecule ID、无效记录和部分读取错误按文档处理。
8. 单线程和并发搜索结果一致。
9. 多次运行结果确定。
10. 若支持增量索引，测试 add、update、delete 后不残留旧结果。
11. 若使用 fingerprint prefilter，必须通过 baseline 证明无假阴性；prefilter 可以有假阳性，但最终 matcher 不能。
12. 序列化、重载或跨进程恢复索引后结果一致。

对索引误差报告必须区分：

- prefilter false negative；
- 最终 matcher false result；
- 缓存键错误；
- 分子预处理不一致；
- 结果截断或分页错误。

## 12. Oracle 与差分测试策略

使用以下证据等级：

- Tier 1：经规范和人工推导的最小确定案例，权重最高。
- Tier 2：多个独立参考实现、在对齐选项后得到的一致结果。
- Tier 3：项目公开声明的兼容目标。
- Tier 4：不依赖外部答案的变形关系和健壮性性质。

若环境允许，差分 runner 至少支持两个参考实现，优先考虑：

- RDKit；
- Open Babel；
- CDK，特别是 Java 项目。

记录：

- 引擎名称和精确版本；
- SMARTS 是否被接受；
- 靶分子是否成功预处理；
- 芳香性、手性、氢和 sanitization 选项；
- existence；
- embeddings 或稳定原子集合；
- 匹配数量和去重模式；
- 异常、警告和超时。

差分结果分类：

- unanimous：参考实现及人工契约一致；
- project_mismatch：参考实现一致而项目不同；
- oracle_disagreement：参考实现互相不一致；
- extension_only：仅特定方言接受；
- preprocessing_disagreement：差异来自靶分子预处理；
- enumeration_disagreement：存在性一致但枚举/去重不同；
- unresolved：证据不足。

只有在芳香性、手性、显式氢、sanitization 和去重选项足够对齐后，才把结果标为 project_mismatch。参考实现的多数票不是规范。

差分工具不应成为普通离线 CI 的硬依赖。允许开发环境生成并人工审核小型 golden corpus，但必须：

- 固定参考版本和选项；
- 保存 provenance；
- 更新使用显式命令；
- 更新前显示差异；
- 禁止测试运行时静默覆盖 golden 数据；
- core CI 可完全离线运行。

## 13. 变形测试

实现不依赖外部引擎的 metamorphic tests，只采用能够证明的关系：

- 相同输入和选项重复执行结果一致；
- 靶分子原子编号置换后，映射回稳定身份的结果一致；
- 合法的等价 SMILES 重排后，匹配存在性一致；
- OR 分支交换顺序后存在性一致；
- 对确定可交换的 AND 条件交换顺序后结果一致；
- 在目标方言允许且语义明确的上下文中，双重否定保持结果；
- 若有查询序列化，parse -> serialize -> parse 后语义一致；
- 给靶分子增加完全无关的 disconnected component 时，普通局部查询的已有匹配不消失；
- batch search 的结果等于逐分子匹配结果的集合。

每条变形性质都要写明适用前提。芳香/Kekulé 转换、手性重排、氢显隐式转换和互变异构不能无条件视为等价。

## 14. 生成式测试与 fuzz

根据项目语言选用现有或合适工具，例如 property-based 框架、libFuzzer、AFL 或项目已有 fuzz 基础设施。

建立三类输入：

### 14.1 按语法生成的合法查询

从受限 AST 生成原子、键、分支、环、逻辑和递归查询。控制：

- 最大深度；
- 最大查询原子数；
- 最大分支数；
- 最大环数；
- 最大递归层数；
- 最大逻辑表达式长度。

合法生成器必须按 grammar 构造，不能只是随机字符拼接。

### 14.2 从合法种子产生的非法变异

执行可证明破坏语法的单点变异，例如：

- 删除闭合方括号或递归括号；
- 删除逻辑操作数；
- 产生悬空键；
- 截断环号；
- 破坏 atom map；
- 插入不允许的控制字符。

只有能够证明非法的变异才断言必须拒绝。其余变异只能作为 robustness_only。

### 14.3 任意字节和查询-分子对

任意字节测试只断言：

- 不崩溃；
- 不越界；
- 不挂起；
- 不发生资源失控；
- 相同输入行为确定。

对合法 SMARTS 与合法小分子进行配对生成，检查 matcher、结果枚举和参考实现差分。

所有崩溃、超时、非确定性和高置信度差分都应：

1. 自动保存 seed；
2. 尽可能最小化；
3. 转换为稳定 regression case；
4. 记录运行版本和选项。

对于 C/C++ 等内存不安全实现，在可行时增加 ASan、UBSan 或等价检查。

## 15. 性能、复杂度和拒绝服务风险

benchmark 至少分开测量：

- 冷启动解析；
- 重复查询编译；
- 单分子首次匹配；
- 缓存后匹配；
- 全部 embeddings 枚举；
- 批量线性扫描；
- fingerprint/index 搜索；
- 并发吞吐。

压力输入至少覆盖：

- 长链和大量重复原子；
- 深层分支；
- 大量环闭合；
- 多层递归；
- 大量 AND/OR；
- 高度对称靶分子；
- 会产生大量 embeddings 的查询；
- 几乎匹配但最终失败的查询；
- 超长非法输入；
- 多线程同时编译或搜索相同查询。

要求：

- 单案例有超时和内存保护；
- heavy 测试与普通 PR CI 分离；
- 固定随机 seed；
- 记录输入规模、耗时分布和峰值内存；
- 优先检查随规模增长的趋势，而不是依赖某台机器的严格毫秒阈值；
- 对基准波动使用预热、多次采样和中位数或分位数；
- 明显指数级退化、栈溢出和无限递归必须报告为安全风险。

不得通过过早截断而让 benchmark “变快”却不报告 truncated。

## 16. 真实规则与大规模分子集

在许可证和环境允许时，可导入：

- Daylight 文档中的可合法使用示例；
- RDKit SMARTS 测试及 FilterCatalog；
- Open Babel SMARTS 测试；
- CDK SMARTS 测试；
- PAINS；
- Brenk structural alerts；
- ChEMBL structural alerts；
- 项目已经合法使用的官能团和结构警报规则。

要求：

- 不将第三方全部行为标成 Daylight 核心；
- 记录来源、版本、许可证和转换脚本；
- 工具专属语法进入 extension 或 disputed；
- 大数据集提供下载/导入脚本、校验和和缓存说明；
- 仓库只提交许可证允许的小型确定性子集；
- 网络不可用时 smoke 和 core 仍可运行。

大规模测试不要默认计算完整规则 × 分子笛卡尔积。使用：

- 分层抽样；
- 固定 seed；
- 阳性富集；
- near-negative；
- 按规则特征和分子规模分桶；
- 可重复分片；
- 用户可配置的样本和时间预算。

对于规则库，至少保存每条规则的一个已确认阳性和若干近似阴性；不能只用随机分子得到大量“无匹配”而宣称覆盖充分。

## 17. 覆盖率和质量门槛

同时报告两类覆盖：

### 17.1 功能覆盖

feature_matrix.yaml 应逐项列出：

- 语法特性；
- 方言；
- 正例数；
- 确定反例数；
- 语义案例数；
- near-negative 数；
- 差分案例数；
- fuzz seed 数；
- 是否有 regression。

功能覆盖比单纯代码行覆盖更重要。

### 17.2 代码覆盖

报告 parser、query compiler、matcher、enumeration 和 search/index 的行覆盖与分支覆盖，并列出未覆盖的重要分支。

建议初始目标：

- 每个核心语法功能均有正例、反例和语义验证；
- 不少于 300 个有意义的合法解析案例；
- 不少于 150 个确定非法案例；
- 不少于 300 个 SMARTS-分子语义配对；
- 每个历史缺陷有独立 regression；
- parser 核心模块尽量达到 90% 行覆盖和 85% 分支覆盖。

这些是质量方向，不是凑数指标。如果项目规模较小或 API 受限，应说明缺口；不得制造大量等价重复案例。

在工具可用时，对 parser 的关键条件增加 mutation testing。若大量明显错误的 parser mutation 仍能通过，说明测试虽然数量多但区分能力不足。

## 18. CI 分级

建立至少三个 profile：

### smoke

- 最关键的合法、非法和语义案例；
- 无网络；
- 无外部参考引擎；
- 每次提交运行；
- 失败直接阻断。

### core

- 完整确定性核心语料；
- regression；
- corpus schema 校验；
- 适配器测试；
- 主要变形测试；
- 覆盖率；
- 无网络且结果稳定；
- PR 或主分支 CI 运行。

### heavy

- 多引擎差分；
- 大型规则和分子集；
- 长时 fuzz；
- 性能、内存和并发；
- sanitizer；
- 定时或手动运行。

所有命令必须使用非零退出码表示强制测试失败。audit/report 模式可以继续收集全部问题，但不能冒充 strict 模式成功。

## 19. 已知失败管理

不要使用无理由 skip、catch-all 或宽泛 xfail。

如果当前实现存在已知不合规行为：

- 保留失败案例；
- 缩减为最小复现；
- 标明实际与期望；
- 引用规范、人工推导或参考实现证据；
- 分类为 bug、unsupported、dialect disagreement 或 preprocessing disagreement；
- 如必须建立 known_failures，每条包含唯一 ID、原因、负责人或追踪链接、加入日期和退出条件；
- strict 模式可以让 known failure 也失败；
- 新失败不得自动加入白名单；
- golden 数据不得根据当前项目输出自动改写。

不得用“这是当前行为”作为唯一理由，把明显缺陷固定成规范。

## 20. 实施顺序

按以下顺序执行，避免过早堆积低可信案例：

1. 仓库发现：读取约束、定位 API、运行已有测试。
2. 契约冻结：编写 SMARTS_CONFORMANCE.md。
3. 适配层：实现 parse、target、match、search 统一接口并自测。
4. 语料 schema：实现 manifest 和 corpus validator。
5. 最小核心集：先覆盖关键语法和明确语义。
6. 结果枚举：解决稳定原子身份、自动同构和 uniquify。
7. 批量 search：与逐分子 baseline 对比。
8. 扩展核心集：补齐 feature matrix。
9. 差分验证：对齐参考工具选项，审核分歧。
10. 变形、生成式和 fuzz。
11. benchmark、并发和资源风险。
12. CI 接入和最终报告。

每完成一层都先运行并修正测试基础设施问题，再进入下一层。不要在适配器和预处理契约尚未确认时批量生成 golden 结果。

## 21. 最终必须运行的命令

根据项目实际命令替换以下逻辑任务，并在报告中给出可直接执行的真实命令：

1. 项目原有测试。
2. SMARTS smoke。
3. SMARTS core strict。
4. corpus validator。
5. SMARTS 模块覆盖率。
6. 短时、固定 seed 的 fuzz/robustness。
7. 合理规模 benchmark。
8. 可用时运行多引擎 differential。
9. 可用时运行 batch/index 与 brute-force baseline 对比。

如果某一步因缺少可选工具不能运行，应说明准确原因、安装方式和剩余风险；不得将其写成已通过。

## 22. 最终报告格式

最终回复和 smarts_conformance_report.md 必须包含：

- 项目最终采用的 SMARTS 方言与预处理契约；
- 新增和修改的文件；
- 可复制的运行命令；
- smoke/core/heavy 的测试数量和结果；
- 功能覆盖矩阵摘要；
- 代码覆盖率；
- 参考引擎及版本；
- 解析问题；
- 查询图编译问题；
- 匹配语义问题；
- 枚举/去重问题；
- 批量搜索或索引问题；
- 参考实现分歧；
- 崩溃、超时、非确定性和性能风险；
- 当前 unsupported 功能；
- 尚未覆盖的部分及原因；
- 如何增加 regression；
- 如何审核更新 golden；
- 是否修改过生产代码；若未获授权，应明确为否。

每个独立问题至少列出：

    Issue ID:
    Classification:
    SMARTS:
    Target SMILES:
    Options:
    Expected:
    Actual:
    Failure phase:
    Evidence:
    Reproducibility:
    Suspected module:

不要只报告“若干测试失败”。

## 23. 完成判据

只有同时满足以下条件，任务才算完成：

- 测试代码已写入仓库，而非只提供建议；
- core 测试可离线、确定性运行；
- 语法和语义没有混为一谈；
- unsupported 和 invalid 没有混为一谈；
- matcher 和 search/index 已分别验证；
- 结果枚举已处理自动同构及稳定原子身份；
- 外部语料有 provenance 和许可证记录；
- 差分结果记录参考版本与选项；
- 至少执行过一次实际测试和短时 robustness 测试；
- 所有失败均有可复现证据；
- 最终报告如实说明通过项、失败项和未执行项。

现在开始检查当前仓库并实施。不要停留在设计或伪代码阶段。
