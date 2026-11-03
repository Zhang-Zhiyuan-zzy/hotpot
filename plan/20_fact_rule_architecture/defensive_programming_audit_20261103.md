# 防御性代码只读审查（2026-11-03）

按 Owner 要求审查“没有现实报错依据而提前添加的防卫”。结论是：确有分散在多个模块的冗余检查和默认替换，但不能把异常处理、输入校验或有限性检查整体判成过度防护。本次确认 **9 组第一类候选、10 组第二类待核对行为**；不据语法命中总量推断冗余比例。

本报告是当前代码的审查清单，不修改业务行为、不修订 implementation plan，不建立新的阶段前置条件。此前提出的 **MCA 推理结果 NaN/Inf 检查尚未加入代码，建议已撤回**，不计入现存问题清单。

## 分类口径与范围

- **第一类：当前受控调用链已保证条件，正常执行几乎不会触发。** 结论限定于表内的构造方式和调用者；私有函数被直接误用、注册表被任意修改、绕过锁修改状态、猴子补丁及未来改变契约不在该保证内。它们是后续所属业务改动中的简化候选，无需再设计一个替代防卫层。
- **第二类：可能发生，留待全部业务实施完成后的统一业务校验。** 不为这些理论可能性提前增设检查、重试或兜底；结合实际业务执行、必要的已有失败记录及明确的预期结果，确认真实错误后才补最小必要处理。现存行为有些已经是默认值约定，尚不能断言错误；列入本类不表示现在全部删除。
- **另列必要契约和已有实际失败。** 原子映射、信息归属、事务回滚、已批准化学规则、真实外部输入与协议约束，不因本次没有找到历史故障就自动成为冗余。错误被兜底吞掉时，“没见过报错”尤其不能证明没有发生问题。

扫描使用 Python 标准库 AST，455 个生产 Python 文件全部解析成功；同时索引了宽泛异常捕获，并人工追踪核心表示、规则、MCA/CBond、FF/xTB、pipeline、转换和 I/O 的相关调用链。另抽查 geometry native 数值分支和通用 ML/数据接口。**这是语法层面全量发现、关键路径人工核查，不是每个分支、全部 native 实现或全部历史记录的穷尽证明。** 未导入产品、运行模型、测试、构建或基准。

## 第一类：正常受控执行几乎不触发的检查

| ID | 代码位置 | 当前检查 | 调用链依据与边界 |
|---|---|---|---|
| A1 | `hotpot/cheminfo/AImodels/mca/featurizer.py:59–64` | 每次 `_single_embedding` 扫描是否超过 embedding size。 | `mol_to_unimolv2` 的三个调用使用固定分类索引。参与 16 宽度编码的原子/键类别均小于 16；pair 输入来自已编码的第一列，远小于 128。当前构造不产生超界值。这里只涉及重复范围扫描，**不改变训练特征、类别或未知类别映射**。 |
| A2 | `hotpot/cheminfo/AImodels/cbond/deploy/cbond_infer_model.py:198` | `assert rings_node_nums.shape[0] >= 0`。 | Tensor 的维度长度不可能为负；不存在该维度会先在取值时自然报错。属于恒真断言，且位于部署/导出模型源码，不能声称它拖慢当前 ONNX 推理。 |
| A3 | `hotpot/cheminfo/forcefields/auto.py:238–239` | 接受 native attempt 时再次检查 `report is None`。 | 唯一定位到的生产调用在 `:476` 的 `status is ACCEPTED` 分支；构造者用同一个 report 生成 accepted 状态。`report=None` 的正常失败分支均不为 ACCEPTED，attempt 为 frozen dataclass。仅此私有后置重复检查是候选，实际 build/优化失败处理保留。 |
| A4 | `hotpot/cheminfo/core.py:3613` | `_bond_order_setter` 断言 `key == 'bond_order'`。 | 正常属性派发在 `core.py:2430` 已处于同一个等值分支，注册键也是该字面量。不是新的业务事实验证；后续事实提交、映射与事务检查不在简化范围。 |
| A5 | `hotpot/cheminfo/rules/mathematics/internal_coordinates.py:138` | 断言内部生成的 Z-matrix 行宽为 4。 | 唯一定位到的生产调用枚举 `zmat_idx`；该矩阵在 `:62` 明确分配为 `(node_count, 4)`。此结论不涵盖零长度向量、数值退化或外部坐标合法性。 |
| A6 | `hotpot/cheminfo/structure/_identity.py:268–270` | `_check_source` 再次核对依赖域名称是否在 `_DOMAINS`。 | 当前 source 由 `_capture_source` 构造，已在 `:250–252` 验证域名称，再保存进不可变记录。候选严格限定为第二次名称检查；`:263` owner 与 `:274` 内容 currentness 检查仍然必要。 |
| A7 | `hotpot/cheminfo/core.py:4493–4496` | 同一属性第二次 `getattr` 又包一层 `AttributeError` 转译。 | 进入该分支意味着第一次相同 `getattr` 已成功，只是随后索引发生 TypeError；两次操作在同一锁内，普通 NumPy/None/标量字段不会在中间消失。保留外层对可选/标量值的既有处理语义。 |
| A8 | `hotpot/cheminfo/operations/io.py:174,179,186` | 每次执行 writer 插件前断言 Callable。 | 正常装饰器注册存入的是函数，直接调用本来就会对非 callable 自然报 TypeError。每次写文件重复检查无额外化学价值；若将来有实际误注册问题，应针对注册边界处理，而非每次重查。 |
| A9 | `hotpot/cheminfo/geometry/_native/primitives.cpp:93–99` | 归一化向量叉积范数再次检查非有限。 | `determine_line_relation` 已验证输入有限并处理无法归一化的方向；当前 `normalized` / `scale_safe_norm` 路径下，常规浮点运算中的两个归一化方向不会产生非有限的叉积范数。只判定这一次内部重复检查；外部坐标有限性、退化几何处理均不据此删除。不声称该分支有显著性能代价。 |

以上无需为“将来代码可能写错”继续添加防卫。没有运行性能测量，不能把这些代码存在直接解释成主要耗时来源。

## 第二类：可能发生，业务收尾统一核对，不提前扩展防卫

| ID | 代码位置 | 当前行为与可能触发 | 后续业务校验应确认什么 |
|---|---|---|---|
| B1 | `hotpot/plugins/xtb/adapter.py:194–209,257` | 对整个 xTB JSON 递归检查有限性，包括 Hotpot 根本不消费的附属字段；随后还单独检查能量等实际结果。某个无关字段的溢出数值可能使完整结果被拒绝。 | 当前 parser 契约确实要求完整有限，因此不能直接删。用实际 backend 输出核对拒绝范围是否应限于消费字段；保留实际能量、坐标、梯度、原子映射的既定要求。本次未找到“无关字段导致拒绝”的实际样本。 |
| B2 | `hotpot/cheminfo/forcefields/stage.py:439–441`；`hotpot/plugins/xtb/stage.py:389–391`，及对应 CLI serializer | 报告中所有非有限标量统一转成 `None` / JSON null。可能把“未计算”的约定哨兵和“计算产生无效值”合并。 | 有些未评估轨迹帧本来就使用 NaN，转 null 可以是合理 JSON 表达。统一核对业务报告是否还能区分不可用与计算失败，不把所有哨兵改成报错，也不凭理论可能性新增状态框架。本次未证明主结果已被掩盖。 |
| B3 | `hotpot/cheminfo/core.py:4471–4479` | `Conformers.__getitem__` 遇到 IndexError/TypeError 就跳过该字段，可能在错误索引或可选字段错位时返回不完整字典。 | 在 conformer 生命周期业务核验中区分合法缺失字段与错误选择；若实际业务失败，再决定最小错误传播，不提前加遍布各层的长度/索引检查。 |
| B4 | `hotpot/cheminfo/adapters/io/formats/g16.py:120–122` | 只有最终帧 Mulliken charge 时，其他坐标帧的 charge 被填成物理数值 0。 | 区分“缺测”与“显式零电荷”。已有构造测试期待这个行为，但测试并不证明科学含义正确；需要实际 log/frame 业务结果决定。属于现存默认替换，**不是建议添加 NaN 防卫**。 |
| B5 | `hotpot/cheminfo/adapters/io/formats/g16.py:94–108` | cclib 的 RuntimeError/AttributeError 先改成 `data=None`，再报笼统 IOError；不同解析失败原因被合并。 | 保留外部解析失败这一事实；用具体失败文件判断是否需要保留原始 cause，而非现在设计完整的异常分类和重试。没有伪造成功分子。 |
| B6 | `hotpot/cheminfo/operations/thermo.py:306–329` | 构造 Chemical 的任意异常变为整行 None，属性访问任意异常变为单格 None。 | 缺少物性数据很正常，None 也有文档约定；需要区分真实缺测与第三方/程序错误。独立 thermo 集成优先级较低，允许按业务收尾统一检查，不扩展更多 fallback。 |
| B7 | `hotpot/ds/public/search.py:263–275`；`hotpot/ds/public/_generation.py:661–668` | SMARTS 搜索的任意异常变成“不匹配”；结构解析的任意异常变成“invalid”。 | 统一核对错误引擎/解析器故障是否被误归为真实化学不匹配/输入无效。允许缺失结构和局部失败，不应仅凭 checker 行为判定业务事实；无实际样本前不新增防卫体系。 |
| B8 | `hotpot/plugins/ml/wf.py:458,1454` | `nan_to_num(corrcoef(...))` 把未定义相关系数变为数字。常量特征就可触发，不是罕见神经网络损坏。 | 明确这是被接受的特征筛选约定还是掩盖未定义观察；不能仅因输出变有限就判正确。该通用 ML 路径不是 MCA 推理结果检查。 |
| B9 | `hotpot/plugins/opti/ev/core.py:101–106,138` | fitness 函数任意异常改成 `-8888888.`；超时改成另一个有限极小数，可能进入普通排序。 | 核对失败能否仍被识别及是否影响优化选择；有限哨兵不等于观测到的 fitness。属于可延后的通用优化模块问题，不据此添加主线工作流框架。 |
| B10 | `hotpot/cheminfo/adapters/visualization/draw.py:244–245,257,271,375–386` | 全 NaN 着色输入退回 ±1 范围，NaN 原子/键或整行被跳过。 | 外部绘图数据确实可能缺失；核对“无观测”是否被误画成普通结果。零值颜色范围需要非退化处理，不与 NaN 分支混为一谈。绘图可延后，不作为主线关卡。 |

除表中明确标为已有构造测试/源代码可触发路径者，本次没有为这些具体风险找到实际业务故障记录；也没有穷尽搜索历史。**不把“没查到记录”写成“从未发生”，不把人工注入失败当作真实故障。** 对已有约定，先验证实际业务含义，不因本清单立刻改 API 或删掉处理。

## 不按“未出错的提前防卫”处理

| 范围 | 结论与依据 |
|---|---|
| MCA 输出 NaN/Inf | 只是已撤回的新增建议。当前 `mca/runtime.py:102–113`、`mca/pipeline.py:63–92`、`calculator/mca_inference.py:20–38` 没有这个通用扫描。不要虚构一个现存检查再删除。 |
| 训练特征的 sentinel / clamp | `mca/featurizer.py:70–87` 的断开图距离 510、范围截断及 `_safe_index` 的分类桶是模型输入编码。断开图正常存在，不能当作出错兜底删除；FF 对 arccos 输入的数学截断、采样分布截断同样需要区分。 |
| 原子/样本归属、进程协议 | MCA GPU result 的设备、batch、encoding key、数量检查（`gpu_backend.py:744–761`），FF worker 数据/退出处理和 xTB 元素/原子数映射，防止把看似正常的结果贴给错误对象。它们与预测数值是否 NaN 是两个问题。 |
| 化学和数值实际边界 | FF/xTB 实际计算的有限能量、合法电荷/电子数/自旋、输入或生成几何、具名 stereo 规则，不是随意添加的可信度标签。外部坐标可以直接包含非法值；MCA 需从 SMILES 生成构象也是真实路径。保留现有业务约束，不扩大到所有内部中间值。 |
| Open Babel partial charge | `operations/conversion.py:178–196` 包括模型可用性、计算状态、原子数和有限值要求。外部 charge model 随模型/输入变化，不能类比受控 MCA 输出直接断言非有限值几乎不可能。有限扫描的真实必要性可随 charge 业务验证核对；本次未证明它冗余，也不扩展更多检查。 |
| 身份、currentness、回滚、持久化 | owner/UID、view/source currentness、提交前漂移和事务 rollback，以及外部文件覆盖、注册 stage 的 artifact 路径/声明内容校验，保护正常生命周期中的数据完整性。哈希只能用于归属和传输契约，不能代表化学正确。A6 不授权删除 source currentness。 |
| MCA cache 的 OSError 回退 | `mca/all_atom_model.py:530–549` 的持久缓存失败回退临时副本，确实可能丢原始原因。`implementation_log.md:2286,2298` 已记录一次 cache-path assertion 与被吞掉的 **unknown OSError**。这是已有真实症状，单独保留原问题；不能说“未报错”，也不能无证据将该错误写成 ENOSPC。按缓存所属业务处理，不重建预防框架；模型内容核验不因此删除。 |
| Open Babel 日志初始化 | `adapters/io/formats/openbabel.py:6–15` 确有首次使用时的全局 `StopLogging()` 调用。仅凭函数名不能证明其对当前绑定的 stderr/日志缓存的全部作用，本次未运行验证；因此不把“所有诊断都被屏蔽”当作已确认事实或凑入上面两类。 |

## 执行与证据

- 检查身份：`git rev-parse HEAD` → `63043dc766d27087bcc3aba3c75367d403b2f3d0`。生产代码最后业务提交仍为 `6d1ea01f4e8d354318a00633a43b85b92cb68e7b`；工作区已有 goal 激活及其他计划记录变更，不据此宣称整个工作区干净。
- `git diff --name-only -- hotpot tests test`：空输出；本次未修改业务代码或测试。
- 使用 `rg`、有界逐行源文件读取及 Python 标准库 AST/JSON 索引做静态检查，455/455 Python 文件解析成功。语法命中数包含嵌套重复，不代表独立错误或应删 guard 数。
- `git diff --check`：通过。未以任何测试、模型、安装包/native 或性能通过率声明本次静态结果。
- 本次不创建连续的纯文档提交；审查记录随下一次有实际业务进展的连贯提交交付。不改变 Stage7B source HOLD 或任何现有阶段状态。

按需证据目录：`/tmp/hotpot_plan20/short/defensive-audit-20261103/`。其中 `core.md`、`workflows.md`、`adapters.md` 是三个本地只读分区的详细线索；`python_guard_index.json` / `broad_except_index.json` 是静态发现索引。**本报告是主 agent 复核后的分类**：例如将已发生的 cache 问题单列，未把未证实的 charge 冗余或全局日志效果算成定论。临时分区报告不成为新的政策或关卡。
