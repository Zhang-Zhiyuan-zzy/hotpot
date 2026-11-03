# Python3.11 核心工作流实际运行报告（2026-11-03）

结论：本轮实际执行指定的力场、官方 xTB、CBond 集成链及 MCA 模型路径，最终22个测试通过；原始1个集成测试失败保留。完整链的计算并未在该失败处出错，错误来自测试将 GFN-FF 的“不使用 UHF”误写成数值0。两处断言已按既有方法合约纠正。以下仅支持所列案例与条件，不构成全计划、全化学域或安装包验收。

## 来源与执行条件

- 当前源码起点：`6e3fe41e780fba922f757b225568e83125d5e89e`；生产代码包含已提交的多记录 I/O 修复。本轮未改生产代码、模型权重或科学算法。
- 仅在首轮链测试终止后修改 `tests/test_plugin/test_xtb/test_official_pipeline_integration.py` 的两个 UHF 断言；每次执行期间均无源码修改。原测试与失败产物另存，不覆盖原始结果。
- 使用 `/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python3.11`；xTB 为 `/home/zhangzhiyuan/usr/conda3/envs/pyscf/bin/xtb`，实际版本6.7.1、revision `edcfbbe`。Python、NumPy、RDKit、Open Babel、ONNX Runtime 的实际版本见证据记录。
- MCA 使用本机已有 FP16 模型；CBond 使用仓库指针指定的 `hotpot-models` revision `adf525b6a95d6d296ca0d5c68995b1a0ce14d5fe`，两份模型约10.3MB，下载至 `/tmp/hotpot_plan20/medium/cbond-model-cache-20261103`。运行时使用 CPU、显式本地模型路径及离线模式。模型验证用于确认选中产物，不作为预测正确性的判据。
- 数值库环境变量 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 均为1；MCA 自身按现有 CPU 资源策略执行。所有本轮临时执行输出位于 `/tmp/hotpot_plan20`。

## 实际覆盖

| 选择 | 最终结果 | 业务证据及边界 |
|---|---:|---|
| `test_forcefield_integration.py` 中真实锌配合物构建/优化、对象与编号保持、稠合芳香体系几何门控 | 3 passed | 实际 Open Babel 力场计算；检查有限能量、图/对象保持和指定几何门控。 |
| `test_official_integration.py` | 5 passed | 实际 GFN2/GFN-FF 单点、GFN2 优化、氯离子与氢自由基；与相同条件下直接 CLI 的能量/坐标相比较。 |
| `tests/mca/test_api.py` 的指定四项 | 4 passed | 实际 CPU 推理、单个/批量、五类输入的原子/位点 join、批量尺寸5/8/64的数值变化与带电输入默认拒绝。 |
| `test_official_pipeline_integration.py` | 3 passed | 两条 Zn/NCCO 路线及 README Eu/乙酸路线，实际 CBond 模型、FF、可选 GFN-FF、GFN2、落盘与回读。 |
| `test_state_projection.py` | 4 passed | 使用假可执行文件的后端上下文合约检查；不计作真实 xTB 计算。 |
| `test_mca_calculator.py` | 3 passed | 实际模型返回值向 Molecule/Atom 附着、位点副本隔离、空/多位点，以及 Hotpot/SMILES 输入的一致性。 |

22是本轮选择的实际通过数，不是以后验收的固定目标。初次只执行带 GFN-FF 的 Zn 链，测试失败1项；修正后执行全部3个集成场景及4个上下文测试，得到7 passed。其余测试没有因这处断言修改而重复执行。

## 三个真实集成场景

| 场景 | 阶段 | 电荷 | GFN-FF 能量 / Hartree | GFN2 能量 / Hartree | 结果 |
|---|---|---:|---:|---:|---|
| Zn + NCCO | CBond → FF → GFN2 单点 | +2 | 未请求 | -14.26981398 | 所有阶段成功、几何检查通过、最终单记录可回读。 |
| Zn + NCCO | CBond → FF → GFN-FF 优化 → GFN2 单点 | +2 | 1.298971744181723 | -14.27320288 | 所有阶段成功、几何检查通过、GFN-FF 坐标已提交。 |
| Eu + O=C(O)C | CBond → FF → GFN-FF 优化 → GFN2 优化 | +3 | 3.237790501029679 | -13.53210062 | 所有阶段成功、几何检查通过、两次优化坐标已提交。 |

各能量属于各自方法、组成、状态和参考，不能跨方法直接排序为“更正确”。GFN-FF 的后端 UHF 为 `None`，GFN2 在这些案例中为0；后者是执行参数，不证明唯一真实自旋态。几何门控通过及模型选择的配位关系也不证明唯一实验结构。

## 断言修复与保留失败

现有 `XTBBackendContext.consumes_uhf` 对 GFN-FF 为 `False`，投影约定为 `charge_only`；对 GFN2 为 `True`。现有 README 也明确 GFN-FF 不接受 spin 选项。原集成测试却对两个方法都要求 `unpaired_electrons == 0`，与这些合约矛盾。

修复后对具体序列作检查：带 GFN-FF 的链必须为 `[None, 0]`，仅 GFN2 的链必须为 `[0]`。GFN2 的自旋、电荷、能量有限性、收敛、几何、原子顺序、文件输出及回读断言均保留。未修改生产报告以满足错误测试，也未将原失败改记为通过。

CBond 下载与 `verify_artifact` 已完成后，最初的证据记录脚本误假定下载清单含 `artifact` 键，抛出 `KeyError`。改从仓库指针读取该元数据，核对下载清单对应项后写成记录；未重复下载。此错误不计入业务测试通过数，原错误另存。

## 命令与证据入口

[所属证据目录](/tmp/hotpot_plan20/long/stage7a-cp311-core-workflows-20261103/) 中：

- `run-results.json`：六次 pytest 的完整实际 argv、cwd、环境覆盖、选中测试、JUnit 结果/时长、版本、三个场景观察和覆盖局限。
- 六份 XML：`ff-connected.xml`、`official-xtb.xml`、`real-mca.xml`、`official-pipeline-attempt1.xml`、`official-pipeline-attempt2.xml`、`mca-attachment.xml`。
- `attempt1/`：原失败链的完整34份文件及修改前测试；`attempt2/`：最终三个场景的 manifest、阶段报告、SDF 和计算产物。
- `pipeline-test-repair.diff`、`cbond-artifact.json`、`cbond-acquisition-receipt-failure.json`：具体修复及模型取得/记录错误的归因。摘要只用于来源和完整性。

## 仍未解决的业务与交付

- Open Babel 导出丢失未展开 H；推导位点总H与工具包非图H的既有原始字段含义不一致，不能简单加一个 setter 或统一减去图H。涉及现有 Stage7E 的连接消费者迁移与既定公开/模型编码审阅，尚未修复。
- Open Babel 起源的部分同位素/立体事实在导入时缺失，仍由现有 canonical/source transport 工作闭合；上述链通过不能抹去这些表示缺口。
- 仅 CP311 当前源码；没有新增安装包、其他 Python 版本、GPU、干净 native/ABI/SDK、广泛科学语料、性能或长期内存结论。
- 三个集成场景不覆盖所有电荷/金属/配位环境，也未证明全部混合失败批次、恢复和冷持久化路径。MCA 数值是所选模型的预测，已有 golden 是模型特定的回归例，不是实验真值。
- `goal_v2.md` 仍待用户评审，本轮未激活、替换目标或修改主计划。Stage7A 仍为 IN_PROGRESS，后续阶段与整体目标尚未完成。


## 源码目录外的 CP311 安装包补验

当前提交 `97f1f0fca60371a53a90a473e50439430b0a9ab3` 的已跟踪代码通过 `git archive` 放入独立工作目录；未复制用户未跟踪文件。现有构建环境以标准 `build_py` 和 `bdist_wheel --skip-build` 组装当前 Python/资源代码与源码环境实际使用的6个 CP311 native 扩展。得到约6.28MB的 `hotpot_zzy-0.5.4.5-cp311-cp311-linux_x86_64.whl`；新 I/O 代码、6个扩展和5个字体均在包内。扩展摘要只验证复用/传输完整性，不参与化学正确性判断。

wheel 用 `pip install --no-deps --ignore-installed --no-compile` 安装到独立 venv，在源码目录外执行。此 venv 使用 `--system-site-packages` 共享现有运行依赖，未修改原 Conda 环境，也没有取得新的完整依赖解析、干净 native 构建或 manylinux 可移植性结论。构建环境的 setuptools79/pybind11_3.1 与运行环境分开；最初运行环境缺 pybind11、setuptools83超出项目构建范围的观察保留，没有为此升级或改动它。

| 安装边界 | 实际结果 | 覆盖 |
|---|---|---|
| 初始隔离导入 | 134个已加载 Hotpot 模块均位于安装目录 | 当前代码、新 I/O 及6个扩展；OB native 通过现有 accessor 先加载匹配的 Open Babel。 |
| 首次组合 pytest | 73 passed、3 failed；该进程236个 Hotpot 模块均来自安装目录 | 多记录 I/O、转换新鲜性/隔离、Crystal、官方 xTB、实际 MCA 附着等；3个集成链失败原因见下。 |
| 标准 pytest 重跑原3个失败场景 | 3 passed | 两条 Zn 链和 Eu 完整链实际计算、几何、落盘/回读通过；未重跑其余73项。 |
| 现有 installed smoke 的 core/inference/forcefield | 三次均RC0 | 安装资源、入口、实际推理和CLI衔接；格式检查不替代完整H/立体或几何语义验证。 |

首轮错误来自本次临时入口：以 stdin 运行 `pytest.main`，FF 的 spawn 子进程尝试重新读取不存在的 `outside/<stdin>`，导致 pipe EOF。原 XML 和三个失败阶段的文件保留。改回标准 `python -I -B -m pytest`，仅重跑完全相同的3个失败 test ID；没有修改产品、测试断言或包内容。最终覆盖76个不同测试全部通过，不能把这个合并结果改写为“首轮76 passed”。

安装后 Zn 案例的能量与前节所列观察相同。Eu 案例的 GFN-FF 能量为3.237791384957271 Hartree，GFN2为-13.53210061 Hartree，与源码运行分别略有差异。该README路线未显式固定FF seed；既定测试标准是状态/映射正确、实际收敛、能量有限、几何门控及文件完整性，不是与旧能量逐位相等。本轮保留两组值，不据此单独推断差异成因、唯一正确构象或普遍等价。

完整证据位于 `/tmp/hotpot_plan20/long/stage7a-cp311-installed-business-20261103/`：`assembly-inputs.json` 记录当前源码、wheel和复用扩展；`installed-results.json` 记录命令、环境、首轮73/3、精确失败集重跑、三个场景及局限；`installed-origins.json` 与 `pytest-product-origins.json` 是实际导入检查；两个XML、三份smoke日志及 `attempt1/`、`attempt2/` 保留计算/失败产物。wheel、独立源码副本及venv在 `/tmp/hotpot_plan20/medium/20261103-cp311-installed-business/`。

这补齐了当前修复在 CP311 安装边界的指定业务证据，不关闭完整 Stage7A、全版本、native/ABI/SDK、科学、性能或长期内存门；前述 H/同位素/立体运输问题仍未解决。


## Connected H semantics: actual current-source observations

A subsequent diagnostic at documentation HEAD `74a3fc31ff1552f24fa418cd566f8f4ee274cc58` inspected eight explicit fixtures at three points: initial prepared state, first `add_hydrogens()`, and repeated `add_hydrogens()`. Production and test source stayed unchanged. This is 24 observed states, not 24 passing chemical tests. Six RDKit projections raised local H-consistency errors, retained in the evidence. No new normalization, tensor encoding, source owner or Stage7E entry was introduced.

| Origin and represented input | Initial graph H / public H | Graph H after first / repeated materialization | Projection observation |
|---|---|---|---|
| Manual C, omitted H | 0 / 4 | 4 / 4 | Initial RDKit projection preserves 4 bracket H; Open Babel drops omitted H. After expansion RDKit duplicates 4 graph + 4 bracket H (total8). |
| RDKit `C` | 0 / 4 | 4 / 4 | Subsequent RDKit projection rejects stale current implicit H4 versus derived0. |
| RDKit `[H]C`, with graph H retained by parser | 1 / 3 | 3 / 3 | Input had 1 graph + 3 omitted H; materialization reaches only 3 graph H. RDKit projection then rejects requested3 versus derived1. |
| RDKit full graph `[H]C([H])([H])[H]` | 4 / 0 | 4 / 4 | Full-graph control retains total4 H through both materializations and both toolkit projections. |
| RDKit `CCO` | 0 / (3,2,1) | 6 / 6 | Subsequent RDKit projection rejects stale current H at the first carbon. |
| Open Babel `C` | 0 / 4 | 4 / 4 | Initial RDKit projection preserves 4 bracket H; Open Babel drops omitted H. After expansion RDKit duplicates 4 graph + 4 bracket H (total8). |
| Open Babel `[H]C` | 1 / 3 | 3 / 3 | Incomplete graph materialization; RDKit then duplicates 3 graph + 3 bracket H (total6), versus the input total4. |
| Open Babel `CCO` | 0 / (3,2,1) | 6 / 6 | Initial RDKit projection preserves 6 bracket H; Open Babel drops omitted H. After expansion RDKit duplicates 6 graph + 6 bracket H (total12). |

The table compares represented atom/H accounting and named public operations, not a uniquely correct physical protonation state. RDKit `GetNumImplicitHs()==0` alone is not H loss: bracket-explicit and graph H must also be counted. Existing rule-inferred site-total counts and imported omitted-H counts have different scopes. An import-only isotope patch is likewise incomplete: sparse atom isotope storage and the transitional payload feed different consumers, while RDKit rejects unsupported atom-fact source namespaces. Filling either side alone would leave inconsistent observations.

Evidence: `/tmp/hotpot_plan20/long/connected-hydrogen-observations-20261103/execution.json` contains exact argv, cwd, environment, file identities and the six local errors; `observations.json` contains every raw count, presence/source value, graph H, current/source transport record and toolkit output/error. The first run failed only while JSON-serializing an Enum after case execution; its script/log are retained in `attempt1/`. The corrected writer represents enum type/value explicitly, preserves the raw observation pickle, and completed RC0. No product fix or stage acceptance follows from that writer correction.

The next connected decision remains the Stage7E H-slot/source convention and its public/model projection effects. A single setter, producer-code subtraction or sparse/payload mirror is not accepted as a substitute. CP311 core workflows passed earlier remain evidence for their actual cases; they do not prove these untested representation paths correct.


### Isotope origin and projection observations

An additional eight source/fixture pairs used `C`, `[0CH4]`, `[13CH4]` and `[2H][18O][2H]` through both existing toolkit import routes, without H materialization. The diagnostic returned RC0; returned objects do not imply isotope preservation. Atom ordering and elemental identities were retained in these fixtures.

| Origin | Current sparse isotope attributes | Current payload | RDKit projection | Open Babel projection |
|---|---|---|---|---|
| RDKit labelled13C or D/18O | Absent | Label values retained | Label values retained | All isotope API values become0 |
| Open Babel labelled13C or D/18O | Absent | Empty | All isotope API values become0 | All isotope API values become0 |

Both toolkits report isotope0 for both `C` and `[0CH4]`; this observation does not recover source-level explicit-zero versus unspecified declaration presence. A zero value must not be used to fabricate that missing history. These actual results corroborate the distinct sparse/payload consumer gap and rule out claiming end-to-end isotope preservation from an import-only change. No model inference was run; downstream model effects remain a source-traced concern, not measured predictions. Exact command and artifacts are in the same `execution.json`, `isotope-observations.json`, `observe_isotopes.py` and `isotope-run.log`. No public property, canonical storage, model encoding or source fix changed.


## 混合成功/失败记录：真实 xTB CLI 观察

随后使用同一 CP311 当前源码和官方 xTB6.7.1，实际运行一个3记录 SDF 批次：完整图H水、单原子H、完整图H甲烷。统一请求中性、UHF0、GFN2单点，`jobs=2`、`threads=1`、`post-check=off`。水与甲烷的几何只是明确输入；本次没有优化或几何门控通过的结论。

| 原输入索引 | 实际结果 | 保留的业务观察 |
|---|---|---|
| 0：水 | `ok` | 有限能量 -4.80210653 Hartree，原子顺序已核对，单点未提交坐标。 |
| 1：原子H | `error` | `XTBStateResolutionError`，具体诊断 `HARD-013.UHF_PARITY`：该中性单电子输入与UHF0奇偶性不相容。 |
| 2：甲烷 | `ok` | 有限能量 -4.17501064 Hartree，原子顺序已核对，单点未提交坐标。 |

CLI整体返回1，报告仍保留全部3个索引，输出SDF只含水和甲烷且保持原输入顺序。局部状态拒绝没有中断独立样本；没有把返回1改记为业务全成功或删去失败分母。两份成功结果来自真实数值执行，不是假后端。中间记录没有计算报告；观察脚本中原始键名 `no_native_execution_for_rejected_record` 实际只检查报告缺失，不证明没有后端版本/能力探测调用。

精确命令、环境、11个相关生产文件前后身份、输入/输出、三记录报告、stdout/stderr及保留的xTB工作目录见 `/tmp/hotpot_plan20/long/stage7a-cp311-mixed-xtb-20261103/execution.json`。原样归档在 `.plan20-artifacts/20261103/stage7a/mixed-xtb-business-observation/`。这是一组有明确边界的实际观察，不是所有混合输入、方法、几何恢复、安装边界或整个Stage7A的验收。

## H 连接修复的可审阅方案

[待审方案](stages/stage07/preentry-hydrogen/review.md)与[连接消费者矩阵](stages/stage07/preentry-hydrogen/consumer_matrix.md)已保存到项目维护记录，状态为 **PROPOSED / NOT APPROVED / NO STAGE7E ENTRY**。建议保留现有普通非金属推导规则的位点总目标 `T`，把明确的工具包输入规范为 `T=图H+非图H`，各消费者由图关系派生剩余非图H。这样会让部分显式甲烷的公开字段14由3变4、完整图H甲烷由0变4；这是真实公开语义变更，尚未授权实施。

根复核接受问题定位与连接迁移范围作为准备证据，未批准公开H语义、DTO/模型特征含义或历史持久化处理。源历史不得充当当前剩余H的第二真值，不能以关闭RDKit检查器掩盖陈旧运输。模型权重和科学方法保持；具体DTO版本、旧记录的可恢复边界、完整源声明/原子表映射及Stage7E前置所有者冻结仍须在依赖步骤前确定。本轮没有进入Stage7E。

H/同位素原始诊断与失败写入尝试共10份文件、164413字节，已原样归档至 `.plan20-artifacts/20261103/stage7a/connected-hydrogen-observations/`，原来的嵌入路径和失败记录保持。归档摘要仅核对保存完整性，不参与化学合理性判断。


### 持续回归与安装边界

新增 `tests/test_plugin/test_xtb/test_official_cli_integration.py`，通过已有外部执行开关启用真实xTB，分别检查 `jobs=1` 与 `jobs=2` 的上述局部拒绝/独立继续语义。当前源码 CP311：**2 passed、0 failed、0 skipped**；在既有CP311 wheel的隔离安装环境、源码目录外执行相同测试字节：**2 passed、0 failed、0 skipped**。安装进程已加载186个Hotpot模块，全部位于选定安装目录；11个相关生产文件与当前源码一致。源码测试期间492个生产Python/native文件身份保持不变。

测试不锁定精确能量、运行时间或一组“唯一正确”构象。它要求有限能量、合法请求归因、3记录分母、两个成功记录的原顺序/坐标和单点不提交行为；序列化后的能量只与同一次计算报告对应。输入均为完整图H，故不掩盖或解决前述H表示缺陷。安装依然共享既有依赖、复用6个扩展，没有新增干净native构建或全版本结论。

源码证据：`/tmp/hotpot_plan20/long/stage7a-cp311-mixed-xtb-regression-20261103/execution.json`；安装证据：`/tmp/hotpot_plan20/long/stage7a-cp311-mixed-xtb-installed-20261103/execution.json`。两组原始XML、输入/输出、真实计算报告、日志及源身份分别归档于 `.plan20-artifacts/20261103/stage7a/mixed-xtb-regression-source/` 与 `mixed-xtb-regression-installed/`。这些是新增的局部证据，不重写前述22项/76项选择及其原始失败，也不关闭阶段或整体门。
