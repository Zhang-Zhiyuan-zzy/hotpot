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
