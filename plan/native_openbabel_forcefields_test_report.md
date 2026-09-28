# Open Babel 原生力场后端测试报告

## 1. 总结

截至 2026-09-28，原生后端通过单元、集成、标准覆盖率入口以及 Python 3.9/Open Babel 3.1 和 Python 3.11/Open Babel 3.2 的清洁构建测试。187 个 Eu–萃取剂标准基准与 `obWrappers.old` 基线具有完全一致的逐案例状态：175 个通过、3 个质量失败、9 个 CBond 失败；没有新增失败或新增通过。端到端墙钟时间减少 2.911 s（2.10%），力场配对累计时间减少 37.362 s。

## 2. 常规测试

| 测试 | 环境/命令 | 结果 |
|---|---|---|
| obWrappers 原生专项 | Python 3.11，`pytest tests/test_cheminfo/obWrappers` | 38 passed |
| forcefields/API/集成专项 | Python 3.11，相关 forcefield 与 integration targets | 108 passed，1 warning |
| 标准 coverage 测试入口 | `PATH=/tmp/hotpot-native-test/bin:$PATH ./tests/run_coverage.sh` | 1256 passed，2 warnings，49 subtests passed；62.21 s |
| Python 3.9 原生清洁构建 | Python 3.9.25 + Open Babel 3.1.0 | 强制编译成功；38 passed |
| Python 3.11 原生清洁构建 | Python 3.11.16 + Open Babel 3.2.1 | 强制编译成功；38 passed |
| P(V) 病理结构 | 两个上述环境 | 构筑成功；UFF 4×5 步能量有限；P 规则命中 |
| ORCA 环境隔离 | 清除 ORCA 引入的 `LD_LIBRARY_PATH` 后重测 | 通过 |

标准 coverage 入口已包含 `tests/test_cheminfo/obWrappers`。当前全仓覆盖率报告为 32%；该数值受大量未纳入本轮范围的历史模块影响，不能单独代表原生后端覆盖质量。证据文件：

- `tests/coverage/junit.xml`
- `tests/coverage/coverage.xml`
- `tests/coverage/coverage.txt`
- `tests/coverage/html/index.html`

## 3. 重点契约验证

测试覆盖：

- typed buffer 的 dtype、shape、连续性、原子/键顺序和晶胞传输；
- Python `Molecule` 到 C++ `MoleculeData`、再到 `OBMol` 的属性保持；
- 芳香原子/键与形式电荷传输；
- 不可无损表达的 dative bond 明确拒绝；
- P(V) 构筑规则、线性扭转奇点规则及其可审计证据；
- 规则注册顺序、阶段过滤和 inspect/build/optimize 一致性；
- steepest-descent 单段优化及 conjugate/steepest 完整优化；
- increasing-VDW、扰动、停止窗口、best-frame 选择、帧保留与有界标量历史；
- setup 失败、未知能量单位、损坏帧与非有限数值的结构化报告；
- 原生结果到 `ForceFieldRunReport` 和 `ForceFieldTrajectory` 的适配；
- 生产路径不再依赖 Python/SWIG `OBMol` 数值循环。

## 4. Wheel 与安装验证

已完成 clean wheel 构建：

```text
/tmp/hotpot-wheel-check.uMNoEM/dist/
└── hotpot_zzy-0.5.3.0-cp311-cp311-linux_x86_64.whl
```

wheel 内容检查通过：包含 `_ob_native`、类型 stub、Python facade 和原生源文件；不包含已删除的 `_ob_rules` 或 `snapshot.py`。

该 wheel 已在 `/tmp/hotpot-wheel-check.uMNoEM/venv` 中完成隔离安装和运行验收：

- `hotpot` 与 `_ob_native` 均从虚拟环境 `site-packages` 加载，无源码树遮蔽；
- 清除 `PYTHONPATH` 和 `LD_LIBRARY_PATH` 后仍可正常加载；
- 编译和运行时 Open Babel 均为 3.2.1，C++ ABI 为 0；
- `libopenbabel.so.8` 正确解析到隔离环境的 `openbabel/lib`；
- CCO 原生构筑成功且坐标有限；UFF 优化在 2 个 epoch 后收敛，最优能量为 `6.873761168643569e-10 kJ/mol`；
- 安装 wheel 声明的依赖后，`pip check` 输出 `No broken requirements found.`。

测试环境最初使用的 `tests/requirements-inference.txt` 不含项目元数据声明的 `openpyxl`；以 `--no-deps` 安装 wheel 时需额外补装它。正常依赖解析安装 wheel 不存在这一缺项。

## 5. 187 个 Eu–萃取剂标准基准

### 5.1 输入与运行

- 输入：`molecules/extractant/extractants.smi`
- 样本：187
- 并行 worker：16
- Python：3.11.16
- Open Babel：3.2.1
- ONNX Runtime：1.30.0，`CPUExecutionProvider`
- 金属：Eu
- CBond threshold：-0.125
- seed：20260921
- 轨迹起点：`ligand_build`
- 每个主优化：100 epochs × 100 steps/epoch

执行命令：

```bash
/tmp/hotpot-native-test/bin/python \
  movie/extractants_eu_readme_full_16c_20260928/run_validation.py \
  --input molecules/extractant/extractants.smi \
  --output movie/extractants_eu_native_cpp_16c_20260928 \
  --workers 16
```

### 5.2 新后端结果

| 指标 | 结果 |
|---|---:|
| 总样本 | 187 |
| CBond 完成 / 力场完成 | 178 / 178 |
| 质量通过 | 175 |
| CBond 失败 | 9 |
| 质量失败 | 3 |
| 全样本质量通过率 | 93.5829% |
| CBond 成功样本中的质量通过率 | 98.3146% |
| 力场报告收敛 | 159 |
| 质量通过且收敛 | 158 |
| 完整轨迹 archive | 178 |
| 主轨迹总帧数 | 8484 |
| 主轨迹帧数中位数 / 最大值 | 39 / 129 |
| 配体构筑分支 / 分支帧数 | 186 / 1480 |
| 端到端墙钟时间 | 135.842 s |
| 全案例累计时间 | 2062.645 s |
| 单案例中位数 / 最大值 | 10.155 s / 69.633 s |

### 5.3 与 `obWrappers.old` 基线对比

基线输出：`movie/extractants_eu_obwrappers_16c_20260928`。新后端输出：`movie/extractants_eu_native_cpp_16c_20260928`。两次运行设置完全一致，共享全部 187 个案例。

| 指标 | `obWrappers.old` | 原生 C++ 后端 | 变化 |
|---|---:|---:|---:|
| passed | 175 | 175 | 0 |
| failed_quality | 3 | 3 | 0 |
| failed_cbond | 9 | 9 | 0 |
| forcefield_converged | 157 | 159 | +2 |
| 墙钟时间 | 138.753 s | 135.842 s | -2.911 s（-2.10%，1.021×） |
| 累计案例时间 | 2100.904 s | 2062.645 s | -38.259 s（-1.82%，1.019×） |
| FF 累计时间 | 2066.643 s | 2029.281 s | -37.362 s |
| FF 中位时间 | 10.358 s | 10.167 s | -0.192 s |

逐案例状态转换完全一致：175 个 `passed -> passed`、9 个 `failed_cbond -> failed_cbond`、3 个 `failed_quality -> failed_quality`。178 个完成力场的案例中，139 个更快、39 个更慢；配对 FF 时间下降中位数为 0.08994 s。此次迁移的主要收益是执行结构与边界简化。

2.10% 的墙钟下降和 1.82% 的累计案例时间下降来自同机、同参数的一次成对全量运行。该结果证明新后端没有出现整体性能回退，并给出了约 2% 的观测加速；但单次运行不足以支持“稳定加速”的统计结论。若要发布稳定性能声明，应进行多次交错重复运行，报告均值、中位数、离散程度及置信区间。

### 5.4 失败案例

9 个 CBond 失败案例为 22、134、136、139、141、182、185、186、187；它们在基线和新后端中相同，且未进入力场阶段。

3 个质量失败案例为：

| 案例 | 终止 | 主要证据 |
|---|---|---|
| 54 | converged | 原子 8–54 距离 1.144 Å，低于 1.678 Å 阈值；另有数学不确定的 bond–ring relation warning |
| 61 | budget_exhausted | 两条 Eu–N 键约 1.507/1.501 Å，低于 1.749 Å 阈值，键长比约 0.560/0.558 |
| 109 | budget_exhausted | Eu–N 塌缩至 0.351 Å；并伴随过近原子、短键和键长比失败 |

这些失败的种类、案例和事件计数均与基线一致。失败结构仍以 `last_finite_failure_frame` 保存，便于后续定位，不被静默丢弃。

## 6. 产物与完整性

根目录：`movie/extractants_eu_native_cpp_16c_20260928`

关键产物：

- `summary.json`：汇总指标和环境；
- `results.csv`：187 个逐案例结果；
- `integrity.json`：产物完整性校验；
- `comparison_to_obwrappers_old.json` / `.md`：与基线的逐项比较；
- `optimized_all.sdf`、`optimized_passed.sdf`：集中结构输出；
- `final.png`：178 个完成构筑案例的总览图；
- `cases/<index>/report.json`：逐案例证据；
- `cases/<index>/trajectory/main/coordinates.npz`：主流程坐标轨迹；
- `cases/<index>/trajectory/ligand_build_attempts/<attempt>/coordinates.npz`：配体构筑分支轨迹；
- `cases/<index>/optimized.mol2`、`optimized.sdf`、`final.png`：末态结构及图片。

`integrity.json` 的结果：187/187 报告存在；178 个 CBond 成功案例均有轨迹 archive 和优化结构；无缺失或意外案例；汇总状态计数与逐案例报告一致。

根目录 `final.png` 已人工目视打开：178 个完成构筑/优化的结构图和 9 个 CBond 失败占位均可见，图像数量与状态汇总一致。

## 7. 已知限制与发布前检查

1. Python 3.9/3.11 已做清洁编译和专项测试；3.10、3.12、3.13、3.14 的完整兼容矩阵尚未在本报告中声明为已验证。
2. Open Babel 3.2 的低层、同进程重复 seed 不能重置其函数静态 RNG；受支持的高层可复现路径通过全新 `spawn` 子进程在首次 Open Babel 初始化前设置 seed，相关真实流程测试已通过。
3. Python 3.11 wheel 已完成隔离安装、rpath 和最小公开构筑/优化闭环；尚未在已安装 wheel 上重复运行完整测试套件，也尚未验证其他 Python 版本的 wheel 产物。
4. 187 基准显示化学质量没有退化，也没有修复既有 3 个质量失败；案例 61 和 109 仍需独立处理金属–供体键过短/塌缩问题。
