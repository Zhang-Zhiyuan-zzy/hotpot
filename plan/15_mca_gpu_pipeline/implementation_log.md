# Plan 15 实施日志

## 记录规则

本日志随实施更新，不在完成后倒填。每个阶段记录计划结果、实际工作、验证证据、
意外发现、方案调整及对应提交。

只修改独立 worktree `/tmp/hotpot-mca-gpu-pipeline` 中的
`perf/mca-gpu-pipeline` 分支；主仓库现有未提交内容不属于本阶段。

## 基线

| 项目 | 结果 |
| --- | --- |
| 日期 | 2026-10-20 |
| 工作分支 | `perf/mca-gpu-pipeline` |
| 独立 worktree | `/tmp/hotpot-mca-gpu-pipeline` |
| 起始提交 | `88ccec5` |
| 上一轮 MCA 优化 | `d8138aa` |
| 基线 MCA 测试 | `77 passed in 14.15s` |

## 阶段记录

### 阶段 0 — 固化已批准方案

状态：完成

已完成：

- 从主仓库当前提交建立独立分支与 worktree；
- 明确不引入 profile；直接保留每卡 9/7 worker 默认值；
- 明确 GPU 选择、CPU 数量推导、显存 batch 档位及 OOM 行为；
- 明确删除旧接口与旧 GPU 后端，不设置兼容开关；
- 固化 all-atom、logical atom-row batch、padding、顺序和 `is_Nuc_site` 边界；
- 创建实施方案、实施日志和最终报告入口。
- 执行全部现有 MCA 与 calculator 集成测试，确认改动前基线为 77 项通过。

验证：

- `git status --short --branch` 显示当前分支为 `perf/mca-gpu-pipeline`；
- `git log -1` 显示起点为 `88ccec5`。
- `PYTHONPATH=/tmp/hotpot-onnx-inspect:. python -m pytest -q tests/mca tests/test_cheminfo/test_mca_calculator.py`
- 结果：`77 passed in 14.15s`。

意外发现：

- 新 worktree 不包含 Git 忽略的 CPython native extension，首次收集因此失败；
- 从同一源提交的主 worktree 复制三个 CPython 3.11 构建产物后恢复测试。这些
  `.so` 仅用于本地验证，不纳入提交，也不改变产品代码或测试范围。

方案调整：无。

提交：`8935208 docs(mca): define GPU pipeline implementation plan`。

### 阶段 1 — 资源、显存与 batching 原语

状态：完成

实际工作：

- 新增 `resources.py`，实现 `gpu_count`/`gpu_indices` 二选一、每卡 9/7
  默认 worker、按 affinity 稳定划分 CPU block、单 worker 单 CPU 绑核及不足时
  的结构化错误；
- 新增 `gpu_memory.py`，实现只按总显存的六档 batch、异构 GPU 取最小档、
  `nvidia-smi` 逻辑 index 映射及 `MCAGpuOutOfMemoryError`；
- 新增 `batching.py`，实现可接收乱序完成窗口的 stateful logical atom-row
  planner、跨窗口尾块、`pad_width`、encoding key 和 scatter；
- 显式 batch 限定为 `64/128/256/512/1024/2048`，OOM 只建议下一档，
  不自动重试或回落 CPU。

验证：

- foundations 聚焦测试：`36 passed`；
- 显存阈值逐边界、异构卡、CPU 不足、GPU index 顺序、worker 绑核、跨窗口和
  乱序 planner 均有单元测试；
- `compileall` 与 `git diff --check` 通过。

意外发现：无。

方案调整：无。

### 阶段 2 — 显式 runtime 与流式 CPU backend

状态：完成

实际工作：

- `ModelStore` 只负责 artifact 与 variant，不再解析 device/provider；
- runtime 改为显式 CPU/CUDA 工厂；CPU 使用最多 16 个 intra-op thread、
  inter-op=1、顺序执行，GPU 使用 1/1 thread、顺序执行及显式 `device_id`；
- runtime 强制 all-atom layout，出现 `atom_index` 输入时失败，不走 selected-atom
  旧路径；
- 新增 `MCACpuPredictor`，以同一个 stateful planner 实现真正的有界
  `iter_predict()`，保持旧 flattened atom-row、padding 和 `(mol,pad)` cache；
- CPU predictor 提供幂等 close、context manager 和 non-reentrant 约束。

验证：

- 新旧 CPU ORT session 对同一输入 bitwise equal，`max_abs=0`；
- 固化 batch 5、8、64 的 provider+batch 专属 SHA-256；
- CPU/API/DTO/calculator 聚焦回归：`48 passed`；
- streaming 测试证明 batch=2 时取得首个结果只消费两个单原子输入。

意外发现：同一 provider 的结果也会随 logical batch padding 改变，因此三个
batch 分别保存 golden，不能用 batch=64 的 hash 验收其他 batch。

方案调整：把 CPU `iter_predict()` 从仅迭代一个已 materialize 列表改为 stateful
有界执行；数值 batch 边界保持不变。

### 阶段 3 — DTO 与 spawn-safe P1/P2

状态：完成

实际工作：

- 新增版本化、不可变、可 pickle 的双表示 `MoleculeDTO`；
- Hotpot 图快照保留电荷、芳香性、隐式氢、价态、同位素、坐标、bond kind、
  dative direction 及金属配位语义；RDKit binary 保留立体和构象；
- 新增不导入 runtime/CUDA 的顶层 P1/P2 task、result 和 worker 函数；
- 入口只规范化一次，task/result 都携带全局 molecule index。

验证：

- DTO/worker 聚焦测试：`22 passed`；
- spawn executor 实际 round trip 通过；
- SMILES、Hotpot、RDKit、Pybel、OBMol、MoleculeGraph、iterable adapter 全覆盖；
- 芳香、手性、双键立体、同位素、自由配体、零阶/单键配位、螯合和 metal-star
  的位点语义回归通过。

意外发现：OpenBabel 不能输出部分 dative 图的 canonical SMILES。为严格保持原有
功能边界，没有增加 RDKit SMILES fallback；这类输入继续传播原有 Hotpot
`ValueError`，不把原先失败的输入静默扩展为成功。

方案调整：采用 Hotpot 图快照 + RDKit binary 双表示，而不是假设任一 toolkit
对象有稳定通用 pickle 契约。

### 阶段 4 — 单卡与多卡 GPU pipeline

状态：完成

实际工作：

- CPU pool 在任何 CUDA consumer 前用 `spawn` 建立；
- 建立全局 P1/P2 pool、每卡一个独占 CUDA process/session、bounded request
  queue 和公共 result queue；
- coordinator 以原子数预算形成有界窗口，冻结全局 logical batches；P1/P2 使用
  `apply_async` 增量回收，首个连续 ordinal 所需的 P2 feature 就绪后立即送 GPU，
  不等待整窗 P2 或慢 P1；
- logical batch 跨 GPU round-robin 分发，并按 request、chunk 和 molecule index
  三层重组；
- 同一 `(molecule_index, pad_width)` 在 pending/cache 中只计算一次；
- predictor 异常或生成器提前关闭时清理 pool、process 和 queue；所有等待都有
  deadline，worker/consumer 退出或替换会 fail-fast；
- coordinator、所有 CPU worker 和 GPU consumer 均限制在推导出的 CPU block，
  close 后恢复调用线程的原 affinity。
- 模型 resolve、完整性校验和 all-atom materialize 只在 coordinator 执行一次；
  所有 GPU consumer 复用同一已解析路径，但每卡仍拥有独立 ORT session；
- 多 GPU 关闭使用共享 graceful/terminate/kill deadline，最坏等待不随 GPU 数量
  线性增加。

ai-14 A800 验证（`hp-gpu`，固定 256 分子样本，允许样本中的域外带电分子）：

| 配置 | 原子 | 位点 | MCA SHA-256 | site SHA-256 | 结果 |
| --- | ---: | ---: | --- | --- | --- |
| 单卡 0，batch 2048 | 6,486 | 1,915 | `fa8e55f5…158e0` | `5d6150ef…10b3` | 与固定 reference 完全一致 |
| 双卡 0,1，batch 2048 | 6,486 | 1,915 | `fa8e55f5…158e0` | `5d6150ef…10b3` | 与单卡/reference 完全一致 |

单卡持久池重复运行：

- 自动显存档：A800 80 GiB 正确选择 2048；
- startup：约 5.44 s；
- 稳态本轮：20.51 s，12.48 molecules/s；
- 旧串行实测：151.796 s，1.686 molecules/s；
- 当前安全实现相对旧串行约 7.4x；reference 原型含共同解析约
  13.44 molecules/s，本实现因 DTO 重建、进程边界和严格资源约束略低。

CPU 超限检查：运行中枚举完整进程树，coordinator/resource tracker/GPU consumer
的 allowed CPU 均为 `0-15`，16 个 P1/P2 worker 各只允许一个不同 CPU `0..15`；
未发现第 17 个 CPU 或 ORT affinity 越界。

意外发现：

1. ai-14 的 `hp` 环境当前安装的是 CPU-only ONNX Runtime；真实 GPU 验收必须使用
   已配置 CUDA provider 和 CUDA library activation hook 的 `hp-gpu` 环境。
2. 固定 256 分子样本在位置 164 含带电分子，原 benchmark hash 包含该域外输入；
   验证脚本必须显式 `--allow-charged`，生产默认仍拒绝带电分子。
3. 64 分子窗口会增加同步边界，稳态约 11.15 molecules/s；改为按原子数预算的
   有界窗口后提升至约 12.48 molecules/s，同时不依赖总输入规模。

方案调整：窗口容量从固定分子数改为
`max(4 * gpu_batch_size, 100 * total_cpu_workers)` 个 atom rows 的预算；这保持
内存有界，并减少 P1/P2 barrier 和 GPU 饥饿。

### 阶段 5 — API、CLI、calculator 与文档切换

状态：完成

实际工作：

- 公共面只保留 `MCACpuPredictor`、`MCAGpuPredictor`、`predict_mca_cpu`、
  `predict_mca_gpu` 和要求显式 backend 的 `predict_mca`；
- 删除 `MCAPredictor`、`device=auto/cpu/cuda`、`HOTPOT_MCA_DEVICE`、calculator
  LRU runtime cache 和旧调用约定；
- CLI 使用 `--backend` 严格分离 CPU/GPU 参数，并流式读取、推理和写出；
- calculator 改为显式 `backend` 的 one-shot 调用；批量 GPU 使用者显式持有
  predictor；
- 更新根 README、MCA README/CLI 文档、model pointer 和两份 usage 文档。

验证：

- CLI 表格继续保留 `is_Nuc_site` 列并联合 atom/site 值；
- 旧 `--device` 被 parser 拒绝，包根不存在 `MCAPredictor`；
- MCA README 例程和 CLI `--doc` 聚焦测试：`3 passed`；
- MCA/计算器/README 组合运行中 166 项通过；另 2 项 CBond README 测试因当前
  sandbox 中 CBond artifact cache 为只读且本地无 artifact 而失败，与 MCA 改动无关。

方案调整：无 compatibility facade 或 feature flag。

### 阶段 6 — 完整验证、独立审计与报告

状态：完成

独立审计及修复：

- 生命周期审计复现并修复残留大 payload 时 `Queue.join_thread()` 永久等待、
  consumer `exitcode=0` 无限等待、CPU Pool worker 替换后 affinity ID queue 耗尽、
  CLI 输出文件打开失败泄漏 predictor；
- 架构审计发现整窗口 P2 barrier，改为按 ordinal 的持续 P2→GPU conveyor；
- 单一实现审计后删除死代码/影子状态，CPU/GPU 共用输入识别、P1/P2 纯函数和
  `build_prediction()`；后者是 atom/site MCA 与 `is_Nuc_site` join 的唯一实现；
- 冗余审计后把模型 resolve/verify/materialize 从 coordinator + 每卡一次收敛为
  coordinator 一次，并统一 9/7、batch tiers、conformer seed、window budget 和
  affinity helper 的配置来源；
- 关闭路径改为共享 deadline，并由末轮只读审计确认 temporary/persistent artifact
  生命周期和多进程清理不存在 blocker。

本地最终验证：

- `PYTHONPATH=. python -m pytest -q tests/mca`：`172 passed in 13.47s`；
- calculator + MCA README 聚焦：`5 passed, 12 deselected`；
- command loading、calculator package、import safety、MCA import layout：
  `10 passed`；
- `compileall`、`git diff --check` 通过；
- 全 README 组合中的 2 个失败来自未修改的 CBond artifact 缓存只读；打包元数据
  的 1 个失败来自仓库原有 `requirements.txt` 与 `pyproject.toml` 的
  `cryptography` 不一致，均非本轮回归。

ai-14 最终验收（A800、自动 batch=2048、`hp-gpu`）：

| 配置 | 分子/窗口 | startup(s) | run(s) | mol/s | 联合结果 |
| --- | --- | ---: | ---: | ---: | --- |
| GPU 0 | 256 / 1 | 4.914 | 20.372、20.645 | 12.57、12.40 | 固定三类 hash 一致 |
| GPU 0,1 | 256 / 1 | 5.758 | 14.983 | 17.09 | 固定三类 hash 一致 |
| GPU 0 | 512 / 2 | 5.114 | 38.522 | 13.29 | 跨窗口三类 hash 基线 |
| GPU 0,1 | 512 / 2 | 5.980 | 28.959 | 17.68 | 与单卡跨窗口逐位一致 |
| GPU 0..7 | 256 / 1 | 9.157 | 10.974 | 23.33 | 固定三类 hash 一致 |

固定 256 样本：

- MCA：`fa8e55f55519fac1afca9874f1535de6ee936c89ee88f9bbcda298e31ad158e0`；
- sites：`5d6150efed59cbf96cc83324c4c35c387630e71e25826bad5382378dfe4a10b3`；
- atom/site/`is_Nuc_site` 联合：
  `8fe4ede2dad5e7caf1575432f348f9b10ec4f251eb14c88c41d2fb5b790f87e1`。

跨窗口 512 样本（12,972 atoms，窗口预算 8,192）：

- MCA：`0fac229ef0192f24faef1d8a7c80ba7c9e0eb07114deba8f16b07e7dbb64a71a`；
- sites：`47c05d0da1948740fda2e0571b3c349493633999dba34343bcedc751bdc8aa40`；
- 联合：`d5af2956c4aaf6386b3c1d25d0d002a5694cd5492b66d3fe3aaa91f0a52a5fe3`。

所有远端运行均确认 caller CPU affinity 在 close 后恢复。8 卡配置按批准公式推导
128 个 CPU worker，并在 128 个 affinity-visible CPU 上成功启动；GPU consumer
和 coordinator 共享该 CPU block，不额外暴露 `cpu_limit` 参数。

方案偏差：任意第三方 GPU 进程异常类型不能跨进程安全重建；非 OOM GPU 错误以
`RuntimeError` 保留远端类型名、message 和 traceback。CPU worker 本地异常保持
原类型传播。OOM 仍使用结构化 `MCAGpuOutOfMemoryError`，且不重试、不回退 CPU。

提交：核心实现为 `676bc31 perf(mca): add bounded multi-GPU inference pipeline`；
最终报告位于本日志的收尾文档提交中。

### 阶段 7 — CLI 默认 CPU

状态：完成

根据 2026-10-21 补充批复，CLI 的 `--backend` 从必选项改为默认 `cpu`；只有 GPU
路径需要显式写 `--backend gpu`。Python `predict_mca()` 与 calculator 的 backend
参数继续保持显式，未扩大本次修改范围。同步更新根 README、MCA CLI 文档、usage
文档、实施方案和实施报告，并用无 `--backend` 的真实顶层 CLI 回归覆盖默认行为。

验证：

- CLI 与 README MCA 聚焦测试：`14 passed in 6.12s`；
- 完整 MCA 测试：`172 passed in 13.31s`；
- `git diff --check` 通过。

### 阶段 8 — 双门控滚动准入与 GPU 启动阈值

状态：已完成本地回归与 ai-14 验收。

根据 2026-10-21 补充批复，本阶段不增加硬件或并发资源，实施两项优化：

1. 保留既有 atom-row window budget，但去除“整窗完成才继续读取”的 barrier。
   coordinator 使用 preprocessing credit 与 resident hard cap 双门控：P1 site
   detection 和最后一个 feature batch handoff 都完成后即可复用 preprocessing
   credit；resident credit 直到结果按序输出才归还。9/7 worker、每卡 CUDA
   session、request queue 和 `2 × GPU 数` 的最大 in-flight 均不增加。
2. `backend="gpu"` 改为 GPU-preferred。每次调用先规范化并检测位点；累计少于
   256 个位点且输入结束时，复用已准备的 DTO/site 直接进行 CPU 推理，不初始化
   CUDA，默认不输出警告；debug 日志记录 requested/effective backend、实际位点数
   和阈值。累计达到 256 后才建立 CPU pool/CUDA consumer；启动错误和 OOM 直接
   失败，不回退 CPU。

实际工作：

- `MCAGpuPredictor` 构造阶段只校验 GPU index、worker 数和显式 batch 参数，CUDA
  provider、CPU affinity、显存查询、模型解析及 worker/session 均推迟到阈值；
- 增加 `PreparedMolecule` 单一中间表示，CPU 与 GPU 路径直接复用阈值判断期间已
  完成的规范化与 site detection，不重复业务计算；
- 阈值前缀以 `SpooledTemporaryFile` 保存，内存上限 8 MiB，超过后转临时文件，
  避免低位点长输入无界占用内存；达到阈值后不额外预读生成器；
- 滚动 coordinator 维持全局 planner、completion queue 与 pending batch deque；
  preprocessing credit 只有在 P1 完成且覆盖最后 atom row 的 logical feature batch
  已提交后才释放，resident atom credit 在结果按输入顺序输出后归还；
- resident hard cap 为
  `window_budget + max_inflight * (gpu_batch_size + 2 * max_atoms)`，防止 GPU 结果
  乱序或首结果阻塞时轻量状态无界累积；该上限位于旧实现的名义资源包络内，但
  不宣称严格总内存字节数不变；
- 顶层 `-d/--debug` 现在把日志等级真实配置为 DEBUG，使默认静默的 CPU 路由可在
  高信息模式下观察；普通 stdout 与 MCA 表格不受影响；
- 根 README、MCA README/CLI 文档、usage/development skill 和 Plan 15 文档同步
  GPU-preferred 的窄例外及启动后零回退边界。

本地验证：

- 滚动准入聚焦测试以延迟 GPU 结果的门控队列确认：前一批 GPU 结果尚未返回时，
  已完成 CPU 预处理的 credit 可以接纳后续分子，同时保持 batch ordinal、输出
  顺序和值；测试同时覆盖慢 site 与即时 site 两种完成顺序；
- 阈值测试覆盖 `<256` CPU、`=256` GPU、CUDA 惰性启动、前缀复用、生成器不预吞、
  默认静默/debug 日志和启动错误不回退；
- `PYTHONPATH=. python -m pytest -q tests/mca`：`183 passed in 13.43s`；
- `git diff --check`：通过；当前环境没有 Ruff，未执行该可选检查。

实现期间的 ai-14 首轮运行暴露了一个边界：一轮 resident state 已完全输出而输入
尚未结束时，coordinator 会错误进入阻塞读取并报告 stopped making progress。
提交 `9011f87` 在所有阻塞式 queue 读取前优先回到准入循环，并增加参数化回归。
首版 credit 直到有序输出才释放，4096 分子稳态仅为 `29.864 mol/s`（相对旧实现
`+3.67%`）；随后提交 `9494657` 按上述安全边界提前复用 preprocessing credit，
并用独立 resident hard cap 保持有界。

ai-14 最终验收使用 4 张 A800、64 个 CPU（CPU 64–127）、9/7 workers/card、
batch 2048，源码提交 `9494657`，环境 `hp-gpu`：

- 255 个预测位点：GPU-preferred 与显式 CPU 结果完全一致；`gpu_started=false`、
  无 resource plan、无 GPU 子进程。默认静默，debug 精确记录
  `requested=gpu effective=cpu site_count=255 threshold=256`；
- 256 个预测位点：确实启动 4 卡 GPU；MCA、site 与联合 hash 与优化前完全一致；
- 4096 分子（103,776 atom rows、30,640 sites）：首轮含惰性启动
  `99.481 s / 41.174 mol/s`，warm 稳态 `92.002 s / 44.521 mol/s`；旧实现相同配置
  稳态均值为 `28.808 mol/s`，因此吞吐提高 `54.55%`、耗时降低 `35.29%`；
- 最终 4096 分子的 MCA/site/joint hashes 分别为
  `ad00da4e45c7e4462f7d5b9eab1ed6918989e18dec8a6a4a1299659084b81544`、
  `a7339ccdce12c9f53ac13ddebe650868d82332ae748f27202eeee063232482e2`、
  `b81031d437b6766bf96282dac21410f1b0e55fda2117f5c856ff88b67c18d7eb`，
  与优化前逐项一致；joint hash 覆盖 `is_Nuc_site`、site type 以及 atom/site MCA；
- 四卡平均利用率为 74.26%–77.78%，四卡同时 busy 为 75.17%，四卡同时不低于
  90% 为 55.17%；选定 64 CPU 平均 busy 为 45.72%。测试前后四卡显存读数完全
  相同，无 OOM、无残留进程，CPU affinity 恢复正常。

远端结果保存在 `~/mca_test/results/rolling_13b836f/`；实现提交依次为
`13b836f`（滚动准入与阈值主体）、`9011f87`（空转边界修复）和 `9494657`
（提前复用 preprocessing credit 与 resident hard cap）。

