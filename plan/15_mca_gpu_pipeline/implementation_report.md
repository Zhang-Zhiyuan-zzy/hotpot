# Plan 15 实施报告

状态：已完成
完成日期：2026-10-21

## 1. 结论

Plan 15 已在独立分支 `perf/mca-gpu-pipeline` 和独立 worktree
`/tmp/hotpot-mca-gpu-pipeline` 完成。实现用一套新的有界 CPU/GPU 流水线替换旧
MCA device 路径，保留 all-atom 计算、logical atom-row batch、padding/cache、
输入顺序、24 条位点规则和 `is_Nuc_site` 语义。

分支基线为 `88ccec5`。此前的 `perf/mca-all-atom-inference` 分支仍原样保留，
Plan 15 没有改写或删除该分支。

最终状态满足本轮批准的五项关键决定：

1. 不引入 profile，9/7 worker 等参数直接作为默认值；
2. 产品代码只有单一实现和单一公共接口，不保留兼容 facade 或旧 GPU 后端；
3. GPU batch 按单卡总显存自动选档，OOM 报告当前档并建议下一较低档；
4. 不公开 `cpu_limit`，CPU 数由 GPU 数与两类每卡 worker 数直接推导；
5. 旧实现只由 Git 历史保存，不在当前代码中显式留存。

本分支尚未嫁接或合并到主仓库；主仓库原有未提交内容未被修改。

## 2. 批复落地情况

| 原判断项 | 批复 | 实施结果 |
| --- | --- | --- |
| 2 | 默认值保留，不引入 profile | 默认 `site_cpu_per_gpu=9`、`infer_cpu_per_gpu=7`；没有 profile 层 |
| 7 | 单一实现、单一接口 | CPU/GPU 共用核心业务函数；旧类、旧参数、旧 GPU 路径已删除 |
| 8 | 显存动态 batch；OOM 给出降档建议 | 按每张卡总显存选六档，异构多卡取最小档；OOM 不自动重试或回退 |
| 10 | GPU 选择与每卡 CPU 数直接推导上限 | 支持 GPU 数量或具体 index；所需 CPU 为 `GPU 数 × (site + infer)` |
| 15 | Git 管理旧实现 | 当前源码没有 legacy backend、feature flag 或兼容开关 |

这里的 profile 是“一组有名字的配置预设”。例如 `balanced` 可能在内部同时改变
worker 数、batch 和线程数。调用方只看到一个名字，却必须额外理解该名字在当前
版本展开成哪些隐含参数，复现和排错也多一层映射。本实现没有引入这层抽象，所有
可调项都以原子参数和明确默认值出现。

批复中的 `inter_cpu_per_gpu` 在实现中命名为 `infer_cpu_per_gpu`，表示 inference
特征构造阶段的 CPU worker 数；这样可避免与 ONNX Runtime 的 `inter_op_num_threads`
混淆。

## 3. 最终架构

GPU 路径的生产流程为：

```text
有界输入窗口
  -> 全局 P1 位点识别池
  -> 全局 P2 构象/特征池
  -> coordinator 冻结全局 atom-row logical batches
  -> 每张 GPU 一个独立进程和 CUDA ORT session
  -> 按 batch ordinal、molecule index 有序合并
  -> 流式输出 MoleculeMCAResult
```

主要设计结果：

- P1/P2 使用 `spawn` CPU pool，并在任何 CUDA consumer 创建前启动；
- P2 feature 一旦满足最早连续 logical batch 就立即送入 GPU，不等待整窗完成；
- 输入窗口、请求队列、in-flight batch 和重排缓存均有界；
- logical batch 可跨输入窗口，窗口边界不改变 padding 或数值语义；
- 每张 GPU 独占一个进程和一个显式 `device_id` 的 ORT session；
- 模型 artifact 只由 coordinator 解析、校验和 materialize 一次，各卡复用同一路径；
- CPU/GPU 共用输入识别、P1、P2 和 `build_prediction()`；atom MCA、site MCA 与
  `is_Nuc_site` 只在 `build_prediction()` 中进行一次联合构造；
- 所有等待有 deadline；异常、迭代提前停止和正常关闭都会清理 pool、process、
  queue，并恢复调用线程原 CPU affinity。

CPU 路径使用相同的 stateful batch planner 和结果构造逻辑，以有界方式消费输入，
不再由旧的通用 device predictor 间接分派。

## 4. 公共接口与运行行为

当前公共接口只有：

- `MCACpuPredictor`
- `MCAGpuPredictor`
- `predict_mca_cpu`
- `predict_mca_gpu`
- `predict_mca(..., backend="cpu" | "gpu")`

已删除 `MCAPredictor`、`device=auto/cpu/cuda`、`--device`、
`HOTPOT_MCA_DEVICE` 和 calculator 中持有重型 predictor 的 LRU cache。CLI 的
`--backend` 默认使用 CPU，GPU 必须显式指定，并对输入、推理和输出全程流式处理。

这是按“单一接口”批复实施的有意破坏性切换：旧 Python 调用方必须改用显式
backend；CLI 调用方省略 `--backend` 时稳定使用 CPU。没有过渡期别名、弃用代理
或双实现。

GPU 可以通过 `gpu_count` 或有序的 `gpu_indices` 指定，二者互斥；均未给出时使用
逻辑 GPU 0。CPU 预算为：

```text
required_cpu = gpu_number * (site_cpu_per_gpu + infer_cpu_per_gpu)
```

实现只从当前进程 affinity 可见的 CPU 中稳定分配。可用量不足时在创建 worker 或
CUDA session 前失败并报告所需量和可用量，不静默缩容，也不越过 cpuset/cgroup。
coordinator 与 GPU consumer 共享已分配 CPU block，因此不额外引入公开
`cpu_limit`；MCA 无法约束的宿主进程其他线程不计入该公式。

`gpu_batch_size=None` 时读取每张卡总显存，按
`64/128/256/512/1024/2048` 六档选择；多卡取最小档。这里的 batch 单位是扁平
atom rows，不是分子数。CUDA/ORT OOM 会抛出 `MCAGpuOutOfMemoryError`，包含 GPU
index、发生阶段、当前 batch 和下一较低建议档位；不会自动重试，也不会切换 CPU。

## 5. 文件改动

核心实现提交共改动 42 个文件，主要分组如下：

| 范围 | 结果 |
| --- | --- |
| `hotpot/cheminfo/AImodels/mca/batching.py` | logical atom-row batch、padding、cache/scatter |
| `cpu_backend.py` | 有界流式 CPU predictor |
| `gpu_backend.py`、`gpu_consumer.py` | coordinator、多 GPU conveyor、生命周期与有序 join |
| `resources.py`、`gpu_memory.py` | GPU 选择、CPU affinity、显存档位与结构化 OOM |
| `serialization.py`、`workers.py` | 版本化不可变 DTO 与 spawn-safe P1/P2 |
| `pipeline.py` | CPU/GPU 共用业务计算与唯一结果构造路径 |
| `runtime.py`、`model_store.py` | 显式 CPU/CUDA session 与单次 artifact 解析 |
| `api.py`、`__init__.py`、`cli.py` | 新公共接口、严格 backend 参数和流式 CLI |
| `calculator/mca_inference.py` | 显式 backend 的 one-shot calculator |
| README、CLI/usage/model 文档 | 删除旧 device 用法，说明当前接口和资源语义 |
| `tests/mca/` 及集成测试 | batch、DTO、资源、runtime、单/多 GPU、故障和 API 回归 |
| `plan/15_mca_gpu_pipeline/` | 方案、实时日志、验证脚本与本报告 |

旧 GPU backend 没有被重命名或复制到 legacy 文件；其历史只存在于 Git。

## 6. 正确性验证

### 6.1 本地验证

- MCA 全套：`172 passed in 13.47s`；
- calculator + MCA README 聚焦：`5 passed, 12 deselected in 13.09s`；
- command loading、calculator package、import safety、MCA import layout：
  `10 passed in 7.93s`；
- `compileall` 和 `git diff --check` 通过。

CPU provider 按 batch 分别固化结果；不同 padding/batch 的浮点结果不能错误地共用
一个通用 hash：

| CPU batch | SHA-256 |
| ---: | --- |
| 5 | `1b39a16d8c84f718184d89ed623ba56ec69aa5ffeacd6eb84138ed5949538a7f` |
| 8 | `ddeb4f8635c47a7f40aa8fff8576a5dbf94c8c2bb525381752d952d0a44c4f11` |
| 64 | `6e60a9f38359c05702f503f76bbbd710476469eac0d3837377a7044b5d73ec92` |

DTO 与业务回归覆盖 SMILES、Hotpot、RDKit、Pybel、OBMol、MoleculeGraph、芳香、
手性、双键立体、同位素、自由配体、零阶/单键配位、螯合和 metal-star。dative 图
继续保持原有 Hotpot SMILES 失败边界，没有用 fallback 静默扩大功能范围。

### 6.2 ai-14 A800 验证

验证样本为 256 个分子、6,486 个原子和 1,915 个位点；A800 80 GiB 自动选择
batch 2048。固定结果为：

- MCA：`fa8e55f55519fac1afca9874f1535de6ee936c89ee88f9bbcda298e31ad158e0`；
- sites：`5d6150efed59cbf96cc83324c4c35c387630e71e25826bad5382378dfe4a10b3`；
- atom/site/`is_Nuc_site` 联合：
  `8fe4ede2dad5e7caf1575432f348f9b10ec4f251eb14c88c41d2fb5b790f87e1`。

单卡、双卡和八卡运行的三类 hash 完全一致，证明多 GPU 调度未改变原子结果、位点
结果或 `is_Nuc_site` join。

联合校验不仅比较输出行，还检查 site index 的有效性与重复情况、同一 site 在
atom/site 视图中的 MCA 逐位一致，并把 `site_type` 和 `is_Nuc_site` 一并纳入 hash。

另用重复后的 512 分子、12,972 原子样本强制跨越两个有界窗口：

- MCA：`0fac229ef0192f24faef1d8a7c80ba7c9e0eb07114deba8f16b07e7dbb64a71a`；
- sites：`47c05d0da1948740fda2e0571b3c349493633999dba34343bcedc751bdc8aa40`；
- 联合：`d5af2956c4aaf6386b3c1d25d0d002a5694cd5492b66d3fe3aaa91f0a52a5fe3`。

跨窗口单卡与双卡逐位一致，验证了跨窗口 logical chunk、padding/cache 和有序
重组。

## 7. 性能与资源验证

| GPU 配置 | 样本 | startup | run | 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 256 | 4.914 s | 20.372 / 20.645 s | 12.57 / 12.40 mol/s |
| 0,1 | 256 | 5.758 s | 14.983 s | 17.09 mol/s |
| 0 | 512，跨窗口 | 5.114 s | 38.522 s | 13.29 mol/s |
| 0,1 | 512，跨窗口 | 5.980 s | 28.959 s | 17.68 mol/s |
| 0..7 | 256 | 9.157 s | 10.974 s | 23.33 mol/s |

旧串行实测为 1.686 mol/s；当前单卡约 7.4 倍、双卡约 10.1 倍、八卡资源路径约
13.8 倍。八卡的 256 分子样本只有 4 个完整 logical batches，不足以同时持续喂满
8 张推理卡，因此 23.33 mol/s 是资源与功能验收结果，不是八卡吞吐上限。

进程树检查确认单卡默认配置只使用 16 个 affinity-visible CPU：16 个 P1/P2
worker 各绑定一个不同 CPU，coordinator、resource tracker 和 GPU consumer 也限制
在同一 `0-15` block，没有出现第 17 个 CPU。八卡按公式需要 128 个 worker，并在
128 个可见 CPU 上成功启动。所有远端运行在 predictor 关闭后都恢复了调用方原
affinity。

## 8. 独立审计与修复

实施期间进行了生命周期、架构、冗余和最终只读审计，已修复：

- queue 中残留大 payload 时 `join_thread()` 永久等待；
- consumer 正常退出后 coordinator 无限等待；
- Pool 替换 worker 后 affinity ID queue 耗尽；
- CLI 输出文件打开失败时 predictor 泄漏；
- 整个输入窗口完成后才启动 GPU 的 P2 barrier；
- coordinator 与每张 GPU 重复解析、校验和 materialize 模型；
- 多卡关闭超时按卡线性累加；
- 重复配置、死 helper、影子状态及只为测试暴露的生产接口。

末轮审计确认：临时和持久 artifact 生命周期均覆盖所有 CUDA session 创建；每张
卡仍创建独立 session；关闭的 graceful/terminate/kill 三阶段分别共享 deadline，
GPU 进程阶段总等待最坏约 15 秒，而不是每卡约 15 秒。

## 9. 已知边界与方案偏差

1. 任意第三方异常对象不能可靠地跨进程重建。CPU pool 的本地业务异常保持原类型；
   GPU consumer 的非 OOM 异常统一为 `RuntimeError`，其中完整保留远端类型名、
   message 和 traceback。OOM 保持专用 `MCAGpuOutOfMemoryError`。这是相对“所有
   非 OOM 原类型传播”计划的唯一显式偏差。
2. ai-14 的 `hp` 当前是 CPU-only ONNX Runtime；真实 CUDA 验收使用已有 CUDA
   provider 与 activation hook 的 `hp-gpu`。本轮只从远端临时源码目录验收，没有
   把未合并分支安装为 ai-14 的正式 `hp` 包；最终验收源码目录为
   `/tmp/hotpot-plan15-final.JiYkNu`。
3. 固定远端样本含一个带电分子；为复验既有 benchmark 显式使用
   `--allow-charged`，生产默认仍拒绝带电输入。
4. 自动 batch 依据总显存而非瞬时空闲显存，保证选择规则可复现；若同卡其他任务
   占用显存，用户按 OOM 信息手动降低一个档位。
5. 全 README 组合测试中两个失败来自未修改的 CBond artifact 缓存只读/缺失；一项
   打包检查失败来自仓库既有 `requirements.txt` 与 `pyproject.toml` 的
   `cryptography` 不一致。均不是 MCA 回归。
6. 当前本地环境没有 `ruff` 和 `build` 可执行工具，因此未运行这两个可选工具；
   已用 `compileall`、完整 MCA 测试和聚焦集成测试覆盖本轮代码。

## 10. 提交与交付

| 提交 | 内容 |
| --- | --- |
| `8935208` | 创建 Plan 15、批准方案、实时日志和报告入口 |
| `676bc31` | 完整 CPU/GPU pipeline、接口切换、测试与文档 |
| 本报告所在提交 | 完成状态、最终实施日志和实施报告 |

回退旧实现应通过 Git 选择历史提交完成，不在产品代码中维护双路径。当前分支可以在
审核通过后再按仓库流程合并或嫁接；本轮没有对主分支执行该操作。

