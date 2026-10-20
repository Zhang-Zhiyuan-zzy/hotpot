# MCA GPU 流水线实施方案

状态：已批准实施。

## 1. 目标与不可变边界

目标是在不改变 MCA 结果与业务功能的前提下，消除 GPU 等待 CPU 位点识别、
构象生成和特征构造的串行空洞，并支持单卡、多卡和有界流式输入。

以下约束是硬边界：

1. 保持 24 条位点规则的原顺序、单分子内 first-match 行为、位点输出顺序及
   `is_Nuc_site` 语义；同一原子与位点结果必须联合校验。
2. 保持 all-atom 推理、扁平 atom-row logical batch、每个 logical batch 的
   最大原子数 padding，以及 `(molecule_index, pad_width)` 复用语义。
3. worker 完成顺序不得改变全局输入顺序或 logical batch 边界。
4. CPU 与 GPU、不同 batch size 可以有各自的固定 golden；不得把一个 provider
   或 batch 的 hash 当成另一运行条件的通用 golden。
5. GPU OOM 不自动重试、不切换 CPU；错误必须报告当前 batch，并明确建议下一
   个较低 batch 档位。
6. 旧 GPU 后端、feature flag、`MCAPredictor(device=...)` 和旧
   `predict_mca(device=...)` 不保留。Git 负责历史回退，产品代码只有一套当前实现。

## 2. 已批准的配置模型

### 2.1 不引入 profile

profile 是把多项配置收进一个命名预设，例如 `balanced` 可能暗中选择 worker
数量、batch 和线程数。它会形成第二层配置解释规则。本实现直接使用以下默认值：

- `site_cpu_per_gpu=9`
- `infer_cpu_per_gpu=7`
- `gpu_batch_size=None`：按单卡总显存自动选择

用户需要调优时直接修改这些原子参数，不通过预设名间接修改。

### 2.2 GPU 与 CPU 资源

GPU 通过以下二选一参数指定：

- `gpu_count=N`：使用当前进程可见的逻辑 GPU `0..N-1`；
- `gpu_indices=(...)`：使用明确且有序的逻辑 GPU index。

两者都省略时使用逻辑 GPU 0。两者同时给出属于配置错误。

所需 CPU 数量只由配置推导，不暴露 `cpu_limit`：

```text
required_cpu = gpu_number * (site_cpu_per_gpu + infer_cpu_per_gpu)
```

创建任何 worker 或 CUDA session 前读取当前进程的 CPU affinity。可用 CPU 少于
所需数量时立即失败，并报告 GPU 数、两类每卡 worker 数、所需和可用 CPU。
不会静默缩容，也不会越过 cpuset/cgroup。CPU ID 按 affinity 的稳定顺序分配，
每个 worker 固定一个 CPU，避免超过推导出的上限。

CPU backend 不公开 `cpu_limit`。ONNX Runtime 使用显式线程配置和顺序执行，
避免其隐式 affinity 与外层进程池叠加。

### 2.3 显存与 batch

自动 batch 只读取每张卡的总显存，不依赖瞬时空闲显存：

| 单卡总显存（二进制 GiB） | batch |
| --- | ---: |
| `< 8` | 64 |
| `[8, 16)` | 128 |
| `[16, 24)` | 256 |
| `[24, 40)` | 512 |
| `[40, 80)` | 1024 |
| `>= 80` | 2048 |

多卡显存不同时取各卡自动档位的最小值，使任一全局 logical batch 都能发送给
任一卡。用户可显式选择 `64/128/256/512/1024/2048` 覆盖自动值。

发生真正的 CUDA/ORT OOM 时抛出 `MCAGpuOutOfMemoryError`，至少包含 GPU index、
当前 batch、下一较低建议档位和操作阶段。batch=64 时明确说明已经没有更低的
受支持档位。非 OOM 异常原样传播。

## 3. 单一执行架构

```text
有界输入窗口（带全局 index）
    -> P1 全局 site worker pool：GPU 数 * site_cpu_per_gpu
    -> P2 全局 conformer/feature worker pool：GPU 数 * infer_cpu_per_gpu
    -> coordinator 按输入顺序冻结全局 atom-row logical batches
    -> 每张 GPU 一个独占 CUDA consumer process / ORT session
    -> 按 batch ordinal 和 molecule index 合并
    -> 有序流式结果
```

统一使用 `multiprocessing` 的 `spawn` context。CPU worker 必须在 CUDA consumer
之前启动，worker 模块不得导入或初始化 CUDA runtime。每个 CUDA consumer 是
唯一持有对应 GPU session 的进程；provider 显式设置 `device_id`。

P1、P2 是全局池而非每卡嵌套池。队列与重排缓冲区有界。`iter_predict()` 是
真正的流式 API；`predict()` 只是将其 materialize 为列表的便捷入口。

### 3.1 序列化边界

进程间只传输带版本号的不可变 DTO，不直接依赖 Hotpot、OpenBabel 或 pybel
对象的 pickle 行为。入口只规范化一次。DTO 必须保留原子序号、元素、形式电荷、
芳香性、键级/芳香键、立体信息以及位点识别需要的配位/金属连接语义。

### 3.2 logical batch 与缓存

`gpu_batch_size` 的单位是扁平 atom-row 数，不是分子数。coordinator 必须跨输入
窗口保留未填满的 logical chunk，因此窗口边界不得改变推理 batch。

同一分子在不同 logical chunk 中可能需要不同 `pad_width`，必须按
`(molecule_index, pad_width)` 构造或复用编码。同一 key 只计算一次；不能把一种
宽度的编码错误复用于另一宽度。

## 4. 公共接口

当前接口只有显式 backend，不保留 `device="auto"`：

```python
class MCACpuPredictor:
    def __init__(self, *, batch_size: int = 64): ...
    def iter_predict(self, molecules): ...
    def predict(self, molecules): ...
    def close(self): ...

class MCAGpuPredictor:
    def __init__(
        self,
        *,
        gpu_count: int | None = None,
        gpu_indices: Sequence[int] | None = None,
        site_cpu_per_gpu: int = 9,
        infer_cpu_per_gpu: int = 7,
        gpu_batch_size: int | None = None,
    ): ...
    def iter_predict(self, molecules): ...
    def predict(self, molecules): ...
    def close(self): ...

def predict_mca_cpu(molecules, *, batch_size: int = 64): ...
def predict_mca_gpu(molecules, *, gpu_count=None, gpu_indices=None,
                    site_cpu_per_gpu=9, infer_cpu_per_gpu=7,
                    gpu_batch_size=None): ...
def predict_mca(molecules, *, backend: Literal["cpu", "gpu"], **kwargs): ...
```

两个 predictor 都支持 context manager，`close()` 幂等。单个 predictor 的
`iter_predict/predict` 不可重入；运行异常必须清理子进程和队列。一次性函数用
`try/finally` 关闭资源。

CLI 删除 `--device`，改为必选 `--backend {cpu,gpu}`。GPU 参数与上述构造器一一
对应；CPU 只接受 `--batch-size`。CLI 消费 `iter_predict()`，不预先读取全部输入。

calculator 采用显式 backend，不用 LRU 缓存持有重型 GPU pool。批量使用者应
显式持有 predictor 生命周期。

## 5. 模块边界

| 模块 | 职责 |
| --- | --- |
| `api.py` | 当前公共类型、显式 factory/one-shot 函数 |
| `cpu_backend.py` | 单进程 CPU all-atom 推理 |
| `gpu_backend.py` | coordinator、生命周期、ordered streaming |
| `batching.py` | provider 无关的 logical batch 规划与 scatter/join |
| `serialization.py` | 版本化分子 DTO 与重建 |
| `workers.py` | 不接触 CUDA 的 P1/P2 spawn worker |
| `gpu_consumer.py` | 每卡 CUDA session、collation 和 ORT 执行 |
| `resources.py` | GPU index、CPU affinity 和 pinning 计划 |
| `gpu_memory.py` | 显存档位、OOM 识别与结构化错误 |
| `runtime.py` | 显式 CPU/CUDA ONNX Runtime session |
| `model_store.py` | 模型 artifact 解析，不再解析 device |

## 6. 实施阶段与验收

### 阶段 0 — 固化方案与基线

- 建立本计划、实时实施 log 和最终报告模板；
- 运行现有 MCA 测试作为改动前基线；
- 记录分支、提交和环境限制。

### 阶段 1 — 纯资源、显存与 batching 原语

- 实现 CPU/GPU 资源计划、显存档位和 OOM 错误；
- 抽取保持现有数值语义的 logical batch planner；
- 用纯单元测试覆盖显存边界、异构卡、CPU 不足、乱序回收和跨窗口 chunk。

### 阶段 2 — 显式 runtime 与 CPU backend

- 显式配置 CPU/CUDA provider、线程、顺序执行和 device ID；
- 迁移为 `MCACpuPredictor`，删除旧 device resolver；
- 对所有现有输入 adapter、batch size 和结果字段做 CPU 回归。

### 阶段 3 — DTO 与 spawn-safe P1/P2

- 建立版本化 DTO；
- 抽出 top-level P1/P2 worker 函数；
- 验证 spawn round trip、位点顺序、立体/电荷/芳香/配位和错误传播。

### 阶段 4 — 单卡与多卡 GPU pipeline

- 先建立单卡完整流水线，再使用同一 coordinator 扩展多卡；
- 每卡一个 consumer 和 session；
- 保持全局 batch 冻结、有序输出、缓存 key 和异常清理。

### 阶段 5 — 公共 API、CLI 与 calculator 切换

- 发布显式 CPU/GPU 接口；
- 同一提交删除旧类、旧参数和旧 GPU 代码；
- CLI 流式化，calculator 不缓存重型 pool；
- 更新导出、文档和调用点。

### 阶段 6 — 完整验证

至少覆盖：

- CPU 当前 golden 和所有现有 MCA 测试；
- GPU provider+batch 专属 golden；
- `is_Nuc_site` 与 atom/site join 联合一致；
- 单卡、多卡、异构显存、乱序完成、跨窗口 logical batch；
- OOM 当前/下一档信息且零自动重试；
- CPU affinity 不足、worker pinning 和无超限行为；
- context manager、幂等 close、中途异常和子进程无泄漏；
- CLI/API 旧接口确已消失。

已确认的外部性能数据不重复跑来证明方案合理性；实现完成后的 A800 运行仅用于
当前代码的功能、数值和吞吐验收。

## 7. 完成定义

只有同时满足以下条件才完成：

1. 代码库只有当前 CPU/GPU 实现和当前接口；
2. 结果、位点、顺序、`is_Nuc_site` 和 all-atom 语义通过回归；
3. 单卡与多卡 pipeline 的资源上限、关闭和失败行为可证明；
4. CLI 对大输入不全量 materialize；
5. 实施 log 记录每一阶段的实际变更、证据与偏差；
6. 最终实施报告列出提交、文件、测试、性能结果和剩余边界。

