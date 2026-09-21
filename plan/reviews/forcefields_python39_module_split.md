# Force-field Python 3.9 / Open Babel 3.1 隔离方案

> 实施状态（2026-09-21）：已完成。Python 3.9 与 3.14 的真实兼容测试均通过；两套 façade
> 的导出集合和函数签名由专门测试锁定。

## 1. 已确定的支持政策

- 继续支持 Python 3.9-3.14。
- Python 3.9 继续使用 `openbabel-wheel 3.1.1.23`；Python 3.10 及以上使用当前声明的
  Open Babel 3.2.x。
- Open Babel 3.1 的兼容实现不得继续与 3.10+ 主实现混放在同一个 Python 文件中。
- 版本选择只允许发生在 `hotpot.cheminfo.forcefields` 包入口；业务执行过程中不得反复
  判断 Python/Open Babel 版本，也不得使用 `try/except ImportError` 静默回退。
- Python 3.9 与 3.10+ 暴露完全相同的公开名称和函数签名。

## 2. 目标文件结构

```text
hotpot/cheminfo/
├── forcefields/
│   ├── __init__.py   # 唯一版本选择点；重新导出选中实现的公开 API
│   ├── ff.py         # Python >=3.10 / Open Babel 3.2 当前公开实现
│   ├── ff39.py       # Python 3.9 / Open Babel 3.1 等签名公开实现
│   ├── utils.py      # 两条路径共用的业务、数据结构及 3.2 helper
│   └── utils39.py    # 仅 Open Babel 3.1 所需的替代 helper
└── geometry/
```

迁移完成后删除原单文件 `hotpot/cheminfo/forcefields.py`，避免同名 module/package 并存造成
导入解析依赖解释器实现细节。

## 3. 各文件职责

### 3.1 `__init__.py`

该文件只负责一次、显式的解释器版本选择：

```python
import sys

if sys.version_info[:2] == (3, 9):
    from .ff39 import *
    from .ff39 import __all__
else:
    from .ff import *
    from .ff import __all__
```

原则：

- 不探测函数是否存在；
- 不捕获导入异常后改用另一实现；
- 不根据单次运行结果降级；
- 不在 `ff.py` 或 `utils.py` 内再次判断版本；
- `from hotpot.cheminfo import forcefields as ff` 和
  `from hotpot.cheminfo.forcefields import optimize` 保持当前导入形式。

当前 `pyproject.toml` 已拒绝 Python 3.9 以下和 3.15 以上版本，因此入口只需要区分 3.9
与 3.10+；无需添加无条件兜底实现。

### 3.2 `ff.py`

- 只连接 `utils.py`；不得导入 `utils39.py`。
- 定义或显式重新导出完整公共数据类、异常和函数。
- 公共函数保持显式参数，不允许使用 `*args` / `**kwargs` 代理。
- 3.10+ 的 seed/build 路径只使用 Open Babel 3.2 的当前机制。

### 3.3 `ff39.py`

- 可以同时连接 `utils.py` 和 `utils39.py`。
- 与 `ff.py` 提供相同的 `__all__`、参数顺序、keyword-only 边界、默认值、annotation 和
  返回契约。
- 仅在需要 Open Babel 3.1 差异的调用点改用 `utils39.py`，其余工作流调用相同的
  `utils.py` 实现。
- 不允许把 3.9 差异写成“先试 3.2，失败后退回 3.1”。

### 3.4 `utils.py`

放置两条版本路径真正共用的实现：

- report/dataclass 和 exception；
- working-copy、事务提交和回滚；
- 力场选择、能量单位转换和约束空接口；
- optimizer、质量门控连接、配体代理构筑；
- worker result 协议、进程超时与退出管理；
- 3.10+ 使用的 Open Babel 3.2 seed helper；
- 两套公开 façade 调用的内部 workflow 函数。

`utils.py` 本身必须使用 Python 3.9 可解析的语法，因为 `ff39.py` 也会导入它。共享工作流
通过显式传入 seed/build adapter 连接版本差异；不得读取一个可变的模块级“当前后端”全局
变量。

### 3.5 `utils39.py`

只放 Open Babel 3.1 独有或必须替换的实现，首个已确认内容为：

- `ctypes.CDLL(None)` / `srand()`；
- 初始化 Open Babel 3.1 函数局部 RNG 的 `vector3.randomUnitVector()`；
- 如 spawn target 必须具有可 pickle 的模块限定名，则提供最小的 3.9 worker entry，而把
  实际业务委托给 `utils.py`。

`utils39.py` 不复制 optimizer、事务提交、质量门控或配体构筑算法。若后续发现 3.1 与
3.2 的新差异，应先证明它确实是后端接口差异，再加入这里。

## 4. 避免复制整套业务代码

`ff.py` 与 `ff39.py` 的公开签名必须重复声明，以便 IDE、文档生成器和 Python 自身都看到
真实签名；但函数体只做薄分派。共同业务实现放在 `utils.py`，版本差异通过明确 adapter
参数传入，例如：

```text
ff.build3d(...)
└─ utils.build3d_workflow(..., seed_builder=utils.seeded_ob_build)

ff39.build3d(...)
└─ utils.build3d_workflow(..., seed_builder=utils39.seeded_ob_build)
```

同理，络合物 worker 需要 seed 时，由 façade 选择 top-level、可 pickle 的版本专用 worker
入口。这样版本差异在进程创建前已经决定，worker 内没有版本判断或异常兜底。

公共 dataclass/exception 应只在 `utils.py` 定义一次，再由两个 façade 显式重新导出，以保证
跨模块的 `isinstance()`、pickle 类型标识和异常捕获一致。不得在 `ff.py` 与 `ff39.py`
分别复制同名 class。

## 5. 公共接口一致性清单

两个 façade 的 `__all__` 必须逐项相等。当前共 44 个公开符号：

- 类型：`OptimizationAlgorithm`、`TerminationReason`、`ForceFieldDiagnosticValue`；
- report/data：`ForceFieldRunReport`、`Build3DReport`、`CandidateRejection`、
  `ComplexBuildDiagnostics`、`BuildWorkerResult`、`ForceFieldWorkflowReport`、
  `BuildAndOptimizeReport`、`ComplexBuildReport`、`ForceFieldSetupReport`、
  `AcceptanceCheck`、`StructureAcceptanceThresholds`、`ForceFieldAcceptanceEvidence`、
  `AtomTopologySignature`、`BondTopologySignature`、`TopologyReference`、
  `ForceFieldValidationReport`、
  `CoordinationEnvironment`、`CoordinationGeometryCandidate`、
  `CoordinationGeometryResult`；
- exception：`ForceFieldError`、`ForceFieldSetupError`、`BuildWorkerError`、
  `BuildTimeoutError`、`ComplexBuildError`、`ComplexBuildWorkerError`、
  `ComplexBuildTimeoutError`、`GeometryQualityError`；
- warning：`GeometryQualityWarning`；
- 函数：`perturb`、`collect_coordination_environments`、
  `prepare_coordination_geometry`、`build3d`、`optimize`、`build_complex3d`、
  `optimize_complex`、`complexes_build`、`build_and_optimize`、`auto_optimize`。

执行 C001 后，`complexes_build()` 在两个文件中都直接使用当前显式签名，不再接受旧参数
或任意 `**options`。

## 6. 导入、测试和打包影响

### 6.1 保持不变的调用

- `core.py` 的 `from . import forcefields as ff`；
- 测试和外部用户的 `from hotpot.cheminfo import forcefields as ff`；
- 通过包入口访问的全部公开类型和函数。

### 6.2 必须迁移的内部测试

当前测试会 monkeypatch `ff._ob_build`、`ff._complexes_build_impl` 等私有实现。包入口完成
版本选择后，不应再重新导出私有 helper。测试应分为：

- 公共契约测试只调用 `hotpot.cheminfo.forcefields`；
- 共用 helper 单元测试直接导入 `hotpot.cheminfo.forcefields.utils`；
- Python 3.9 专用测试直接导入 `utils39`，并只在 3.9/Open Babel 3.1 job 执行；
- worker/spawn 测试 patch 实际定义 target 的模块，而不是 package façade。

### 6.3 CI 和打包

- 将 workflow path filter 从 `hotpot/cheminfo/forcefields.py` 改为
  `hotpot/cheminfo/forcefields/**`；
- setuptools 的 `hotpot*` package discovery 会包含新子包，但 wheel/sdist 测试必须确认
  四个文件全部入包；
- 删除旧 module 后清理源码树中的 `__pycache__/forcefields.*.pyc`，它们不是发布文件；
- 文档链接和旧的行号引用需要更新为 package 内路径。

## 7. 必须新增的围栏

1. **选择测试**：Python 3.9 导出 `ff39`，3.10-3.14 导出 `ff`；检查未选择模块没有被
   package 入口意外导入。
2. **签名测试**：对 `ff.py` 与 `ff39.py` 的全部公开函数执行 AST/`inspect.signature`
   对比；比较参数名、顺序、kind、默认值和返回 annotation。
3. **符号测试**：两个模块的 `__all__` 完全一致，package `__all__` 与选中模块一致。
4. **行为测试**：同一无 seed 输入的 report 类型、单位、异常和拓扑结果一致。
5. **版本专用 seed 测试**：分别在 Open Babel 3.1 和 3.2 环境验证同 seed 重复运行得到
   相同坐标，不同 seed 可产生不同构型。
6. **隔离测试**：3.10+ 导入路径不得加载 `utils39`，`utils.py` 不得导入 `ctypes` 或调用
   libc `srand()`。
7. **spawn 测试**：两个版本的 worker target 均可 pickle，父进程环境变量恢复，timeout、
   abnormal exit 和 malformed envelope 仍显式报错。
8. **完整矩阵**：继续运行 Python 3.9-3.14；3.9 job 明确打印 Open Babel 3.1.x，其余 job
   明确打印 Open Babel 3.2.x，防止装错 backend 后测试仍通过。

## 8. 推荐迁移顺序

1. 先删除旧 `complexes_build` 参数兼容层并恢复唯一显式签名；
2. 创建 package 和 `utils.py`，原代码只做机械迁移，保持行为；
3. 建立 `ff.py` 当前 façade 并先让 3.10+ 测试全部通过；
4. 抽出 `utils39.py` 的 RNG 差异，建立等签名 `ff39.py`；
5. 修改私有 helper 测试目标和 workflow path filters；
6. 分别运行 3.9/Open Babel 3.1 与 3.10+/Open Babel 3.2 动态测试；
7. 最后删除旧单文件、旧 bytecode 和过时文档引用。

每一步单独提交。迁移节点不得夹带 force-field 化学参数、几何判据或优化算法变化。
