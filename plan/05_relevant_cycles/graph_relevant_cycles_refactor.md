# `hotpot.cheminfo.graph` Relevant Cycles 原生实现设计

> 状态：图能力阶段（第 1--5 步）已在 `feat/relevant-cycles-native` 实施并验证；
> Core/Geometry 化学策略接入（第 6--7 步）仍保持未启用
>
> 目标：在 Hotpot 内提供一个严格、确定、可安装的 Relevant Cycles 接口；生产运行不依赖 RingDecomposerLib、RDKit 或 Java
>
> 算法基线：Vismara Relevant Cycle Families；RingDecomposerLib 固定提交 `3a7ff93de0d9c4f6a5661508549c6063573f39c7` 仅作为实现参考与差分测试 oracle
>
> 验证报告：`tests/test_cheminfo/graph/rdl_oracle/VALIDATION.md`

## 1. 评审结论

该方案可行，但需要修正两个表述：

1. 可以只暴露一个公开的 Python 函数和一个 C++ 核心函数；
2. 不能把 Relevant Cycles 理解为一个可以从 RingDecomposerLib 单独复制出来的小函数。

精确的 `edges -> all relevant cycles` 至少包含：

```text
simple undirected graph
    -> biconnected components
    -> ordered all-pairs shortest paths / shortest-path DAGs
    -> Vismara odd/even cycle-family candidates
    -> GF(2) relevance filtering by cycle length
    -> expansion over tied shortest paths
    -> canonical ordered vertex cycles
```

两个容易产生错误结果的裁剪方式必须禁止：

- `Vismara candidates` 是 Relevant Cycle Families 的候选超集，未经 GF(2)
  筛选不能作为 Relevant Cycles 返回；
- 一个 family prototype 只代表该 family，未经等长最短路径组合展开，不是该
  family 中的全部 Relevant Cycles。RingDecomposerLib 自带示例中存在
  `4 RCF != 6 RC` 的实际情况。

RingDecomposerLib 去掉测试和 DIMACS 后，能够完成 Relevant Cycles 计算的最小原始
C 依赖闭包仍为 11 个源文件、约 4609 行 C；源文件和头文件约 212 KiB，静态库约
94 KiB。由此可见：库的二进制体积并不是主要问题，真正需要控制的是公共 API、
维护边界和算法正确性。

最终建议是：

- Hotpot 生产包只发布 relevant-only 的现代 C++ 内核和 pybind11 绑定；
- RingDecomposerLib 不作为运行时依赖；
- 开发阶段使用固定版本 RingDecomposerLib 和独立穷举算法做双重 oracle；
- 对外只公开 `relevant_cycles()`，内部保持可验证的职责分层；
- 不提供 NetworkX、RDKit 或其他不同语义算法的静默 fallback。

整体实施难度为中高。pybind11 本身难度低；主要工作在算法等价性、组合爆炸控制、
确定性输出和 Python 3.9--3.14 wheel 验证。

## 2. 当前仓库事实与迁移边界

### 2.1 已有 `graph.py`

当前 [`hotpot/cheminfo/graph.py`](../../hotpot/cheminfo/graph.py) 是活跃模块，包含：

- `linkmat2adj()`、`adj2laplacian()`；
- `calc_electron_config()`、`atoms_electron_configurations()`；
- `calc_spectrum()`、`GraphSpectrum`；
- `graph_dfs_path()`、`graph_dfs_paths()`。

已知生产消费者包括：

- `core.py` 的 `from . import graph`；
- `core.py` 对 `linkmat2adj`、`GraphSpectrum` 和 DFS 的调用；
- `plugins/dl/function/graph.py` 对 `calc_spectrum` 的深层导入。

因此不能直接创建同名 package 后忽略原模块，也不能长期同时维护 `graph.py` 与
`graph/`。实施时必须先做一次行为保持的单文件包化迁移，并由
`graph/__init__.py` 重导出现有公开名称。

### 2.2 当前环语义

当前 `Molecule.rings` 和 `Molecule.rings_for_scope()` 使用
`networkx.cycle_basis()`，只返回一套 cycle basis。`geometry.convert.RingFamily`
也只记录 `NETWORKX_CYCLE_BASIS`。

Relevant Cycles 的加入不能直接替换 `Molecule.rings`，因为该属性还参与：

- `Atom.in_ring`；
- `Bond.is_aromatic`；
- fused/joint ring 判断；
- SMARTS 和其他 Core 化学语义；
- force-field 环扫描。

首轮只建立独立图算法 API。Core 和 Geometry 必须在后续独立节点中通过显式
`RingFamily.RELEVANT_CYCLES` 接入，经过化学回归审查后才能讨论默认值。

### 2.3 输入来源

`Molecule.link_matrix` 是形状 `(n_bonds, 2)` 的 0-based 原子边表，可以直接传给
Python facade。`full_graph` 和 `ligand_skeleton` 的边选择继续由 Core 化学层负责；
原生图算法不识别 Atom、Bond、金属或配位键。

## 3. 目标目录结构

```text
hotpot/cheminfo/
├── graph/
│   ├── __init__.py
│   ├── matrix.py
│   ├── spectrum.py
│   ├── traversal.py
│   ├── cycles.py
│   ├── _relevant_cycles.pyi
│   ├── README.md
│   └── _native/
│       ├── bindings.cpp
│       ├── relevant_cycles.hpp
│       ├── relevant_cycles.cpp
│       ├── THIRD_PARTY_NOTICES.md
│       └── LICENSE.RingDecomposerLib
└── graph.py                         # 包化迁移完成后删除

tests/test_cheminfo/graph/
├── __init__.py
├── oracle.py
├── fixtures.py
├── test_existing_graph_api.py
├── test_relevant_cycles_contract.py
├── test_relevant_cycles_reference.py
├── test_relevant_cycles_invariance.py
└── test_relevant_cycles_limits.py

tests/performance/
└── test_relevant_cycles_benchmark.py
```

### 3.1 Python 模块职责

| 模块 | 职责 |
|---|---|
| `graph/__init__.py` | 唯一公开入口；通过 `__all__` 重导出现有图 API 和 `relevant_cycles` |
| `matrix.py` | 原 `linkmat2adj`、`adj2laplacian`，本轮迁移不改变行为 |
| `spectrum.py` | 原电子构型、图谱函数和 `GraphSpectrum`，本轮迁移不改变行为 |
| `traversal.py` | 原 DFS 函数，本轮迁移不顺便修复或改写其逻辑 |
| `cycles.py` | 输入规范化、稠密编号映射、资源上限、调用 native、恢复原节点编号、输出规范化 |
| `_relevant_cycles.pyi` | 私有 native 扩展的静态类型契约，不作为公共入口 |
| `README.md` | Relevant Cycles 定义、输入输出契约、算法来源、限制与示例 |

`graph/__init__.py` 的规范公开面为：

```python
__all__ = (
    "calc_electron_config",
    "atoms_electron_configurations",
    "linkmat2adj",
    "adj2laplacian",
    "calc_spectrum",
    "GraphSpectrum",
    "graph_dfs_path",
    "graph_dfs_paths",
    "DEFAULT_RELEVANT_CYCLE_LIMIT",
    "RelevantCycleLimitExceeded",
    "relevant_cycles",
)
```

前八项保持当前 `graph.py` 的现有入口；后三项是本次新增公开面。NumPy、NetworkX、
typing 名称及 native extension 均不得因 `import *` 意外泄漏。

### 3.2 C++ 模块职责

| 文件 | 职责 |
|---|---|
| `relevant_cycles.hpp` | 只声明 C++ 值类型、限制结构和一个核心计算函数 |
| `relevant_cycles.cpp` | 图分解、最短路径、cycle families、GF(2) 筛选、展开和 edge-cycle 结果 |
| `bindings.cpp` | pybind11 类型转换、GIL 释放和原生异常映射；不包含算法 |
| `THIRD_PARTY_NOTICES.md` | 固定上游提交、派生范围与论文引用 |
| `LICENSE.RingDecomposerLib` | 完整 BSD-3-Clause 许可证和原版权声明 |

公开 API 只有一个函数，不等于把所有实现写进一个巨大函数。首版无需把内部 helper
过度拆成大量 translation units；若 `relevant_cycles.cpp` 在实施中超过可审查范围，
再把 `graph_data` 和 `cycle_space` 拆为私有 `.hpp/.cpp`，不改变 Python API。

## 4. Python 公共接口契约

```python
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional


NodeIndex = int
EdgeInput = Sequence[int]
NodeCycle = tuple[NodeIndex, ...]

DEFAULT_RELEVANT_CYCLE_LIMIT = 10_000


class RelevantCycleLimitExceeded(RuntimeError):
    """Raised before an incomplete Relevant Cycle collection is returned."""


def relevant_cycles(
    edges: Iterable[EdgeInput],
    *,
    max_size: Optional[int] = None,
    max_cycles: Optional[int] = DEFAULT_RELEVANT_CYCLE_LIMIT,
) -> tuple[NodeCycle, ...]:
    """Return the exact Relevant Cycles of an undirected unweighted graph."""
```

### 4.1 输入

- `edges` 表示无向、无权、简单图；
- 每项必须恰有两个非负整数节点编号；
- 节点编号允许不连续；
- 空边表和森林返回空 tuple；
- 独立节点与环集合无关，因此首版不需要 `num_vertices`；
- 自环、重复无向边和平行边均显式报 `ValueError`，不得静默删除；
- NumPy `(n_edges, 2)` 整数数组和普通 Python edge iterable 使用同一入口；
- 首版不接受 NetworkX Graph、Molecule 或任意 hashable 节点对象。适配由上层完成。

### 4.2 Relevant Cycle 定义

首版只实现单位边权定义：

> 一个 simple cycle 是 Relevant Cycle，当且仅当它属于至少一套 Minimum Cycle
> Basis；等价地，其 edge-incidence vector 不能由严格更短的 cycles 在
> $\mathrm{GF}(2)$ 上线性组合得到。

环长度为边数。键级、芳香性、元素、坐标和配位语义均不进入图算法。

### 4.3 输出

- 返回 `tuple[tuple[int, ...], ...]`；
- 内层 tuple 是沿环边界排列的节点，不重复首节点；
- 相邻元素以及末尾—首位必须对应输入边；
- 每个环在所有旋转与反向表示中取字典序最小形式；
- 全部环按 `(len(cycle), cycle)` 排序；
- 输入边顺序和每条边的方向不得影响结果；
- 不公开 native edge-id，因为它依赖输入边顺序且不适合直接构造 Hotpot `Ring`。

示例：

```python
from hotpot.cheminfo.graph import relevant_cycles

cycles = relevant_cycles([(4, 7), (7, 9), (9, 4)])
assert cycles == ((4, 7, 9),)
```

### 4.4 限界语义

- `max_size=None` 返回所有尺寸的 Relevant Cycles；
- `max_size=k` 返回全局 Relevant Cycles 中长度不超过 `k` 的精确子集；
- 相关性必须先按完整图定义确定，禁止把“只对长度不超过 `k` 的候选做局部判断”
  冒充完整语义；
- C++ 层应跳过大于 `max_size` 的 family 展开，避免先物化再由 Python 过滤；
- `max_cycles` 约束本次实际返回的环数；
- 超限时抛出 `RelevantCycleLimitExceeded`，不返回部分结果；
- `max_cycles=None` 表示调用者显式接受无上限展开；
- 默认上限暂定 10,000，实施 benchmark 后可以在不改变异常语义的前提下调整。

这项限制是必要的：Relevant Cycles 的数量可能随图规模指数增长，完整枚举不可能具有
与输出数量无关的多项式时间或内存保证。

## 5. 私有 C++ 接口

C++ 内核只处理由 Python 层压缩后的连续节点编号：

```cpp
namespace hotpot::graph {

using VertexId = std::uint32_t;
using Edge = std::array<VertexId, 2>;
using CycleEdges = std::vector<std::size_t>;

struct RelevantCycleLimits {
    std::optional<std::size_t> max_size;
    std::optional<std::size_t> max_cycles;
};

std::vector<CycleEdges> relevant_cycles(
    const std::vector<Edge>& edges,
    const RelevantCycleLimits& limits
);

}  // namespace hotpot::graph
```

C++ 内核优先返回 edge-id cycle，而不是假定 prototype 中的边已经按环周向排序。Python
facade 或 binding 层从每个 degree-2 cycle subgraph 恢复有序节点环，然后执行旋转/
反向 canonicalization。

实现要求：

- 所有状态属于函数调用本地对象；
- 不使用可变全局 logger 或全局 scratch buffer；
- 计算期间释放 GIL；
- C++ 异常必须映射为明确的 Python 异常；
- 不返回裸指针或要求 Python 调用释放函数；
- 使用 C++17；
- 不依赖 Open Babel、RDKit、NetworkX 或 NumPy C ABI；
- native 函数不承担 Hotpot 化学对象转换。

## 6. 算法内部阶段

### 6.1 图规范化

1. Python 层验证 edge shape 和整数节点；
2. 将稀疏原节点编号映射到 `[0, n)`；
3. 无向边端点规范化为 `(min(u, v), max(u, v))`；
4. 明确拒绝 self-loop 和 duplicate edge；
5. C++ 建立稳定 adjacency 和 edge-id 映射。

### 6.2 双连通分量

使用 Tarjan 算法拆分 edge-biconnected components。桥和无环分量不进入后续阶段；
不同分量的 Relevant Cycles 取并集。

### 6.3 最短路径结构

对每个 BCC 运行适用于无权图的 BFS/APSP，并保存所有影响 cycle-family 展开的等长
最短路径关系。只保存一个 predecessor 会丢失高对称图中的 Relevant Cycles。

### 6.4 Vismara family candidates

按稳定顶点次序构造 odd/even cycle-family candidates 和每个 family 的 prototype
edge bit vector。顶点排序只能用于确定性，不能改变无标号图上的最终环集合。

### 6.5 GF(2) relevance filtering

按 family weight 分组。对于长度为 `w` 的 prototype，检查它是否属于所有严格短于
`w` 的 cycle vectors 所张成的空间：

- 属于该空间：该 family 不是 Relevant Cycle Family；
- 不属于该空间：保留该 family，并更新相应消元结构。

位向量使用 packed `uint64_t` blocks，消元操作使用 XOR。

### 6.6 family 展开

枚举一个已保留 family 两侧所有满足约束的等长最短路径组合，生成 simple cycle。
必须拒绝除规定根节点外产生额外共享顶点的组合，避免生成非 simple cycle。

在展开前利用 family weight 实施 `max_size`；在产生第 `max_cycles + 1` 个唯一环时立即
终止并抛异常。

### 6.7 输出恢复

将 BCC edge-id 映射回原图 edge-id，验证 cycle edge subgraph 中每个节点 degree=2，
恢复周向顶点序列，映射回原始节点编号，canonicalize、去重并稳定排序。

## 7. RDL 参考范围与许可证

若实现参考、翻译或改写 RingDecomposerLib 源码，则视为派生实现：

- `relevant_cycles.cpp` 保留上游版权和 BSD-3-Clause notice；
- `LICENSE.RingDecomposerLib` 收录完整许可证；
- `THIRD_PARTY_NOTICES.md` 记录来源仓库、固定 commit 和裁剪范围；
- sdist 与 wheel 必须包含上述文件；
- 项目文档引用 Vismara 论文，以及 RingDecomposerLib 2017 实现论文。

BSD-3-Clause 与 Hotpot 的 MIT 许可证兼容。论文引用属于科研与来源规范；保留版权和
许可证文本则是再发布条件。

开发用 RDL oracle 不进入运行时依赖。不要直接把上游 `.c` 改名为 `.cpp` 编译；上游
C90 代码包含 `malloc` 隐式转换和 C++ 关键字冲突，必须保持 C 编译或真正移植。

## 8. 构建与发布

当前根项目使用 setuptools，`setup.py` 只是 metadata shim；`cxxpy/` 是未接入生产包
的实验目录，`_clib/` 也不能作为新的构建范例。

首版沿用 setuptools，避免为一个扩展同时引入 CMake/scikit-build：

```toml
[build-system]
requires = [
    "setuptools>=77,<82",
    "wheel>=0.43",
    "pybind11>=3,<4",
]
```

根 `setup.py` 使用：

```python
Pybind11Extension(
    "hotpot.cheminfo.graph._relevant_cycles",
    [
        "hotpot/cheminfo/graph/_native/bindings.cpp",
        "hotpot/cheminfo/graph/_native/relevant_cycles.cpp",
    ],
    cxx_std=17,
)
```

构建规则：

- `pybind11` 只进入 build requirements，不进入 Hotpot runtime dependencies；
- `MANIFEST.in` 包含 C++ 源码、头文件、notice 和 BSD license；
- 不提交 `.o`、`.so` 或本机生成文件；
- 不使用平台硬编码的 `-std=c++17`，由 `Pybind11Extension(cxx_std=17)` 处理；
- 首版不采用 `abi3`；CPython 3.9--3.14 分别构建 ABI wheel；
- 使用 `cibuildwheel` 生成发布 wheel，并在源码目录外安装验证。

源码安装时 native 编译失败必须使安装失败。源码检出环境可延迟导入 native 模块，
但调用 `relevant_cycles()` 时必须报告明确的 native-extension unavailable 错误；不得
切换到 `cycle_basis()`、`minimum_cycle_basis()`、`chordless_cycles()` 或暴力实现。

## 9. 测试围栏

### 9.1 独立小图 oracle

测试代码枚举小图的全部 simple cycles，并按如下严格定义判断：

```text
C is relevant
iff edge_vector(C) is not in
    span_GF(2)({edge_vector(S) | length(S) < length(C)})
```

该实现只用于测试小图，不进入生产 fallback。对 NetworkX Graph Atlas 中可控规模的
图逐一比较 native 输出。

### 9.2 固定参考图

| 图 | 预期 Relevant Cycles |
|---|---|
| 空图、树 | 0 |
| triangle | 1 个三元环 |
| square + diagonal | 2 个三元环，不含外围四元环 |
| naphthalene topology | 2 个六元环，不含外围十元环 |
| anthracene topology | 3 个六元环 |
| cubane topology | 6 个四元环；cycle rank 只有 5 |
| $K_4$ | 4 个三元环 |
| equal-path theta graph | 3 个环均 relevant |
| disconnected graph | 各分量结果的稳定并集 |

### 9.3 不变量与错误契约

- 输入 edge 顺序置换不改变输出；
- 每条 edge 反向不改变输出；
- 节点重标号后结果保持图同构；
- 稀疏节点编号正确映射回来；
- 每个返回 cycle 是 simple cycle 且边均存在；
- 返回集合不存在旋转/反向重复；
- self-loop、duplicate/parallel edge 显式失败；
- `max_size` 只筛选完整 Relevant Cycle 集合；
- `max_cycles` 超限不返回部分集合；
- 多线程并发调用结果一致；
- ASan、UBSan 无错误。

### 9.4 上游差分与化学案例

- 固定 RDL commit 作为开发 oracle；
- 引入其 Relevant Cycles golden fixtures 时保留来源与许可证；
- 比较的是规范化后的 cycle edge sets，而不是上游内部 RCF 数量；
- 对苯、萘、蒽、cubane 及代表性配位图做 Hotpot `link_matrix` 闭环测试；
- 分别验证完整分子图与移除 metal--ligand edges 后的 ligand skeleton；
- 不把 RDKit SymmSSSR 当作 Relevant Cycles oracle。

### 9.5 构建矩阵

现有兼容脚本直接从源码树导入，没有构建 native extension；现有 workflow 也只在
Python 3.11 构建一次 wheel。实施后必须新增：

- CPython 3.9、3.10、3.11、3.12、3.13、3.14 的 extension build；
- 每个版本的 installed-wheel API smoke test；
- 至少 Linux x86_64 manylinux wheel；
- sdist 在干净环境中构建 wheel；
- wheel 外部执行 `relevant_cycles()`，证明 native artifact 实际被打包。

## 10. Core 与 Geometry 后续接入设计

首轮 native API 完成后，另开节点接入，不与算法实现混在同一提交。

建议的 Core 方向：

```python
class RingFamily(str, Enum):
    CYCLE_BASIS = "cycle_basis"
    RELEVANT_CYCLES = "relevant_cycles"


def rings_for_scope(
    self,
    ring_scope: RingScope,
    *,
    ring_family: RingFamily = RingFamily.CYCLE_BASIS,
    max_size: Optional[int] = None,
) -> list[Ring]:
    ...
```

其中：

- `ring_scope` 决定哪些化学边进入图；
- `ring_family` 决定使用 cycle basis 还是 Relevant Cycles；
- `max_size` 直接下推 native，避免 FF 逐帧扫描时先枚举全部大环；
- 缓存 key 必须覆盖 scope、family、max size 和完整边签名；
- `Molecule.rings` 默认语义首轮保持不变；
- Geometry 报告必须记录实际 `RingFamily`，不能继续写死
  `NETWORKX_CYCLE_BASIS`。

这一步会改变化学业务覆盖范围，必须独立审查。Relevant Cycles 是更合适的环集合候选，
但不自动等于所有调用者都应采用的默认 ring semantics。

## 11. 分阶段实施与提交边界

1. `refactor(graph): convert graph module into package`
   - 只移动现有实现并重导出；
   - 不增加 Relevant Cycles，不修复旧 DFS 或谱逻辑；
   - 验证现有导入与测试完全不变。
2. `test(graph): add relevant-cycle oracle and contract fixtures`
   - 独立小图 oracle、RDL golden fixtures、输入与确定性契约；
   - 此时 native 测试预期尚未通过或通过明确 marker 隔离。
3. `feat(graph): add native relevant-cycle implementation`
   - C++ relevant-only 核心、pybind11、Python facade、许可证；
   - 不接入 Core 默认 rings。
4. `build(graph): package native graph extension`
   - build requirements、setup、manifest、wheel smoke test。
5. `ci(graph): validate native wheels across Python 3.9-3.14`
   - build matrix、sanitizers、installed-wheel test。
6. `feat(core): expose relevant ring family explicitly`
   - Core selector、缓存和 chemistry integration tests。
7. `feat(geometry): scan relevant cycles for bond-ring relations`
   - Geometry/FF 显式选择、覆盖率报告和性能验证。

每个节点独立提交、可单独回退。第 1--5 步只建立图能力；第 6--7 步才改变上层可选
行为，默认语义变化需要另行批准。

## 12. 完成标准

- `from hotpot.cheminfo.graph import relevant_cycles` 是唯一生产入口；
- 现有 graph API 在包化后保持可导入且行为不变；
- native 结果与独立 GF(2) oracle、固定 RDL oracle 一致；
- 对称图不会因只保留一个 shortest-path predecessor 而漏环；
- 输出严格确定、无方向或旋转重复；
- 超限显式失败，不返回不完整集合；
- 无语义不同的 fallback；
- BSD notice 和许可证进入 sdist/wheel；
- Python 3.9--3.14 的 wheel 构建和包外运行验证通过；
- Core 默认 ring 语义在未经独立批准前保持不变。
