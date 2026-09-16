# `complexes_build` 络合物构筑与力场优化修补手册

> 状态：实施前设计冻结稿
>
> 工作分支：`fix/complexes-build-pipeline`
>
> 基线提交：`41099db9b3e665b3ed11c87e34ff7adfac71a762`
>
> 本文只规定后续修改和验收方式；建立本文时不得修改业务代码。

## 0. 目标与授权边界

本轮修补的目标不是删除 `complexes_build` 的拓扑解结思路，而是把它整理为一条可终止、可诊断、可验证、不会在失败时污染调用方分子的完整络合物构筑流程。

已经批准直接实施的内容：

1. 子进程超时、异常传播、资源回收和并发隔离。
2. 重试计数器、最近环边选择、VDW 插值、收敛判断等确定性程序错误。
3. 冗余类、冗余入口、冗余参数和参数没有真正传递的问题。
4. `build3d()` 改为只负责分子类型分派。
5. 络合物力场选择统一经过一个很小的 helper；当前任何请求均静默解析为 UFF。
6. 增加可选随机种子，默认仍为非确定性运行。
7. 将现有 `steps`/`step_size` 明确改名为 `epochs`/`steps_per_epoch`。
8. 默认只保留最低能帧，显式 `save_movie=True` 时才保留逐 epoch 轨迹。
9. 清除当前错误的 Open Babel 约束映射实现，只保留稳定的空接口。
10. 建立分层、结构化的几何质量门控。

本轮不得擅自加入的化学假设：

- 不自动推断金属氧化态或改写总电荷。
- 不根据配位数强制指定线性、四面体、平方平面、八面体等几何。
- 不引入新的金属力场或把 RDKit、OpenMM、ASE 变成新的计算后端。
- 不把绝对总能量设为跨分子的统一合格线。
- 在氢原子策略通过第 5 节的测试和人工确认前，不把“金属配位必然去质子化”固化为规则。

任何失败都必须显式抛出带报告的异常；不得使用宽泛 `try/except`、静默忽略错误或无条件切换算法来制造“成功”结果。唯一允许的静默调整是用户明确要求的络合物力场解析 helper。

## 1. 当前真实调用链及问题边界

当前完整批处理入口为：

```text
hotpot/works/convert.py::_build3d
  -> Molecule.complexes_build_optimize_()
    -> forcefields.complexes_build()
      -> subprocess: _run_complexes_build()
         -> clone molecule
         -> hide metal-ligand bonds
         -> split ligand components
         -> repeated ob_build + ligand UFF optimization
         -> copy ligand coordinates into clone
      -> copy coordinates into caller molecule
    -> OBFF_(UFF).optimize(caller molecule)
```

另有两条不一致路径：

- `Molecule.build3d()` 对 `not self.is_organic` 的分子只调用 `complexes_build()`，缺少全体系 UFF 阶段。
- `Molecule.optimize_complexes()` 是另一套没有调用 `complexes_build()` 的旧实现。

规划基线上的直接定位如下；实施过程中行号会随修改变化，因此提交信息还应记录函数名：

| 位置 | 当前职责或缺陷 |
|---|---|
| `hotpot/cheminfo/forcefields.py:53-159` | 配体代理 worker |
| `hotpot/cheminfo/forcefields.py:103-108` | 克隆、隐藏金属键、拆分组件 |
| `hotpot/cheminfo/forcefields.py:111-122` | 硬编码 UFF，且异常分支再次调用同一个 UFF |
| `hotpot/cheminfo/forcefields.py:113-145` | 重试状态；第 126 行错误地执行 `+= 0` |
| `hotpot/cheminfo/forcefields.py:154-159` | 选配体最低能候选并回填代理坐标 |
| `hotpot/cheminfo/forcefields.py:162-222` | 公共 builder 与当前 Queue/timeout 协议 |
| `hotpot/cheminfo/forcefields.py:225-374` | `OBFF_` epoch 优化与错误约束映射 |
| `hotpot/cheminfo/forcefields.py:377-574` | 与 `OBFF_` 高度重复的 `OBFF` |
| `hotpot/cheminfo/core.py:870-901` | 当前 `build3d()` 分派不完整 |
| `hotpot/cheminfo/core.py:1145-1246` | 旧 `optimize_complexes()` 实现 |
| `hotpot/cheminfo/core.py:1248-1349` | 当前真正的两阶段络合物入口 |
| `hotpot/cheminfo/core.py:3242-3291` | 当前加氢及金属邻居扣减规则 |
| `hotpot/cheminfo/core.py:5509-5514` | 最近环边错误使用 `argmax` |
| `hotpot/works/convert.py:56-95` | 批处理对金属分子调用两阶段入口 |

因此必须先冻结以下定义：

- “配体代理构筑”是内部阶段，不是公开完成态。
- “完整络合物构筑”必须包含：代理配体构筑、恢复完整拓扑、全体系 UFF 优化、最终质量门控、原子化提交。
- 只有通过最终门控后，调用方传入的 `Molecule` 才允许被更新。

## 2. 目标架构

目标数据流如下：

```text
Molecule.build3d()
  |
  +-- mol.has_metal is False
  |     -> ob_build(mol)
  |     -> ob_optimize(mol, requested forcefield, requested steps)
  |
  +-- mol.has_metal is True
        -> complexes_build(mol, ...)
             1. capture immutable input/topology signature
             2. create full working clone before any H/topology operation
             3. build ligand proxies with metal-ligand edges hidden
             4. map proxy coordinates into the full working clone
             5. restore/retain the complete original graph
             6. resolve complex forcefield through tiny helper -> UFF
             7. run full-system epoch optimizer on working clone
             8. run global geometry quality gate
             9. commit accepted state atomically to caller molecule
```

### 2.1 为什么保留代理拆分

拆除金属–配体边后，`OBBuilder` 看到的是其更擅长处理的普通共价配体图。该步骤能避免金属连接关系干扰配体的环、杂化和初始三维构筑。历史提交 `7d5ab681` 从一开始就采用了 `copy -> break metal-ligand bonds -> build components`，所以“代理结构用于把配体送回普通力场适用域”的回忆与代码历史一致。

问题不在于“使用代理”，而在于当前代理契约不完整：

1. `mol.add_hydrogens()` 发生在创建代理之前，因此调用方本体已经被修改，代理没有保护氢拓扑。
2. 各配体独立构筑后没有完成围绕金属的可靠装配；多个组分可占据同一空间。
3. `complexes_build()` 自身在这里只返回代理坐标，没有执行全体系 UFF。
4. 只有 `complexes_build_optimize_()` 额外对本体执行全体系 UFF，而 `build3d()` 没有。
5. 全体系优化失败或结果不合格时，没有回滚和验收。

### 2.2 是否应在代理完成后对本体执行完整力场优化

化学意义上的答案是“是”：代理配体坐标回填之后，必须在恢复了全部金属–配体边的完整拓扑上执行 UFF，否则代理构筑结果不是完整络合物。

实现上不应直接、逐步修改用户传入的本体。推荐方式是：

1. 在“完整 working clone”上恢复原始拓扑；
2. 在该完整 working clone 上执行全体系 UFF；
3. 对 working clone 做最终门控；
4. 仅在成功后一次性把坐标、最低能量和按选项保存的构象提交给本体。

这在化学上等价于“用本体的完整结构优化”，同时具备事务语义：超时、异常或门控失败不会留下半优化、少氢或已破坏的调用方分子。

## 3. 公开入口与接口收敛

### 3.1 `Molecule.build3d()`

`build3d()` 只做明确分派，不再包含另一套构筑逻辑：

```python
if self.has_metal:
    return forcefields.complexes_build(self, ...)

forcefields.ob_build(self)
return forcefields.ob_optimize(self, forcefield=forcefield, steps=steps, ...)
```

要求：

- 判断条件必须是 `has_metal`，不能继续使用 `not is_organic`。
- 普通有机分子直接调用模块级 `ob_build()` 和 `ob_optimize()`。
- 络合物调用的 `complexes_build()` 必须已经是完整两阶段流程。
- `forcefield`、优化步数和所有认可的关键字参数必须真实传递。
- 未识别参数应显式报错，不得因 `**kwargs` 被静默丢弃。

### 3.2 唯一的络合物入口

`forcefields.complexes_build()` 成为唯一实现完整流程的函数。

现有入口的处理：

- `Molecule.complexes_build_optimize_()`：改为薄兼容代理，仅转发到 `forcefields.complexes_build()`，不保留独立实现；标记弃用。
- `Molecule.optimize_complexes()`：同样转发到唯一实现，或在确认没有外部调用后于后续主版本删除。
- `works.convert._build3d()`：统一调用 `mol.build3d()`，不再自行选择另一套络合物流程。
- `OBFF_` 与 `OBFF`：合并为一个内部 Open Babel 优化器实现；暂时保留旧类名作为直接别名，不复制逻辑。

弃用层只能负责参数名翻译和一次调用，不得捕获底层异常或改变结果。

### 3.3 络合物力场解析 helper

建立唯一且很小的纯函数，例如：

```python
def _resolve_complex_forcefield(requested: str | None) -> str:
    return "UFF"
```

规则：

- 当前 `None`、`UFF`、`MMFF94`、`MMFF94s`、`GAFF`、`Ghemical` 均静默解析为 `UFF`。
- 解析只发生一次，得到的值必须继续传入配体候选优化、候选精修和完整体系优化。
- 不允许各层再次硬编码或各自重新选择力场。
- 将来出现经过验证的络合物力场时，只修改该 helper 和对应测试矩阵。
- 运行报告同时记录 `requested_forcefield` 和 `effective_forcefield`，静默调整不等于隐藏事实。

普通有机分子不经过此 helper，用户指定的兼容力场应原样传给 `ob_optimize()`。

## 4. 进程、异常与重试修补

### 4.1 子进程结果协议

禁止继续使用“先 `join()`、后无条件阻塞 `queue.get()`”的协议。大对象还可能因 Queue feeder 未排空而形成即使成功也死锁的情况。

定义结构化结果信封：

```text
BuildWorkerResult
  status: "ok" | "error"
  coordinates: ndarray | None
  conformers: serialized conformer data | None
  diagnostics: build attempts, rejected candidates, timings
  error_type: str | None
  error_message: str | None
  traceback: str | None
```

推荐使用单向 `multiprocessing.Pipe`：父进程先 `poll(timeout)`，有消息后 `recv()`，最后 `join()`；不得在读取大结果前等待子进程完全退出。

父进程必须在 `finally` 中完成：

1. 若子进程仍存活，`terminate()`；
2. 有界 `join()`；
3. 关闭父子通信端；
4. 检查 `exitcode`；
5. 将子进程错误重新抛为带原始类型、消息和 traceback 的 `ComplexBuildWorkerError`。

使用 `try/except` 仅限跨进程传输真实异常，不得把失败转换成另一种计算路径。

### 4.2 重试语义

- 修正 `rebuild_time += 0`。
- 统一命名为 `attempt_count` 和 `max_attempts`。
- 每次调用 `ob_build()` 都消耗一次 attempt，无论失败原因是穿环、力场初始化失败还是门控拒绝。
- `accepted_candidates` 与 `attempt_count` 分开统计。
- 达到上限时抛出 `ComplexBuildError`，报告每次拒绝的具体原因。
- 不再根据 `lst_energy` 长度重置失败计数器。
- `build_times` 改名为 `candidate_count`；旧名只在兼容入口转换一次。

### 4.3 超时和并发

- `timeout` 必须从 `Molecule.build3d()` 穿透到子进程管理层。
- 同时运行多个构筑任务时，每次调用拥有独立 Pipe、随机数生成器、临时状态和 Open Babel 力场对象。
- 不使用模块级可变优化器或全局 NumPy seed。
- 有 seed 时优先使用 `spawn` context；worker 启动后在第一次调用 `OBBuilder` 前设置 `OB_RANDOM_SEED`。
- 同一 seed 只承诺在相同 Hotpot/Open Babel/平台版本内重现；不承诺跨 Open Babel 版本逐位一致。

### 4.4 纯程序冗余清单

以下内容与化学语义无关，应在对应提交中直接清理：

- 删除“先运行 UFF，捕获 `RuntimeError` 后再运行同一个 UFF”的伪 fallback。
- 删除依赖 `current_length` 的调试 `print(min/mean/max energy)`；诊断写入结构化报告。
- 删除不被消费的 `**kwargs`，或者把它替换为明确的命名参数。
- 删除成功进程已经退出后才执行的无意义 `terminate()`。
- 删除两套优化器中重复的 setup、perturb、constraint 和循环代码。
- 删除 `optimize_complexes()` 中注释掉的调用及先设置 constraint、随后由错误映射消费的死逻辑。
- 将 `print(UserWarning(...))` 改为真正的薄入口转发；不再用标准输出表达控制流。
- 公共函数返回统一的运行报告；调用方如果不接收返回值，原有原位更新行为仍成立。

## 5. 代理分子、氢原子和化学身份

### 5.1 当前问题的准确机制

当前顺序是：

```text
caller mol.add_hydrogens()
caller mol.refresh_atom_id()
spawn worker with caller molecule
worker clone = copy(mol)
clone.hide_metal_ligand_bonds()
```

所以克隆只隔离了之后的隐藏键和配体构筑，并没有隔离加氢操作。

`Atom._add_hydrogens()` 对 O 或芳香 N 先按隐式氢计算数量，再减去金属邻居数。`rm_polar_hs=False` 只阻止“删除已经存在的氢”分支，不能撤销这次减法，也不能补回因此没有添加的氢。

### 5.2 必须保留的设计

- 保留完整工作克隆。
- 保留在工作克隆上隐藏金属–配体边、拆分普通配体、反复 `ob_build()` 的策略。
- 使用稳定 atom ID 将配体代理坐标映射回完整工作克隆。
- 在完整工作克隆上恢复原始金属–配体拓扑并执行全体系 UFF。
- 失败时丢弃整个 working clone，不修改本体。

### 5.3 具体质子化案例

必须建立以下回归案例并同时记录优化前后的元素计数、显式氢数、形式电荷和键集合：

| 输入化学身份 | 添加金属配位键后的当前行为 | 风险 |
|---|---|---|
| `O` 代表水，O–Zn | O 原本应有 2 H；金属邻居使待添加数量减 1，结果为 HO–Zn | 中性水配体被静默变成羟基型配体，形式电荷却未同步 |
| `CO` 代表甲醇，O–Zn | O 原本应补 1 H；减去一个金属邻居后补 0 H，结果为 CH₃O–Zn | 甲醇被变成甲氧基型结构 |
| 苯酚 O–金属 | 与甲醇相同 | 中性酚配体可能被静默去质子化 |
| 羧酸羟基 O–金属 | O–H 可能被删除或不再补回 | 配体电荷和配位模式被改变 |
| 氨 `N`–Zn | 非芳香 N 不属于当前 `polar_hydrogen_site`，通常仍得到 NH₃ | O 与普通 N 的规则不一致 |
| `[nH]`–金属 | 芳香 N 属于 polar site | 可能删除关键芳香氢并改变芳香/电荷语义 |
| 羰基 O–金属 | O 的正常隐式氢为 0 | 不应凭配位键再改变氢数，是负对照 |

每个案例都要覆盖：

- 隐式氢输入；
- 显式氢输入；
- `rm_polar_hs=True/False`；
- 单齿与双齿连接；
- 代理阶段失败时本体完全不变。

### 5.4 推荐氢策略及审批点

推荐引入明确的 `hydrogen_policy`，不再让布尔值同时承担“补氢”和“去除极性氢”两种语义：

- `preserve`：默认候选。严格保留调用方显式原子、形式电荷和键图；代理不得用金属邻居数推断去质子化。
- `complete`：只在 working clone 上按未配位的共价配体图补足隐式氢；新增 H 带 provenance。是否把新增 H 提交给本体由公开契约明确决定。
- `deprotonate`：未来显式、可审计的化学操作；必须由位点规则或用户指定，不能由“存在金属邻居”自动触发。

在实施化学行为改变前需要确认两点：

1. `build3d()` 的返回分子是否应默认显式补齐隐式 H；
2. `complete` 模式新增的代理 H 是否应提交到本体，还是只用于优化后丢弃并回填原始原子的坐标。

在这两个问题确认前，先完成进程、API、优化器和质量门控修复，并通过测试冻结当前差异；不得继续扩大自动去质子化行为。

## 6. 解结与完整体系优化

### 6.1 最近环边

把 `Ring.closest_edge_to_bond()` 中的 `np.argmax` 改成 `np.argmin`，除此之外不改变该对象接口。

测试必须使用几何上距离不同的六元环和探针键，明确断言返回距离最小的边，不能只断言返回值属于环。

### 6.2 环语义

解结与最终门控使用已有 `Molecule.ligand_rings`，即移除金属–配体边后的配体骨架环；不能把螯合形成的金属环当成待拆的有机环。

阶段检查：

- 配体代理阶段：检查配体内部共价键穿过 `ligand_rings`。
- 完整体系阶段：检查所有非环键，包括金属–配体键，是否穿过 `ligand_rings`。
- 不得在完整体系阶段隐藏金属–配体键后再做最终检查。

### 6.3 全体系优化

配体代理完成后，在完整 working clone 上运行 UFF。该阶段负责：

- 消除不同配体组件之间的空间重叠；
- 使金属–供体键长度进入 UFF 的局部极小区域；
- 在不改变图拓扑的情况下松弛配体相对方位；
- 为质量门控提供能量、梯度、收敛和 epoch 轨迹。

UFF 能得到局部极小值不等于结构具有正确配位化学。ZnCl₂ 得到 109.47°而 PtCl₄ 得到平方平面，说明当前不得把 UFF 输出直接解释为已验证的配位几何。几何模板/配位场模型属于后续化学增强，不纳入本轮静默修补。

## 7. 优化循环、参数命名与结果选择

### 7.1 参数重命名

当前 `OBFF_.steps` 是外层记录循环，`step_size` 是每轮交给 Open Babel 的真实优化步数。目标名称：

| 旧名称 | 新名称 | 语义 |
|---|---|---|
| `steps` | `epochs` | 构型优化/记录循环次数 |
| `step_size` | `steps_per_epoch` | 每个 epoch 内 Open Babel 执行的优化步数 |
| `perturb_steps` | `perturb_interval` | 每隔多少 epoch 扰动一次；`None` 表示不扰动 |
| `save_screenshot` | `save_movie` | 是否保存所有 epoch 帧 |
| `build_times` | `candidate_count` | 每个配体需要接受的候选构象数 |
| `init_opt_steps` | `candidate_warmup_steps` | 每次候选初始松弛步数 |
| `second_opt_steps` | `candidate_score_steps` | 合格候选用于评分的步数 |
| `min_energy_opt_steps` | `best_candidate_refine_steps` | 最佳配体候选的最终精修步数 |

旧参数只在一个兼容层翻译。新旧名称同时传入且值冲突时必须报错，不能猜测优先级。

### 7.2 标准 Open Babel 迭代接口

不再每个 epoch 重新调用一次 `ConjugateGradients(n)` 或 `SteepestDescent(n)`。使用 Open Babel 的标准分段接口：

1. `ConjugateGradientsInitialize(total_steps, energy_tolerance)` 或对应 steepest 初始化；
2. 每个 epoch 调用 `*TakeNSteps(steps_per_epoch)`；
3. 读取坐标、能量和梯度；
4. 无扰动时保留优化器状态；
5. 发生扰动后重新 `Setup` 和初始化优化器，因为目标坐标已被外部修改。

每个 epoch 的顺序冻结为：

```text
optional scheduled perturbation
  -> force-field setup/reinitialize when needed
  -> optimize steps_per_epoch
  -> retrieve coordinates, energy, gradient and backend status
  -> quality observation
  -> update best frame
  -> optionally append movie frame
```

### 7.3 随机种子

新增 `seed: int | None = None`：

- `None` 保持业务默认的随机探索。
- 使用 `np.random.default_rng(seed)`，不得修改 NumPy 全局随机状态。
- seed 传入 worker；worker 在首次 `OBBuilder` 调用前设置 `OB_RANDOM_SEED`。
- Open Babel 的 `vector3::randomUnitVector()` 使用静态 PRNG，因此有 seed 的 worker 应使用 `spawn` 启动，避免继承父进程中已初始化的 PRNG 状态。
- 测试只要求同版本、同平台、相同 seed 在容差内一致。
- 扰动应使用双侧 `clip(-2*sigma, +2*sigma)`；不能只裁剪正尾部。

### 7.4 最低能帧与 movie

- 始终只维护一个 `best_coordinates`/`best_energy`，因此默认内存为 O(atom_count)。
- 默认 `save_movie=False`：最终 `mol.coordinates` 和 `mol.energy` 为通过门控的最低能帧；构象容器最多保存该一帧。
- `save_movie=True`：保存每个 epoch 的坐标、统一单位后的能量和质量指标；最终活动坐标仍必须是最低能帧，而不是最后一帧。
- 原始 Open Babel 能量单位必须从 `GetUnit()` 读取并记录；比较不同后端前显式换算，禁止默认假设所有力场单位相同。

### 7.5 VDW 插值与收敛

VDW cutoff 使用：

```text
start + (epoch + 1) / epochs * (end - start)
```

最终值必须严格等于 `end`。移除当前基于多维数组和 Python `max()` 的平衡判断，统一使用：

- Open Babel 分段优化返回状态；
- 能量变化；
- 从 `Energy(True)`/`GetGradient(atom)` 得到的 RMS 与最大梯度；
- 第 9 节质量门控的收敛等级。

## 8. 清理约束实现

当前 Hotpot 0-based 索引被直接传给 Open Babel 1-based 约束接口，已经确认会约束错误原子。

本轮处理方式：

- 删除 `OBFF_` 和 `OBFF` 内把 Atom/Bond/Angle/Torsion flags 翻译为 Open Babel constraints 的循环。
- 保留一个稳定的 `_make_constraints(mol)` 或 `_add_constraints(mol)` 空接口。
- 空接口只返回一个空 `ob.OBFFConstraints()`，不得读取对象上的 constraint flags。
- `Setup` 仍可接收这个空容器，避免以后重新加入正确实现时改变优化器调用结构。
- 暂不删除 `Atom.constraint`、`Bond.constraint` 等核心数据字段，避免破坏序列化和外部 API；文档明确本轮力场后端不执行这些约束。
- 删除 `optimize_complexes()` 中设置非金属原子 constraint 的死逻辑。

重新启用约束必须作为独立功能提交，并至少具备 atom、distance、angle、torsion 的 0/1-based 映射测试。

## 9. 标准化几何质量门控

### 9.1 外部实现调研结论

本轮以以下标准代码库作为设计参考，但不增加新的运行时后端：

1. Open Babel：
   - `Setup()` 提供力场是否能初始化；
   - `Energy()`/`GetUnit()` 提供能量及单位；
   - `GetGradient()` 可计算逐原子梯度；
   - `ConjugateGradientsInitialize/TakeNSteps` 和 steepest 对应接口支持分段优化；
   - `DetectExplosion()` 的官方实现只检查坐标是否 finite 以及是否存在长度超过 30 Å 的键，所以只能作为最低级信号，不能独立判断化学合理性。
2. RDKit：
   - `UFFHasAllMoleculeParams()` 在优化前检查参数覆盖；
   - UFF minimizer 明确返回 `0=converged`、`1=needs more iterations`；
   - 该模式说明参数覆盖和收敛状态应成为正式结果，而不是日志文本。
3. OpenMM：
   - `LocalEnergyMinimizer` 以所有力分量的 RMS 作为停止标准，默认 10 kJ mol⁻¹ nm⁻¹，即 1 kJ mol⁻¹ Å⁻¹；
   - 约束误差与势能分开报告。
4. ASE：
   - 通常以最大原子力 `fmax` 判断收敛；
   - 说明仅看相邻两步能量不足以判定几何已经稳定。

参考源码：

- <https://github.com/openbabel/openbabel/blob/master/src/forcefield.cpp>
- <https://github.com/openbabel/openbabel/blob/master/include/openbabel/forcefield.h>
- <https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ForceFieldHelpers/UFF/UFF.h>
- <https://github.com/rdkit/rdkit/blob/master/Code/ForceField/ForceField.h>
- <https://github.com/openmm/openmm/blob/master/openmmapi/include/openmm/LocalEnergyMinimizer.h>
- <https://ase-lib.org/ase/optimize.html>

### 9.2 数据结构与纯函数接口

建议新增独立模块 `hotpot/cheminfo/geometry_quality.py`，避免继续扩大 `forcefields.py`：

```text
QualityLevel = "off" | "basic" | "standard" | "strict"

GeometryCheck
  name
  passed
  severity
  measured
  threshold
  atom_indices / bond_indices
  message

GeometryQualityReport
  level
  passed
  checks
  metrics

evaluate_geometry_quality(
    mol,
    *,
    level,
    topology_reference=None,
    forcefield_report=None,
    thresholds=None,
) -> GeometryQualityReport
```

该函数必须是只读纯函数：不加氢、不改键、不重算并写回价态、不调用优化器。

另由优化器产生：

```text
ForceFieldRunReport
  requested_forcefield
  effective_forcefield
  setup_succeeded
  converged
  epochs_completed
  steps_completed
  final_energy
  best_energy
  energy_unit
  rms_gradient
  max_gradient
```

### 9.3 不可关闭的运行完整性检查

即使 `quality_level="off"`，以下内容也属于程序正确性，不允许关闭：

- 返回坐标 shape 与原子数一致；
- 全部坐标、能量和梯度为有限数；
- worker 正常完成且消息协议完整；
- 按所选 hydrogen policy，原子 ID、元素、形式电荷、键端点和键类型满足拓扑契约；
- Open Babel `Setup()` 成功。

### 9.4 分层门控

#### `off`

只执行上述不可关闭检查。用于研究性诊断，不代表结构合格。

#### `basic`

增加：

- Open Babel `DetectExplosion()` 必须为 false；
- 任意两个不同原子不得具有近零距离；
- 所有显式键长度必须为有限正数且不得超过 Open Babel 的 30 Å explosion 上限；
- 不得有明显的全原子硬碰撞，初始建议绝对下限 0.40 Å；
- 最终能量必须有限，但不设置绝对能量上限。

#### `standard`（目标默认）

包含 basic，并增加：

- 对非键原子使用共价半径和绝对下限结合的碰撞检查；初始候选规则为 `distance >= max(0.50 Å, 0.55*(r_cov_i+r_cov_j))`；
- 普通共价键长度与共价半径和之比处于宽松区间，初始候选 `[0.65, 1.45]`；
- 金属–配体键使用独立的宽松区间，初始候选 `[0.65, 1.60]`；
- 所有非环键，包括金属–配体键，不得穿过 `ligand_rings`；
- 记录收敛状态、RMS 梯度和最大梯度；未达到严格梯度阈值可作为 warning，但不得掩盖几何硬失败；
- 输入拓扑和质子化状态必须符合选定 hydrogen policy。

这些半径比例只用于捕获灾难性结构，不宣称验证配位化学正确性。阈值必须先用第 11 节语料校准，避免误杀镧系、长配位键或多中心键。

#### `strict`

包含 standard，并增加：

- 后端必须报告已收敛；
- 建议 RMS 梯度上限从 OpenMM 默认量级开始校准：1 kJ mol⁻¹ Å⁻¹；
- 建议最大梯度参考 ASE 常用量级开始校准：约 5 kJ mol⁻¹ Å⁻¹；
- 最后若干无扰动 epoch 的能量和坐标变化必须处于配置阈值内；
- 所有 standard warning 升格为 failure。

严格阈值在基准集完成统计前不得作为默认值。能量/梯度单位必须先统一为 kJ mol⁻¹ 和 kJ mol⁻¹ Å⁻¹。

### 9.5 门控位置

门控分三次执行：

1. 配体候选门控：用于拒绝穿环或爆炸的单个候选，并消耗一次 attempt。
2. 完整体系 epoch 观察：只记录指标并更新最低能合格帧，不在中途偷偷切换后端。
3. 最终门控：对最低能候选执行所选等级；失败则抛出 `GeometryQualityError(report)`，本体保持原样。

如果最低能帧未通过门控，而较高能帧通过，选择“最低能且通过门控”的帧；若没有任何帧通过则失败。不能先选最低能帧后忽略其结构错误。

## 10. 文件级修改清单

### `hotpot/cheminfo/forcefields.py`

- 抽出 `_resolve_complex_forcefield()`。
- 把 `_run_complexes_build()` 改为只处理 working clone 并返回结构化结果。
- 修复 attempt 计数和最近环边调用。
- 替换 Queue/join 协议，保证异常、超时和大结果均可终止。
- 合并 `OBFF_`/`OBFF` 的重复优化循环。
- 使用 epoch/steps-per-epoch 命名及 Open Babel 分段优化接口。
- 增加局部 RNG、最低能合格帧和 `save_movie`。
- 修复 VDW 线性插值。
- 清空 constraint adapter。
- `complexes_build()` 承担完整 working-clone UFF 与最终门控。

### `hotpot/cheminfo/core.py`

- `build3d()` 改为纯分派。
- `complexes_build_optimize_()` 与 `optimize_complexes()` 收敛为薄兼容入口。
- 修复 `Ring.closest_edge_to_bond()` 的 `argmax -> argmin`。
- 不在本轮删除 constraint 数据字段。
- 不在未批准前改写通用价态或氢规则。

### `hotpot/cheminfo/geometry_quality.py`（新增）

- 定义质量等级、报告对象、阈值对象和纯门控函数。
- 只依赖 NumPy/SciPy 与 Hotpot 自身图和元素半径数据。
- 不导入 RDKit/OpenMM/ASE；这些库仅作为设计参考。

### `hotpot/works/convert.py`

- 删除独立的金属分支，统一调用 `mol.build3d()`。
- 明确 `write_single` 与 `save_movie` 的关系。

### 测试

新增：

```text
tests/test_cheminfo/test_forcefield_optimizer.py
tests/test_cheminfo/test_complexes_build.py
tests/test_cheminfo/test_geometry_quality.py
tests/test_cheminfo/test_complex_hydrogens.py
tests/test_cheminfo/fixtures/complexes/
```

所有运行产物写入 pytest `tmp_path`，不得污染源码树。

## 11. 必须完成的测试矩阵

### 11.1 分派与参数穿透

- 乙醇：`build3d()` 精确调用一次 `ob_build` 和一次 `ob_optimize`。
- 苯：用户指定 UFF/MMFF94s 均原样传入普通有机路径。
- Zn–乙二胺：精确调用一次完整 `complexes_build`。
- 络合物请求 MMFF94s：运行报告显示 requested=MMFF94s、effective=UFF。
- `epochs`、`steps_per_epoch`、timeout、quality level、seed 全部抵达实际消费位置。
- 传入未知参数必须失败，不能静默丢弃。

### 11.2 子进程

- worker 内人为抛出异常：父进程在有界时间内得到 `ComplexBuildWorkerError`。
- dative bond导致 Open Babel 转换失败：不得挂起。
- 极短 timeout：子进程终止并 join；`multiprocessing.active_children()` 不得残留该 PID。
- 发送大坐标数组：不得发生 join-before-receive 死锁。
- 连续和并发执行至少 20 个小体系：结果与进程互不串扰。

### 11.3 计数与解结

- 持续失败候选恰好执行 `max_attempts` 次后终止。
- 成功候选数与总尝试数分别统计。
- 人工六元环测试返回真正最近边。
- 螯合金属环不作为 ligand ring 拆除。
- 金属–配体键穿过真实配体环时，最终门控必须失败。

### 11.4 优化器

- `epochs=3, steps_per_epoch=7` 实际执行不超过 21 个后端步骤。
- backend 提前收敛时停止并报告实际步数。
- VDW cutoff 首值、单调性和末值正确。
- 同一 seed 在同版本环境结果一致；`seed=None` 不要求不同，也不要求相同。
- 默认结果为最低能合格帧，不是最后帧。
- `save_movie=False` 不保存 500 帧；`save_movie=True` 帧数等于已完成 epoch 数。
- UFF 能量报告为 kJ/mol；若普通有机路径使用 MMFF，单位必须被显式记录/换算。

### 11.5 质量门控

最少包括：

- 正常乙醇、苯、环己烷；
- Zn–乙二胺；
- Zn(NH₃)₄；
- ZnCl₂ 和 PtCl₄，作为“几何模板不可由通用 gate 保证”的记录案例；
- README 的 68 原子 Eu–配体实例；
- 人工 0.05/0.12 Å 原子重叠；
- 非有限坐标；
- 断裂到 30 Å 以上的显式键；
- 配体内部共价键穿环；
- 金属–配体键穿过配体环；
- 合法螯合环，不得误报为有机环穿越。

每个失败报告必须包含检查名称、实测值、阈值和相关 atom/bond indices。

### 11.6 氢和拓扑事务

执行第 5.3 节全部案例，并另外断言：

- worker 异常前后本体的原子数、原子 ID、形式电荷和键集合完全一致；
- timeout 前后本体完全一致；
- 最终门控失败前后本体完全一致；
- 成功时只提交契约允许的字段。

### 11.7 回归与性能

- README Eu 默认流程必须通过 standard gate；记录运行时间、最终 Eu–供体距离和能量，但不对随机运行做逐位断言。
- 有 seed 的基线使用距离/能量容差比较。
- 小型 Zn 体系作为 CI smoke；Eu 案例标记为 slow，防止普通 CI 超时。
- 对比修改前后峰值内存，验证默认不再储存完整 movie。
- 执行项目完整测试和 Python 3.9–3.14 兼容脚本。

## 12. 分提交实施顺序

每一步独立提交且测试通过后再进入下一步：

1. `test(complexes): capture current build and failure contracts`

   只加入最小复现、测试 fixture 和预期失败标记。
2. `fix(complexes): make worker failures and timeouts deterministic`

   修复 Pipe 协议、异常传播、终止与 join。
3. `fix(complexes): bound rebuild attempts and select nearest ring edge`

   只修计数器和 `argmin`，便于回溯。
4. `refactor(forcefields): unify optimizer and normalize parameters`

   合并重复类，加入力场 helper，完成 epoch 参数迁移和空 constraint adapter。
5. `feat(forcefields): add seeded epoch optimization and best-frame output`

   加入局部 RNG、正确 VDW 插值、标准分段优化、最低能帧及 movie。
6. `feat(cheminfo): add layered geometry quality gates`

   新增纯门控模块和结构化报告。
7. `refactor(complexes): make proxy build transactional and globally optimized`

   working clone 全流程、最终门控、成功后原子化提交。
8. `refactor(core): make build3d the canonical dispatcher`

   收敛三个公开入口和 `works.convert`。
9. `test(complexes): validate hydrogens, metals, failures, and regression matrix`

   完成完整矩阵；氢策略只按已批准结论实现。
10. `docs(complexes): document forcefield and quality contracts`

    更新公开 API、单位、随机性和限制。

不得把全部修改压成一个提交；任何化学语义调整必须与纯程序修复分开提交。

## 13. 每阶段验收条件

### 阶段 A：程序安全

- 不再存在无限 Queue 阻塞。
- timeout 后没有存活 worker。
- 重试次数严格有界。
- 子进程异常保留原始诊断。

### 阶段 B：优化器正确性

- 参数命名与实际循环一致。
- 请求力场和有效力场可追踪。
- 约束不再错误地作用于相邻原子。
- 默认返回最低能合格帧。
- seed 行为符合声明。

### 阶段 C：络合物流程完整性

- `build3d()` 的金属入口必经完整体系 UFF。
- 代理失败不会污染本体。
- 完整结构经过全局门控。
- README Eu 实例通过，已知穿环/重叠实例被拒绝。

### 阶段 D：化学身份

- 氢、形式电荷和拓扑行为有明确、可测试的公开契约。
- 未经显式策略不再把水/醇静默变成羟基/醇盐型配体。
- 不宣称 UFF 已验证金属配位几何或氧化态。

## 14. 回滚与禁止事项

- 每个提交必须可独立回滚。
- 不删除历史输入 fixture。
- 不以放宽断言、吞异常、返回最后一次尝试或自动换路径解决失败。
- 不在门控失败后把不合格坐标写回本体。
- 不以“UFF 能算出有限能量”代替结构质量判断。
- 不把 RDKit 的成功结果当作 Hotpot/Open Babel 的 oracle；参考实现只用于验证设计原则和非金属差分测试。
- 不在本轮加入未经标定的配位几何模板。

完成全部阶段后，最终报告必须给出：提交列表、测试命令、通过/失败/跳过数量、基准结构指标、已知化学边界及仍需人工决定的 hydrogen policy。
