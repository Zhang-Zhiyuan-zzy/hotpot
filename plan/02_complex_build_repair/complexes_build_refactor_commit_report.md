# `complexes_build` 重构提交复盘报告

## 1. 报告范围

- 分支：`fix/complexes-build-pipeline`
- 起始基线：`41099db`
- 当前终点：`51f70a0`
- 提交范围：`41099db..51f70a0`
- 提交数量：65
- 总体差异：30 个文件，新增 10,305 行，删除 1,085 行

本报告不是只根据 commit message 做摘要，而是结合每个提交的实际 diff、后续测试暴露的问题和后继修正，复盘完整修改过程。文中的“问题”既包括重构前已经存在的问题，也包括本轮实现后通过动态测试、并发测试和真实 Open Babel 运行新发现的问题。

## 2. 总体演进脉络

本轮工作大致经历了七个阶段：

1. 先把用户确认的 API、化学边界、事务语义和测试要求固化到计划。
2. 将几何判断集中到 `geometry.py`，并把 `Molecule` 收束为薄门面。
3. 重写 `forcefields.py`，建立 working copy、子进程、分段优化、质量门控和结构化报告。
4. 用测试补齐身份保持、氢拓扑、并发、随机种子和真实 Eu 络合物基线。
5. 动态测试持续暴露 Open Babel 的步数、终止、随机数和插件实例语义问题，逐项修正。
6. 补强原子化提交、失败回滚和 worker/批处理诊断，消除“失败但污染调用方”与“只知道失败、不知道为什么”的情况。
7. 收束公开接口、统一工作流报告，并修正文档与测试标准自身的矛盾。

最明显的来回调整不是无目的反复，而是以下几类运行时事实只有在真实后端或高并发测试下才能确认：

- Open Babel 的 `MakeNewInstance()` 看似可以隔离力场实例，但真实反复 `Setup()` 会触发不稳定甚至崩溃，因此最终改为稳定插件句柄加串行锁。
- 收到 worker 的 Pipe 结果不代表进程已经完全退出；最初的 `join`、宽限期和 sentinel 检测又经过多次细化。
- Open Babel 的 `TakeNSteps(False)` 同时可能表示收敛或碰到初始化上限，且初始化自身可能消耗一步，不能直接当作普通循环计数。
- Eu 络合物的 UFF 绝对能量跨 Open Babel 版本漂移明显，测试最终从固定能量常数改为结构、拓扑和质量门控不变量。
- 初版补氢修正只覆盖 N/O，真实配体路径表明 P/S/As/Se 也需要按金属键隐藏后的共价骨架重算。

## 3. 按提交逐项复盘

### 阶段 A：规划和边界冻结

#### 1. `0696c21 docs(plan): specify complexes build pipeline repair`

- 修改了什么：新建 `plan/complexes_build_repair.md`，首次完整列出旧调用链、代理配体流程、worker 协议、重试计数、优化 epoch、质量门控、测试矩阵和分阶段提交方案。
- 为什么修改：前期实测已经确认旧 `complexes_build` 同时混有构筑、优化、约束、异常处理和对象写回，且存在计数器不增长、超时后进程残留、参数没有实际消费、失败污染本体等问题。先冻结修补边界，避免直接在大函数中继续打补丁。

#### 2. `b717c7c docs(plan): refine forcefield API and chemistry boundaries`

- 修改了什么：细化 `Molecule.build3d()`/`Molecule.optimize()` 薄门面与 `forcefields` 功能层；明确默认加氢、working clone、完整体系 UFF、配位几何预留 hook、力场解析 helper、`epochs`/`steps_per_epoch`、最低能帧和 `save_movie` 等契约。
- 为什么修改：用户进一步明确了三点：优化前默认加氢是正式行为；代理拆分是为了让配体进入有机力场目标域，但代理完成后仍必须恢复本体完整拓扑再优化；`Molecule` 不应继续承载实际业务。原计划在这些边界上还不够明确，因此先修正文档再实现。

#### 3. `e0c4aa5 docs(plan): centralize geometry validation`

- 修改了什么：把重叠、过近、键穿环、拓扑保持和综合质量门控统一规划到现有 `hotpot/cheminfo/geometry.py`；规定 `core.py` 只保留别名接口，`forcefields.py` 直接调用 `geometry`；补充 `off/basic/standard/strict` 分层语义。
- 为什么修改：早期代码在 `core`、力场流程和几何模块中重复实现相近判断，且布尔结论缺少可诊断的距离、阈值和索引。用户要求复用现有总体抽象，不能再造一个平行的 geometry-quality 模块。

### 阶段 B：几何、核心门面和转换入口

#### 4. `9f8a746 refactor(geometry): centralize structural validation`

- 修改了什么：在 `geometry.py` 增加结构化的 `GeometryCheck`、阈值、拓扑快照和质量报告；实现原子重叠、过近、键穿环、拓扑对比、力场报告聚合和 `evaluate_geometry_quality()`；新增两组几何测试。
- 为什么修改：此前 `Molecule.has_bond_ring_intersection` 等逻辑分散、返回值贫乏，无法为候选拒绝和最终质量门控提供统一证据。该提交建立后续力场重构所需的只读几何基础设施。

#### 5. `354daa8 fix(core): make topology traversal deterministic`

- 修改了什么：调整 `Molecule.components`、隐藏键恢复及相关拓扑遍历的顺序，保证按稳定原子/键索引返回；新增确定性回归测试。
- 为什么修改：代理配体拆分和坐标回填依赖位置对应。原实现受 set/图遍历顺序影响，同一结构可能产生不同组件或键顺序，既破坏 seed 复现，也可能把坐标写回错误原子。

#### 6. `72579bc feat(geometry): complete layered structure quality gates`

- 修改了什么：完善平面与凹多边形判定、非平面环面语义、布尔检查的短路、ligand-skeleton ring scope、严格级力场诊断和 core 委托测试。
- 为什么修改：第一版几何实现通过基础案例后，扩展测试发现平移后的共面判断、凹环中心扇面、环边界以及 strict 缺失诊断的处理并不完整。该提交是对首版几何算法的实质补正。

#### 7. `38fc7dd refactor(core): expose canonical forcefield facades`

- 修改了什么：把 `Molecule.build3d()` 和 `Molecule.optimize()` 改为调用 forcefield 层的标准门面；删除 `optimize_complexes()`、`complexes_build_optimize_()` 等旧业务实现；保留必要的几何便捷属性和 ring helper；调整加氢随机源接口。
- 为什么修改：旧 `Molecule` 同时实现分派、补氢、代理构筑、约束和优化，导致同一功能多入口、多套语义。按照计划，核心对象只负责标准接口和自动分派，实际功能必须下沉到 `forcefields.py`。

#### 8. `892729f fix(convert): propagate build timeout and reap workers`

- 修改了什么：重构 `works/convert.py` 的 `_build3d` 和批量转换进程管理；将 timeout 传给实际构筑；默认只写最终帧，`save_movie` 才写轨迹；增加 terminate、kill、join 的清理路径及测试。
- 为什么修改：旧转换层的 timeout 没有可靠抵达内部构筑，超时后也可能遗留子进程；同时每轮帧输出语义与新的“默认最低能帧”契约冲突。这是转换入口与新力场接口的第一轮对齐，后面又进一步补强了诊断和 spawn 隔离。

### 阶段 C：主力场流程重构

#### 9. `30e3ff6 refactor(forcefields): unify complex build pipeline`

- 修改了什么：大规模重写 `forcefields.py`。加入结构化报告/异常、working-copy 补氢、代理配体构筑、配位环境收集、通用 Open Babel 优化器、分段 epoch、VDW schedule、局部随机扰动、最低能合格帧、worker Pipe 协议，以及 `build3d`、`optimize`、`build_complex3d`、`optimize_complex`、`complexes_build`、`build_and_optimize`、`auto_optimize` 等功能层入口；清空错误约束映射，只保留空 adapter。
- 为什么修改：旧代码由 `OBFF_`、`OBFF`、`OBBuilder` 和多个 `Molecule` 方法重复承担相同工作，步骤命名混乱，完整络合物恢复后没有统一的全体系事务优化，也缺少可观察报告。这是本轮的主体实现，但其后大量小提交是在真实测试中校正这里首次建立的边界语义。

#### 10. `d790cc5 fix(forcefields): preserve caller objects on commit`

- 修改了什么：增加面向 worker 的干净结构代理和更细粒度的 working-copy 提交；成功时更新原始 Atom/Bond 对象而不是整体替换；补充水、醇、双齿配体、失败不变性和对象 identity 测试。
- 为什么修改：主体重构的首版成功写回可能替换调用方持有的 Atom/Bond 对象，破坏外部引用、自定义 atom id 和缓存；失败路径也需要确保本体不被 working copy 的补氢或拓扑变化污染。

#### 11. `26963bd fix(forcefields): honor optimization step budgets`

- 修改了什么：显式记录 `steps_submitted` 与 `initialization_steps`；把共轭梯度初始化消耗的一步纳入总预算；新增模拟 Open Babel 后端的优化器测试矩阵。
- 为什么修改：Open Babel 的共轭梯度初始化本身会执行一步。首版按 `epochs × steps_per_epoch` 再提交全部 TakeNSteps，会超过用户预算，并把无法观测的实际完成步数伪装为精确计数。

#### 12. `de69876 fix(forcefields): harden worker process protocol`

- 修改了什么：加强 `_receive_worker_result()`、worker 坐标形状/有限性校验、异常序列化、超时终止和大消息 Pipe 接收；新增大量协议、事务、重试、配位环境和进程清理测试。
- 为什么修改：先 join 再读取大 Pipe 消息可能发生 feeder/pipe 死锁；worker 还可能返回畸形对象、不完整成功结果或错误坐标。必须先接收、验证协议，再有界回收进程，并保留原始异常诊断。

#### 13. `65a0b68 fix(forcefields): preserve requested forcefield in reports`

- 修改了什么：修正 `auto_optimize()` 的参数转发，使报告同时保留调用者请求的力场和内部实际采用的力场；新增 API 契约测试。
- 为什么修改：络合物必须由 helper 静默解析为 UFF，但报告不能因此把用户原始请求覆盖掉，否则无法判断一次运行是否发生过后端调整。

#### 14. `422511b fix(forcefields): refine viable candidates deterministically`

- 修改了什么：候选构筑与评分改为确定性顺序；被初步判为可行但精修失败时继续尝试下一个候选；环边按稳定端点顺序隐藏；补充相应测试。
- 为什么修改：首版可能只精修单个候选并在失败后直接结束，浪费已经生成的其他可行候选；交叉环边处理还受遍历顺序影响。这会造成不必要的构筑失败和 seed 不稳定。

#### 15. `94c9855 feat(forcefields): translate legacy complex options`

- 修改了什么：将兼容层与 `_complexes_build_impl()` 分离，集中翻译旧参数名到新的 `epochs`、候选步数等参数；新旧参数冲突时明确报错。
- 为什么修改：删除旧 `Molecule` 入口后，现有调用者仍可能使用历史参数。兼容逻辑若散落在主流程中会继续污染新 API，因此只在边界翻译一次，并禁止含糊覆盖。

#### 16. `d3b4d45 refactor(forcefields): remove redundant wrapper classes`

- 修改了什么：删除只做一层转发的 `OBBuilder` 和 `ForceFields` wrapper class，更新旧入口不存在性的测试。
- 为什么修改：主体函数化后，这两个类不再持有有效状态或独立抽象，只增加重复入口和维护负担。

### 阶段 D：回归围栏、文档与真实运行

#### 17. `76a7a3c test(forcefields): add chemistry quality regression matrix`

- 修改了什么：加入 pytest `slow` 选项、真实 README Eu 络合物慢测试，以及理想有机环和典型配位几何的质量门控参数化测试。
- 为什么修改：仅用人工小图无法证明门控不会误杀正常化学结构，也不能覆盖项目最重要的真实络合物主流程，因此建立快/慢分层围栏。

#### 18. `80fe956 test(forcefields): lock object and Eu regression baselines`

- 修改了什么：加强成功提交后的对象身份断言，并给 Eu 全流程增加当时环境下的能量基线。
- 为什么修改：需要尽早锁住两个容易回归的结果：外部 Atom/Bond 引用不能失效，真实 Eu 构筑不能悄然改变。后来发现绝对能量跨 Open Babel 版本不可移植，`76150c9` 对这一测试标准作了修正。

#### 19. `15e9b21 docs(forcefields): document complex build contracts`

- 修改了什么：更新 README、`doc/cheminfo.md` 和两个示例，展示新的 `build3d`/`complexes_build` 用法、working-copy 事务、质量等级、seed、movie 和报告语义。
- 为什么修改：代码入口已经收束，文档和示例仍引用旧方法或旧参数，会继续诱导用户进入已删除路径。

#### 20. `86e6eac test(ci): cover forcefield and conversion workflows`

- 修改了什么：把几何、力场、络合物和转换测试加入 coverage 快捷脚本与 GitHub Actions；补充 Python 版本兼容工作流选项。
- 为什么修改：新增主流程若不进入常规 CI，后续改动很容易只通过轻量单测而漏掉进程和化学工作流回归。

#### 21. `6b4c2ef test(forcefields): verify real identity-preserving commit`

- 修改了什么：用真实乙醇/Open Babel 优化验证自定义 atom id、原始 Atom 对象和 Bond 对象在成功提交后仍保持 identity。
- 为什么修改：此前对象保持主要由 mock 测试覆盖；需要确认真实后端的坐标、构象和加氢路径不会绕开原子化提交契约。

#### 22. `11fc54b docs(api): remove retired forcefield methods`

- 修改了什么：从生成的 HTML/JS API 索引中移除 `optimize_complexes` 和 `complexes_build_optimize_`。
- 为什么修改：源码已删除旧入口，但静态 API 文档仍把它们列为可用方法，形成错误公开接口。

#### 23. `2423826 test(forcefields): complete worker reproducibility matrix`

- 修改了什么：增加连续 24 个 worker 不串结果/不泄漏进程、8 线程并发构筑及真实同 seed 复现测试；复现比较采用距离矩阵以排除刚体旋转和平移。
- 为什么修改：单次进程测试不能覆盖 PID 回收、跨请求串包和 Open Babel 静态随机状态。并发压力测试随后也帮助发现了 exitcode 与生命周期同步问题。

### 阶段 E：契约补漏和并发/优化器语义修正

#### 24. `837e037 fix(geometry): enforce hydrogen topology policy`

- 修改了什么：`capture_topology()` 增加 `allow_added_hydrogens`；拓扑门控可按调用契约允许或拒绝新增 H/X-H。
- 为什么修改：第一版拓扑门控无条件允许补氢，导致显式 `add_hydrogens=False` 时仍可能把意外新增氢判为合法。

#### 25. `7ab2d64 fix(forcefields): propagate hydrogen topology policy`

- 修改了什么：所有构筑/优化入口把 `add_hydrogens` 传给拓扑快照；代理候选自身固定不允许再新增氢；补充参数传播测试。
- 为什么修改：只有 geometry 支持策略还不够，若业务入口没有传递实际选择，门控仍会按错误契约验收。

#### 26. `798f480 fix(forcefields): restrict complex-only entrypoints`

- 修改了什么：增加 `_require_explicit_complex()`，要求 complex-only API 必须有金属且至少一条显式金属—配体键。
- 为什么修改：`[Zn].N` 或普通有机物进入专用流程时，代理拆分与配位语义根本不成立；此前可能在更深处以难理解的方式失败。

#### 27. `d462723 feat(forcefields): retain candidate rejection evidence`

- 修改了什么：`CandidateRejection` 保存结构化 `quality_failures`，文本同时包含实测值、阈值、原子和键索引。
- 为什么修改：此前候选只留下“geometry gate failed”字符串，无法判断是过近、爆炸、拓扑变化还是其他原因，也无法据此调整算法或门限。

#### 28. `5d3d6ff fix(forcefields): seed legacy Open Babel builders`

- 修改了什么：增加 `_seed_openbabel_random()`；兼容 Open Babel 3.2 的 `OB_RANDOM_SEED` 和 3.1 使用 C RNG 的旧实现；同步更新计划和文档。
- 为什么修改：真实同 seed 测试表明 Open Babel 3.1 的 `OBBuilder` 不读取环境变量，且静态 RNG 会先被时间初始化，仅设置 NumPy seed 或环境变量不能复现。

#### 29. `e48d812 docs(plan): record observable optimizer counters`

- 修改了什么：在计划中区分 `steps_submitted`、`initialization_steps` 和不可观测的 `steps_completed=None`。
- 为什么修改：Open Babel 只返回“继续/停止”，不公开一次 `TakeNSteps(n)` 实际完成了多少步。文档必须避免把估计值包装成观测值，也不能为了计数把 C++ 分段调用拆成低效逐步调用。

#### 30. `045911f docs(forcefields): clarify complex entry contracts`

- 修改了什么：文档明确三个 complex-only 入口要求显式配位键，并说明候选拒绝会保留完整几何证据。
- 为什么修改：这是对 `798f480` 和 `d462723` 的公开契约同步，避免用户把自动分派入口与强制络合物入口混用。

#### 31. `3ee7ca0 fix(forcefields): validate coordination hook inputs`

- 修改了什么：尚未实现的 `prepare_coordination_geometry()` 也先调用 `_require_explicit_complex()`；测试普通有机物会在 strategy 处理前失败。
- 为什么修改：预留接口虽然当前对非空 strategy 抛 `NotImplementedError`，但如果连输入域都不验证，未来实现时会形成与其他 complex-only API 不一致的入口。

#### 32. `3aecd63 fix(forcefields): allow bounded worker exit grace`

- 修改了什么：worker 发送结果后增加独立、有限的退出宽限时间，而不是结果一到就要求进程已经完全消失。
- 为什么修改：动态测试发现 Pipe 消息可先于 Python 子进程清理完成；把“已返回正确结果但稍晚退出”误判为故障会造成负载相关的假失败。

#### 33. `dd2cb69 fix(convert): keep build timeout out of ligand optimization`

- 修改了什么：转换层不再把外层构筑 timeout 错传给拆出配体后的普通 `optimize()`；测试精确记录 build 与 optimize 收到的参数。
- 为什么修改：timeout 属于 3D 构筑 worker 生命周期，不是普通配体优化参数。首轮转换重构把同一 options 无差别复用，可能产生无效参数或改变优化行为。

#### 34. `feea1e4 test(forcefields): verify worker exit cleanup`

- 修改了什么：新增“worker 已发送成功消息但拒绝退出”的测试 double，要求仍执行 terminate/kill/join 清理。
- 为什么修改：`3aecd63` 只证明正常慢退出应有宽限；还必须覆盖相反情况，防止成功消息掩盖僵死进程。

#### 35. `847f599 fix(forcefields): synchronize worker exit detection`

- 修改了什么：使用进程 sentinel 等待真实退出，再有界刷新 exitcode；调整并发复现测试为两轮多 seed 距离矩阵比较。
- 为什么修改：单纯 `join(timeout)`/`is_alive()` 在并发 child cleanup 下仍可能短暂得到 `exitcode=None`。这解释了 Python 3.14/高负载运行中偶发的“结果正确但 shutdown 失败”。

#### 36. `cde42b0 fix(forcefields): preserve carbon and halide protonation`

- 修改了什么：隐藏金属键后只对当时确认的中性 N/O donor 重算隐式氢，不再对所有中性配位原子统一重算；加入 C/卤素负对照。
- 为什么修改：首版 donor 重算范围过宽，会把金属—C 或金属—卤素共价关系误当配位键并错误再质子化。这个收窄随后因 P/S/As/Se 正例又在 `d0a8a73` 中扩展为显式元素集合。

#### 37. `deecc57 test(forcefields): distinguish convergence from step limits`

- 修改了什么：初始化时给 Open Babel 内部步数上限保留 sentinel 余量，并把变量语义改成 backend convergence；增加专用后端测试。
- 为什么修改：`TakeNSteps(False)` 既可能是收敛，也可能只是碰到 `Initialize(limit)` 的内部上限。若初始化上限恰好等于 Hotpot 计划提交量，预算结束会被错误报告成收敛。

#### 38. `f8f72d4 fix(forcefields): resume scheduled perturbation segments`

- 修改了什么：引入 `segment_active`，一次 segment 收敛后不再永久 break；若后面安排了 perturbation，会重新初始化并继续新 segment。
- 为什么修改：原循环把局部优化收敛等同于整个带扰动搜索结束，导致 `perturb_interval` 之后的计划段永远无法执行。

#### 39. `42f1012 fix(forcefields): evaluate strict stability within segments`

- 修改了什么：能量/坐标稳定历史在每个 perturbation segment 内重置；报告增加 segment epoch 信息；strict 允许“首个 segment 第一 epoch 后端明确收敛”的无历史特例。
- 为什么修改：把扰动前后的巨大坐标差纳入稳定性窗口会误判；而一轮即明确收敛时又没有两帧可算 delta，不能因缺历史否定真实收敛。

#### 40. `afacb32 refactor(forcefields): generalize build worker protocol`

- 修改了什么：把只面向络合物的 worker 接收器抽象成通用 build-worker 协议；允许普通坐标 worker 不携带复杂构筑 diagnostics；增加通用错误与 timeout 类型。
- 为什么修改：后续普通有机物的 seeded `OBBuilder` 也必须进入隔离进程。复制一套 Pipe/超时/回收实现会重新引入协议分叉。

#### 41. `c9fb40e fix(forcefields): isolate seeded ordinary builders`

- 修改了什么：普通 `build3d(seed=...)` 改为 spawn 独立 worker，在首次 `OBBuilder` 前设 seed；无 seed 仍走直接构筑；新增普通分子 seed 复现测试。
- 为什么修改：动态审计证明同一进程内 Open Babel 静态 RNG 已初始化后，即使再次设 seed 也不能保证结果；只比较距离矩阵仍显示真实不复现，因此必须隔离进程状态。

#### 42. `560d938 fix(forcefields): serialize worker lifecycle transitions`

- 修改了什么：用 `_WORKER_LIFECYCLE_LOCK` 串行化 Process start、reap 和 exitcode 刷新；异常清理也在同一锁下执行。
- 为什么修改：`multiprocessing.Process.start()` 会触发全局 child cleanup。并发线程可能替另一个 Process 执行 waitpid，造成结果已到但对应对象仍短暂显示 `exitcode=None`。

#### 43. `d0a8a73 fix(forcefields): infer pnictogen and chalcogen donor hydrogens`

- 修改了什么：把中性 donor 重算从 N/O 扩展到 N、O、P、S、As、Se，并抽为 `_recalculate_neutral_donor_valence()`；新增组装路径与直接解析路径的一致性测试。
- 为什么修改：`cde42b0` 为防止 C/卤素误质子化而收窄过头，实际含膦、硫醚、As/Se donor 的配体在两种输入路径中出现不同氢数。正确边界应是明确的中性主族 donor 集合，而不是仅 N/O。

#### 44. `eefe3ce fix(forcefields): constrain temporary ring openings`

- 修改了什么：只允许临时打开单键且非稠合的环边；每次候选尝试前恢复隐藏共价键；没有安全边时不强行开环；增加相关几何和失败恢复测试。
- 为什么修改：解穿环逻辑原先可能选择芳香键、双键或稠合共享边，并在 builder 失败后把隐藏状态带到下一次尝试，导致化学拓扑损伤和候选间串扰。

#### 45. `2aac455 fix(forcefields): enable standard quality gate by default`

- 修改了什么：`ff.optimize()` 默认 `quality_level` 从 `off` 改为 `standard`，并锁定转发测试。
- 为什么修改：代码默认值与已经批准的计划、`build3d`/络合物流程的安全标准不一致，使普通优化默认绕过过近、穿环等结构门控。

### 阶段 F：真实 Open Babel、诊断与事务边界再加固

#### 46. `a375974 fix(forcefields): isolate Open Babel backend instances`

- 修改了什么：最初把 `OBForceField.FindType()` 返回的 prototype 经 `MakeNewInstance()` 克隆，使每次优化获得独立 backend；增加实例独立性测试。
- 为什么修改：Open Babel 插件对象有可变 setup/constraint/cutoff 状态，理论上共享 singleton 会让线程和连续调用相互污染。当时依据 API 表面语义选择实例克隆。
- 后续修正：真实反复运行发现部分 Open Babel 版本的克隆实例在 `Setup()` 中不稳定，甚至可能段错误，因此该方案在 `49a4fe3` 被撤回，改为稳定插件句柄加全调用串行化。

#### 47. `f9be127 fix(forcefields): synchronize Open Babel builder seeding`

- 修改了什么：增加 builder 调用串行装饰器，使父进程内直接 `_ob_build` 与 seeded worker 启动时的环境变量窗口互斥。
- 为什么修改：启动 seeded worker 时会暂时改变 `OB_RANDOM_SEED`。另一个线程若恰好调用直接 `OBBuilder`，可能观察到不属于自己的 seed，造成非 seeded 请求被污染。

#### 48. `cb9dd72 test(forcefields): reproduce builder seed exclusion`

- 修改了什么：加入基于 `threading.Event` 的确定性并发测试，主动暂停在 seed 环境窗口并启动直接 builder，验证后者不能进入。
- 为什么修改：普通线程时序测试容易“碰巧通过”，不能真正证明锁覆盖了危险窗口。该提交把之前难复现的竞争条件变成稳定回归测试。

#### 49. `49a4fe3 fix(forcefields): retain stable serialized plugin handles`

- 修改了什么：撤销 `MakeNewInstance()` 路线，恢复 `FindType()` 的稳定插件句柄；所有 backend 操作继续由锁串行；补充 OBMol atom iterator helper并同步计划/文档/测试。
- 为什么修改：真实 Open Babel 3.1/3.2 运行表明“指针不同”不等于实例生命周期安全，克隆后反复 `Setup()` 存在 native crash 风险。最终策略优先保证后端句柄稳定，再用进程/线程边界解决隔离。这是本轮最明确的一次由实测推翻初始设计。

#### 50. `657874b fix(geometry): validate forcefield reports by stage`

- 修改了什么：质量门控增加 candidate/final stage；候选阶段只要求其实际可提供的 setup、有限能量和 explosion 信息，最终阶段即使 `quality_level=off` 也要求完整关键诊断；增加 fail-closed 测试。
- 为什么修改：候选快速评分没有梯度和完整收敛历史，若按最终报告要求会全部误杀；反过来，最终报告缺字段时又不应在低质量等级下默认放行。

#### 51. `33dfb1e fix(forcefields): retain structured intersection failures`

- 修改了什么：新增 `bond_ring_intersection_checks()`，把每个穿环转换成带稳定原子/键索引的 `GeometryCheck`；候选拒绝同时保留穿环与其他 gate 失败，不再只留字符串。
- 为什么修改：`d462723` 已保存一般质量失败，但穿环走单独布尔/对象路径，仍丢失结构化证据，而且可能覆盖同一次候选的其他失败。

#### 52. `6b4da26 fix(forcefields): roll back failed working-copy commits`

- 修改了什么：将提交拆为预验证 payload、调用方完整快照、原子化写入和失败恢复；覆盖原子/键对象、邻接关系、图、atom-pair、构象和缓存；增加故意在写回中途抛错的测试。
- 为什么修改：此前“优化失败不修改本体”已成立，但“优化成功、最终 commit 中途失败”仍可能留下新增氢、部分坐标或损坏缓存。事务边界必须覆盖最后写回本身，而不只是前面的 working copy。

#### 53. `60b4b22 fix(convert): aggregate conversion worker failures`

- 修改了什么：批量转换收集每个任务的 nonzero-exit/timeout 失败，先回收全部 worker，再统一抛 `ConversionBatchError`；不再只打印后继续。
- 为什么修改：批处理中静默失败会让调用者误以为所有输出均已生成；遇到第一个异常立即退出又可能遗留其他活动进程。此版先解决聚合与清理，详细子进程异常在 `6e1ec66` 补齐。

#### 54. `b482264 refactor(forcefields): complete worker result envelope`

- 修改了什么：`BuildWorkerResult` 增加可选 `conformers` 序列化字段及测试。
- 为什么修改：计划定义的通用 worker envelope 包含坐标、构象和诊断，但主体实现漏了构象字段。虽然当前路径主要回传坐标，协议必须完整，便于后续 movie/多构象 worker 使用。

#### 55. `76150c9 test(forcefields): stabilize europium acceptance criteria`

- 修改了什么：重写 Eu 慢测试，记录耗时并检查原始对象身份、拓扑、配位键、门控结果、有限能量和结构尺度；移除固定 UFF 绝对能量断言。
- 为什么修改：`80fe956` 的单点能量在 Open Babel 3.1/3.2 间出现显著漂移，即便结构和流程都正确也会失败。UFF 绝对能量不是跨版本验收标准，测试应锁化学与程序不变量。

#### 56. `8a48e4d fix(forcefields): propagate seeded build timeout`

- 修改了什么：`build3d()` 暴露 timeout，并传入 `_seeded_ob_build_coordinates()` 和通用 worker 接收器；组合有机流程也完整透传；移除内部固定 1000 秒常量。
- 为什么修改：seeded 普通构筑改为独立进程后，公开 timeout 仍在该分支被固定常量吞掉，用户无法控制真正可能卡住的 worker。

#### 57. `02432c1 fix(geometry): fail closed on short bonds and explosions`

- 修改了什么：将拓扑键异常短独立报告为 `short_bond`，不与 nonbonded too-close 混合；当 backend explosion 状态缺失时 basic/final gate 失败关闭。
- 为什么修改：枚举中虽然已有短键概念，第一版综合门控却没有真正执行；同时“没有 explosion 字段”曾被当作“没有爆炸”，会让不完整报告静默通过。

#### 58. `aec6088 fix(forcefields): report setup failures structurally`

- 修改了什么：增加 `ForceFieldSetupReport`，区分 lookup 与 setup 阶段，并让 `ForceFieldSetupError` 携带结构化诊断；增加真实模拟 setup failure 测试。
- 为什么修改：未知力场或 `backend.Setup()` 失败此前主要依赖异常字符串，无法供上层、日志和测试稳定地区分失败阶段与 requested/effective forcefield。

#### 59. `b45d64f refactor(forcefields): unify build workflow reports`

- 修改了什么：引入 `ForceFieldWorkflowReport`、`BuildAndOptimizeReport`，普通与络合物组合流程统一拥有 `build`/`optimization` 结构；普通 `build_and_optimize()` 不再丢弃 build report。
- 为什么修改：此前普通组合流程只返回优化报告，络合物返回包含两阶段的 `ComplexBuildReport`，同类高层 API 形状不一致，也无法查看普通 3D 嵌入阶段信息。

### 阶段 G：公开接口收口、转换 IPC 完成与文档校准

#### 60. `707bc9b refactor(forcefields): privatize Open Babel primitives`

- 修改了什么：公开的 `ob_build`/`ob_optimize` 改名为 `_ob_build`/`_ob_optimize`，内部调用和测试同步更新。
- 为什么修改：公开 API 已规定用户使用 `ff.build3d()`、`ff.optimize()` 等标准函数；裸 Open Babel primitive 绕过补氢、事务、门控和报告，继续公开会形成第二套不安全入口。

#### 61. `6e1ec66 fix(convert): preserve worker exception diagnostics`

- 修改了什么：转换 worker 增加结构化 `ConversionWorkerResult` 和 `_ActiveConversion`；每个 worker 用单独 Pipe 返回异常类型、消息和 traceback；父进程先 drain Pipe 再 join；增加超过 1 MiB 异常消息测试。
- 为什么修改：`60b4b22` 只能报告 exit code/timeout，丢失真正 Python 异常；若用 Queue 或先 join，大 traceback 又可能因缓冲区填满死锁。

#### 62. `dd85db2 docs(forcefields): align workflow acceptance contracts`

- 修改了什么：同步 README、cheminfo 文档和计划：默认候选数、step/convergence 可观测性、strict 首段收敛例外、当前空约束、统一报告类型，以及 UFF 绝对能量仅作同版本参考。
- 为什么修改：多轮动态修正后，早期文档仍残留固定能量、旧候选数量和过度简化的收敛描述，需要让用户看到的契约与真实实现一致。

#### 63. `630fe3c fix(convert): spawn isolated conversion workers`

- 修改了什么：批量转换显式使用 `multiprocessing.get_context("spawn")` 创建 Pipe/Process，并让测试 double 注入相同 context。
- 为什么修改：默认 fork 在多线程 Python/Open Babel 环境中会继承锁和 native 状态，Python 3.12+ 还会警告；转换任务需要与父进程完全隔离。

#### 64. `c996f66 fix(convert): prioritize completed worker results`

- 修改了什么：父进程先处理已经到达的 worker result，再判断 `is_alive` 或 timeout；结果后执行有界 join；`Process.start()` 失败时关闭 Pipe 两端；增加真实 spawn、结果/超时竞争和 FD 清理测试。
- 为什么修改：上一版可能在结果已经成功写入 Pipe 时，因为检查时钟或 `is_alive=True` 而误报 timeout；启动异常还会泄漏文件描述符。这是转换 IPC 的最后一个竞态修正。

#### 65. `51f70a0 docs(plan): align hydrogen acceptance matrix`

- 修改了什么：重写计划中的氢测试矩阵，改为按风险机制选择显式/隐式氢代表案例，并明确新 API 已删除 `rm_polar_hs`，不再要求无意义的笛卡尔积。
- 为什么修改：计划 §5.3 曾要求每类分子同时覆盖 `rm_polar_hs=True/False`，但 §5.4 又明确该参数从新接口删除，两处自相矛盾；按旧矩阵执行会测试不存在的业务分支。

## 4. 关键波折与最终结论

### 4.1 Open Babel 力场实例：克隆方案被真实运行推翻

演进链：`a375974` → `49a4fe3`

- 初始判断：共享 `FindType()` prototype 可能泄漏可变状态，因此调用 `MakeNewInstance()`。
- 实测问题：克隆对象虽有不同指针，但某些 Open Babel 版本反复 `Setup()` 不稳定，出现 native crash 风险。
- 最终策略：保留稳定插件句柄，所有 backend 操作在锁内完成；需要更强隔离的构筑路径使用 spawn worker。

### 4.2 worker 生命周期：从“收到结果”逐步收紧到完整状态机

演进链：`de69876` → `3aecd63` → `feea1e4` → `847f599` → `560d938`

- 先解决大消息 Pipe 的 receive-before-join 和 timeout 强制回收。
- 再确认消息抵达与进程退出不是同一事件，增加独立 exit grace。
- 随后加入“发完消息但僵死”的反例，不能让成功结果掩盖未退出 worker。
- 最后处理多线程 `multiprocessing` 的全局 child cleanup 竞争，用 sentinel 与生命周期锁稳定 exitcode。

### 4.3 优化器终止：不是一个布尔值能表达

演进链：`26963bd` → `deecc57` → `f8f72d4` → `42f1012`

- 初始化可能消耗物理步，必须纳入预算。
- `TakeNSteps(False)` 可能是收敛，也可能是撞到内部 limit，需 sentinel 容量区分。
- 单个 segment 收敛不代表带后续扰动的全局流程结束。
- strict 稳定性只能比较同一 segment 内的相邻帧，首帧明确收敛需要单独语义。

### 4.4 补氢规则：先防止过度修复，再补足 donor 元素范围

演进链：`d790cc5` → `cde42b0` → `d0a8a73` → `51f70a0`

- 首先确立隐藏金属键后按配体共价骨架补氢，并在 working copy 上执行。
- 发现对所有中性配位原子重算会错误质子化 C/卤素，于是先收窄为 N/O。
- 再由 P/S/As/Se 配体测试证明 N/O 范围过窄，改为明确的中性 donor 元素集合。
- 最后修正计划测试矩阵，删除已经不存在的 `rm_polar_hs` 组合要求。

### 4.5 Eu 基线：从固定数值改为化学与程序不变量

演进链：`80fe956` → `76150c9` → `dd85db2`

- 初始用单一环境的 UFF 能量锁回归。
- Open Babel 3.1/3.2 给出不同绝对值，但拓扑、配位和几何均正常，证明固定值不可作为跨版本门槛。
- 最终验收对象是：流程完成、门控通过、能量有限、拓扑/对象 identity 保持、配位键与距离合理；绝对能量只在同版本作参考。

### 4.6 批量转换：从进程回收到完整可诊断 IPC

演进链：`892729f` → `60b4b22` → `6e1ec66` → `630fe3c` → `c996f66`

- 先实现 timeout 传递与 terminate/kill/join。
- 再聚合全部任务失败，避免静默丢样本。
- 然后用 per-worker Pipe 保留异常类型、消息和 traceback，并避免大消息死锁。
- 再切到 spawn 隔离 Open Babel/native 状态。
- 最后修正结果到达与 timeout/is_alive 的竞态，以及 start 失败时的 FD 泄漏。

## 5. 最终状态与仍然明确保留的边界

当前实现最终形成以下稳定边界：

- `Molecule.build3d()` 和 `Molecule.optimize()` 是薄门面；业务在 `hotpot.cheminfo.forcefields`。
- 络合物完整流程在 working copy 上完成，成功原子化提交，任一步失败均回滚调用方对象。
- 构筑/优化默认补氢；络合物先隐藏金属—配体键，再按配体共价骨架补氢。
- 络合物流程强制 UFF，但报告保留 requested/effective forcefield。
- 几何与拓扑判断统一在 `geometry.py`，并返回结构化证据。
- 有 seed 的 Open Babel 构筑通过 spawn worker 隔离；同版本、同平台承诺复现，不承诺跨 Open Babel 版本逐位一致。
- 默认质量等级为 `standard`；最终报告缺少必要 backend 诊断时 fail closed。
- 默认只保留最低能合格帧，`save_movie=True` 才保留逐 epoch 轨迹。
- 配位数驱动的中心原子几何调整仍只是明确的预留 hook，非空 strategy 会显式失败，尚未伪装成已实现功能。
- Open Babel 原子/距离/角度/扭转约束映射仍保持空 adapter；旧版 0-based 到 1-based 的错误映射没有被保留。

## 6. 关于提交粒度的结论

65 个提交看起来较多，主要原因是本轮刻意采用“小步提交、每个发现独立回溯”的策略。大部分后续提交不是重复改同一个表面问题，而是新增测试把隐藏语义分离出来，例如：

- “结果已返回”与“进程已退出”是两个状态；
- “局部 segment 收敛”与“整个扰动计划结束”是两个状态；
- “后端没有报告爆炸”与“后端明确报告未爆炸”不是同一件事；
- “不同插件指针”与“native 实例可安全独立使用”不是同一件事；
- “同 seed 坐标不完全相同”需要先排除刚体旋转，再判断是否真不复现。

因此，建议保留当前提交链用于审计和回溯。若未来为了合并主分支需要 squash，可按上述七个阶段压缩，而不建议压成单个提交；至少应保留“规划/几何/核心 API/力场主流程/worker 与 seed/事务与诊断/文档测试”七个逻辑节点。
