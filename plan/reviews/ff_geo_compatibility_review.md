# `forcefields.py` / `geometry.py` 兼容性代码审查

## 1. 审查目标和边界

本审查响应以下项目约束：Hotpot 当前不承担旧 force-field API 的历史兼容责任；业务代码
应直接采用当前确定的名称、签名、对象和后端接口，而不是在运行时翻译旧名称或猜测多种
对象形态。

审查范围包括：

- `hotpot/cheminfo/forcefields.py`；
- `hotpot/cheminfo/geometry.py`；
- 两个模块的直接调用方、测试、依赖声明和用户文档；
- 不只搜索 `legacy` / `compatibility` 字样，也检查参数翻译、版本分支、鸭子类型回退、
  多种数据表示归一化、闲置包装器和为保留旧行为而设置的测试。

本文件只记录事实、分类和候选整改，不修改业务代码。

## 2. 总结

当前发现四类明确的历史兼容实现：

1. `complexes_build()` 的 11 个旧参数名翻译，以及由此产生的双层入口；
2. 已经私有化、但完全没有调用者的 `_ob_optimize()` 兼容残留；
3. Open Babel 3.1/3.2 双 RNG 实现；
4. `geometry.py` 对非平面环旧判定语义的显式保留。

另有四组“结构兼容”代码：它们没有标注为 legacy，却允许同一入口接受 Hotpot 当前对象以外
的形态。这些包括原子索引回退、环接口回退、键类型回退，以及力场报告同时接受 mapping
和属性对象。若确立“只接收当前 Hotpot 契约”，这些分支也应收束。

其中 Open Babel 3.1 不能立即按死代码删除：项目目前明确支持 Python 3.9，而依赖声明为
Python 3.9 安装 `openbabel-wheel 3.1.1.23`。若要求只适配 Open Babel 3.2 当前接口，必须
同时改变 Python/依赖支持矩阵；否则删除该分支会破坏仍被声明为受支持的平台。

## 3. 明确的旧 API 兼容层

### C001：旧 `complexes_build` 参数翻译

**位置**

- `forcefields.py:1747-1759`：`_LEGACY_COMPLEX_BUILD_OPTIONS`；
- `forcefields.py:1762-1774`：`_translate_legacy_complex_build_options()`；
- `forcefields.py:2041-2051`：接受任意 `**options` 的公开包装器；
- `tests/test_cheminfo/test_forcefield_api.py:131-179`：专门锁定旧参数翻译和冲突处理；
- `doc/cheminfo.md:258-261`：向用户公开说明兼容入口。

当前翻译表共保留 11 个旧名称：

| 旧名称 | 当前名称 |
|---|---|
| `steps` | `epochs` |
| `step_size` | `steps_per_epoch` |
| `perturb_steps` | `perturb_interval` |
| `save_screenshot` | `save_movie` |
| `build_times` | `candidate_count` |
| `init_opt_steps` | `candidate_warmup_steps` |
| `second_opt_steps` | `candidate_score_steps` |
| `min_energy_opt_steps` | `best_candidate_refine_steps` |
| `increasing_Vdw` | `increasing_vdw` |
| `Vdw_cutoff_start` | `vdw_cutoff_start` |
| `Vdw_cutoff_end` | `vdw_cutoff_end` |

**判断**

这是确定无疑的历史兼容层。它由提交 `94c9855` 专门引入，并将原本具有完整显式签名的
`complexes_build()` 改名为 `_complexes_build_impl()`，再在外部增加 `**options` 包装器。
仓库中的生产代码和可执行示例没有使用这些旧名称；用户文档只说明了这套映射，只有兼容性
测试实际调用它们。

该兼容层还是不完整的。旧实现中的 `equilibrium`、`equi_check_steps`、
`equi_max_displace`、`equi_max_energy`、`equi_threshold`、`max_iter`、`print_energy`、
`rm_polar_hs` 和 `sophisticated` 等名称并未映射。保留一个部分兼容层既扩大签名，又无法
形成可靠的旧接口契约。

**候选整改**

1. 删除 `_LEGACY_COMPLEX_BUILD_OPTIONS` 和
   `_translate_legacy_complex_build_options()`；
2. 将 `_complexes_build_impl()` 提升/改名为唯一的公开 `complexes_build()`，保留其当前
   显式 keyword-only 签名；
3. 删除“翻译旧参数”和“新旧参数冲突”测试，改为验证当前显式签名，并让未知参数由
   Python 正常抛出 `TypeError`；
4. 删除 `doc/cheminfo.md:258-261` 的旧参数说明；
5. `build_and_optimize()` 继续直接调用这个唯一入口，无需中转 `dict` 或 `**options`。

这里不建议另加 deprecated alias、warning 或版本探测；它们会重新引入用户已经明确拒绝
承担的历史包袱。

### C002：`_ob_optimize()` 是未清除的兼容残留

**位置**：`forcefields.py:489-491`。

该函数只有一行，调用 `_single_ob_optimization()` 后仅返回 `.energy`。全仓库没有生产、
测试或文档调用 `_ob_optimize()`。Git 历史显示它原名为公开 `ob_optimize()`，文档字符串
明确称其为 “Compatibility primitive”；提交 `707bc9b` 只把它改成私有名称，没有删除
实现。

**候选整改**：直接删除 `_ob_optimize()`。不要给它换一个新名字，也不要为其增加调用者。

同一提交私有化的 `_ob_build()` 情况不同：它仍被普通 3D 构筑、seed worker 和配体代理
构筑实际调用，是当前 Open Babel 构筑 primitive，不能随 `_ob_optimize()` 一起删除。
后续只需依照新类型/命名规则把参数收紧为 `working_mol: Molecule` 或其他准确生命周期名。

### C003：`complexes_build` 名称本身不是已经证实的兼容别名

`complexes_build()` 的当前包装方式是兼容层，但名称本身同时是：

- `forcefields.__all__` 中的公开业务入口；
- `build_and_optimize()` 的络合物分支目标；
- 用户此前明确认定的络合物主流程名称。

因此不能仅因其包装器 docstring 写着 “Compatibility entry” 就断言函数名也必须删除。
最小整改是保留 `complexes_build` 这一当前入口名，只清除旧参数翻译和 `_impl` 双层结构。
若后续决定改成 `build_and_optimize_complex`，依据“无历史包袱”原则应一次改完所有内部调用
和文档，而不是再保留一个 `complexes_build` alias。

## 4. 后端版本兼容

### C004：Open Babel 3.1 C RNG 与 3.2 `OB_RANDOM_SEED` 双路径

**位置**

- `forcefields.py:5`：仅为旧分支引入的 `ctypes`；
- `forcefields.py:435-449`：`_seed_openbabel_random()`；
- `pyproject.toml:11,33-34` 和 `requirements.txt:5-6`：按 Python 版本选择
  Open Babel 3.1/3.2；
- `README.md:164-180`、`doc/cheminfo.md:199-205`：明确说明双版本支持；
- `.github/workflows/inference_compatibility.yml:96`：Python 3.9-3.14 矩阵。

函数当前同时执行：

1. 设置新版 Open Babel 使用的 `OB_RANDOM_SEED`；
2. 初始化 3.1 的函数局部 RNG，再通过 libc `srand()` 重置旧 C RNG。

这是明确的后端版本兼容代码，但目前不是无调用死分支。项目声明 Python 3.9-3.14；
Python 3.9 依赖被固定为 `openbabel-wheel>=3.1.1.23,<3.2`，因为项目当前认为 Open Babel
3.2 没有对应 Python 3.9 包。

**候选整改分两种互斥路线**

- 若继续支持 Python 3.9：保留该分支，并把它定义为“当前支持矩阵的后端适配”，不能按
  历史 API 残留删除；但应把版本差异封装成明确的 backend adapter，而不是用“额外操作
  对新版无害”作为长期依据。
- 若只支持当前 Open Babel 3.2 接口：同步将 Python 下限提高到 3.10，删除
  `openbabel-wheel` 3.1 依赖、`ctypes`/`randomUnitVector`/`srand` 路径、对应测试和双版本
  文档，只保留经 Open Babel 3.2 实测有效的 seed 流程。

在未改变已经批准的 Python 3.9-3.14 支持矩阵前，不应单独删除 3.1 路径。

### C005：能量单位多拼写归一化不是旧版本兼容层

`forcefields.py:410-416` 的 `_energy_factor_to_kj()` 接受 `kJ/mol`、`kJ mol^-1`、
`kcal/mol` 等后端单位拼写。这些表示来自当前可选 Open Babel force-field plugin，不是
Hotpot 旧 API 参数名。统一输出为 kJ/mol 是当前公共契约，应保留；后续只需根据实际支持
的 Open Babel 版本收紧并测试允许集合。

## 5. 对象形态和数据契约兼容

以下代码不一定源于已发布旧版本，但都具有“缺少当前属性时走另一套表示”的兼容式结构。
既然 forcefield/geometry 正式输入应为 Hotpot 原生对象，就不应让测试替身反向扩大业务
接口。

### C006：原子索引缺失时回退到遍历位置

**位置**：`geometry.py:196-197` 的 `_atom_index(atom, fallback)`。

当前 `Atom.idx` 是 Hotpot `Atom` 的正式属性，但该 helper 在属性不存在时静默使用枚举
位置。这个分支主要使 `SimpleNamespace` 一类测试对象也能运行，并会掩盖传错对象或对象
契约损坏。

**候选整改**：参数标为 `Atom`，直接读取 `atom.idx`；涉及 malformed coordinate 的测试
也使用真实 `Molecule`/`Atom` 构造异常状态，不把无 `idx` 的匿名对象塑造成受支持输入。

### C007：环查询在私有当前接口和旧属性之间回退

**位置**：`geometry.py:578-584` 的 `_rings_for_scope()`。

当前 `Molecule` 明确具有 `_uncached_rings(ligand_skeleton=...)`。geometry 为避免质量检查
修改 ring cache，需要使用这一无缓存路径；但函数仍在该方法不存在时回退到 `mol.rings`
或 `mol.ligand_rings`。正式生产对象不需要该回退，而且两条路径的副作用不同：属性路径会
访问/填充缓存。

**候选整改**：确立一个当前、稳定的 Core 查询接口（优先将无缓存、按 scope 查询的能力
作为明确内部契约），geometry 只调用该接口；删除 `getattr` 和属性回退。不要仅为了支持
测试 double 保留两套语义。

### C008：键类型同时接受 `BondKind`、字符串和缺失属性

**位置**

- `geometry.py:620-622` 的 `_bond_kind()`；
- `forcefields.py:584-590` 的 `_bond_commit_signature()`。

当前 `Bond.bond_kind` 总是由 Core 的 `_coerce_bond_kind()` 规范为 `BondKind`。现有代码却
使用 `getattr(kind, "value", kind)`，geometry 甚至在 `bond_kind` 缺失时返回空字符串。
这允许旧字符串/不完整 fake bond 悄悄进入拓扑签名。

**候选整改**：将参数收紧为 `Bond`，直接使用 `bond.bond_kind.value`。若某个导入器仍产生
字符串，应在创建 `Bond` 的唯一边界完成转换，而不是在 forcefield/geometry 再次兼容。

### C009：力场质量数据同时接受 mapping 和属性对象

**位置**

- `geometry.py:786-789` 的 `_report_value()`；
- `geometry.py:792-927` 的 `_forcefield_checks()`；
- `geometry.py:1584-1592`、`1821-1829` 的公开参数 `forcefield_report: Any`；
- `forcefields.py:863-881`、`1194-1205`、`1263-1274`：生产代码实际传入 dict。

`_report_value()` 同时支持 `Mapping` 和任意属性对象，缺失字段一律得到 `None`。这是双数据
模型适配，也正是 `Any` 扩散到整个质量门控的来源。生产代码目前统一构造 dict；测试也
主要使用 dict，并没有证据表明公开入口必须兼容任意报告对象。

**候选整改**：定义唯一的当前质量输入契约。建议在 geometry 侧定义精确的不可变数据类
或 `TypedDict`，由候选打分和优化器观测共同构造；`_forcefield_checks()` 只读这一种结构。
不要继续用动态字段名、`getattr(..., None)` 或 `Any` 兼容未知对象。

### C010：质量阈值同时接受 dataclass 和任意 mapping

**位置**：`geometry.py:776-783`、`1584-1592`、`1821-1829`，以及 forcefield 各公开入口的
`quality_thresholds: Optional[Mapping[str, float]]`。

这不是已经证实的旧 API 支持，而是当前便利输入设计；但它仍维持两种配置表示，并造成
类型不准确——两个 bond-ratio 阈值实际为 `tuple[float, float]`，不是 `float`。

**候选整改**：在整改计划中明确选择一种公共契约。若“直接适配当前接口”优先，建议公开
使用 `GeometryQualityThresholds`，需要关键字便利时提供一个明确构造函数；不要继续让所有
mapping 直接穿透 `dataclasses.replace()`。这是接口收束，不应与 C001 的旧参数翻译混在
一次机械删除中。

## 6. 显式保留旧行为的几何逻辑

### C011：非平面环的 historical center-fan surface

**位置**

- `geometry.py:436-476`，尤其 `463` 行注释；
- `tests/test_cheminfo/test_geometry.py:144-154` 的
  `test_nonplanar_ring_keeps_center_fan_surface_semantics()`。

平面环采用投影后的真实多边形包含判定；非平面环仍用环顶点均值与每条边形成三角扇面，
注释明确写为保留 historical behavior，测试名也明确锁定该旧语义。这是本轮发现的、并非
参数命名形式的兼容行为。

它不能像 C001 一样直接删除，因为“非平面环的内部表面”本身没有唯一化学定义。当前行为
会参与 bond-ring intersection 质量门控，改变它可能改变络合物候选的接受/拒绝结果。

**候选整改**：先定义当前业务语义，再替换测试：例如明确采用按环序 triangulation、
best-fit plane 投影，或仅对足够平面的环执行穿环判定。决定后测试应描述所选几何规则，
不再以“保持历史语义”为验收目标。该项属于需要用户审议的化学/几何行为，不纳入无条件
纯代码清理。

## 7. 看似兼容、但实际属于当前实现约束的代码

以下项目不应因为“去兼容”而误删。

| 位置 | 表面现象 | 实际职责 | 结论 |
|---|---|---|---|
| `forcefields.py:356-372` | 两层锁和包装器 | Open Babel plugin/OBBuilder 的 native 并发隔离 | 保留 |
| `forcefields.py:375-388` | `None` 和不同力场名被改写 | 有机分子默认 MMFF94s、络合物强制 UFF 的当前化学后端政策 | 保留并独立审议化学策略 |
| `forcefields.py:420-432` | `_find_forcefield_prototype()` 薄包装 | 隔离 plugin lookup，测试可稳定模拟缺失 plugin | 不是旧 API alias |
| `forcefields.py:497-503` | 手工复制 metadata | 补偿当前 `Molecule.copy()` 不复制 charge/properties/runtime metadata | 不是版本兼容；应另行审查 Core copy 契约 |
| `forcefields.py:572-576` | 另建结构代理并刷新 ID | 避免把不可 pickle 的运行时 metadata 发送到 spawn worker | 当前进程边界要求，保留 |
| `forcefields.py:1362` | `getattr(exc, "diagnostics", None)` | 将任意 worker 异常装入统一 IPC envelope | 当前错误传输策略，不是旧对象兼容 |
| `forcefields.py:1399-1510` | generic/complex worker 两套异常参数 | 同一可靠 IPC 生命周期供 ordinary build 和 complex build 复用 | 泛化实现，不是旧入口兜底 |
| `forcefields.py:2080,2159` | 根据金属自动分派 | `Molecule.build3d()` / `optimize()` 当前标准 façade 的核心业务 | 不是兼容分支 |
| `geometry.py:776-783` | `None` 生成默认阈值 | 当前可选配置默认值 | 本身不是兼容；只有 dataclass/mapping 双表示需要决策 |
| `geometry.py:1821-1838` | `is_geometry_reasonable()` 只返回 `.passed` | 为详细报告提供明确的布尔查询接口 | 当前便利 API，不是旧名称 alias |
| `from __future__ import annotations`、`Optional`、`Union` | 旧式 typing 写法 | 当前声明支持 Python 3.9 | 在 Python 下限改变前保留 |

### 值得另行清理、但不能冒充兼容问题的残留

- `BuildWorkerResult.conformers`（`forcefields.py:139`）只有存储测试，没有生产写入或读取。
  它是闲置 schema 字段，而非旧版本转换器；应在 worker protocol 清理项中决定是否删除。
- `Point`（`geometry.py:981-983`）没有调用者；它是死的旧几何对象，不是 compatibility
  adapter，可直接列入后续 dead-code 清理。
- `Line`、`Plane`、`CyclePlanes` 和相关函数属于较早的对象式几何层，但当前 Core 的
  `Bond.line`、芳香性判定、`Ring.cycle_places` 仍在调用。它们是待统一的重复抽象，不是
  可以未经引用树迁移便删除的兼容包装器。已有接口报告中记录了其引用关系。
- `Molecule.copy()` 当前只重建 atoms/bonds；`_copy_molecule_metadata()` 是这一不完整 copy
  契约的局部修补。长期更合理的整改是先定义并修正 Core copy 语义，再删除 forcefields
  中的局部桥接，但该动作影响全库，不应夹在旧参数删除中完成。

## 8. 已经清理完成的旧入口

以下旧接口已不存在；测试中的 `hasattr(...)=False` 是防止回归的负向围栏，不是兼容代码：

- `Molecule.optimize_complexes()`；
- `Molecule.complexes_build_optimize_()`；
- `forcefields.OBBuilder`；
- `forcefields.ForceFields`；
- 公开 `forcefields.ob_build()`；
- 公开 `forcefields.ob_optimize()`。

这些负向断言可保留。唯一例外是私有 `_ob_optimize()` 仍残留，见 C002。

## 9. 建议整改顺序

### 可直接批准的纯接口清理

1. 删除 C001 的旧参数表、translator、`**options` wrapper、对应测试和文档；
2. 将 `_complexes_build_impl()` 收束为唯一显式签名的 `complexes_build()`；
3. 删除无调用的 `_ob_optimize()`；
4. 按新类型/命名规范处理 C006-C009，测试使用真实 Hotpot 对象或精确 typed fixture，
   不让不完整 duck type 决定生产接口。

### 需要先确定支持政策或业务语义

1. C004：Python 3.9/Open Babel 3.1 是否仍属于当前支持矩阵；
2. C010：阈值公开接口只收 dataclass，还是保留精确 TypedDict 作为正式第二种输入；
3. C011：非平面环表面的当前化学/几何定义。

### 实施时的测试原则

- 删除兼容测试，不把旧参数应继续工作作为验收条件；
- 增加 `inspect.signature()` 或直接调用测试，锁定当前显式参数名；
- 对未知参数只依赖 Python 的标准 `TypeError`，不再自写翻译/冲突分支；
- Open Babel 版本整改必须在实际保留的 Python/OB 矩阵中做 seed 重复性测试；
- C011 的测试名和 fixture 应描述选定的新几何规则，而不是“preserve historical”。
