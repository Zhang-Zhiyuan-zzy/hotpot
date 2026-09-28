# Hotpot 使用指南（面向 LLM 调用者）

[开发规范](development.md) · [CLI 设计规范](cli_designer.md)

本文档是 **调用（使用）** Hotpot 的权威参考，面向以 Hotpot 完成化学信息学任务的 LLM / 自动化代理，
也适用于人类用户。它回答的是"如何正确调用 Hotpot 的 Python API 与 CLI"，而不是"如何修改 Hotpot"
（见 `development.md`）或"如何设计新命令"（见 `cli_designer.md`）。

文中所有 API 签名均**照抄自当前源码**并核对过行为；凡与 `README.md` 叙述不一致处，以本文为准，
并在 [§3.2](#32-当前不可用--路线图接口切勿据-readme-生成) 明确标注。**不要**从 README 的效果展示
推断可调用接口。

> **总原则**：Hotpot 把 AI 与力场后端隐藏在 `Molecule` / `Atom` / `Bond` 对象背后。你几乎只和这些
> 化学对象打交道，但它们的许多结果是**模型预测或经验力场输出**，带有明确适用域。忠实报告单位、
> 索引基准、后端选择与失败状态，是使用 Hotpot 的硬约束。

---

## 0. 面向 LLM 调用者的黄金规则

生成任何 Hotpot 代码、命令或结论前，MUST 遵守：

1. **`Molecule` 是化学事实源**。外部输入（SMILES、文件、RDKit/OpenBabel 对象）先经
   `hp.to_hotpot_mol(...)` 或 `hp.read_mol(...)` 归一化，再进入后续逻辑。
2. **原子索引默认 0-based**（`atom.idx`）。面向人的输出可能改用 1-based：`hotpot mca` 表格的 `No.`
   列是 1-based，而 `hotpot cbond --bond-detail` 的 `AtomIdx` 是 0-based。**在任何回答里都要写明用的
   是哪一种基准。**
3. **单位必须显式**。能量 kJ/mol（见 `report.optimization.energy_unit`）、梯度 kJ/(mol·Å)、MCA
   kJ/mol、温度 K、压强 Pa。不要凭空换算或省略单位。
4. **AI / 力场输出不是已验证事实**。MCA 值、CBond 连接与概率、`build3d` 几何都是模型/经验结果，
   须按适用域谨慎表述，不得包装成实验测定值或热力学量。
5. **绝不编造数值**。需要 MCA、配位键、能量等具体数字时，实际调用对应 API 或命令；不确定接口时，
   以 `hotpot <cmd> --help` / `--doc` 或源码签名为准。
6. **`build3d` / 复合物构建会派生子进程**。可执行脚本 MUST 使用 `if __name__ == "__main__":` 保护。
7. **Shell 中给 SMILES 加引号**（含 `()=#` 等元字符）。
8. **不要混装 Open Babel**：PyPI `openbabel`、`openbabel-wheel`、Conda `openbabel` 互斥，三选一。
9. **显式后端严格执行**：`--device cuda` 不可用即失败，不会静默回退到 CPU；只有 `auto` 允许自动选择。

---

## 1. 核心心智模型

- 一切围绕 `hotpot.cheminfo.core.Molecule`（顶层可用 `hp.Molecule`）。`Atom` / `Bond` / `Ring` 通过
  `Molecule` 的属性访问。
- 分子文件读取走 Open Babel/pybel（`read_mol` 支持其全部格式）；金属配位、site detection、环感知、
  结果挂载等语义由 Hotpot 自己的图完成。
- 与 RDKit / OpenBabel / PyG 的互转只在边界层发生，且保留源原子顺序。已是 `Molecule` 的输入不会被
  无谓地经 SMILES 往返而丢失坐标或键元数据。

---

## 2. 安装与运行环境

```bash
conda create -n hp python=3.11 pip -y
conda activate hp
python -m pip install hotpot-zzy      # 或源码目录内 pip install -e .
```

- 支持 Python 3.9–3.14（3.10+ 用 Open Babel 3.2.x，3.9 用 `openbabel-wheel` 3.1.1.23）。
- 可选依赖组：`optimize`、`datasets`、`complexformer`、`onnx-export`、`legacy-search`、`dev`、`all`
  （如 `pip install 'hotpot-zzy[optimize]'`）。
- GPU 推理：安装后 `pip uninstall onnxruntime && pip install onnxruntime-gpu`。
- 相关环境变量：`HOTPOT_MCA_DEVICE`、`HOTPOT_MCA_MODEL_DIR`、`HOTPOT_CBOND_MODEL_DIR`。

---

## 3. Python API 速查

### 3.1 顶层入口（`import hotpot as hp`）

以下名字均由 `hotpot/__init__.py` 的 `from .cheminfo import *` 导出，可直接 `hp.<name>` 使用：

| 名称 | 类型 | 用途 |
|:-----|:-----|:-----|
| `read_mol(src, fmt=None, **kw)` | 函数 | 读取并返回 `src` 中**第一个** `Molecule` |
| `to_hotpot_mol(source, fmt=None)` | 函数 | 把任意受支持输入归一化成 `Molecule` |
| `is_molecule_input(obj)` | 函数 | 判断对象能否被转换为 `Molecule` |
| `Molecule` `Atom` `Bond` `BondKind` | 类 | 核心化学对象与键语义枚举 |
| `MolReader` `MolWriter` | 类 | 底层读/写迭代器 |
| `MolBundle` | 类 | 分子集合容器（**仅**打包 + 导出 PyG，见 §3.2） |
| `to_pyg_dataset` | 函数 | 由分子列表构建 PyG 数据集 |
| `Searcher` `Substructure` `Query` `QueryAtom` `QueryBond` `Hit` `Hits` | 类 | 自定义子结构检索 |
| `SmartsSemantics` `SmartsSyntaxError` `UnsupportedSmartsError` | 枚举/异常 | SMARTS 语义与错误 |
| `ComplexStatistics` | 类 | 配合物统计 |
| `Fragment` `EdgeShoulder` `AtomLink` `AlkylGraft` `BondAdding` `AtomReplace` `AssembleFactory` | 类 | 分子组装（见 §10） |
| `atom_link_atom_action` `shoulder_bond_action` `bond_order_add` `atom_replace` `ring_wedge` | 函数 | 组装 action 函数 |

非顶层但常用（需显式导入）：

```python
from hotpot.calculator import mca, formal_charge, MolChargeCalculator   # §9
from hotpot.cheminfo.AImodels.cbond.apply import (                      # §9 进阶
    auto_build_cbond, build_all_possible_cbond, get_cbond_runtime,
)
from hotpot.cheminfo.mol_assemble import RingWedge, alkyl_generator     # RingWedge 未经 import* 导出
```

### 3.2 当前不可用 / 路线图接口（切勿据 README 生成）

`README.md` 含若干**尚未合并**的效果展示，其接口在当前源码中**不存在**。生成代码时务必避开：

| README 里出现 | 现状 | 应改用 |
|:--------------|:-----|:-------|
| `MolBundle.optimize(...)` | **不存在**。`MolBundle` 只有 `__init__(list_mol)` 与 `to_pyg_dataset(...)` | CLI `hotpot optimize`（§11）；或对每个 `mol` 调 `mol.optimize(...)` |
| `Molecule.add_envs(...)` | **不存在** | —（环境变量优化属路线图） |
| `Molecule.descriptors` | **不存在**（仅 README 散文提及） | 用 `mol.graph_spectral()`、`mol.to_pyg_data()` 等真实表示 |
| `Molecule.read_from(...)` | **不存在**（仅旧插件/2023 README 引用） | `hp.read_mol(...)` 或 `Molecule.read_one(...)` |

若用户明确要求"环境感知的贝叶斯/进化优化"这类能力，应说明当前公版仅提供 CLI 参数优化与逐分子
`Molecule.optimize`，完整的 `MolBundle` 级优化在路线图上（README 目标约 2026 年底），而非直接生成
不可运行的示例。

---

## 4. 读取、转换与写出分子

```python
import hotpot as hp

# 读取：src 可为 SMILES/路径/字符串/bytes/StringIO/BytesIO 等
mol = hp.read_mol('c1ccc(O)cc1')            # 无后缀字符串默认按 'smi' 解析
mol = hp.read_mol('c1ccc(O)cc1', 'smi')     # 显式格式（第二个位置参数即 fmt）
mol = hp.read_mol('ligand.mol2')            # 路径按扩展名推断格式
mol = hp.read_mol('struct.data', fmt='mol2')# 无标准扩展名时显式指定

# 归一化任意输入（SMILES/路径/RDKit Mol/OBMol/Pybel/实现 to_rdmol() 的对象/Molecule）
mol = hp.to_hotpot_mol(rdmol)
assert hp.to_hotpot_mol(mol) is mol         # 已是 Molecule 原样返回，不复制

# 多记录文件：MolReader 是迭代器
for m in hp.MolReader('library.sdf'):
    ...

# 写出（fmt 缺省按扩展名；overwrite 控制覆盖）
mol.write('out.mol2')
mol.write('out.sdf', overwrite=True)

# 互转（保留源原子顺序）
obmol = mol.to_obmol()
rdmol = mol.to_rdmol()
data  = mol.to_pyg_data()                   # PyG Data：x / edge_index / rings_attr / coordinates 等
```

- `read_mol(src, fmt=None, **kwargs)` 等价于 `next(hp.MolReader(src, fmt, **kwargs))`，即取第一条记录。
  需要全部记录时遍历 `MolReader`。
- `Molecule.read_one(src, fmt=None, **kwargs)` 是同一功能的静态方法。
- **格式转换**优先用 `read_mol` + `write`（或 `MolReader` / `MolWriter`）。CLI 的 `hotpot convert`
  当前有参数名缺陷（见 §12.4），修复前不建议在自动化中依赖。

---

## 5. 检查分子结构

所有下列成员，**属性**用点号无括号，**方法**要加括号：

```python
mol.atoms            # 属性 → list[Atom]（内部列表的拷贝）
mol.bonds            # 属性 → list[Bond]
mol.c_bonds          # 属性 → 仅配位键
mol.metals           # 属性 → list[Atom]，全部金属原子
mol.components        # 属性 → list[Molecule]，各连通分量
mol.link_matrix      # 属性 → np.ndarray，形状 (n, 2) 的成键原子索引对
mol.heavy_atoms      # 属性 → 重原子
mol.has_metal        # 属性 → bool
mol.formula          # 属性 → 分子式

# 环族：三种语义各有含义
mol.rings                 # 属性 → 全图 Relevant Cycles
mol.ligand_rings          # 属性 → 去金属-配体边后的配体骨架 Relevant Cycles
mol.cycle_basis_rings     # 属性 → 传统 NetworkX cycle-basis（遗留 ring 特征模型仍用它）
mol.rings_for_scope(      # 方法
    "full_graph",         # 或 "ligand_skeleton"
    max_cycles=None,      # 仅当确需无界枚举时；默认有 1 万环安全上限
)
```

- Relevant-Cycle 访问要么返回完整族，要么在默认 1 万环上限处抛 `RelevantCycleLimitExceeded`，
  **不会静默截断**。
- 临时隐藏/恢复配位键（把配合物暂看作 [配体, 金属] 两个片段）：

```python
assert len(pair.components) == 1
pair.hide_metal_ligand_bonds()               # 方法（可选 clear_conformers=False）
assert len(pair.components) == 2
pair.recover_hided_metal_ligand_bonds()
assert len(pair.components) == 1

eu = pair.metals[0]
print(eu.neighbours)                         # 配位原子邻居
```

---

## 6. SMARTS 子结构检索

```python
from hotpot import SmartsSemantics

hits = mol.search_substructure('[Ln](n)(n)(n)O')     # 默认 full_graph 语义 → 返回 Hits
print(len(hits), hits[0].atoms)

# 显式语义 profile（也可用字符串 "full_graph" / "ligand_skeleton"）
full = mol.search_substructure('[N;D4;X4]', semantics=SmartsSemantics.FULL_GRAPH)
lig  = mol.search_substructure('[N;D3;X3]', semantics=SmartsSemantics.LIGAND_SKELETON)
```

- `search_substructure(smarts, *, semantics="full_graph")` → `Hits`。`FULL_GRAPH` 保留完整分子图；
  `LIGAND_SKELETON` 仅在计算配体局部描述符时移除金属–非金属边，**不改动分子本身**。
- 两个 profile 对 `D`/`X`/`v`/`R`/`r` 的取义不同（详见 `README.md` 的语义表与
  `hotpot/cheminfo/smarts.md`）。不支持的立体化学、方向键、同位素约束会**显式报错**
  （`UnsupportedSmartsError`），不会被当作无约束。

配位化学扩展通配符：

| 符号 | 含义 | 示例 |
|:-----|:-----|:-----|
| `M` / `!M` | 任意金属 / 非金属 | `[M]~[O]`、`[!M]` |
| `Ln` / `An` | 镧系（La–Lu）/ 锕系（Ac–Lr） | `[Ln](n)(n)(n)` |
| `NP<n>` | 第 n 周期（支持范围） | `[NP4]`、`[NP3-5]` |
| `NG<n>` | 第 n 族（支持范围） | `[NG1]`、`[NG1-2]` |

需要更复杂的匹配逻辑时，用 `hp.Searcher` / `hp.Substructure` / `hp.QueryAtom` 手工构造检索器
（见 §10 的组装示例）。

---

## 7. 金属配位与 3D 构建

### 7.1 自动配位（2D 连接，AI 模型）

```python
pair = ligand.auto_pair_metal('Eu')                     # → Molecule（配合物）
pair, prob = ligand.auto_pair_metal('Eu', probability=True)
```

`auto_pair_metal(metal, threshold=-0.125, greedy=True, probability=False)`：用 CBond 模型自动加配位键。
`metal` 可为元素符号或原子序数。`probability=True` 时返回 `(Molecule, float)`。这是"预测连接性"，
不涉及氧化态推断或几何优化。

### 7.2 生成并优化 3D 结构

```python
import hotpot as hp

SMILES = "O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C(C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3"

def main():
    pair = hp.read_mol(SMILES).auto_pair_metal("Eu")
    report = pair.build3d(
        seed=20260916,
        max_attempts=20,
        epochs=20,
        steps_per_epoch=500,
    )
    opt = report.optimization                    # 可能为 None，访问前判空
    if opt is not None:
        print(opt.best_energy, opt.energy_unit)  # 能量单位来自对象本身（kJ/mol）
    print(report.effective_forcefield)           # 实际使用的力场
    print(report.quality_report.passed)          # 几何验收是否通过
    pair.write("./Eu-pair.mol2")

if __name__ == "__main__":                       # 复合物构建会派生子进程，必须有 main guard
    main()
```

`build3d(...)` 关键参数（完整签名见 `core.py` 的 `build3d`）：

| 参数 | 默认 | 说明 |
|:-----|:-----|:-----|
| `forcefield` | `'UFF'` | `'UFF'/'MMFF94'/'MMFF94s'/'GAFF'/'Ghemical'`；复合物请求当前解析为 UFF |
| `epochs` / `steps_per_epoch` | `100` / `100` | 优化预算 |
| `quality_level` | `'standard'` | `'off'/'basic'/'standard'/'strict'` 几何验收强度 |
| `seed` | `None` | 随机种子（可复现构象） |
| `max_attempts` | `50` | 尝试次数 |
| `candidate_count` | `None` | **保留参数，当前无效**（多构象在路线图上） |
| `add_hydrogens` | `True` | 缺失氢在事务性工作副本上补齐 |
| `save_movie` | `False` | 保存优化轨迹 |

- 返回对象为 `ForceFieldWorkflowReport` 的子类（简单分子 `BuildAndOptimizeReport`，配合物
  `ComplexBuildReport`）。常用字段：`report.optimization`（`ForceFieldRunReport`：`best_energy`、
  `final_energy`、`energy_unit`、`converged`、`epochs_completed`、`termination_reason`…）、
  `report.effective_forcefield` / `report.requested_forcefield`、`report.quality_report`
  （`.passed` / `.failures` / `.warnings` / `.to_dict()`）。
- 若没有尝试通过基本几何门，Hotpot 会告警并把"确认的键-环穿刺数最少、能量最低"的可用尝试送入下一
  阶段。该流程**不推断氧化态，也不保证配位场几何**。
- 已有可用 3D 坐标时，用 `mol.optimize(forcefield=None, *, epochs=100, steps_per_epoch=100, seed=None, ...)`
  只做优化，不重建结构。

---

## 8. 分子性质、描述符与表示

### 8.1 热力学性质（借助 `thermo` 库）

```python
th = mol.get_thermo(temp=298.15, pressure=101325.0)   # 温度 K，压强 Pa
print(th.Tc)      # 临界温度 (K)
print(th.Psat)    # 饱和蒸气压 (Pa)
```

### 8.2 图谱表示与相似度

```python
s1 = mol1.graph_spectral()          # 方法，返回 GraphSpectrum（默认 norm='l2'）
s2 = mol2.graph_spectral()
print(s1 | s2)                      # __or__ → 相似度 float（等价 s1.similarity(s2)）
print(s1.vectors.shape)             # .vectors 是谱矩阵 np.ndarray 的别名
```

`GraphSpectrum` 支持：`|`（相似度）、`.vectors` / `.spectrum`（谱矩阵）、`.similarity(other)`、
`.width`、`.norm`。相似度对原子重排不变（同一分子不同原子序 → 1.0）。

### 8.3 PyG 图表示

`mol.to_pyg_data(prefix="", with_batch=True)` 返回 PyG `Data`，含 `x` / `x_names` / `edge_index` /
`edge_attr` / `pair_index` / `rings_node_index` / `rings_attr`（列 `['is_aromatic','has_metal']`）/
`coordinates` 等字段。批量构建数据集用 `hp.to_pyg_dataset(list_mol, dataset_root, prefix='')`。

---

## 9. AI 预测的 Python 接口

### 9.1 MCA（甲基阳离子亲和能，kJ/mol）

```python
from hotpot import read_mol
from hotpot.calculator import mca

mol = read_mol("c1ccccc1CN")
prediction = mca(mol)                       # 就地把结果挂到原子/分子上
for atom in mol.atoms:
    print(atom.idx, atom.symbol, atom.mca)  # 全原子 MCA（kJ/mol），atom.idx 为 0-based
for atom, value in mol.mca_sites.items():
    print(atom, value)                      # 经适用域规则筛出的可靠亲核位点
```

`mca(mol, *, device=None, allow_charged=False)`：`device` 缺省读 `HOTPOT_MCA_DEVICE`（默认 `auto`）。
**适用域**：模型在中性分子上验证；带电分子默认拒绝，`allow_charged=True` 得到的是显式域外估计，须
谨慎。显式氢不是有效预测目标；单分子最多 512 个原子。**全原子输出**与**可靠位点选择**是两件事——
后者刻意排除金属及直接连金属的原子。

### 9.2 形式电荷

```python
from hotpot.calculator import formal_charge

charges = formal_charge(mol, model="valence")            # → tuple[int, ...]，按原子顺序
# model: "valence" | "valence-constrained" | "preserve"
# 金属：metal_model="default" | "preserve" | Callable(atom, mol) -> int
```

从 Hotpot 原生图用经典价键规则赋整数形式电荷，**不调用** RDKit/OpenBabel 电荷器。含金属时先按
配位键拆分，配体片段与金属分别处理。该模型不选质子化态、不算部分电荷、不推断过渡金属氧化态。

### 9.3 CBond 进阶（枚举/细节）

高层用 §7.1 的 `mol.auto_pair_metal(...)`。需要枚举全部终态或拿到每步分数时，用底层 API：

```python
from hotpot.cheminfo.AImodels.cbond.apply import (
    get_cbond_runtime, auto_build_cbond, build_all_possible_cbond,
)
runtime = get_cbond_runtime(device=None, model_dir=None)
result  = auto_build_cbond(mol, "Eu", threshold=-0.125, greedy=True, runtime=runtime,
                           return_details=True)
```

候选配位原子当前限于 **O、N、S、P、Si、B**；单次仅支持一个金属中心 + 一个配体图；打包运行时最多
32 个配体环、单环最多 64 原子。分数是原始 logit，排名概率是归一化路径权重（非校准物理概率）。

---

## 10. 虚拟分子组装

`hotpot.cheminfo.mol_assemble` 依据框架分子（`Molecule`）与片段（`Fragment`）迭代生成虚拟结构。
详细教程见 `hotpot/cheminfo/mol_assemble/README.md`。

```python
import hotpot as hp

frame = hp.read_mol('c1ccccc1')                 # 苯作为框架
asm   = hp.EdgeShoulder(hp.read_mol('c1ccc[nH]1'), action_points=(0, 1))
products = asm.graft(frame)                      # → dict[str, Molecule]，键为产物 SMILES（去重）
```

预定义 Assembler（均为 `Fragment` 子类，都有 `.graft(frame)`）：

| 类 | 构造 | action_points |
|:---|:-----|:--------------|
| `EdgeShoulder` | `(mol, action_points: tuple[int,int])` | 需 2 个 |
| `AtomLink` | `(mol, action_points: tuple[int])` | 需 1 个 |
| `AlkylGraft` | `(mol)` | 无（`AtomLink` 特例；批量用 `alkyl_generator(lengths)`） |
| `BondAdding` | `()` | 无 |
| `AtomReplace` | `(element: str)` | 无（`element` ∈ `{'N','O','Si','S'}`） |
| `RingWedge` | `(mol, action_points: tuple[int])` | 需 1 个（**未经 `import *` 导出，需显式导入**） |

- 自定义策略：`Fragment(mol, searcher, action_points, action_func, exclude_searcher=None)`，其中
  `searcher` 是 `hp.Searcher`，`action_func` 签名固定为
  `f(mol, hit: list[int], frag: Molecule, action_points: list[int]) -> Molecule`。
- 高通量：`hp.AssembleFactory(assembler, iter_step=5, mode='random', seed=None, ...)`，用 `.make(frames)`
  单核或 `.mp_make(frames, nproc=..., batch_size=...)` 多进程；`AssembleFactory.load_assembler_file(json)`
  可从模板批量定义。签名细节以模块 README 与源码为准。

---

## 11. 实验参数优化

**纯参数优化走 CLI**（Excel 进，`output_dir` 出优化推荐与流形可视化）：

```bash
hotpot optimize <input_excel> <output_dir> [--flags ...]
hotpot optimize --help          # 权威参数以此为准
```

`input_excel` 组织为 `feature1 … featureN, target` 的表格。

**逐分子几何优化**用 `Molecule.optimize(...)`（§7.2）。README 中"分子结构参与的贝叶斯优化"
（`MolBundle.optimize(...)` + `Molecule.add_envs(...)`）属**路线图接口，当前源码不可用**（见 §3.2），
不要据此生成代码。

---

## 12. 命令行接口（CLI）

统一入口 `hotpot`，Git 风格子命令：`hotpot <command> <inputs> [options]`。全局：`-v/--version`、
`-d/--debug`、`-b/--background`。已注册子命令：`convert`、`optimize`、`ml_train`、`mca`、`cbond`。

### 12.1 总则与流契约

- **stdout 只放承诺的结果数据**；日志、警告、进度、ONNX telemetry 一律走 stderr。
- `-o/--output FILE` 写入的 payload 与默认 stdout 相同，且用 `-o` 时 stdout 为空；文本输出 UTF-8。
- 退出码：成功 0，失败非零。`--device cuda` 严格，不可用即失败。
- `--help` 简明、随包可得；`mca` / `cbond` 另有 `--doc` 展示详细 Markdown 文档，且缺业务参数时仍可退出。

### 12.2 `hotpot mca` — 原子级 MCA 预测

```bash
hotpot mca 'c1ccccc1CN'                       # 单分子（SMILES 要加引号）
hotpot mca 'CN' 'CCO' mols.smi inputs/*.mol2  # 多 SMILES / 文件 / 多记录混合，批量推理
hotpot mca mols.sdf -o mca.txt                # 保存文本表
hotpot mca 'c1ccccc1CN' --plot mca.png        # 仅着色检出的亲核位点
hotpot mca 'c1ccccc1CN' --plot all.png --all-site   # 着色全部原子
hotpot mca '[NH4+]' --allow-charged           # 显式请求域外估计
```

输出表：`No.`（**1-based** 原子序号）、`Atom`、`MCA(kJ/mol)`、`is_Nuc_site`。`is_Nuc_site=True` 只表示
命中亲核规则，不影响是否给出 MCA 值。选项：`-o/--output`、`--plot IMAGE`、`--all-site`、
`--input-format`、`--device {auto,cpu,cuda}`、`--variant {fp32,fp16}`（默认打包 fp16）、
`--batch-size 64`（每次 ONNX 调用的原子-位点行数）、`--conformer-seed 42`、`--model-dir`、
`--allow-charged`。

### 12.3 `hotpot cbond` — 金属-配体配位键构建

```bash
hotpot cbond Eu 'CN'                                  # 默认贪心，输出一行 SMILES
hotpot cbond 63 'CN'                                  # 金属可用原子序数
hotpot cbond Eu ligand.mol2 -o complex.smi
hotpot cbond Eu 'NCC(O)CO' --all-structures           # 枚举并排序全部终态
hotpot cbond Eu 'CN' --bond-detail                    # 显示每键原始分数
hotpot cbond --doc                                    # 完整文档
```

要点：
- 默认贪心结果**不等于** `--all-structures` 归并后的 Rank 1。
- `--all-structures` 的 `Prob` 是**归一化相对路径权重**，不是校准物理概率/平衡布居/热力学量。不同
  `--threshold` 下的 `Prob` 不可直接比较。
- `--bond-detail` 的 `Score` 是**原始 logit**，`AtomIdx` 是 **0-based** Hotpot 索引（非文件中的原子序号）。
- `--threshold` 默认 `-0.125`，严格 `score > threshold`；`--max-states` 默认 4096（仅 `--all-structures`
  用，超限抛 `RuntimeError`）；`--no-greedy`、`--device`、`--model-dir`。
- **空结果语义有别**：单结构模式退出码 1 并抛 `ValueError`；`--all-structures` 模式退出码 0 并打印
  "No coordination structures exceeded the threshold."
- 只预测连接性，不做几何优化/结合能/稳定性。加了 rank 或 detail 后输出已非单记录 SMILES，请用文本类
  扩展名保存。

### 12.4 `hotpot optimize` / `ml_train` / `convert`

- `hotpot optimize <excel> <output_dir>`：见 §11，详参 `hotpot optimize --help`。
- `hotpot ml_train ...`：标准 ML 训练流程，参数以 `hotpot ml_train --help` 为准。
- `hotpot convert <infile> ...`：**当前存在缺陷**——分发逻辑读取 `args.outputs_format`，而 parser 注册的
  是 `--output-format`（dest 为 `output_format`），会触发 `AttributeError`。修复前，格式转换请用
  Python 的 `hp.read_mol` + `mol.write`（§4）。

### 12.5 `--help` / `--doc` 是当前接口的权威来源

CLI 选项、默认值、单位与退出语义可能随版本演进。**面向自动化时，先以 `hotpot <cmd> --help`
（`mca`/`cbond` 再加 `--doc`）确认当前接口**，再据此生成命令，而不是照抄本文或 README 的历史示例。

---

## 13. 任务 → 接口对照表

| 任务 | Python | CLI |
|:-----|:-------|:----|
| 读入分子 | `hp.read_mol` / `hp.to_hotpot_mol` / `hp.MolReader` | — |
| 格式转换 / 写出 | `mol.write` | （`hotpot convert` 暂缺陷，见 §12.4） |
| 转 RDKit / OpenBabel / PyG | `mol.to_rdmol` / `mol.to_obmol` / `mol.to_pyg_data` | — |
| 找配位中心 / 子结构 | `mol.search_substructure(SMARTS)` | — |
| 金属-配体配位键（2D） | `mol.auto_pair_metal` / `cbond.apply.*` | `hotpot cbond` |
| 生成 / 优化 3D | `mol.build3d` / `mol.optimize` | — |
| MCA / 亲核位点 | `hotpot.calculator.mca` | `hotpot mca` |
| 形式电荷 | `hotpot.calculator.formal_charge` | — |
| 热力学性质 | `mol.get_thermo` | — |
| 图谱相似度 | `mol.graph_spectral` | — |
| 虚拟分子组装 | `hp.EdgeShoulder`/`AtomLink`/… + `AssembleFactory` | — |
| 实验参数优化 | （路线图，见 §11） | `hotpot optimize` |
| 构建 PyG 数据集 | `hp.to_pyg_dataset` | — |

---

## 14. 常见反模式（LLM 调用者须避免）

- ❌ 依据 README 生成 `MolBundle.optimize(...)`、`mol.add_envs(...)`、`mol.descriptors`、
  `Molecule.read_from(...)` —— 这些在当前源码中不存在（§3.2）。
- ❌ 把 `mol.rings` / `mol.metals` / `mol.smiles` 当方法加括号，或把 `mol.graph_spectral` /
  `mol.rings_for_scope` 当属性不加括号 —— 前者是属性、后者是方法。
- ❌ 把 MCA / CBond `Prob` / `build3d` 能量当成实验值或热力学量陈述。
- ❌ 报告原子索引却不写明 0-based 还是 1-based；给出能量/MCA 却省略单位。
- ❌ 在无 `if __name__ == "__main__":` 的脚本里跑 `build3d` / 复合物构建。
- ❌ 对带电分子直接跑 MCA 而不提适用域，或把 `--allow-charged` 结果当可靠值。
- ❌ 在 shell 里不加引号传含 `()=#` 的 SMILES。
- ❌ 假设 `--device cuda` 会在无 GPU 时回退到 CPU。
- ❌ 需要具体数字时凭记忆编造，而不实际调用 API / 命令。
