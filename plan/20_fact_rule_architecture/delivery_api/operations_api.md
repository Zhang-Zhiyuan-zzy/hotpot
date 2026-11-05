# 命名操作、科学状态、力场与 I/O：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot.cheminfo.elements`](#hotpot-cheminfo-elements)
- [`hotpot.cheminfo.operations`](#hotpot-cheminfo-operations)
- [`hotpot.cheminfo.operations._aromatic_protocol`](#hotpot-cheminfo-operations--aromatic-protocol)
- [`hotpot.cheminfo.operations._carrier_signatures`](#hotpot-cheminfo-operations--carrier-signatures)
- [`hotpot.cheminfo.operations.assembly`](#hotpot-cheminfo-operations-assembly)
- [`hotpot.cheminfo.operations.assembly.action_func`](#hotpot-cheminfo-operations-assembly-action-func)
- [`hotpot.cheminfo.operations.assembly.assembler`](#hotpot-cheminfo-operations-assembly-assembler)
- [`hotpot.cheminfo.operations.assembly.fragment`](#hotpot-cheminfo-operations-assembly-fragment)
- [`hotpot.cheminfo.operations.calculation`](#hotpot-cheminfo-operations-calculation)
- [`hotpot.cheminfo.operations.calculation.scientific_state`](#hotpot-cheminfo-operations-calculation-scientific-state)
- [`hotpot.cheminfo.operations.calculation.scientific_state._rule_serialization`](#hotpot-cheminfo-operations-calculation-scientific-state--rule-serialization)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition`](#hotpot-cheminfo-operations-calculation-scientific-state-composition)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition._artifacts`](#hotpot-cheminfo-operations-calculation-scientific-state-composition--artifacts)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition._candidate_space_serialization`](#hotpot-cheminfo-operations-calculation-scientific-state-composition--candidate-space-serialization)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition._decision_snapshot`](#hotpot-cheminfo-operations-calculation-scientific-state-composition--decision-snapshot)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition._serialization`](#hotpot-cheminfo-operations-calculation-scientific-state-composition--serialization)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.candidate_space`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-candidate-space)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.compose`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-compose)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.constraints`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-constraints)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.coupling`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-coupling)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.decision_search`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-decision-search)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.derive`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-derive)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.expand`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-expand)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.resolution`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-resolution)
- [`hotpot.cheminfo.operations.calculation.scientific_state.composition.validation`](#hotpot-cheminfo-operations-calculation-scientific-state-composition-validation)
- [`hotpot.cheminfo.operations.calculation.scientific_state.contracts`](#hotpot-cheminfo-operations-calculation-scientific-state-contracts)
- [`hotpot.cheminfo.operations.calculation.scientific_state.defaults`](#hotpot-cheminfo-operations-calculation-scientific-state-defaults)
- [`hotpot.cheminfo.operations.calculation.scientific_state.execution_selection`](#hotpot-cheminfo-operations-calculation-scientific-state-execution-selection)
- [`hotpot.cheminfo.operations.calculation.scientific_state.facts`](#hotpot-cheminfo-operations-calculation-scientific-state-facts)
- [`hotpot.cheminfo.operations.calculation.scientific_state.manifest`](#hotpot-cheminfo-operations-calculation-scientific-state-manifest)
- [`hotpot.cheminfo.operations.calculation.scientific_state.operation_requirements`](#hotpot-cheminfo-operations-calculation-scientific-state-operation-requirements)
- [`hotpot.cheminfo.operations.calculation.scientific_state.profile_decision`](#hotpot-cheminfo-operations-calculation-scientific-state-profile-decision)
- [`hotpot.cheminfo.operations.calculation.scientific_state.profile_eligibility`](#hotpot-cheminfo-operations-calculation-scientific-state-profile-eligibility)
- [`hotpot.cheminfo.operations.calculation.scientific_state.profiles`](#hotpot-cheminfo-operations-calculation-scientific-state-profiles)
- [`hotpot.cheminfo.operations.calculation.scientific_state.rule_controller`](#hotpot-cheminfo-operations-calculation-scientific-state-rule-controller)
- [`hotpot.cheminfo.operations.calculation.scientific_state.rule_registry`](#hotpot-cheminfo-operations-calculation-scientific-state-rule-registry)
- [`hotpot.cheminfo.operations.calculation.scientific_state.rule_validation`](#hotpot-cheminfo-operations-calculation-scientific-state-rule-validation)
- [`hotpot.cheminfo.operations.calculation.scientific_state.state_resolver`](#hotpot-cheminfo-operations-calculation-scientific-state-state-resolver)
- [`hotpot.cheminfo.operations.calculation.scientific_state.state_selection`](#hotpot-cheminfo-operations-calculation-scientific-state-state-selection)
- [`hotpot.cheminfo.operations.chemistry`](#hotpot-cheminfo-operations-chemistry)
- [`hotpot.cheminfo.operations.collections`](#hotpot-cheminfo-operations-collections)
- [`hotpot.cheminfo.operations.conformers`](#hotpot-cheminfo-operations-conformers)
- [`hotpot.cheminfo.operations.conformers.contracts`](#hotpot-cheminfo-operations-conformers-contracts)
- [`hotpot.cheminfo.operations.conformers.openbabel`](#hotpot-cheminfo-operations-conformers-openbabel)
- [`hotpot.cheminfo.operations.conformers.rdkit`](#hotpot-cheminfo-operations-conformers-rdkit)
- [`hotpot.cheminfo.operations.conformers.sampler`](#hotpot-cheminfo-operations-conformers-sampler)
- [`hotpot.cheminfo.operations.conversion`](#hotpot-cheminfo-operations-conversion)
- [`hotpot.cheminfo.operations.crystal`](#hotpot-cheminfo-operations-crystal)
- [`hotpot.cheminfo.operations.crystal_generation`](#hotpot-cheminfo-operations-crystal-generation)
- [`hotpot.cheminfo.operations.crystal_geometry`](#hotpot-cheminfo-operations-crystal-geometry)
- [`hotpot.cheminfo.operations.crystal_io`](#hotpot-cheminfo-operations-crystal-io)
- [`hotpot.cheminfo.operations.crystal_matching`](#hotpot-cheminfo-operations-crystal-matching)
- [`hotpot.cheminfo.operations.descriptors`](#hotpot-cheminfo-operations-descriptors)
- [`hotpot.cheminfo.operations.forcefields`](#hotpot-cheminfo-operations-forcefields)
- [`hotpot.cheminfo.operations.forcefields.acceptance`](#hotpot-cheminfo-operations-forcefields-acceptance)
- [`hotpot.cheminfo.operations.forcefields.attempts`](#hotpot-cheminfo-operations-forcefields-attempts)
- [`hotpot.cheminfo.operations.forcefields.auto`](#hotpot-cheminfo-operations-forcefields-auto)
- [`hotpot.cheminfo.operations.forcefields.backend`](#hotpot-cheminfo-operations-forcefields-backend)
- [`hotpot.cheminfo.operations.forcefields.contracts`](#hotpot-cheminfo-operations-forcefields-contracts)
- [`hotpot.cheminfo.operations.forcefields.coordinates`](#hotpot-cheminfo-operations-forcefields-coordinates)
- [`hotpot.cheminfo.operations.forcefields.coordination`](#hotpot-cheminfo-operations-forcefields-coordination)
- [`hotpot.cheminfo.operations.forcefields.ff`](#hotpot-cheminfo-operations-forcefields-ff)
- [`hotpot.cheminfo.operations.forcefields.ligand`](#hotpot-cheminfo-operations-forcefields-ligand)
- [`hotpot.cheminfo.operations.forcefields.native_adapters`](#hotpot-cheminfo-operations-forcefields-native-adapters)
- [`hotpot.cheminfo.operations.forcefields.optimizer`](#hotpot-cheminfo-operations-forcefields-optimizer)
- [`hotpot.cheminfo.operations.forcefields.repair`](#hotpot-cheminfo-operations-forcefields-repair)
- [`hotpot.cheminfo.operations.forcefields.stage`](#hotpot-cheminfo-operations-forcefields-stage)
- [`hotpot.cheminfo.operations.forcefields.trajectory`](#hotpot-cheminfo-operations-forcefields-trajectory)
- [`hotpot.cheminfo.operations.forcefields.workers`](#hotpot-cheminfo-operations-forcefields-workers)
- [`hotpot.cheminfo.operations.forcefields.workflows`](#hotpot-cheminfo-operations-forcefields-workflows)
- [`hotpot.cheminfo.operations.forcefields.working_copy`](#hotpot-cheminfo-operations-forcefields-working-copy)
- [`hotpot.cheminfo.operations.geometry`](#hotpot-cheminfo-operations-geometry)
- [`hotpot.cheminfo.operations.io`](#hotpot-cheminfo-operations-io)
- [`hotpot.cheminfo.operations.native_packing`](#hotpot-cheminfo-operations-native-packing)
- [`hotpot.cheminfo.operations.search`](#hotpot-cheminfo-operations-search)
- [`hotpot.cheminfo.operations.search.native`](#hotpot-cheminfo-operations-search-native)
- [`hotpot.cheminfo.operations.search.predicates`](#hotpot-cheminfo-operations-search-predicates)
- [`hotpot.cheminfo.operations.search.search`](#hotpot-cheminfo-operations-search-search)
- [`hotpot.cheminfo.operations.search.semantics`](#hotpot-cheminfo-operations-search-semantics)
- [`hotpot.cheminfo.operations.search.smarts`](#hotpot-cheminfo-operations-search-smarts)
- [`hotpot.cheminfo.operations.solvent_properties`](#hotpot-cheminfo-operations-solvent-properties)
- [`hotpot.cheminfo.operations.stereochemistry`](#hotpot-cheminfo-operations-stereochemistry)
- [`hotpot.cheminfo.operations.thermo`](#hotpot-cheminfo-operations-thermo)
- [`hotpot.cheminfo.operations.topology`](#hotpot-cheminfo-operations-topology)
- [`hotpot.cheminfo.operations.topology_queries`](#hotpot-cheminfo-operations-topology-queries)
- [`hotpot.cheminfo.operations.visualization`](#hotpot-cheminfo-operations-visualization)
- [`hotpot.cheminfo.pubchem`](#hotpot-cheminfo-pubchem)
- [`hotpot.cheminfo.rdworks`](#hotpot-cheminfo-rdworks)

<a id="hotpot-cheminfo-elements"></a>
## `hotpot.cheminfo.elements`

`hotpot/cheminfo/elements.py`；显式 __all__: ('elements', 'element_properties', 'PropertiesSheet')


<a id="hotpot-cheminfo-elements-propertiessheet"></a>
### `PropertiesSheet`

定义 owner `hotpot.cheminfo.elements.PropertiesSheet`；基类 无。

- `__init__(self, sheet: pd.DataFrame=None)`
- `__getitem__(self, item)`
- `get_by_properties(self, properties: Union[str, list])`
- `get_by_elements(self, ele: Union[str, list])`
- `drop_units(self)`
- `get_units(self)`
- `complete_values(self)`
- `number_values(self)`
- `shared_sheet(self)`
- `normalize_sheet(self)`

<a id="hotpot-cheminfo-elements-element"></a>
### `Element`

定义 owner `hotpot.cheminfo.elements.Element`；基类 无。

- `__dir__(self)`
- `__getattr__(self, item)`
- `__getitem__(self, item: Union[int, str])`
- `symbols` = `_reference._ELEMENT_SYMBOLS` [类声明值；可重绑定不构成事实写许可]
- `atomic_orbital` = `_mutable_copy(_reference._ATOMIC_ORBITAL)` [类声明值；可重绑定不构成事实写许可]
- `default_valence` = `_mutable_copy(_reference._DEFAULT_VALENCE)` [类声明值；可重绑定不构成事实写许可]
- `valence_dict` = `_mutable_copy(_reference._VALENCE_DICT)` [类声明值；可重绑定不构成事实写许可]
- `atomic_radii` = `_mutable_copy(_reference._ATOMIC_RADII)` [类声明值；可重绑定不构成事实写许可]
- `ionic_radii` = `_mutable_copy(_reference._IONIC_RADII)` [类声明值；可重绑定不构成事实写许可]
- `ionization_energies` = `_mutable_copy(_reference._IONIZATION_ENERGIES)` [类声明值；可重绑定不构成事实写许可]
- `element_densities` = `_mutable_copy(_reference._ELEMENT_DENSITIES)` [类声明值；可重绑定不构成事实写许可]
- `electronegativity` = `_mutable_copy(_reference._ELECTRONEGATIVITY)` [类声明值；可重绑定不构成事实写许可]
- `electron_configs` = `_reference._ELECTRON_CONFIGS` [类声明值；可重绑定不构成事实写许可]
- `Z_TO_GROUP` = `_mutable_copy(_reference._Z_TO_GROUP)` [类声明值；可重绑定不构成事实写许可]
- `alkali_metals` = `set(_reference._ALKALI_METALS)` [类声明值；可重绑定不构成事实写许可]
- `alkaline_earth_metals` = `set(_reference._ALKALINE_EARTH_METALS)` [类声明值；可重绑定不构成事实写许可]
- `transition_metals` = `set(_reference._TRANSITION_METALS)` [类声明值；可重绑定不构成事实写许可]
- `post_transition_metals` = `set(_reference._POST_TRANSITION_METALS)` [类声明值；可重绑定不构成事实写许可]
- `lanthanides` = `set(_reference._LANTHANIDES)` [类声明值；可重绑定不构成事实写许可]
- `actinides` = `set(_reference._ACTINIDES)` [类声明值；可重绑定不构成事实写许可]
- `metal` = `set(_reference._METAL)` [类声明值；可重绑定不构成事实写许可]
- `metalloid_1st` = `set(_reference._METALLOID_1ST)` [类声明值；可重绑定不构成事实写许可]
- `metalloid_2nd` = `set(_reference._METALLOID_2ND)` [类声明值；可重绑定不构成事实写许可]
- `metalloid` = `set(_reference._METALLOID)` [类声明值；可重绑定不构成事实写许可]
- `nonmetals` = `_mutable_copy(_reference._NONMETALS)` [类声明值；可重绑定不构成事实写许可]
- `metalloids` = `_mutable_copy(_reference._METALLOIDS)` [类声明值；可重绑定不构成事实写许可]
- `noble_gases` = `_mutable_copy(_reference._NOBLE_GASES)` [类声明值；可重绑定不构成事实写许可]
- `halogens` = `_mutable_copy(_reference._HALOGENS)` [类声明值；可重绑定不构成事实写许可]
- `covalent_radii` = `_mutable_copy(_reference._COVALENT_RADII)` [类声明值；可重绑定不构成事实写许可]
- `density` = `_mutable_copy(_reference._DENSITY)` [类声明值；可重绑定不构成事实写许可]

模块值/类型别名（声明存在不构成事实写许可）：

- `element_properties` = `_element_properties_snapshot()`
- `elements` = `Element()`


<a id="hotpot-cheminfo-operations"></a>
## `hotpot.cheminfo.operations`

`hotpot/cheminfo/operations/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations--aromatic-protocol"></a>
## `hotpot.cheminfo.operations._aromatic_protocol`

`hotpot/cheminfo/operations/_aromatic_protocol.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations--carrier-signatures"></a>
## `hotpot.cheminfo.operations._carrier_signatures`

`hotpot/cheminfo/operations/_carrier_signatures.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-assembly"></a>
## `hotpot.cheminfo.operations.assembly`

`hotpot/cheminfo/operations/assembly/__init__.py`；显式 __all__: ('Fragment', 'EdgeShoulder', 'AtomLink', 'AlkylGraft', 'BondAdding', 'AtomReplace', 'RingWedge', 'AssembleFactory', 'alkyl_generator', 'atom_link_atom_action', 'shoulder_bond_action', 'bond_order_add', 'atom_replace', 'ring_wedge')

- 导出 `Fragment` → `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- 导出 `EdgeShoulder` → `hotpot.cheminfo.operations.assembly.assembler.EdgeShoulder`
- 导出 `AtomLink` → `hotpot.cheminfo.operations.assembly.assembler.AtomLink`
- 导出 `AlkylGraft` → `hotpot.cheminfo.operations.assembly.assembler.AlkylGraft`
- 导出 `BondAdding` → `hotpot.cheminfo.operations.assembly.assembler.BondAdding`
- 导出 `AtomReplace` → `hotpot.cheminfo.operations.assembly.assembler.AtomReplace`
- 导出 `RingWedge` → `hotpot.cheminfo.operations.assembly.assembler.RingWedge`
- 导出 `AssembleFactory` → `hotpot.cheminfo.operations.assembly.assembler.AssembleFactory`
- 导出 `alkyl_generator` → `hotpot.cheminfo.operations.assembly.assembler.alkyl_generator`；`alkyl_generator(lengths: Iterable[int])`
- 导出 `atom_link_atom_action` → `hotpot.cheminfo.operations.assembly.action_func.atom_link_atom_action`；`atom_link_atom_action(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])`
- 导出 `shoulder_bond_action` → `hotpot.cheminfo.operations.assembly.action_func.shoulder_bond_action`；`shoulder_bond_action(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])`
- 导出 `bond_order_add` → `hotpot.cheminfo.operations.assembly.action_func.bond_order_add`；`bond_order_add(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])`
- 导出 `atom_replace` → `hotpot.cheminfo.operations.assembly.action_func.atom_replace`；`atom_replace(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])`
- 导出 `ring_wedge` → `hotpot.cheminfo.operations.assembly.action_func.ring_wedge`；`ring_wedge(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])`


<a id="hotpot-cheminfo-operations-assembly-action-func"></a>
## `hotpot.cheminfo.operations.assembly.action_func`

`hotpot/cheminfo/operations/assembly/action_func.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-assembly-action-func-actions-register"></a>
### `actions_register`

```python
actions_register(register_key: str)
```

<a id="hotpot-cheminfo-operations-assembly-action-func-atom-link-atom-action"></a>
### `atom_link_atom_action`

```python
atom_link_atom_action(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])
```

<a id="hotpot-cheminfo-operations-assembly-action-func-bond-order-add"></a>
### `bond_order_add`

```python
bond_order_add(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])
```

<a id="hotpot-cheminfo-operations-assembly-action-func-atom-replace"></a>
### `atom_replace`

```python
atom_replace(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])
```

<a id="hotpot-cheminfo-operations-assembly-action-func-shoulder-bond-action"></a>
### `shoulder_bond_action`

```python
shoulder_bond_action(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])
```

<a id="hotpot-cheminfo-operations-assembly-action-func-ring-wedge"></a>
### `ring_wedge`

```python
ring_wedge(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int])
```

模块值/类型别名（声明存在不构成事实写许可）：

- `DefaultAction` = `Literal['atom_link', 'bond_shoulder']`
- `ActionFuncTemplate` = `Union[Callable[[Molecule, list[int], Molecule, list[int]], Molecule], DefaultAction]`


<a id="hotpot-cheminfo-operations-assembly-assembler"></a>
## `hotpot.cheminfo.operations.assembly.assembler`

`hotpot/cheminfo/operations/assembly/assembler.py`；显式 __all__: ('EdgeShoulder', 'AtomLink', 'AlkylGraft', 'BondAdding', 'AtomReplace', 'RingWedge', 'AssembleFactory')


<a id="hotpot-cheminfo-operations-assembly-assembler-has-hydrogen"></a>
### `has_hydrogen`

```python
has_hydrogen(atom: Atom) -> bool
```

<a id="hotpot-cheminfo-operations-assembly-assembler-max-heavy-bond-order"></a>
### `max_heavy_bond_order`

```python
max_heavy_bond_order(atom: Atom, _max_value: int) -> bool
```

<a id="hotpot-cheminfo-operations-assembly-assembler-edgeshoulder"></a>
### `EdgeShoulder`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.EdgeShoulder`；基类 `Fragment`。

- `__init__(self, mol, action_points: tuple[int, int])`
- `searcher` = `Searcher(_subs)` [类声明值；可重绑定不构成事实写许可]
- `__repr__(self)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `graft(self, frame: Molecule) -> dict[str, Molecule]` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_sub(self, sub: Substructure)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_mol(self, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_smi(self, smi: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `frag` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_points` = `list(action_points)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_func` = `action_func` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `exclude_searcher` = `exclude_searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`

<a id="hotpot-cheminfo-operations-assembly-assembler-atomlink"></a>
### `AtomLink`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.AtomLink`；基类 `Fragment`。

- `__init__(self, mol, action_points: tuple[int])`
- `searcher` = `Searcher(_subs)` [类声明值；可重绑定不构成事实写许可]
- `__repr__(self)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `graft(self, frame: Molecule) -> dict[str, Molecule]` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_sub(self, sub: Substructure)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_mol(self, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_smi(self, smi: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `frag` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_points` = `list(action_points)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_func` = `action_func` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `exclude_searcher` = `exclude_searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`

<a id="hotpot-cheminfo-operations-assembly-assembler-alkylgraft"></a>
### `AlkylGraft`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.AlkylGraft`；基类 `AtomLink`。

- `__init__(self, mol: Molecule)`
- `create_alkyl_collection(alkyl_length: Iterable[int])` [staticmethod]
- `searcher` = `Searcher(_subs)` [类声明值；可重绑定不构成事实写许可]
- `__repr__(self)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `graft(self, frame: Molecule) -> dict[str, Molecule]` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_sub(self, sub: Substructure)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_mol(self, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_smi(self, smi: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `frag` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_points` = `list(action_points)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_func` = `action_func` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `exclude_searcher` = `exclude_searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`

<a id="hotpot-cheminfo-operations-assembly-assembler-alkyl-generator"></a>
### `alkyl_generator`

```python
alkyl_generator(lengths: Iterable[int])
```

<a id="hotpot-cheminfo-operations-assembly-assembler-bondadding"></a>
### `BondAdding`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.BondAdding`；基类 `Fragment`。

- `__init__(self)`
- `searcher` = `Searcher(_subs)` [类声明值；可重绑定不构成事实写许可]
- `__repr__(self)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `graft(self, frame: Molecule) -> dict[str, Molecule]` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_sub(self, sub: Substructure)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_mol(self, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_smi(self, smi: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `frag` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_points` = `list(action_points)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_func` = `action_func` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `exclude_searcher` = `exclude_searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`

<a id="hotpot-cheminfo-operations-assembly-assembler-atomreplace"></a>
### `AtomReplace`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.AtomReplace`；基类 `Fragment`。

- `__init__(self, element: str)`
- `__repr__(self)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `graft(self, frame: Molecule) -> dict[str, Molecule]` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_sub(self, sub: Substructure)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_mol(self, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_smi(self, smi: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `frag` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `searcher` = `searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_points` = `list(action_points)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_func` = `action_func` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `exclude_searcher` = `exclude_searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`

<a id="hotpot-cheminfo-operations-assembly-assembler-not-amidoxime-nito"></a>
### `not_amidoxime_nito`

```python
not_amidoxime_nito(atom: Atom)
```

<a id="hotpot-cheminfo-operations-assembly-assembler-ringwedge"></a>
### `RingWedge`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.RingWedge`；基类 `Fragment`。

- `__init__(self, mol: Molecule, action_points: tuple[int])`
- `searcher` = `Searcher(_sub)` [类声明值；可重绑定不构成事实写许可]
- `__repr__(self)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `graft(self, frame: Molecule) -> dict[str, Molecule]` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_sub(self, sub: Substructure)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_mol(self, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `add_exclude_from_smi(self, smi: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `frag` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_points` = `list(action_points)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `action_func` = `action_func` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- `exclude_searcher` = `exclude_searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.assembly.fragment.Fragment`

<a id="hotpot-cheminfo-operations-assembly-assembler-molbatch"></a>
### `MolBatch`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.MolBatch`；基类 无。

- `__init__(self, iter_smiles: Iterable[Molecule], batch_size: int=100)`
- `__len__(self)`
- `__next__(self)`
- `__iter__(self)`
- `refresh(self)`
- `list_smiles` = `list(iter_smiles)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `batch_size` = `batch_size` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-assembly-assembler-assemblefactory"></a>
### `AssembleFactory`

定义 owner `hotpot.cheminfo.operations.assembly.assembler.AssembleFactory`；基类 无。

- `__init__(self, assembler: Iterable[Fragment], iter_step: int=5, mode: Literal['random', 'permutations']='random', seed: Optional[int]=None, sample_weights: Optional[Iterable[float]]=None, max_running: Optional[int]=3000000, save_per_step: Optional[int]=100000, catch_path: Optional[Union[str, os.PathLike]]=None)`
- `get_desc(self, epoch, results)`
- `make(self, mol_iter: Iterable[Molecule])`
- `mp_make(self, mol_iter: Iterable[Union[Molecule, str]], nproc: Optional[int]=None, timeout: int=1800, batch_size: int=10)`
- `load_default_assembler(cls)` [classmethod]
- `init_with_default_assembler(cls, assembler: Optional[Iterable[Fragment]]=None, iter_step: int=5, mode: Literal['random', 'permutations']='random', seed: Optional[int]=None, sample_weights: Optional[Iterable[float]]=None, max_results: int=3000000, save_per_step: Optional[int]=100000, catch_path: Optional[Union[str, os.PathLike]]=None) -> 'AssembleFactory'` [classmethod]
- `load_assembler_file(cls, f: Union[str, os.PathLike]) -> list[Fragment]` [classmethod]
- `load_assembler_contents(cls, contents: list[dict]) -> list[Fragment]` [classmethod]
- `assembler` = `list(assembler)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `max_step` = `iter_step` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `mode` = `mode` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `seed` = `seed` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `sample_weights` = `sample_weights` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `max_running` = `max_running` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `save_per_step` = `save_per_step` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `catch_path` = `catch_path` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

模块值/类型别名（声明存在不构成事实写许可）：

- `phen_smi` = `'n1c(C(=O)N)ccc2c1c3c(cc2)cccn3'`
- `rw_assembler` = `RingWedge(read_mol('C1CCCC1'), action_points=(0,))`
- `res` = `rw_assembler.graft(read_mol(phen_smi, fmt='smi'))`


<a id="hotpot-cheminfo-operations-assembly-fragment"></a>
## `hotpot.cheminfo.operations.assembly.fragment`

`hotpot/cheminfo/operations/assembly/fragment.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-assembly-fragment-fragment"></a>
### `Fragment`

定义 owner `hotpot.cheminfo.operations.assembly.fragment.Fragment`；基类 无。

- `__init__(self, mol: Molecule, searcher: Searcher, action_points: Iterable[int], action_func: Optional[ActionFuncTemplate], exclude_searcher: Optional[Searcher]=None)`
- `__repr__(self)`
- `graft(self, frame: Molecule) -> dict[str, Molecule]`
- `add_exclude_from_sub(self, sub: Substructure)`
- `add_exclude_from_mol(self, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)`
- `add_exclude_from_smi(self, smi: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)`
- `frag` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `searcher` = `searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `action_points` = `list(action_points)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `action_func` = `action_func` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `exclude_searcher` = `exclude_searcher` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-operations-calculation"></a>
## `hotpot.cheminfo.operations.calculation`

`hotpot/cheminfo/operations/calculation/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-operations-calculation-scientific-state"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state`

`hotpot/cheminfo/operations/calculation/scientific_state/__init__.py`；显式 __all__: ('extract_molecular_facts', 'resolve_scientific_state', 'project_state', 'resolve_and_project_state', 'MolecularFacts', 'StateRequest', 'StatePolicy', 'ScientificOperationRequirements', 'ScientificStateBundle', 'BackendStateRequest', 'BackendProjectionResult', 'ScientificStateWorkflowResult')

- 导出 `extract_molecular_facts` → `hotpot.cheminfo.operations.calculation.scientific_state.facts.extract_molecular_facts`；`extract_molecular_facts(mol: Molecule) -> MolecularFacts`
- 导出 `resolve_scientific_state` → `hotpot.cheminfo.operations.calculation.scientific_state.state_resolver.resolve_scientific_state`；`resolve_scientific_state(facts: MolecularFacts, request: StateRequest=StateRequest(), *, policy: StatePolicy=StatePolicy.STRICT, requirements: Optional[ScientificOperationRequirements]=None, provider_set: ProviderSet=ProviderSet(), provided_results: ProvidedScientificResults=ProvidedScientificResults(), rule_set: RuleSet=RuleSet(), candidate_budget: CandidateBudget=DEFAULT_CANDIDATE_BUDGET) -> ScientificStateBundle`
- 导出 `project_state` → `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.project_state`；`project_state(state: ScientificStateBundle, backend: BackendContext, *, request: BackendStateRequest=BackendStateRequest(), execution_policies: ExecutionPolicySet=ExecutionPolicySet(), execution_history: Tuple[BackendExecutionEvidence, ...]=()) -> BackendProjectionResult`
- 导出 `resolve_and_project_state` → `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.resolve_and_project_state`；`resolve_and_project_state(facts: MolecularFacts, backend: BackendContext, request: StateRequest=StateRequest(), *, policy: StatePolicy=StatePolicy.STRICT, requirements: Optional[ScientificOperationRequirements]=None, provider_set: ProviderSet=ProviderSet(), provided_results: ProvidedScientificResults=ProvidedScientificResults(), rule_set: RuleSet=RuleSet(), candidate_budget: CandidateBudget=DEFAULT_CANDIDATE_BUDGET, backend_request: BackendStateRequest=BackendStateRequest(), execution_policies: ExecutionPolicySet=ExecutionPolicySet(), execution_history: Tuple[BackendExecutionEvidence, ...]=()) -> ScientificStateWorkflowResult`
- 导出 `MolecularFacts` → `hotpot.cheminfo.structure.snapshots.MolecularFacts`
- 导出 `StateRequest` → `hotpot.cheminfo.rules.chemistry.scientific_state.request.StateRequest`
- 导出 `StatePolicy` → `hotpot.cheminfo.operations.calculation.scientific_state.contracts.StatePolicy`
- 导出 `ScientificOperationRequirements` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificOperationRequirements`
- 导出 `ScientificStateBundle` → `hotpot.cheminfo.operations.calculation.scientific_state.state_resolver.ScientificStateBundle`
- 导出 `BackendStateRequest` → `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendStateRequest`
- 导出 `BackendProjectionResult` → `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendProjectionResult`
- 导出 `ScientificStateWorkflowResult` → `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.ScientificStateWorkflowResult`


<a id="hotpot-cheminfo-operations-calculation-scientific-state--rule-serialization"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state._rule_serialization`

`hotpot/cheminfo/operations/calculation/scientific_state/_rule_serialization.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/__init__.py`；显式 __all__: ('compose_scientific_state_space', 'CandidateDomainScope', 'MaterializationStatus', 'CandidateSearchStatus', 'StateValueOrigin', 'ScientificStateIdentity', 'FactorKey', 'ConstraintValue', 'CandidateConstraint', 'FrontierBound', 'CandidateDomainSummary', 'RejectedPathSummary', 'BranchAssignment', 'StateValueRef', 'DerivedStateValue', 'StateCandidate', 'FactorCandidateValue', 'FactorDomainContribution', 'CandidateFactor', 'ProposedCandidateValue', 'ProposedCandidateFactor', 'ConstraintGuard', 'GuardedCandidateConstraint', 'CandidateProposal', 'CoupledScientificStateCandidate', 'ScientificStateCandidateSpace', 'ScientificResolutionResult')

- 导出 `compose_scientific_state_space` → `hotpot.cheminfo.operations.calculation.scientific_state.composition.compose.compose_scientific_state_space`；`compose_scientific_state_space(facts: MolecularFacts, *, net_charge_results: Tuple[NetChargeResult, ...], lewis_results: Tuple[LewisFormalChargeResult, ...]=(), coordination_counting_results: Tuple[CoordinationElectronCountingResult, ...]=(), oxidation_state_results: Tuple[OxidationStateInferenceResult, ...]=(), electronic_state_results: Tuple[ElectronicStateInferenceResult, ...]=(), candidate_proposals: Tuple[CandidateProposal, ...]=(), constraints: Tuple[CandidateConstraint, ...]=(), candidate_budget: CandidateBudget=DEFAULT_CANDIDATE_BUDGET) -> ScientificStateCandidateSpace`
- 导出 `CandidateDomainScope` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateDomainScope`
- 导出 `MaterializationStatus` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.MaterializationStatus`
- 导出 `CandidateSearchStatus` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateSearchStatus`
- 导出 `StateValueOrigin` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.StateValueOrigin`
- 导出 `ScientificStateIdentity` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ScientificStateIdentity`
- 导出 `FactorKey` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FactorKey`
- 导出 `ConstraintValue` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ConstraintValue`
- 导出 `CandidateConstraint` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateConstraint`
- 导出 `FrontierBound` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FrontierBound`
- 导出 `CandidateDomainSummary` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateDomainSummary`
- 导出 `RejectedPathSummary` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.RejectedPathSummary`
- 导出 `BranchAssignment` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.BranchAssignment`
- 导出 `StateValueRef` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.StateValueRef`
- 导出 `DerivedStateValue` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.DerivedStateValue`
- 导出 `StateCandidate` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.StateCandidate`
- 导出 `FactorCandidateValue` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FactorCandidateValue`
- 导出 `FactorDomainContribution` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FactorDomainContribution`
- 导出 `CandidateFactor` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateFactor`
- 导出 `ProposedCandidateValue` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ProposedCandidateValue`
- 导出 `ProposedCandidateFactor` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ProposedCandidateFactor`
- 导出 `ConstraintGuard` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ConstraintGuard`
- 导出 `GuardedCandidateConstraint` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.GuardedCandidateConstraint`
- 导出 `CandidateProposal` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateProposal`
- 导出 `CoupledScientificStateCandidate` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CoupledScientificStateCandidate`
- 导出 `ScientificStateCandidateSpace` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ScientificStateCandidateSpace`
- 导出 `ScientificResolutionResult` → `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ScientificResolutionResult`


<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition--artifacts"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition._artifacts`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/_artifacts.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition--candidate-space-serialization"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition._candidate_space_serialization`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/_candidate_space_serialization.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition--decision-snapshot"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition._decision_snapshot`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/_decision_snapshot.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition--serialization"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition._serialization`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/_serialization.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-candidate-space"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.candidate_space`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/candidate_space.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-compose"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.compose`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/compose.py`；显式 __all__: ('compose_scientific_state_space',)


<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-compose-compose-scientific-state-space"></a>
### `compose_scientific_state_space`

```python
compose_scientific_state_space(facts: MolecularFacts, *, net_charge_results: Tuple[NetChargeResult, ...], lewis_results: Tuple[LewisFormalChargeResult, ...]=(), coordination_counting_results: Tuple[CoordinationElectronCountingResult, ...]=(), oxidation_state_results: Tuple[OxidationStateInferenceResult, ...]=(), electronic_state_results: Tuple[ElectronicStateInferenceResult, ...]=(), candidate_proposals: Tuple[CandidateProposal, ...]=(), constraints: Tuple[CandidateConstraint, ...]=(), candidate_budget: CandidateBudget=DEFAULT_CANDIDATE_BUDGET) -> ScientificStateCandidateSpace
```


<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-constraints"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.constraints`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/constraints.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-coupling"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.coupling`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/coupling.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-decision-search"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.decision_search`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/decision_search.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-derive"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.derive`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/derive.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-expand"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.expand`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/expand.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-resolution"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.resolution`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/resolution.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-composition-validation"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.composition.validation`

`hotpot/cheminfo/operations/calculation/scientific_state/composition/validation.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-contracts"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.contracts`

`hotpot/cheminfo/operations/calculation/scientific_state/contracts.py`；显式 __all__: ('StatePolicy', 'ProviderToggle', 'ProviderSet', 'ActivatedRule', 'RuleSet')


<a id="hotpot-cheminfo-operations-calculation-scientific-state-contracts-statepolicy"></a>
### `StatePolicy`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.contracts.StatePolicy`；基类 `Enum`。

- `STRICT` = `'strict'` [枚举常量]
- `COORDINATION_SCREENING_V1` = `'coordination_screening_v1'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-calculation-scientific-state-contracts-providertoggle"></a>
### `ProviderToggle`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.contracts.ProviderToggle`；基类 无。

```python
ProviderToggle(provider_id: str, enabled: bool, configuration: Tuple[Tuple[str, ProviderValue], ...] = ())
```

- `provider_id: str` [只读值字段]
- `enabled: bool` [只读值字段]
- `configuration: Tuple[Tuple[str, ProviderValue], ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-contracts-providerset"></a>
### `ProviderSet`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.contracts.ProviderSet`；基类 无。

```python
ProviderSet(toggles: Tuple[ProviderToggle, ...] = ())
```

- `toggles: Tuple[ProviderToggle, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-contracts-activatedrule"></a>
### `ActivatedRule`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.contracts.ActivatedRule`；基类 无。

```python
ActivatedRule(rule: 'InferenceRule', toggle: RuleToggle)
```

- `rule: 'InferenceRule'` [只读值字段]
- `toggle: RuleToggle` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-contracts-ruleset"></a>
### `RuleSet`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.contracts.RuleSet`；基类 无。

```python
RuleSet(layers: Tuple[Tuple[ActivatedRule, ...], ...] = ())
```

- `layers: Tuple[Tuple[ActivatedRule, ...], ...]` = `()` [只读值字段]


<a id="hotpot-cheminfo-operations-calculation-scientific-state-defaults"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.defaults`

`hotpot/cheminfo/operations/calculation/scientific_state/defaults.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection`

`hotpot/cheminfo/operations/calculation/scientific_state/execution_selection.py`；显式 __all__: ('project_state', 'resolve_and_project_state', 'BackendStateValues', 'BackendStateRequest', 'BackendDecisionSource', 'BackendExecutability', 'BackendProjectionFinding', 'BackendExecutionDecision', 'BackendExecutionEvidence', 'ExecutionPolicySet', 'BackendStateProjection', 'BackendProjectionStatus', 'BackendProjectionResult', 'ScientificStateWorkflowResult')


<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendstatevalues"></a>
### `BackendStateValues`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendStateValues`；基类 无。

```python
BackendStateValues(charge: int, uhf: Optional[int], multiplicity: Optional[int])
```

- `charge: int` [只读值字段]
- `uhf: Optional[int]` [只读值字段]
- `multiplicity: Optional[int]` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendstaterequest"></a>
### `BackendStateRequest`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendStateRequest`；基类 无。

```python
BackendStateRequest(uhf: Optional[int] = None)
```

- `uhf: Optional[int]` = `None` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backenddecisionsource"></a>
### `BackendDecisionSource`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendDecisionSource`；基类 `Enum`。

- `EXPLICIT` = `'explicit'` [枚举常量]
- `SCIENTIFIC_STATE` = `'scientific_state'` [枚举常量]
- `EXACT_CONTEXT_HISTORY` = `'exact_context_history'` [枚举常量]
- `EXECUTION_DEFAULT` = `'execution_default'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendexecutability"></a>
### `BackendExecutability`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendExecutability`；基类 `Enum`。

- `NOT_EVALUATED` = `'not_evaluated'` [枚举常量]
- `SUPPORTED` = `'supported'` [枚举常量]
- `UNSUPPORTED_METHOD` = `'unsupported_method'` [枚举常量]
- `UNSUPPORTED_ELEMENT` = `'unsupported_element'` [枚举常量]
- `INVALID_EFFECTIVE_ELECTRON_COUNT` = `'invalid_effective_electron_count'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendprojectionfinding"></a>
### `BackendProjectionFinding`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendProjectionFinding`；基类 无。

```python
BackendProjectionFinding(code: str, message: str, related_identities: Tuple[str, ...] = ())
```

- `code: str` [只读值字段]
- `message: str` [只读值字段]
- `related_identities: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendexecutiondecision"></a>
### `BackendExecutionDecision`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendExecutionDecision`；基类 无。

```python
BackendExecutionDecision(backend_identity: BackendIdentity, requested_state: BackendStateValues, selected_state: BackendStateValues, source: BackendDecisionSource, rationale: Tuple[DecisionReason, ...])
```

- `backend_identity: BackendIdentity` [只读值字段]
- `requested_state: BackendStateValues` [只读值字段]
- `selected_state: BackendStateValues` [只读值字段]
- `source: BackendDecisionSource` [只读值字段]
- `rationale: Tuple[DecisionReason, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendexecutionevidence"></a>
### `BackendExecutionEvidence`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendExecutionEvidence`；基类 无。

```python
BackendExecutionEvidence(scientific_state_identity: ScientificStateIdentity, backend_identity: BackendIdentity, requested_state: BackendStateValues, executed_state: BackendStateValues, artifact_digests: Tuple[str, ...])
```

- `scientific_state_identity: ScientificStateIdentity` [只读值字段]
- `backend_identity: BackendIdentity` [只读值字段]
- `requested_state: BackendStateValues` [只读值字段]
- `executed_state: BackendStateValues` [只读值字段]
- `artifact_digests: Tuple[str, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-executionpolicyset"></a>
### `ExecutionPolicySet`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.ExecutionPolicySet`；基类 无。

```python
ExecutionPolicySet(policy_ids: Tuple[str, ...] = ('EXECUTION-001', 'DEFAULT-004'))
```

- `policy_ids: Tuple[str, ...]` = `('EXECUTION-001', 'DEFAULT-004')` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendstateprojection"></a>
### `BackendStateProjection`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendStateProjection`；基类 无。

```python
BackendStateProjection(scientific_state_identity: ScientificStateIdentity, facts_identity: MolecularFactsIdentity, backend_identity: BackendIdentity, effective_electron_count: int, charge: int, uhf: Optional[int], multiplicity: Optional[int], projection_convention: str, execution_decision: BackendExecutionDecision, executability: BackendExecutability)
```

- `scientific_state_identity: ScientificStateIdentity` [只读值字段]
- `facts_identity: MolecularFactsIdentity` [只读值字段]
- `backend_identity: BackendIdentity` [只读值字段]
- `effective_electron_count: int` [只读值字段]
- `charge: int` [只读值字段]
- `uhf: Optional[int]` [只读值字段]
- `multiplicity: Optional[int]` [只读值字段]
- `projection_convention: str` [只读值字段]
- `execution_decision: BackendExecutionDecision` [只读值字段]
- `executability: BackendExecutability` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendprojectionstatus"></a>
### `BackendProjectionStatus`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendProjectionStatus`；基类 `Enum`。

- `NOT_PROJECTED` = `'not_projected'` [枚举常量]
- `PROJECTED` = `'projected'` [枚举常量]
- `PROJECTION_CONFLICT` = `'projection_conflict'` [枚举常量]
- `BACKEND_UNSUPPORTED` = `'backend_unsupported'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-backendprojectionresult"></a>
### `BackendProjectionResult`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.BackendProjectionResult`；基类 无。

```python
BackendProjectionResult(status: BackendProjectionStatus, backend_identity: BackendIdentity, request: BackendStateRequest, projection: Optional[BackendStateProjection], findings: Tuple[BackendProjectionFinding, ...])
```

- `status: BackendProjectionStatus` [只读值字段]
- `backend_identity: BackendIdentity` [只读值字段]
- `request: BackendStateRequest` [只读值字段]
- `projection: Optional[BackendStateProjection]` [只读值字段]
- `findings: Tuple[BackendProjectionFinding, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-scientificstateworkflowresult"></a>
### `ScientificStateWorkflowResult`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.execution_selection.ScientificStateWorkflowResult`；基类 无。

```python
ScientificStateWorkflowResult(scientific_state: ScientificStateBundle, backend_projection: BackendProjectionResult)
```

- `scientific_state: ScientificStateBundle` [只读值字段]
- `backend_projection: BackendProjectionResult` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-project-state"></a>
### `project_state`

```python
project_state(state: ScientificStateBundle, backend: BackendContext, *, request: BackendStateRequest=BackendStateRequest(), execution_policies: ExecutionPolicySet=ExecutionPolicySet(), execution_history: Tuple[BackendExecutionEvidence, ...]=()) -> BackendProjectionResult
```

<a id="hotpot-cheminfo-operations-calculation-scientific-state-execution-selection-resolve-and-project-state"></a>
### `resolve_and_project_state`

```python
resolve_and_project_state(facts: MolecularFacts, backend: BackendContext, request: StateRequest=StateRequest(), *, policy: StatePolicy=StatePolicy.STRICT, requirements: Optional[ScientificOperationRequirements]=None, provider_set: ProviderSet=ProviderSet(), provided_results: ProvidedScientificResults=ProvidedScientificResults(), rule_set: RuleSet=RuleSet(), candidate_budget: CandidateBudget=DEFAULT_CANDIDATE_BUDGET, backend_request: BackendStateRequest=BackendStateRequest(), execution_policies: ExecutionPolicySet=ExecutionPolicySet(), execution_history: Tuple[BackendExecutionEvidence, ...]=()) -> ScientificStateWorkflowResult
```


<a id="hotpot-cheminfo-operations-calculation-scientific-state-facts"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.facts`

`hotpot/cheminfo/operations/calculation/scientific_state/facts.py`；显式 __all__: ('extract_molecular_facts',)


<a id="hotpot-cheminfo-operations-calculation-scientific-state-facts-extract-molecular-facts"></a>
### `extract_molecular_facts`

```python
extract_molecular_facts(mol: Molecule) -> MolecularFacts
```


<a id="hotpot-cheminfo-operations-calculation-scientific-state-manifest"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.manifest`

`hotpot/cheminfo/operations/calculation/scientific_state/manifest.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-operation-requirements"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.operation_requirements`

`hotpot/cheminfo/operations/calculation/scientific_state/operation_requirements.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-profile-decision"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.profile_decision`

`hotpot/cheminfo/operations/calculation/scientific_state/profile_decision.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-profile-eligibility"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.profile_eligibility`

`hotpot/cheminfo/operations/calculation/scientific_state/profile_eligibility.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-profiles"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.profiles`

`hotpot/cheminfo/operations/calculation/scientific_state/profiles.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-rule-controller"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.rule_controller`

`hotpot/cheminfo/operations/calculation/scientific_state/rule_controller.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-rule-registry"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.rule_registry`

`hotpot/cheminfo/operations/calculation/scientific_state/rule_registry.py`；显式 __all__: ('InferenceRuleRegistry',)


<a id="hotpot-cheminfo-operations-calculation-scientific-state-rule-registry-inferenceruleregistry"></a>
### `InferenceRuleRegistry`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.rule_registry.InferenceRuleRegistry`；基类 无。

- `__init__(self) -> None`
- `register(self, rule: InferenceRule) -> None`
- `available(self) -> Tuple[RuleSpecification, ...]`
- `activate(self, profile: StatePolicy, toggles: Tuple[RuleToggle, ...]) -> RuleSet`


<a id="hotpot-cheminfo-operations-calculation-scientific-state-rule-validation"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.rule_validation`

`hotpot/cheminfo/operations/calculation/scientific_state/rule_validation.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-resolver"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.state_resolver`

`hotpot/cheminfo/operations/calculation/scientific_state/state_resolver.py`；显式 __all__: ('resolve_scientific_state', 'ProvidedScientificResults', 'ScientificStateBundle')


<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-resolver-providedscientificresults"></a>
### `ProvidedScientificResults`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_resolver.ProvidedScientificResults`；基类 无。

```python
ProvidedScientificResults(net_charge_results: Tuple[NetChargeResult, ...] = (), lewis_results: Tuple[LewisFormalChargeResult, ...] = (), coordination_counting_results: Tuple[CoordinationElectronCountingResult, ...] = (), oxidation_state_results: Tuple[OxidationStateInferenceResult, ...] = (), electronic_state_results: Tuple[ElectronicStateInferenceResult, ...] = ())
```

- `net_charge_results: Tuple[NetChargeResult, ...]` = `()` [只读值字段]
- `lewis_results: Tuple[LewisFormalChargeResult, ...]` = `()` [只读值字段]
- `coordination_counting_results: Tuple[CoordinationElectronCountingResult, ...]` = `()` [只读值字段]
- `oxidation_state_results: Tuple[OxidationStateInferenceResult, ...]` = `()` [只读值字段]
- `electronic_state_results: Tuple[ElectronicStateInferenceResult, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-resolver-scientificstatebundle"></a>
### `ScientificStateBundle`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_resolver.ScientificStateBundle`；基类 无。

```python
ScientificStateBundle(facts: MolecularFacts, net_charge_results: Tuple[NetChargeResult, ...], lewis_results: Tuple[LewisFormalChargeResult, ...], coordination_counting_results: Tuple[CoordinationElectronCountingResult, ...], oxidation_state_results: Tuple[OxidationStateInferenceResult, ...], electronic_state_results: Tuple[ElectronicStateInferenceResult, ...], candidate_space: ScientificStateCandidateSpace, scientific_resolution: ScientificResolutionResult, execution_decision: ExecutionStateDecision, rule_trace: Tuple[RuleTrace, ...])
```

- `facts: MolecularFacts` [只读值字段]
- `net_charge_results: Tuple[NetChargeResult, ...]` [只读值字段]
- `lewis_results: Tuple[LewisFormalChargeResult, ...]` [只读值字段]
- `coordination_counting_results: Tuple[CoordinationElectronCountingResult, ...]` [只读值字段]
- `oxidation_state_results: Tuple[OxidationStateInferenceResult, ...]` [只读值字段]
- `electronic_state_results: Tuple[ElectronicStateInferenceResult, ...]` [只读值字段]
- `candidate_space: ScientificStateCandidateSpace` [只读值字段]
- `scientific_resolution: ScientificResolutionResult` [只读值字段]
- `execution_decision: ExecutionStateDecision` [只读值字段]
- `rule_trace: Tuple[RuleTrace, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-resolver-resolve-scientific-state"></a>
### `resolve_scientific_state`

```python
resolve_scientific_state(facts: MolecularFacts, request: StateRequest=StateRequest(), *, policy: StatePolicy=StatePolicy.STRICT, requirements: Optional[ScientificOperationRequirements]=None, provider_set: ProviderSet=ProviderSet(), provided_results: ProvidedScientificResults=ProvidedScientificResults(), rule_set: RuleSet=RuleSet(), candidate_budget: CandidateBudget=DEFAULT_CANDIDATE_BUDGET) -> ScientificStateBundle
```


<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection"></a>
## `hotpot.cheminfo.operations.calculation.scientific_state.state_selection`

`hotpot/cheminfo/operations/calculation/scientific_state/state_selection.py`；显式 __all__: ('StateDecisionAction', 'StateDecisionSource', 'DecisionCertificationStatus', 'StaticEligibilityFinding', 'DecisionReason', 'StaticProfileEligibility', 'DecisionCertifiability', 'ExecutionStateDecision')


<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-statedecisionaction"></a>
### `StateDecisionAction`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.StateDecisionAction`；基类 `Enum`。

- `NOT_REQUESTED` = `'not_requested'` [枚举常量]
- `SELECT_STATE` = `'select_state'` [枚举常量]
- `NO_AUTOMATIC_STATE` = `'no_automatic_state'` [枚举常量]
- `CONFLICT` = `'conflict'` [枚举常量]
- `INADMISSIBLE` = `'inadmissible'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-statedecisionsource"></a>
### `StateDecisionSource`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.StateDecisionSource`；基类 `Enum`。

- `NOT_SELECTED` = `'not_selected'` [枚举常量]
- `EXPLICIT` = `'explicit'` [枚举常量]
- `SCIENTIFIC_SOURCE` = `'scientific_source'` [枚举常量]
- `EXECUTION_DEFAULT` = `'execution_default'` [枚举常量]
- `EXECUTION_PRIOR` = `'execution_prior'` [枚举常量]
- `CANONICAL_EQUIVALENT_TIE` = `'canonical_equivalent_tie'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-decisioncertificationstatus"></a>
### `DecisionCertificationStatus`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.DecisionCertificationStatus`；基类 `Enum`。

- `NOT_EVALUATED` = `'not_evaluated'` [枚举常量]
- `CERTIFIED` = `'certified'` [枚举常量]
- `NON_EQUIVALENT_TIE` = `'non_equivalent_tie'` [枚举常量]
- `OPEN_OR_UNBOUNDABLE_DOMAIN` = `'open_or_unboundable_domain'` [枚举常量]
- `RESOURCE_LIMIT_UNCERTIFIED` = `'resource_limit_uncertified'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-staticeligibilityfinding"></a>
### `StaticEligibilityFinding`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.StaticEligibilityFinding`；基类 无。

```python
StaticEligibilityFinding(code: str, message: str)
```

- `code: str` [只读值字段]
- `message: str` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-decisionreason"></a>
### `DecisionReason`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.DecisionReason`；基类 无。

```python
DecisionReason(code: str, message: str, related_identities: Tuple[str, ...] = ())
```

- `code: str` [只读值字段]
- `message: str` [只读值字段]
- `related_identities: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-staticprofileeligibility"></a>
### `StaticProfileEligibility`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.StaticProfileEligibility`；基类 无。

```python
StaticProfileEligibility(profile_id: str, effective_profile_identity: str, in_domain: bool, findings: Tuple[StaticEligibilityFinding, ...])
```

- `profile_id: str` [只读值字段]
- `effective_profile_identity: str` [只读值字段]
- `in_domain: bool` [只读值字段]
- `findings: Tuple[StaticEligibilityFinding, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-decisioncertifiability"></a>
### `DecisionCertifiability`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.DecisionCertifiability`；基类 无。

```python
DecisionCertifiability(status: DecisionCertificationStatus, findings: Tuple[DecisionReason, ...])
```

- `status: DecisionCertificationStatus` [只读值字段]
- `findings: Tuple[DecisionReason, ...]` [只读值字段]

<a id="hotpot-cheminfo-operations-calculation-scientific-state-state-selection-executionstatedecision"></a>
### `ExecutionStateDecision`

定义 owner `hotpot.cheminfo.operations.calculation.scientific_state.state_selection.ExecutionStateDecision`；基类 无。

```python
ExecutionStateDecision(selected_state_identity: Optional[ScientificStateIdentity], action: StateDecisionAction, source: StateDecisionSource, static_profile_eligibility: StaticProfileEligibility, decision_certifiability: DecisionCertifiability, rationale: Tuple[DecisionReason, ...])
```

- `selected_state_identity: Optional[ScientificStateIdentity]` [只读值字段]
- `action: StateDecisionAction` [只读值字段]
- `source: StateDecisionSource` [只读值字段]
- `static_profile_eligibility: StaticProfileEligibility` [只读值字段]
- `decision_certifiability: DecisionCertifiability` [只读值字段]
- `rationale: Tuple[DecisionReason, ...]` [只读值字段]


<a id="hotpot-cheminfo-operations-chemistry"></a>
## `hotpot.cheminfo.operations.chemistry`

`hotpot/cheminfo/operations/chemistry.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-collections"></a>
## `hotpot.cheminfo.operations.collections`

`hotpot/cheminfo/operations/collections.py`；显式 __all__: ('to_pyg_dataset',)


<a id="hotpot-cheminfo-operations-collections-to-pyg-dataset"></a>
### `to_pyg_dataset`

```python
to_pyg_dataset(list_mol: tp.Iterable[Molecule], dataset_root: os.PathLike)
```


<a id="hotpot-cheminfo-operations-conformers"></a>
## `hotpot.cheminfo.operations.conformers`

`hotpot/cheminfo/operations/conformers/__init__.py`；显式 __all__: ('ConformerBatch', 'ConformerSampler', 'ConformerSamplingError', 'ConformerSamplingRequest', 'OpenBabelConformerAdapter', 'RDKitConformerAdapter')

- 导出 `ConformerBatch` → `hotpot.cheminfo.structure.conformer.ConformerBatch`
- 导出 `ConformerSampler` → `hotpot.cheminfo.operations.conformers.sampler.ConformerSampler`
- 导出 `ConformerSamplingError` → `hotpot.cheminfo.operations.conformers.contracts.ConformerSamplingError`
- 导出 `ConformerSamplingRequest` → `hotpot.cheminfo.operations.conformers.contracts.ConformerSamplingRequest`
- 导出 `OpenBabelConformerAdapter` → `hotpot.cheminfo.operations.conformers.openbabel.OpenBabelConformerAdapter`
- 导出 `RDKitConformerAdapter` → `hotpot.cheminfo.operations.conformers.rdkit.RDKitConformerAdapter`


<a id="hotpot-cheminfo-operations-conformers-contracts"></a>
## `hotpot.cheminfo.operations.conformers.contracts`

`hotpot/cheminfo/operations/conformers/contracts.py`；显式 __all__: ('ConformerBatch', 'ConformerSamplingError', 'ConformerSamplingRequest')

- 导出 `ConformerBatch` → `hotpot.cheminfo.structure.conformer.ConformerBatch`

<a id="hotpot-cheminfo-operations-conformers-contracts-conformersamplingerror"></a>
### `ConformerSamplingError`

定义 owner `hotpot.cheminfo.operations.conformers.contracts.ConformerSamplingError`；基类 `RuntimeError`。


<a id="hotpot-cheminfo-operations-conformers-contracts-conformersamplingrequest"></a>
### `ConformerSamplingRequest`

定义 owner `hotpot.cheminfo.operations.conformers.contracts.ConformerSamplingRequest`；基类 无。

```python
ConformerSamplingRequest(conformer_count: int = 1, seed: int | None = None)
```

- `conformer_count: int` = `1` [只读值字段]
- `seed: int | None` = `None` [只读值字段]


<a id="hotpot-cheminfo-operations-conformers-openbabel"></a>
## `hotpot.cheminfo.operations.conformers.openbabel`

`hotpot/cheminfo/operations/conformers/openbabel.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-conformers-openbabel-openbabelconformeradapter"></a>
### `OpenBabelConformerAdapter`

定义 owner `hotpot.cheminfo.operations.conformers.openbabel.OpenBabelConformerAdapter`；基类 无。

```python
OpenBabelConformerAdapter()
```

- `sample(self, molecule: 'Molecule', request: ConformerSamplingRequest) -> ConformerBatch`


<a id="hotpot-cheminfo-operations-conformers-rdkit"></a>
## `hotpot.cheminfo.operations.conformers.rdkit`

`hotpot/cheminfo/operations/conformers/rdkit.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-conformers-rdkit-rdkitconformeradapter"></a>
### `RDKitConformerAdapter`

定义 owner `hotpot.cheminfo.operations.conformers.rdkit.RDKitConformerAdapter`；基类 无。

```python
RDKitConformerAdapter(materialized_hydrogens: bool = False)
```

- `sample(self, molecule: 'Molecule', request: ConformerSamplingRequest) -> ConformerBatch`
- `materialized_hydrogens: bool` = `False` [只读值字段]


<a id="hotpot-cheminfo-operations-conformers-sampler"></a>
## `hotpot.cheminfo.operations.conformers.sampler`

`hotpot/cheminfo/operations/conformers/sampler.py`；显式 __all__: ('ConformerSampler',)


<a id="hotpot-cheminfo-operations-conformers-sampler-conformersampler"></a>
### `ConformerSampler`

定义 owner `hotpot.cheminfo.operations.conformers.sampler.ConformerSampler`；基类 `Protocol`。

- `sample(self, molecule: 'Molecule', request: ConformerSamplingRequest) -> ConformerBatch`


<a id="hotpot-cheminfo-operations-conversion"></a>
## `hotpot.cheminfo.operations.conversion`

`hotpot/cheminfo/operations/conversion.py`；显式 __all__: ('is_molecule_input', 'to_hotpot_mol', 'mol2obmol', 'obmol2mol', 'to_rdmol', 'from_rdmol', 'write_by_pybel')


<a id="hotpot-cheminfo-operations-conversion-is-molecule-input"></a>
### `is_molecule_input`

```python
is_molecule_input(value) -> 'bool'
```

<a id="hotpot-cheminfo-operations-conversion-to-hotpot-mol"></a>
### `to_hotpot_mol`

```python
to_hotpot_mol(value, *, fmt=None, **kwargs) -> 'Molecule'
```

<a id="hotpot-cheminfo-operations-conversion-mol2obmol"></a>
### `mol2obmol`

```python
mol2obmol(mol)
```

<a id="hotpot-cheminfo-operations-conversion-obmol2mol"></a>
### `obmol2mol`

```python
obmol2mol(obmol, mol)
```

<a id="hotpot-cheminfo-operations-conversion-to-rdmol"></a>
### `to_rdmol`

```python
to_rdmol(mol, kekulize: bool=True, sanitize: bool=False)
```

<a id="hotpot-cheminfo-operations-conversion-from-rdmol"></a>
### `from_rdmol`

```python
from_rdmol(rdmol, mol)
```

<a id="hotpot-cheminfo-operations-conversion-write-by-pybel"></a>
### `write_by_pybel`

```python
write_by_pybel(mol, fmt='smi', filename=None, overwrite=False, opt=None)
```


<a id="hotpot-cheminfo-operations-crystal"></a>
## `hotpot.cheminfo.operations.crystal`

`hotpot/cheminfo/operations/crystal.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-crystal-generation"></a>
## `hotpot.cheminfo.operations.crystal_generation`

`hotpot/cheminfo/operations/crystal_generation.py`；显式 __all__: 无



<a id="hotpot-cheminfo-operations-crystal-geometry"></a>
## `hotpot.cheminfo.operations.crystal_geometry`

`hotpot/cheminfo/operations/crystal_geometry.py`；显式 __all__: 无



<a id="hotpot-cheminfo-operations-crystal-io"></a>
## `hotpot.cheminfo.operations.crystal_io`

`hotpot/cheminfo/operations/crystal_io.py`；显式 __all__: 无



<a id="hotpot-cheminfo-operations-crystal-matching"></a>
## `hotpot.cheminfo.operations.crystal_matching`

`hotpot/cheminfo/operations/crystal_matching.py`；显式 __all__: 无



<a id="hotpot-cheminfo-operations-descriptors"></a>
## `hotpot.cheminfo.operations.descriptors`

`hotpot/cheminfo/operations/descriptors.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields"></a>
## `hotpot.cheminfo.operations.forcefields`

`hotpot/cheminfo/operations/forcefields/__init__.py`；显式 __all__: ('TrajectoryPath', 'TrajectoryStart', 'TrajectoryStage', 'TrajectoryEvent', 'AtomIdentity', 'BondTopology', 'BondTopologyRevision', 'RingFrameEvidence', 'CoordinationFrameEvidence', 'OptimizationFrameEvidence', 'FrameEvidence', 'ForceFieldFrame', 'ForceFieldTrajectory', 'ForceFieldTrajectoryArchive', 'OptimizationAlgorithm', 'OptimizationRoute', 'OptimizationAttemptStatus', 'OptimizationAttemptReport', 'OptimizationRoutingReport', 'ConvergenceLevel', 'DEFAULT_CONVERGENCE_LEVEL', 'OptimizationStoppingCriteria', 'TerminationReason', 'ForceFieldDiagnosticValue', 'ForceFieldRunReport', 'Build3DReport', 'CandidateRejection', 'RingUntanglingReport', 'CoordinationBondRestorationReport', 'ComplexBuildDiagnostics', 'ForceFieldWorkflowReport', 'BuildAndOptimizeReport', 'ComplexBuildReport', 'ForceFieldSetupReport', 'ForceFieldSetupStage', 'ForceFieldWorkflowStage', 'AcceptanceCheck', 'StructureAcceptanceThresholds', 'ForceFieldAcceptanceEvidence', 'AtomTopologySignature', 'TopologyReference', 'ForceFieldValidationReport', 'CoordinationEnvironment', 'ForceFieldError', 'ForceFieldSetupError', 'BuildWorkerError', 'BuildTimeoutError', 'ComplexBuildError', 'ComplexBuildWarning', 'ComplexBuildWorkerError', 'ComplexBuildTimeoutError', 'GeometryQualityError', 'GeometryQualityWarning', 'capture_topology', 'evaluate_structure_acceptance', 'is_structure_accepted', 'perturb', 'collect_coordination_environments', 'build3d', 'optimize', 'build_complex3d', 'optimize_complex', 'complexes_build', 'build_and_optimize', 'auto_optimize')

- 导出 `TrajectoryPath` → `hotpot.cheminfo.operations.forcefields.contracts.TrajectoryPath`
- 导出 `TrajectoryStart` → `hotpot.cheminfo.structure.forcefields.TrajectoryStart`
- 导出 `TrajectoryStage` → `hotpot.cheminfo.structure.forcefields.TrajectoryStage`
- 导出 `TrajectoryEvent` → `hotpot.cheminfo.structure.forcefields.TrajectoryEvent`
- 导出 `AtomIdentity` → `hotpot.cheminfo.structure.forcefields.AtomIdentity`
- 导出 `BondTopology` → `hotpot.cheminfo.structure.forcefields.BondTopology`
- 导出 `BondTopologyRevision` → `hotpot.cheminfo.structure.forcefields.BondTopologyRevision`
- 导出 `RingFrameEvidence` → `hotpot.cheminfo.structure.forcefields.RingFrameEvidence`
- 导出 `CoordinationFrameEvidence` → `hotpot.cheminfo.structure.forcefields.CoordinationFrameEvidence`
- 导出 `OptimizationFrameEvidence` → `hotpot.cheminfo.structure.forcefields.OptimizationFrameEvidence`
- 导出 `FrameEvidence` → `hotpot.cheminfo.structure.forcefields.FrameEvidence`
- 导出 `ForceFieldFrame` → `hotpot.cheminfo.structure.forcefields.ForceFieldFrame`
- 导出 `ForceFieldTrajectory` → `hotpot.cheminfo.operations.forcefields.trajectory.ForceFieldTrajectory`
- 导出 `ForceFieldTrajectoryArchive` → `hotpot.cheminfo.operations.forcefields.trajectory.ForceFieldTrajectoryArchive`
- 导出 `OptimizationAlgorithm` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAlgorithm`
- 导出 `OptimizationRoute` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationRoute`
- 导出 `OptimizationAttemptStatus` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAttemptStatus`
- 导出 `OptimizationAttemptReport` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAttemptReport`
- 导出 `OptimizationRoutingReport` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationRoutingReport`
- 导出 `ConvergenceLevel` → `hotpot.cheminfo.obWrappers.contracts.ConvergenceLevel`
- 导出 `DEFAULT_CONVERGENCE_LEVEL` → `hotpot.cheminfo.obWrappers.contracts.DEFAULT_CONVERGENCE_LEVEL`
- 导出 `OptimizationStoppingCriteria` → `hotpot.cheminfo.rules.physics.forcefields.contracts.OptimizationStoppingCriteria`
- 导出 `TerminationReason` → `hotpot.cheminfo.operations.forcefields.contracts.TerminationReason`
- 导出 `ForceFieldDiagnosticValue` → `hotpot.cheminfo.rules.physics.forcefields.contracts.ForceFieldDiagnosticValue`
- 导出 `ForceFieldRunReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldRunReport`
- 导出 `Build3DReport` → `hotpot.cheminfo.operations.forcefields.contracts.Build3DReport`
- 导出 `CandidateRejection` → `hotpot.cheminfo.operations.forcefields.contracts.CandidateRejection`
- 导出 `RingUntanglingReport` → `hotpot.cheminfo.operations.forcefields.contracts.RingUntanglingReport`
- 导出 `CoordinationBondRestorationReport` → `hotpot.cheminfo.operations.forcefields.contracts.CoordinationBondRestorationReport`
- 导出 `ComplexBuildDiagnostics` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildDiagnostics`
- 导出 `ForceFieldWorkflowReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- 导出 `BuildAndOptimizeReport` → `hotpot.cheminfo.operations.forcefields.contracts.BuildAndOptimizeReport`
- 导出 `ComplexBuildReport` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildReport`
- 导出 `ForceFieldSetupReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupReport`
- 导出 `ForceFieldSetupStage` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupStage`
- 导出 `ForceFieldWorkflowStage` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowStage`
- 导出 `AcceptanceCheck` → `hotpot.cheminfo.rules.physics.forcefields.contracts.AcceptanceCheck`
- 导出 `StructureAcceptanceThresholds` → `hotpot.cheminfo.rules.physics.forcefields.contracts.StructureAcceptanceThresholds`
- 导出 `ForceFieldAcceptanceEvidence` → `hotpot.cheminfo.rules.physics.forcefields.contracts.ForceFieldAcceptanceEvidence`
- 导出 `AtomTopologySignature` → `hotpot.cheminfo.structure.forcefields.AtomTopologySignature`
- 导出 `TopologyReference` → `hotpot.cheminfo.structure.forcefields.TopologyReference`
- 导出 `ForceFieldValidationReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldValidationReport`
- 导出 `CoordinationEnvironment` → `hotpot.cheminfo.structure.forcefields.CoordinationEnvironment`
- 导出 `ForceFieldError` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- 导出 `ForceFieldSetupError` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupError`
- 导出 `BuildWorkerError` → `hotpot.cheminfo.operations.forcefields.contracts.BuildWorkerError`
- 导出 `BuildTimeoutError` → `hotpot.cheminfo.operations.forcefields.contracts.BuildTimeoutError`
- 导出 `ComplexBuildError` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildError`
- 导出 `ComplexBuildWarning` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildWarning`
- 导出 `ComplexBuildWorkerError` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildWorkerError`
- 导出 `ComplexBuildTimeoutError` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildTimeoutError`
- 导出 `GeometryQualityError` → `hotpot.cheminfo.operations.forcefields.contracts.GeometryQualityError`
- 导出 `GeometryQualityWarning` → `hotpot.cheminfo.operations.forcefields.contracts.GeometryQualityWarning`
- 导出 `capture_topology` → `hotpot.cheminfo.adapters.structure.forcefields.capture_topology`；`capture_topology(mol: 'Molecule', *, allow_added_hydrogens: bool=True) -> TopologyReference`
- 导出 `evaluate_structure_acceptance` → `hotpot.cheminfo.operations.forcefields.acceptance.evaluate_structure_acceptance`；`evaluate_structure_acceptance(mol: 'Molecule', *, level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> ForceFieldValidationReport`
- 导出 `is_structure_accepted` → `hotpot.cheminfo.operations.forcefields.acceptance.is_structure_accepted`；`is_structure_accepted(mol: 'Molecule', *, level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> bool`
- 导出 `perturb` → `hotpot.cheminfo.operations.forcefields.coordinates.perturb`；`perturb(mol: 'Molecule', *, sigma: float=0.5, seed: Optional[int]=None) -> np.ndarray`
- 导出 `collect_coordination_environments` → `hotpot.cheminfo.adapters.structure.forcefields.collect_coordination_environments`；`collect_coordination_environments(mol: 'Molecule') -> Tuple[CoordinationEnvironment, ...]`
- 导出 `build3d` → `hotpot.cheminfo.operations.forcefields.workflows.build3d`；`build3d(mol: 'Molecule', *, add_hydrogens: bool=True, seed: Optional[int]=None, timeout: float=1000.0) -> Build3DReport`
- 导出 `optimize` → `hotpot.cheminfo.operations.forcefields.workflows.optimize`；`optimize(mol: 'Molecule', forcefield: Optional[str]='UFF', *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=1, steps_per_epoch: int=100, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.FINAL_OPTIMIZATION, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport`
- 导出 `build_complex3d` → `hotpot.cheminfo.operations.forcefields.workflows.build_complex3d`；`build_complex3d(mol: 'Molecule', forcefield: Optional[str]=None, *, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, timeout: float=1000.0, add_hydrogens: bool=True, seed: Optional[int]=None, perturb_sigma: float=0.5, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION, trajectory_path: Optional[TrajectoryPath]=None) -> ComplexBuildReport`
- 导出 `optimize_complex` → `hotpot.cheminfo.operations.forcefields.workflows.optimize_complex`；`optimize_complex(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, complex_untangling_attempts: int=30, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COMPLEX_UNTANGLING, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport`
- 导出 `complexes_build` → `hotpot.cheminfo.operations.forcefields.workflows.complexes_build`；`complexes_build(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30, timeout: float=1000.0, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ComplexBuildReport`
- 导出 `build_and_optimize` → `hotpot.cheminfo.operations.forcefields.workflows.build_and_optimize`；`build_and_optimize(mol: 'Molecule', forcefield: Optional[str]='UFF', *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, timeout: float=1000.0, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional[TrajectoryStart]=None, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30) -> ForceFieldWorkflowReport`
- 导出 `auto_optimize` → `hotpot.cheminfo.operations.forcefields.auto.auto_optimize`；`auto_optimize(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30, timeout: float=1000.0, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional[TrajectoryStart]=None, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport`


<a id="hotpot-cheminfo-operations-forcefields-acceptance"></a>
## `hotpot.cheminfo.operations.forcefields.acceptance`

`hotpot/cheminfo/operations/forcefields/acceptance.py`；显式 __all__: ('evaluate_structure_acceptance', 'is_structure_accepted')


<a id="hotpot-cheminfo-operations-forcefields-acceptance-evaluate-structure-acceptance-at-checkpoint"></a>
### `evaluate_structure_acceptance_at_checkpoint`

```python
evaluate_structure_acceptance_at_checkpoint(mol: 'Molecule', *, bond_ring_report: 'geometry_adapter.BondRingScreeningReport[Ring, Bond]', level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> ForceFieldValidationReport
```

<a id="hotpot-cheminfo-operations-forcefields-acceptance-evaluate-structure-acceptance-at-native-checkpoint"></a>
### `evaluate_structure_acceptance_at_native_checkpoint`

```python
evaluate_structure_acceptance_at_native_checkpoint(mol: 'Molecule', *, bond_ring_report: NativeRingCheckpointReport, level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> ForceFieldValidationReport
```

<a id="hotpot-cheminfo-operations-forcefields-acceptance-evaluate-structure-acceptance"></a>
### `evaluate_structure_acceptance`

```python
evaluate_structure_acceptance(mol: 'Molecule', *, level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> ForceFieldValidationReport
```

<a id="hotpot-cheminfo-operations-forcefields-acceptance-is-structure-accepted"></a>
### `is_structure_accepted`

```python
is_structure_accepted(mol: 'Molecule', *, level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> bool
```


<a id="hotpot-cheminfo-operations-forcefields-attempts"></a>
## `hotpot.cheminfo.operations.forcefields.attempts`

`hotpot/cheminfo/operations/forcefields/attempts.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields-auto"></a>
## `hotpot.cheminfo.operations.forcefields.auto`

`hotpot/cheminfo/operations/forcefields/auto.py`；显式 __all__: ('auto_optimize',)


<a id="hotpot-cheminfo-operations-forcefields-auto-auto-optimize"></a>
### `auto_optimize`

```python
auto_optimize(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30, timeout: float=1000.0, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional[TrajectoryStart]=None, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport
```


<a id="hotpot-cheminfo-operations-forcefields-backend"></a>
## `hotpot.cheminfo.operations.forcefields.backend`

`hotpot/cheminfo/operations/forcefields/backend.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields-contracts"></a>
## `hotpot.cheminfo.operations.forcefields.contracts`

`hotpot/cheminfo/operations/forcefields/contracts.py`；显式 __all__: ('TrajectoryPath', 'OptimizationAlgorithm', 'ConvergenceLevel', 'DEFAULT_CONVERGENCE_LEVEL', 'TerminationReason', 'OptimizationRoute', 'OptimizationAttemptStatus', 'OptimizationAttemptReport', 'OptimizationRoutingReport', 'ForceFieldDiagnosticValue', 'ForceFieldRunReport', 'Build3DReport', 'CandidateRejection', 'RingUntanglingReport', 'CoordinationBondRestorationReport', 'ComplexBuildDiagnostics', 'ForceFieldWorkflowReport', 'BuildAndOptimizeReport', 'ComplexBuildReport', 'ForceFieldSetupReport', 'ForceFieldSetupStage', 'ForceFieldWorkflowStage', 'AcceptanceCheck', 'StructureAcceptanceThresholds', 'ForceFieldAcceptanceEvidence', 'ForceFieldValidationReport', 'CoordinationEnvironment', 'ForceFieldError', 'ForceFieldSetupError', 'BuildWorkerError', 'BuildTimeoutError', 'ComplexBuildError', 'ComplexBuildWarning', 'ComplexBuildWorkerError', 'ComplexBuildTimeoutError', 'GeometryQualityError', 'GeometryQualityWarning')

- 导出 `ConvergenceLevel` → `hotpot.cheminfo.obWrappers.contracts.ConvergenceLevel`
- 导出 `DEFAULT_CONVERGENCE_LEVEL` → `hotpot.cheminfo.obWrappers.contracts.DEFAULT_CONVERGENCE_LEVEL`
- 导出 `ForceFieldDiagnosticValue` → `hotpot.cheminfo.rules.physics.forcefields.contracts.ForceFieldDiagnosticValue`
- 导出 `AcceptanceCheck` → `hotpot.cheminfo.rules.physics.forcefields.contracts.AcceptanceCheck`
- 导出 `StructureAcceptanceThresholds` → `hotpot.cheminfo.rules.physics.forcefields.contracts.StructureAcceptanceThresholds`
- 导出 `ForceFieldAcceptanceEvidence` → `hotpot.cheminfo.rules.physics.forcefields.contracts.ForceFieldAcceptanceEvidence`
- 导出 `CoordinationEnvironment` → `hotpot.cheminfo.structure.forcefields.CoordinationEnvironment`

<a id="hotpot-cheminfo-operations-forcefields-contracts-optimizationroute"></a>
### `OptimizationRoute`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.OptimizationRoute`；基类 `str`, `Enum`。

- `NATIVE_FAST` = `'native_fast'` [枚举常量]
- `ORDINARY` = `'ordinary'` [枚举常量]
- `COMPLEX_WORKFLOW` = `'complex_workflow'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-forcefields-contracts-optimizationattemptstatus"></a>
### `OptimizationAttemptStatus`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAttemptStatus`；基类 `str`, `Enum`。

- `ACCEPTED` = `'accepted'` [枚举常量]
- `BUILD_FAILED` = `'build_failed'` [枚举常量]
- `QUALITY_REJECTED` = `'quality_rejected'` [枚举常量]
- `NUMERICAL_FAILURE` = `'numerical_failure'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-operations-forcefields-contracts-forcefieldvalidationreport"></a>
### `ForceFieldValidationReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldValidationReport`；基类 无。

```python
ForceFieldValidationReport(level: AcceptanceLevel, passed: bool, checks: Tuple[AcceptanceCheck, ...], metrics: Mapping[str, ForceFieldDiagnosticValue] = field(default_factory=dict))
```

- `to_dict(self) -> dict[str, ForceFieldDiagnosticValue]`
- `failures` [只读]: `Tuple[AcceptanceCheck, ...]`
- `warnings` [只读]: `Tuple[AcceptanceCheck, ...]`
- `level: AcceptanceLevel` [只读值字段]
- `passed: bool` [只读值字段]
- `checks: Tuple[AcceptanceCheck, ...]` [只读值字段]
- `metrics: Mapping[str, ForceFieldDiagnosticValue]` = `field(default_factory=dict)` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-optimizationattemptreport"></a>
### `OptimizationAttemptReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAttemptReport`；基类 无。

```python
OptimizationAttemptReport(route: OptimizationRoute, status: OptimizationAttemptStatus, quality_report: Optional[ForceFieldValidationReport] = None, termination_reason: Optional[TerminationReason] = None, build_succeeded: Optional[bool] = None, build_diagnostics: Optional['ComplexBuildDiagnostics'] = None, message: Optional[str] = None, elapsed_seconds: float = 0.0)
```

- `route: OptimizationRoute` [只读值字段]
- `status: OptimizationAttemptStatus` [只读值字段]
- `quality_report: Optional[ForceFieldValidationReport]` = `None` [只读值字段]
- `termination_reason: Optional[TerminationReason]` = `None` [只读值字段]
- `build_succeeded: Optional[bool]` = `None` [只读值字段]
- `build_diagnostics: Optional['ComplexBuildDiagnostics']` = `None` [只读值字段]
- `message: Optional[str]` = `None` [只读值字段]
- `elapsed_seconds: float` = `0.0` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-optimizationroutingreport"></a>
### `OptimizationRoutingReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.OptimizationRoutingReport`；基类 无。

```python
OptimizationRoutingReport(attempts: Tuple[OptimizationAttemptReport, ...], selected_route: OptimizationRoute)
```

- `attempts: Tuple[OptimizationAttemptReport, ...]` [只读值字段]
- `selected_route: OptimizationRoute` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-forcefieldrunreport"></a>
### `ForceFieldRunReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldRunReport`；基类 无。

```python
ForceFieldRunReport(requested_forcefield: Optional[str], effective_forcefield: str, setup_succeeded: bool, converged: bool, epochs_completed: int, steps_submitted: int, initialization_steps: int, steps_completed: Optional[int], final_energy: float, best_energy: float, energy_unit: str, rms_gradient: float, max_gradient: float, exploded: bool, quality_report: Optional[ForceFieldValidationReport] = None, backend_energy_unit: Optional[str] = None, gradient_unit: str = 'kJ/(mol*angstrom)', energy_changes: Tuple[float, ...] = (), max_displacements: Tuple[float, ...] = (), best_epoch: int = 0, selected_segment_epochs_completed: int = 0, epoch_energies: Tuple[float, ...] = (), termination_reason: TerminationReason = 'budget_exhausted', terminal_converged: bool = False, untangling: Optional['RingUntanglingReport'] = None, trajectory: Optional[ForceFieldTrajectoryArchive] = None, elapsed_seconds: float = 0.0, convergence_level: ConvergenceLevel = DEFAULT_CONVERGENCE_LEVEL, routing_report: Optional[OptimizationRoutingReport] = None)
```

- `requested_forcefield: Optional[str]` [只读值字段]
- `effective_forcefield: str` [只读值字段]
- `setup_succeeded: bool` [只读值字段]
- `converged: bool` [只读值字段]
- `epochs_completed: int` [只读值字段]
- `steps_submitted: int` [只读值字段]
- `initialization_steps: int` [只读值字段]
- `steps_completed: Optional[int]` [只读值字段]
- `final_energy: float` [只读值字段]
- `best_energy: float` [只读值字段]
- `energy_unit: str` [只读值字段]
- `rms_gradient: float` [只读值字段]
- `max_gradient: float` [只读值字段]
- `exploded: bool` [只读值字段]
- `quality_report: Optional[ForceFieldValidationReport]` = `None` [只读值字段]
- `backend_energy_unit: Optional[str]` = `None` [只读值字段]
- `gradient_unit: str` = `'kJ/(mol*angstrom)'` [只读值字段]
- `energy_changes: Tuple[float, ...]` = `()` [只读值字段]
- `max_displacements: Tuple[float, ...]` = `()` [只读值字段]
- `best_epoch: int` = `0` [只读值字段]
- `selected_segment_epochs_completed: int` = `0` [只读值字段]
- `epoch_energies: Tuple[float, ...]` = `()` [只读值字段]
- `termination_reason: TerminationReason` = `'budget_exhausted'` [只读值字段]
- `terminal_converged: bool` = `False` [只读值字段]
- `untangling: Optional['RingUntanglingReport']` = `None` [只读值字段]
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [只读值字段]
- `elapsed_seconds: float` = `0.0` [只读值字段]
- `convergence_level: ConvergenceLevel` = `DEFAULT_CONVERGENCE_LEVEL` [只读值字段]
- `routing_report: Optional[OptimizationRoutingReport]` = `None` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-build3dreport"></a>
### `Build3DReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.Build3DReport`；基类 无。

```python
Build3DReport(atom_count: int, added_hydrogen_count: int, quality_report: ForceFieldValidationReport)
```

- `atom_count: int` [只读值字段]
- `added_hydrogen_count: int` [只读值字段]
- `quality_report: ForceFieldValidationReport` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-candidaterejection"></a>
### `CandidateRejection`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.CandidateRejection`；基类 无。

```python
CandidateRejection(component_index: int, attempt: int, reason: str, quality_failures: Tuple[AcceptanceCheck, ...] = ())
```

- `component_index: int` [只读值字段]
- `attempt: int` [只读值字段]
- `reason: str` [只读值字段]
- `quality_failures: Tuple[AcceptanceCheck, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-ringuntanglingreport"></a>
### `RingUntanglingReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.RingUntanglingReport`；基类 无。

```python
RingUntanglingReport(attempt_limit: int, attempts_completed: int, initial_piercing_count: int, final_piercing_count: int, minimum_piercing_count: int, resolved: bool, warning_messages: Tuple[str, ...] = ())
```

- `attempt_limit: int` [只读值字段]
- `attempts_completed: int` [只读值字段]
- `initial_piercing_count: int` [只读值字段]
- `final_piercing_count: int` [只读值字段]
- `minimum_piercing_count: int` [只读值字段]
- `resolved: bool` [只读值字段]
- `warning_messages: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-coordinationbondrestorationreport"></a>
### `CoordinationBondRestorationReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.CoordinationBondRestorationReport`；基类 无。

```python
CoordinationBondRestorationReport(attempt_limit: int, attempts_completed: int, bond_count: int, metal_relocation_attempt_count: int, relocated_metal_indices: Tuple[int, ...], infeasible_metal_indices: Tuple[int, ...], forced_bond_keys: Tuple[Tuple[int, int], ...], rejected_piercing_trial_count: int, undetermined_trial_count: int, excluded_ring_observation_count: int, warning_messages: Tuple[str, ...] = (), elapsed_seconds: float = 0.0)
```

- `attempt_limit: int` [只读值字段]
- `attempts_completed: int` [只读值字段]
- `bond_count: int` [只读值字段]
- `metal_relocation_attempt_count: int` [只读值字段]
- `relocated_metal_indices: Tuple[int, ...]` [只读值字段]
- `infeasible_metal_indices: Tuple[int, ...]` [只读值字段]
- `forced_bond_keys: Tuple[Tuple[int, int], ...]` [只读值字段]
- `rejected_piercing_trial_count: int` [只读值字段]
- `undetermined_trial_count: int` [只读值字段]
- `excluded_ring_observation_count: int` [只读值字段]
- `warning_messages: Tuple[str, ...]` = `()` [只读值字段]
- `elapsed_seconds: float` = `0.0` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-complexbuilddiagnostics"></a>
### `ComplexBuildDiagnostics`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildDiagnostics`；基类 无。

```python
ComplexBuildDiagnostics(attempt_count: int, accepted_candidates: int, rejected_candidates: Tuple[CandidateRejection, ...], elapsed_seconds: float, warning_messages: Tuple[str, ...] = (), ligand_untangling: Tuple[RingUntanglingReport, ...] = (), coordination_restoration: Optional[CoordinationBondRestorationReport] = None, ligand_build_elapsed_seconds: float = 0.0)
```

- `attempt_count: int` [只读值字段]
- `accepted_candidates: int` [只读值字段]
- `rejected_candidates: Tuple[CandidateRejection, ...]` [只读值字段]
- `elapsed_seconds: float` [只读值字段]
- `warning_messages: Tuple[str, ...]` = `()` [只读值字段]
- `ligand_untangling: Tuple[RingUntanglingReport, ...]` = `()` [只读值字段]
- `coordination_restoration: Optional[CoordinationBondRestorationReport]` = `None` [只读值字段]
- `ligand_build_elapsed_seconds: float` = `0.0` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-buildworkerresult"></a>
### `BuildWorkerResult`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.BuildWorkerResult`；基类 无。

```python
BuildWorkerResult(status: Literal['ok', 'error'], coordinates: Optional[np.ndarray] = None, diagnostics: Optional[ComplexBuildDiagnostics] = None, error_type: Optional[str] = None, error_message: Optional[str] = None, traceback: Optional[str] = None, ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = (), sample_failure_code: Optional[str] = None)
```

- `status: Literal['ok', 'error']` [只读值字段]
- `coordinates: Optional[np.ndarray]` = `None` [只读值字段]
- `diagnostics: Optional[ComplexBuildDiagnostics]` = `None` [只读值字段]
- `error_type: Optional[str]` = `None` [只读值字段]
- `error_message: Optional[str]` = `None` [只读值字段]
- `traceback: Optional[str]` = `None` [只读值字段]
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [只读值字段]
- `sample_failure_code: Optional[str]` = `None` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-forcefieldworkflowreport"></a>
### `ForceFieldWorkflowReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`；基类 无。

```python
ForceFieldWorkflowReport(requested_forcefield: Optional[str], effective_forcefield: str, build: Union[Build3DReport, ComplexBuildDiagnostics], optimization: Optional[ForceFieldRunReport], quality_report: ForceFieldValidationReport, trajectory: Optional[ForceFieldTrajectoryArchive] = None)
```

- `requested_forcefield: Optional[str]` [只读值字段]
- `effective_forcefield: str` [只读值字段]
- `build: Union[Build3DReport, ComplexBuildDiagnostics]` [只读值字段]
- `optimization: Optional[ForceFieldRunReport]` [只读值字段]
- `quality_report: ForceFieldValidationReport` [只读值字段]
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-buildandoptimizereport"></a>
### `BuildAndOptimizeReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.BuildAndOptimizeReport`；基类 `ForceFieldWorkflowReport`。

```python
BuildAndOptimizeReport(requested_forcefield: Optional[str], effective_forcefield: str, quality_report: ForceFieldValidationReport, trajectory: Optional[ForceFieldTrajectoryArchive] = None, build: Build3DReport, optimization: ForceFieldRunReport)
```

- `build: Build3DReport` [只读值字段]
- `optimization: ForceFieldRunReport` [只读值字段]
- `requested_forcefield: Optional[str]` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- `effective_forcefield: str` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- `quality_report: ForceFieldValidationReport` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`

<a id="hotpot-cheminfo-operations-forcefields-contracts-complexbuildreport"></a>
### `ComplexBuildReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildReport`；基类 `ForceFieldWorkflowReport`。

```python
ComplexBuildReport(requested_forcefield: Optional[str], effective_forcefield: str, optimization: Optional[ForceFieldRunReport], quality_report: ForceFieldValidationReport, trajectory: Optional[ForceFieldTrajectoryArchive] = None, build: ComplexBuildDiagnostics)
```

- `build: ComplexBuildDiagnostics` [只读值字段]
- `requested_forcefield: Optional[str]` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- `effective_forcefield: str` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- `optimization: Optional[ForceFieldRunReport]` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- `quality_report: ForceFieldValidationReport` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [只读值字段] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`

<a id="hotpot-cheminfo-operations-forcefields-contracts-forcefieldsetupreport"></a>
### `ForceFieldSetupReport`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupReport`；基类 无。

```python
ForceFieldSetupReport(requested_forcefield: Optional[str], effective_forcefield: str, stage: ForceFieldSetupStage, setup_succeeded: bool = False, workflow_stage: Optional[ForceFieldWorkflowStage] = None)
```

- `requested_forcefield: Optional[str]` [只读值字段]
- `effective_forcefield: str` [只读值字段]
- `stage: ForceFieldSetupStage` [只读值字段]
- `setup_succeeded: bool` = `False` [只读值字段]
- `workflow_stage: Optional[ForceFieldWorkflowStage]` = `None` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-contracts-forcefielderror"></a>
### `ForceFieldError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`；基类 `RuntimeError`。

- `__init__(self, message: str) -> None`
- `sample_failure_code: Optional[str]` = `None` [字段声明；写权限依具体合同]
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-forcefields-contracts-forcefieldsetuperror"></a>
### `ForceFieldSetupError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupError`；基类 `ForceFieldError`。

- `__init__(self, message: str, report: Optional[ForceFieldSetupReport]=None, diagnostics: Optional[ComplexBuildDiagnostics]=None) -> None`
- `sample_failure_code: Optional[str]` = `'FF-SETUP-UNAVAILABLE'` [字段声明；写权限依具体合同]
- `report` = `report` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `diagnostics` = `diagnostics` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`

<a id="hotpot-cheminfo-operations-forcefields-contracts-buildworkererror"></a>
### `BuildWorkerError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.BuildWorkerError`；基类 `ForceFieldError`。

- `__init__(self, error_type: str, error_message: str, worker_traceback: Optional[str], diagnostics: Optional[ComplexBuildDiagnostics]=None) -> None`
- `error_type` = `error_type` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `error_message` = `error_message` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `worker_traceback` = `worker_traceback` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `diagnostics` = `diagnostics` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `sample_failure_code: Optional[str]` = `None` [字段声明；写权限依具体合同] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`

<a id="hotpot-cheminfo-operations-forcefields-contracts-buildtimeouterror"></a>
### `BuildTimeoutError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.BuildTimeoutError`；基类 `ForceFieldError`, `TimeoutError`。

- `sample_failure_code: Optional[str]` = `'FF-BUILD-TIMEOUT'` [字段声明；写权限依具体合同]
- `__init__(self, message: str) -> None` ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`

<a id="hotpot-cheminfo-operations-forcefields-contracts-complexbuilderror"></a>
### `ComplexBuildError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildError`；基类 `ForceFieldError`。

- `__init__(self, message: str, diagnostics: Optional[ComplexBuildDiagnostics]=None) -> None`
- `diagnostics` = `diagnostics` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `sample_failure_code: Optional[str]` = `None` [字段声明；写权限依具体合同] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`

<a id="hotpot-cheminfo-operations-forcefields-contracts-complexbuildworkererror"></a>
### `ComplexBuildWorkerError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildWorkerError`；基类 `ComplexBuildError`。

- `__init__(self, error_type: str, error_message: str, worker_traceback: Optional[str], diagnostics: Optional[ComplexBuildDiagnostics]=None) -> None`
- `error_type` = `error_type` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `error_message` = `error_message` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `worker_traceback` = `worker_traceback` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `diagnostics` = `diagnostics` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildError`
- `sample_failure_code: Optional[str]` = `None` [字段声明；写权限依具体合同] ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`

<a id="hotpot-cheminfo-operations-forcefields-contracts-complexbuildtimeouterror"></a>
### `ComplexBuildTimeoutError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildTimeoutError`；基类 `ComplexBuildError`, `TimeoutError`。

- `sample_failure_code: Optional[str]` = `'FF-COMPLEX-BUILD-TIMEOUT'` [字段声明；写权限依具体合同]
- `__init__(self, message: str, diagnostics: Optional[ComplexBuildDiagnostics]=None) -> None` ← `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildError`
- `diagnostics` = `diagnostics` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildError`
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`

<a id="hotpot-cheminfo-operations-forcefields-contracts-complexbuildwarning"></a>
### `ComplexBuildWarning`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildWarning`；基类 `UserWarning`。


<a id="hotpot-cheminfo-operations-forcefields-contracts-geometryqualityerror"></a>
### `GeometryQualityError`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.GeometryQualityError`；基类 `ForceFieldError`。

- `__init__(self, report: Optional[ForceFieldValidationReport]) -> None`
- `sample_failure_code: Optional[str]` = `'FF-NO-USABLE-FRAME'` [字段声明；写权限依具体合同]
- `report` = `report` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `trajectory: Optional[ForceFieldTrajectoryArchive]` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`

<a id="hotpot-cheminfo-operations-forcefields-contracts-geometryqualitywarning"></a>
### `GeometryQualityWarning`

定义 owner `hotpot.cheminfo.operations.forcefields.contracts.GeometryQualityWarning`；基类 `UserWarning`。


模块值/类型别名（声明存在不构成事实写许可）：

- `OptimizationAlgorithm` = `Literal['steepest', 'conjugate']`
- `TrajectoryPath` = `Union[str, os.PathLike[str]]`
- `TerminationReason` = `Literal['converged', 'budget_exhausted', 'stability_reached', 'topology_blocked', 'nonfinite_coordinates', 'nonfinite_energy', 'nonfinite_gradients', 'explosion_detected']`
- `ForceFieldWorkflowStage` = `Literal['coordination_restoration', 'complex_optimization']`
- `ForceFieldSetupStage` = `Literal['lookup', 'setup', 'preflight-validation']`


<a id="hotpot-cheminfo-operations-forcefields-coordinates"></a>
## `hotpot.cheminfo.operations.forcefields.coordinates`

`hotpot/cheminfo/operations/forcefields/coordinates.py`；显式 __all__: ('perturb',)


<a id="hotpot-cheminfo-operations-forcefields-coordinates-perturb"></a>
### `perturb`

```python
perturb(mol: 'Molecule', *, sigma: float=0.5, seed: Optional[int]=None) -> np.ndarray
```


<a id="hotpot-cheminfo-operations-forcefields-coordination"></a>
## `hotpot.cheminfo.operations.forcefields.coordination`

`hotpot/cheminfo/operations/forcefields/coordination.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields-ff"></a>
## `hotpot.cheminfo.operations.forcefields.ff`

`hotpot/cheminfo/operations/forcefields/ff.py`；显式 __all__: ('TrajectoryPath', 'TrajectoryStart', 'TrajectoryStage', 'TrajectoryEvent', 'AtomIdentity', 'BondTopology', 'BondTopologyRevision', 'RingFrameEvidence', 'CoordinationFrameEvidence', 'OptimizationFrameEvidence', 'FrameEvidence', 'ForceFieldFrame', 'ForceFieldTrajectory', 'ForceFieldTrajectoryArchive', 'OptimizationAlgorithm', 'OptimizationRoute', 'OptimizationAttemptStatus', 'OptimizationAttemptReport', 'OptimizationRoutingReport', 'ConvergenceLevel', 'DEFAULT_CONVERGENCE_LEVEL', 'OptimizationStoppingCriteria', 'TerminationReason', 'ForceFieldDiagnosticValue', 'ForceFieldRunReport', 'Build3DReport', 'CandidateRejection', 'RingUntanglingReport', 'CoordinationBondRestorationReport', 'ComplexBuildDiagnostics', 'ForceFieldWorkflowReport', 'BuildAndOptimizeReport', 'ComplexBuildReport', 'ForceFieldSetupReport', 'ForceFieldSetupStage', 'ForceFieldWorkflowStage', 'AcceptanceCheck', 'StructureAcceptanceThresholds', 'ForceFieldAcceptanceEvidence', 'AtomTopologySignature', 'TopologyReference', 'ForceFieldValidationReport', 'CoordinationEnvironment', 'ForceFieldError', 'ForceFieldSetupError', 'BuildWorkerError', 'BuildTimeoutError', 'ComplexBuildError', 'ComplexBuildWarning', 'ComplexBuildWorkerError', 'ComplexBuildTimeoutError', 'GeometryQualityError', 'GeometryQualityWarning', 'capture_topology', 'evaluate_structure_acceptance', 'is_structure_accepted', 'perturb', 'collect_coordination_environments', 'build3d', 'optimize', 'build_complex3d', 'optimize_complex', 'complexes_build', 'build_and_optimize', 'auto_optimize')

- 导出 `TrajectoryPath` → `hotpot.cheminfo.operations.forcefields.contracts.TrajectoryPath`
- 导出 `TrajectoryStart` → `hotpot.cheminfo.structure.forcefields.TrajectoryStart`
- 导出 `TrajectoryStage` → `hotpot.cheminfo.structure.forcefields.TrajectoryStage`
- 导出 `TrajectoryEvent` → `hotpot.cheminfo.structure.forcefields.TrajectoryEvent`
- 导出 `AtomIdentity` → `hotpot.cheminfo.structure.forcefields.AtomIdentity`
- 导出 `BondTopology` → `hotpot.cheminfo.structure.forcefields.BondTopology`
- 导出 `BondTopologyRevision` → `hotpot.cheminfo.structure.forcefields.BondTopologyRevision`
- 导出 `RingFrameEvidence` → `hotpot.cheminfo.structure.forcefields.RingFrameEvidence`
- 导出 `CoordinationFrameEvidence` → `hotpot.cheminfo.structure.forcefields.CoordinationFrameEvidence`
- 导出 `OptimizationFrameEvidence` → `hotpot.cheminfo.structure.forcefields.OptimizationFrameEvidence`
- 导出 `FrameEvidence` → `hotpot.cheminfo.structure.forcefields.FrameEvidence`
- 导出 `ForceFieldFrame` → `hotpot.cheminfo.structure.forcefields.ForceFieldFrame`
- 导出 `ForceFieldTrajectory` → `hotpot.cheminfo.operations.forcefields.trajectory.ForceFieldTrajectory`
- 导出 `ForceFieldTrajectoryArchive` → `hotpot.cheminfo.operations.forcefields.trajectory.ForceFieldTrajectoryArchive`
- 导出 `OptimizationAlgorithm` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAlgorithm`
- 导出 `OptimizationRoute` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationRoute`
- 导出 `OptimizationAttemptStatus` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAttemptStatus`
- 导出 `OptimizationAttemptReport` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationAttemptReport`
- 导出 `OptimizationRoutingReport` → `hotpot.cheminfo.operations.forcefields.contracts.OptimizationRoutingReport`
- 导出 `ConvergenceLevel` → `hotpot.cheminfo.obWrappers.contracts.ConvergenceLevel`
- 导出 `DEFAULT_CONVERGENCE_LEVEL` → `hotpot.cheminfo.obWrappers.contracts.DEFAULT_CONVERGENCE_LEVEL`
- 导出 `OptimizationStoppingCriteria` → `hotpot.cheminfo.rules.physics.forcefields.contracts.OptimizationStoppingCriteria`
- 导出 `TerminationReason` → `hotpot.cheminfo.operations.forcefields.contracts.TerminationReason`
- 导出 `ForceFieldDiagnosticValue` → `hotpot.cheminfo.rules.physics.forcefields.contracts.ForceFieldDiagnosticValue`
- 导出 `ForceFieldRunReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldRunReport`
- 导出 `Build3DReport` → `hotpot.cheminfo.operations.forcefields.contracts.Build3DReport`
- 导出 `CandidateRejection` → `hotpot.cheminfo.operations.forcefields.contracts.CandidateRejection`
- 导出 `RingUntanglingReport` → `hotpot.cheminfo.operations.forcefields.contracts.RingUntanglingReport`
- 导出 `CoordinationBondRestorationReport` → `hotpot.cheminfo.operations.forcefields.contracts.CoordinationBondRestorationReport`
- 导出 `ComplexBuildDiagnostics` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildDiagnostics`
- 导出 `ForceFieldWorkflowReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowReport`
- 导出 `BuildAndOptimizeReport` → `hotpot.cheminfo.operations.forcefields.contracts.BuildAndOptimizeReport`
- 导出 `ComplexBuildReport` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildReport`
- 导出 `ForceFieldSetupReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupReport`
- 导出 `ForceFieldSetupStage` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupStage`
- 导出 `ForceFieldWorkflowStage` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldWorkflowStage`
- 导出 `AcceptanceCheck` → `hotpot.cheminfo.rules.physics.forcefields.contracts.AcceptanceCheck`
- 导出 `StructureAcceptanceThresholds` → `hotpot.cheminfo.rules.physics.forcefields.contracts.StructureAcceptanceThresholds`
- 导出 `ForceFieldAcceptanceEvidence` → `hotpot.cheminfo.rules.physics.forcefields.contracts.ForceFieldAcceptanceEvidence`
- 导出 `AtomTopologySignature` → `hotpot.cheminfo.structure.forcefields.AtomTopologySignature`
- 导出 `TopologyReference` → `hotpot.cheminfo.structure.forcefields.TopologyReference`
- 导出 `ForceFieldValidationReport` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldValidationReport`
- 导出 `CoordinationEnvironment` → `hotpot.cheminfo.structure.forcefields.CoordinationEnvironment`
- 导出 `ForceFieldError` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldError`
- 导出 `ForceFieldSetupError` → `hotpot.cheminfo.operations.forcefields.contracts.ForceFieldSetupError`
- 导出 `BuildWorkerError` → `hotpot.cheminfo.operations.forcefields.contracts.BuildWorkerError`
- 导出 `BuildTimeoutError` → `hotpot.cheminfo.operations.forcefields.contracts.BuildTimeoutError`
- 导出 `ComplexBuildError` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildError`
- 导出 `ComplexBuildWarning` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildWarning`
- 导出 `ComplexBuildWorkerError` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildWorkerError`
- 导出 `ComplexBuildTimeoutError` → `hotpot.cheminfo.operations.forcefields.contracts.ComplexBuildTimeoutError`
- 导出 `GeometryQualityError` → `hotpot.cheminfo.operations.forcefields.contracts.GeometryQualityError`
- 导出 `GeometryQualityWarning` → `hotpot.cheminfo.operations.forcefields.contracts.GeometryQualityWarning`
- 导出 `capture_topology` → `hotpot.cheminfo.adapters.structure.forcefields.capture_topology`；`capture_topology(mol: 'Molecule', *, allow_added_hydrogens: bool=True) -> TopologyReference`
- 导出 `evaluate_structure_acceptance` → `hotpot.cheminfo.operations.forcefields.acceptance.evaluate_structure_acceptance`；`evaluate_structure_acceptance(mol: 'Molecule', *, level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> ForceFieldValidationReport`
- 导出 `is_structure_accepted` → `hotpot.cheminfo.operations.forcefields.acceptance.is_structure_accepted`；`is_structure_accepted(mol: 'Molecule', *, level: AcceptanceLevel='standard', topology_reference: Optional[TopologyReference]=None, forcefield_report: Optional[ForceFieldAcceptanceEvidence]=None, forcefield_stage: ForceFieldStage='final', thresholds: Optional[StructureAcceptanceThresholds]=None) -> bool`
- 导出 `perturb` → `hotpot.cheminfo.operations.forcefields.coordinates.perturb`；`perturb(mol: 'Molecule', *, sigma: float=0.5, seed: Optional[int]=None) -> np.ndarray`
- 导出 `collect_coordination_environments` → `hotpot.cheminfo.adapters.structure.forcefields.collect_coordination_environments`；`collect_coordination_environments(mol: 'Molecule') -> Tuple[CoordinationEnvironment, ...]`
- 导出 `build3d` → `hotpot.cheminfo.operations.forcefields.workflows.build3d`；`build3d(mol: 'Molecule', *, add_hydrogens: bool=True, seed: Optional[int]=None, timeout: float=1000.0) -> Build3DReport`
- 导出 `optimize` → `hotpot.cheminfo.operations.forcefields.workflows.optimize`；`optimize(mol: 'Molecule', forcefield: Optional[str]='UFF', *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=1, steps_per_epoch: int=100, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.FINAL_OPTIMIZATION, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport`
- 导出 `build_complex3d` → `hotpot.cheminfo.operations.forcefields.workflows.build_complex3d`；`build_complex3d(mol: 'Molecule', forcefield: Optional[str]=None, *, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, timeout: float=1000.0, add_hydrogens: bool=True, seed: Optional[int]=None, perturb_sigma: float=0.5, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION, trajectory_path: Optional[TrajectoryPath]=None) -> ComplexBuildReport`
- 导出 `optimize_complex` → `hotpot.cheminfo.operations.forcefields.workflows.optimize_complex`；`optimize_complex(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, complex_untangling_attempts: int=30, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COMPLEX_UNTANGLING, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport`
- 导出 `complexes_build` → `hotpot.cheminfo.operations.forcefields.workflows.complexes_build`；`complexes_build(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30, timeout: float=1000.0, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ComplexBuildReport`
- 导出 `build_and_optimize` → `hotpot.cheminfo.operations.forcefields.workflows.build_and_optimize`；`build_and_optimize(mol: 'Molecule', forcefield: Optional[str]='UFF', *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, timeout: float=1000.0, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional[TrajectoryStart]=None, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30) -> ForceFieldWorkflowReport`
- 导出 `auto_optimize` → `hotpot.cheminfo.operations.forcefields.auto.auto_optimize`；`auto_optimize(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30, timeout: float=1000.0, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional[TrajectoryStart]=None, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport`


<a id="hotpot-cheminfo-operations-forcefields-ligand"></a>
## `hotpot.cheminfo.operations.forcefields.ligand`

`hotpot/cheminfo/operations/forcefields/ligand.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields-native-adapters"></a>
## `hotpot.cheminfo.operations.forcefields.native_adapters`

`hotpot/cheminfo/operations/forcefields/native_adapters.py`；显式 __all__: ('NATIVE_WARNING_MESSAGES', 'NativeOptimizationPerturbationStreams', 'NativePerturbationStreams', 'apply_native_selected_structure', 'coordination_restoration_report', 'forcefield_run_report', 'ingest_native_trajectory', 'native_coordination_offsets', 'native_optimization_offsets', 'native_perturbation_streams', 'native_warning_messages', 'ring_untangling_report')


<a id="hotpot-cheminfo-operations-forcefields-native-adapters-nativeoptimizationperturbationstreams"></a>
### `NativeOptimizationPerturbationStreams`

定义 owner `hotpot.cheminfo.operations.forcefields.native_adapters.NativeOptimizationPerturbationStreams`；基类 无。

```python
NativeOptimizationPerturbationStreams(untangling: NDArray[np.float64], optimization: NDArray[np.float64])
```

- `untangling: NDArray[np.float64]` [只读值字段]
- `optimization: NDArray[np.float64]` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-nativeperturbationstreams"></a>
### `NativePerturbationStreams`

定义 owner `hotpot.cheminfo.operations.forcefields.native_adapters.NativePerturbationStreams`；基类 无。

```python
NativePerturbationStreams(coordination: NDArray[np.float64], untangling: NDArray[np.float64], optimization: NDArray[np.float64])
```

- `coordination: NDArray[np.float64]` [只读值字段]
- `untangling: NDArray[np.float64]` [只读值字段]
- `optimization: NDArray[np.float64]` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-native-coordination-offsets"></a>
### `native_coordination_offsets`

```python
native_coordination_offsets(atom_count: int, seed: Optional[int], *, options: CoordinationStageOptions=CoordinationStageOptions()) -> NDArray[np.float64]
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-native-optimization-offsets"></a>
### `native_optimization_offsets`

```python
native_optimization_offsets(atom_count: int, seed: Optional[int], *, options: ComplexOptimizationOptions=ComplexOptimizationOptions()) -> NativeOptimizationPerturbationStreams
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-native-perturbation-streams"></a>
### `native_perturbation_streams`

```python
native_perturbation_streams(atom_count: int, seed: Optional[int], *, coordination_options: CoordinationStageOptions=CoordinationStageOptions(), optimization_options: ComplexOptimizationOptions=ComplexOptimizationOptions()) -> NativePerturbationStreams
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-native-warning-messages"></a>
### `native_warning_messages`

```python
native_warning_messages(codes: Sequence[str]) -> Tuple[str, ...]
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-coordination-restoration-report"></a>
### `coordination_restoration_report`

```python
coordination_restoration_report(result: CoordinationStageResult) -> CoordinationBondRestorationReport
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-ring-untangling-report"></a>
### `ring_untangling_report`

```python
ring_untangling_report(result: ComplexOptimizationResult) -> RingUntanglingReport
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-forcefield-run-report"></a>
### `forcefield_run_report`

```python
forcefield_run_report(result: ComplexOptimizationResult, *, requested_forcefield: Optional[str], effective_forcefield: str, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, trajectory: Optional[ForceFieldTrajectoryArchive]=None) -> ForceFieldRunReport
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-apply-native-selected-structure"></a>
### `apply_native_selected_structure`

```python
apply_native_selected_structure(mol: 'Molecule', session_input: ComplexSessionInput, result: _SelectedStructureResult) -> None
```

<a id="hotpot-cheminfo-operations-forcefields-native-adapters-ingest-native-trajectory"></a>
### `ingest_native_trajectory`

```python
ingest_native_trajectory(trajectory: ForceFieldTrajectory, batch: NativeTrajectoryBatch, session_input: ComplexSessionInput) -> Tuple[int, ...]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `NATIVE_WARNING_MESSAGES: Mapping[str, str]` = `<长模块值；见 hotpot/cheminfo/operations/forcefields/native_adapters.py:64>`


<a id="hotpot-cheminfo-operations-forcefields-optimizer"></a>
## `hotpot.cheminfo.operations.forcefields.optimizer`

`hotpot/cheminfo/operations/forcefields/optimizer.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields-repair"></a>
## `hotpot.cheminfo.operations.forcefields.repair`

`hotpot/cheminfo/operations/forcefields/repair.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields-stage"></a>
## `hotpot.cheminfo.operations.forcefields.stage`

`hotpot/cheminfo/operations/forcefields/stage.py`；显式 __all__: ('ForceFieldRouteOperations', 'ForceFieldRouteOptions', 'STAGE', 'execute_forcefield_route', 'get_stage', 'json_value')


<a id="hotpot-cheminfo-operations-forcefields-stage-forcefield-error-evidence"></a>
### `forcefield_error_evidence`

```python
forcefield_error_evidence(error: ForceFieldError) -> dict[str, object]
```

<a id="hotpot-cheminfo-operations-forcefields-stage-forcefieldrouteoptions"></a>
### `ForceFieldRouteOptions`

定义 owner `hotpot.cheminfo.operations.forcefields.stage.ForceFieldRouteOptions`；基类 无。

```python
ForceFieldRouteOptions(rebuild: bool, optimize_only: bool, route: str, forcefield: Optional[str], algorithm: str, epochs: int, steps_per_epoch: int, add_hydrogens: bool, quality_level: str, seed: Optional[int], timeout: float, trajectory_start: Optional[TrajectoryStart], convergence_level: ConvergenceLevel = ConvergenceLevel.FAST)
```

- `rebuild: bool` [只读值字段]
- `optimize_only: bool` [只读值字段]
- `route: str` [只读值字段]
- `forcefield: Optional[str]` [只读值字段]
- `algorithm: str` [只读值字段]
- `epochs: int` [只读值字段]
- `steps_per_epoch: int` [只读值字段]
- `add_hydrogens: bool` [只读值字段]
- `quality_level: str` [只读值字段]
- `seed: Optional[int]` [只读值字段]
- `timeout: float` [只读值字段]
- `trajectory_start: Optional[TrajectoryStart]` [只读值字段]
- `convergence_level: ConvergenceLevel` = `ConvergenceLevel.FAST` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-stage-forcefieldrouteoperations"></a>
### `ForceFieldRouteOperations`

定义 owner `hotpot.cheminfo.operations.forcefields.stage.ForceFieldRouteOperations`；基类 无。

```python
ForceFieldRouteOperations(build3d: _BuildOperation, build_and_optimize: _RouteOperation, auto_optimize: _RouteOperation, complexes_build: _RouteOperation, optimize_complex: _RouteOperation, optimize: _RouteOperation)
```

- `build3d: _BuildOperation` [只读值字段]
- `build_and_optimize: _RouteOperation` [只读值字段]
- `auto_optimize: _RouteOperation` [只读值字段]
- `complexes_build: _RouteOperation` [只读值字段]
- `optimize_complex: _RouteOperation` [只读值字段]
- `optimize: _RouteOperation` [只读值字段]

<a id="hotpot-cheminfo-operations-forcefields-stage-execute-forcefield-route"></a>
### `execute_forcefield_route`

```python
execute_forcefield_route(mol: 'Molecule', options: ForceFieldRouteOptions, *, trajectory_path: Optional[TrajectoryPath]=None, operations: Optional[ForceFieldRouteOperations]=None) -> ForceFieldRouteReport
```

<a id="hotpot-cheminfo-operations-forcefields-stage-json-value"></a>
### `json_value`

```python
json_value(value: object) -> JSONValue
```

<a id="hotpot-cheminfo-operations-forcefields-stage-get-stage"></a>
### `get_stage`

```python
get_stage() -> MolecularStage
```

模块值/类型别名（声明存在不构成事实写许可）：

- `ForceFieldRouteReport` = `Union[ForceFieldRunReport, ForceFieldWorkflowReport]`
- `STAGE` = `_ForceFieldStage()`


<a id="hotpot-cheminfo-operations-forcefields-trajectory"></a>
## `hotpot.cheminfo.operations.forcefields.trajectory`

`hotpot/cheminfo/operations/forcefields/trajectory.py`；显式 __all__: ('TrajectoryStart', 'TrajectoryStage', 'TrajectoryEvent', 'AtomIdentity', 'BondTopology', 'BondTopologyRevision', 'RingFrameEvidence', 'CoordinationFrameEvidence', 'OptimizationFrameEvidence', 'FrameEvidence', 'ForceFieldFrame', 'ForceFieldTrajectory', 'ForceFieldTrajectoryArchive')

- 导出 `TrajectoryStart` → `hotpot.cheminfo.structure.forcefields.TrajectoryStart`
- 导出 `TrajectoryStage` → `hotpot.cheminfo.structure.forcefields.TrajectoryStage`
- 导出 `TrajectoryEvent` → `hotpot.cheminfo.structure.forcefields.TrajectoryEvent`
- 导出 `AtomIdentity` → `hotpot.cheminfo.structure.forcefields.AtomIdentity`
- 导出 `BondTopology` → `hotpot.cheminfo.structure.forcefields.BondTopology`
- 导出 `BondTopologyRevision` → `hotpot.cheminfo.structure.forcefields.BondTopologyRevision`
- 导出 `RingFrameEvidence` → `hotpot.cheminfo.structure.forcefields.RingFrameEvidence`
- 导出 `CoordinationFrameEvidence` → `hotpot.cheminfo.structure.forcefields.CoordinationFrameEvidence`
- 导出 `OptimizationFrameEvidence` → `hotpot.cheminfo.structure.forcefields.OptimizationFrameEvidence`
- 导出 `FrameEvidence` → `hotpot.cheminfo.structure.forcefields.FrameEvidence`
- 导出 `ForceFieldFrame` → `hotpot.cheminfo.structure.forcefields.ForceFieldFrame`

<a id="hotpot-cheminfo-operations-forcefields-trajectory-forcefieldtrajectory"></a>
### `ForceFieldTrajectory`

定义 owner `hotpot.cheminfo.operations.forcefields.trajectory.ForceFieldTrajectory`；基类 无。

- `__init__(self, atoms: Sequence[AtomIdentity], *, start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION) -> None`
- `from_molecule(cls, mol: 'Molecule', *, start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION) -> 'ForceFieldTrajectory'` [classmethod]
- `__len__(self) -> int`
- `__iter__(self) -> Iterator[ForceFieldFrame]`
- `__getitem__(self, index: int) -> ForceFieldFrame`
- `records(self, stage: TrajectoryStage) -> bool`
- `coordinates(self, frame_index: int) -> np.ndarray`
- `topology(self, frame_index: int) -> BondTopologyRevision`
- `record(self, coordinates: np.ndarray, bonds: Sequence[BondTopology], *, stage: TrajectoryStage, event: TrajectoryEvent, energy_kj_mol: Optional[float]=None, component_index: Optional[int]=None, attempt: Optional[int]=None, step: Optional[int]=None, evidence: Optional[FrameEvidence]=None) -> ForceFieldFrame`
- `record_molecule(self, mol: 'Molecule', *, stage: TrajectoryStage, event: TrajectoryEvent, energy_kj_mol: Optional[float]=None, component_index: Optional[int]=None, attempt: Optional[int]=None, step: Optional[int]=None, evidence: Optional[FrameEvidence]=None) -> ForceFieldFrame`
- `select(self, frame_index: int) -> ForceFieldFrame`
- `set_terminal(self, frame_index: int) -> ForceFieldFrame`
- `ingest_native_batch(self, batch: 'NativeTrajectoryBatch', session_input: 'ComplexSessionInput') -> Tuple[int, ...]`
- `materialize(self, mol: 'Molecule', *, keep_all: bool) -> None`
- `write(self, path: Union[str, Path], *, include_sdf: bool=True) -> None`
- `read(cls, path: Union[str, Path]) -> 'ForceFieldTrajectory'` [classmethod]
- `write_sdf(self, path: Union[str, Path]) -> None`
- `atoms` [只读]: `Tuple[AtomIdentity, ...]`
- `start` [只读]: `TrajectoryStart`
- `frames` [只读]: `Tuple[ForceFieldFrame, ...]`
- `topology_revisions` [只读]: `Tuple[BondTopologyRevision, ...]`
- `coordinate_revision_count` [只读]: `int`
- `topology_revision_count` [只读]: `int`
- `selected_index` [只读]: `Optional[int]`
- `selected_frame` [只读]: `Optional[ForceFieldFrame]`
- `terminal_index` [只读]: `Optional[int]`
- `terminal_frame` [只读]: `Optional[ForceFieldFrame]`

<a id="hotpot-cheminfo-operations-forcefields-trajectory-forcefieldtrajectoryarchive"></a>
### `ForceFieldTrajectoryArchive`

定义 owner `hotpot.cheminfo.operations.forcefields.trajectory.ForceFieldTrajectoryArchive`；基类 无。

```python
ForceFieldTrajectoryArchive(main: ForceFieldTrajectory, ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = (), preliminary_attempts: Tuple[ForceFieldTrajectory, ...] = ())
```

- `write(self, path: Union[str, Path], *, include_sdf: bool=True) -> None`
- `read(cls, path: Union[str, Path]) -> 'ForceFieldTrajectoryArchive'` [classmethod]
- `main: ForceFieldTrajectory` [只读值字段]
- `ligand_build_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [只读值字段]
- `preliminary_attempts: Tuple[ForceFieldTrajectory, ...]` = `()` [只读值字段]


<a id="hotpot-cheminfo-operations-forcefields-workers"></a>
## `hotpot.cheminfo.operations.forcefields.workers`

`hotpot/cheminfo/operations/forcefields/workers.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-forcefields-workflows"></a>
## `hotpot.cheminfo.operations.forcefields.workflows`

`hotpot/cheminfo/operations/forcefields/workflows.py`；显式 __all__: ('build3d', 'optimize', 'build_complex3d', 'optimize_complex', 'complexes_build', 'build_and_optimize')


<a id="hotpot-cheminfo-operations-forcefields-workflows-build3d"></a>
### `build3d`

```python
build3d(mol: 'Molecule', *, add_hydrogens: bool=True, seed: Optional[int]=None, timeout: float=1000.0) -> Build3DReport
```

<a id="hotpot-cheminfo-operations-forcefields-workflows-optimize"></a>
### `optimize`

```python
optimize(mol: 'Molecule', forcefield: Optional[str]='UFF', *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=1, steps_per_epoch: int=100, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.FINAL_OPTIMIZATION, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport
```

<a id="hotpot-cheminfo-operations-forcefields-workflows-build-complex3d"></a>
### `build_complex3d`

```python
build_complex3d(mol: 'Molecule', forcefield: Optional[str]=None, *, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, timeout: float=1000.0, add_hydrogens: bool=True, seed: Optional[int]=None, perturb_sigma: float=0.5, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION, trajectory_path: Optional[TrajectoryPath]=None) -> ComplexBuildReport
```

<a id="hotpot-cheminfo-operations-forcefields-workflows-optimize-complex"></a>
### `optimize_complex`

```python
optimize_complex(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, complex_untangling_attempts: int=30, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COMPLEX_UNTANGLING, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ForceFieldRunReport
```

<a id="hotpot-cheminfo-operations-forcefields-workflows-complexes-build"></a>
### `complexes_build`

```python
complexes_build(mol: 'Molecule', forcefield: Optional[str]=None, *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30, timeout: float=1000.0, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: TrajectoryStart=TrajectoryStart.COORDINATION_RESTORATION, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5) -> ComplexBuildReport
```

<a id="hotpot-cheminfo-operations-forcefields-workflows-build-and-optimize"></a>
### `build_and_optimize`

```python
build_and_optimize(mol: 'Molecule', forcefield: Optional[str]='UFF', *, algorithm: OptimizationAlgorithm='conjugate', epochs: int=100, steps_per_epoch: int=100, add_hydrogens: bool=True, quality_level: AcceptanceLevel='standard', quality_thresholds: Optional[StructureAcceptanceThresholds]=None, seed: Optional[int]=None, timeout: float=1000.0, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional[OptimizationStoppingCriteria]=None, convergence_level: ConvergenceLevel=DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional[TrajectoryStart]=None, trajectory_path: Optional[TrajectoryPath]=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30) -> ForceFieldWorkflowReport
```


<a id="hotpot-cheminfo-operations-forcefields-working-copy"></a>
## `hotpot.cheminfo.operations.forcefields.working_copy`

`hotpot/cheminfo/operations/forcefields/working_copy.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-geometry"></a>
## `hotpot.cheminfo.operations.geometry`

`hotpot/cheminfo/operations/geometry.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-io"></a>
## `hotpot.cheminfo.operations.io`

`hotpot/cheminfo/operations/io.py`；显式 __all__: ('MolReader', 'MolWriter', 'read_mol')


<a id="hotpot-cheminfo-operations-io-molreader"></a>
### `MolReader`

定义 owner `hotpot.cheminfo.operations.io.MolReader`；基类 `IoBase`。

- `__init__(self, src, fmt=None, **kwargs)`
- `get_generator(self)`
- `refresh(self)`
- `__next__(self)`
- `__getitem__(self, item: int)`
- `__iter__(self)`
- `fmt: cython.char` [字段声明；写权限依具体合同]
- `src: cython.char` [字段声明；写权限依具体合同]
- `src_type` = `'bytes' if isinstance(self.src, bytes) else 'str'` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `ob_opt` = `kwargs.pop('ob_opt', {})` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.adapters.io.base.IoBase`
- `kwargs` = `kwargs` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.adapters.io.base.IoBase`

<a id="hotpot-cheminfo-operations-io-molwriter"></a>
### `MolWriter`

定义 owner `hotpot.cheminfo.operations.io.MolWriter`；基类 `IoBase`。

- `__init__(self, fp: Union[str, Path], fmt: str=None, overwrite: bool=False, **kwargs) -> None`
- `obmol_write_string(self, obmol: 'ob.OBMol')`
- `to_write(self, script: str)`
- `write(self, mol: 'Molecule', *args, **kwargs) -> Optional[str]`
- `add_plugin(fmt: str, where: Literal['pre', 'write', 'post']='post')` [staticmethod]
- `ob_conv` = `None` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `fp` = `self.src` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `src` = `src` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.adapters.io.base.IoBase`
- `src_type` = `self._src_checks(src)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.adapters.io.base.IoBase`
- `fmt` = `self._determine_fmt(src, fmt)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.adapters.io.base.IoBase`
- `ob_opt` = `kwargs.pop('ob_opt', {})` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.adapters.io.base.IoBase`
- `kwargs` = `kwargs` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.adapters.io.base.IoBase`

<a id="hotpot-cheminfo-operations-io-read-mol"></a>
### `read_mol`

```python
read_mol(src, fmt=None, **kwargs)
```

<a id="hotpot-cheminfo-operations-io-to-gjf-preprocess"></a>
### `to_gjf_preprocess`

```python
to_gjf_preprocess(writer, mol, *args, **kwargs)
```

<a id="hotpot-cheminfo-operations-io-to-gjf"></a>
### `to_gjf`

```python
to_gjf(writer, script, mol, *args, **kwargs) -> str
```


<a id="hotpot-cheminfo-operations-native-packing"></a>
## `hotpot.cheminfo.operations.native_packing`

`hotpot/cheminfo/operations/native_packing.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-native-packing-pack-molecule"></a>
### `pack_molecule`

```python
pack_molecule(mol: Molecule, fields: frozenset[NativeMoleculeField]) -> NativeMoleculeBuffers
```


<a id="hotpot-cheminfo-operations-search"></a>
## `hotpot.cheminfo.operations.search`

`hotpot/cheminfo/operations/search/__init__.py`；显式 __all__: ('Query', 'QueryAtom', 'QueryBond', 'Substructure', 'Searcher', 'Hits', 'Hit', 'SmartsSyntaxError', 'UnsupportedSmartsError', 'SmartsSemantics')

- 导出 `Query` → `hotpot.cheminfo.operations.search.search.Query`
- 导出 `QueryAtom` → `hotpot.cheminfo.operations.search.search.QueryAtom`
- 导出 `QueryBond` → `hotpot.cheminfo.operations.search.search.QueryBond`
- 导出 `Substructure` → `hotpot.cheminfo.operations.search.search.Substructure`
- 导出 `Searcher` → `hotpot.cheminfo.operations.search.search.Searcher`
- 导出 `Hits` → `hotpot.cheminfo.operations.search.search.Hits`
- 导出 `Hit` → `hotpot.cheminfo.operations.search.search.Hit`
- 导出 `SmartsSyntaxError` → `hotpot.cheminfo.rules.chemistry.substructure.errors.SmartsSyntaxError`
- 导出 `UnsupportedSmartsError` → `hotpot.cheminfo.rules.chemistry.substructure.errors.UnsupportedSmartsError`
- 导出 `SmartsSemantics` → `hotpot.cheminfo.structure.substructure.SmartsSemantics`


<a id="hotpot-cheminfo-operations-search-native"></a>
## `hotpot.cheminfo.operations.search.native`

`hotpot/cheminfo/operations/search/native.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-search-predicates"></a>
## `hotpot.cheminfo.operations.search.predicates`

`hotpot/cheminfo/operations/search/predicates.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-search-predicates-match-atom-expression"></a>
### `match_atom_expression`

```python
match_atom_expression(expression, atom)
```

<a id="hotpot-cheminfo-operations-search-predicates-match-bond-expression"></a>
### `match_bond_expression`

```python
match_bond_expression(expression, bond)
```


<a id="hotpot-cheminfo-operations-search-search"></a>
## `hotpot.cheminfo.operations.search.search`

`hotpot/cheminfo/operations/search/search.py`；显式 __all__: ('Query', 'QueryAtom', 'QueryBond', 'Substructure', 'Searcher', 'Hits', 'Hit')


<a id="hotpot-cheminfo-operations-search-search-query"></a>
### `Query`

定义 owner `hotpot.cheminfo.operations.search.search.Query`；基类 无。

- `__init__(self, **kwargs: Union[Container, Any])`
- `__repr__(self)`
- `match(self, obj)`
- `label` [只读]
- `kwargs` = `{n: v if isinstance(v, Callable) else set(v) for n, v in kwargs.items()}` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-search-search-queryatom"></a>
### `QueryAtom`

定义 owner `hotpot.cheminfo.operations.search.search.QueryAtom`；基类 `Query`。

- `__init__(self, sub: 'Substructure'=None, map_number: int=None, **attrs)`
- `from_atom(cls, atom: Atom, include_attrs: Sequence[str]=None, exclude_attrs: Sequence[Literal['atomic_number', 'is_aromatic']]=None)` [classmethod]
- `map_number` [只读]
- `label` [只读]
- `idx` [只读]: `int`
- `sub` = `sub` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `__repr__(self)` ← `hotpot.cheminfo.operations.search.search.Query`
- `match(self, obj)` ← `hotpot.cheminfo.operations.search.search.Query`
- `kwargs` = `{n: v if isinstance(v, Callable) else set(v) for n, v in kwargs.items()}` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.search.search.Query`

<a id="hotpot-cheminfo-operations-search-search-querybond"></a>
### `QueryBond`

定义 owner `hotpot.cheminfo.operations.search.search.QueryBond`；基类 `Query`。

- `__init__(self, atom1: QueryAtom, atom2: QueryAtom, **attrs)`
- `match(self, obj)`
- `label` [只读]
- `a1idx` [只读]: `int`
- `a2idx` [只读]: `int`
- `sub` [只读]
- `atom1` = `atom1` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `atom2` = `atom2` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `__repr__(self)` ← `hotpot.cheminfo.operations.search.search.Query`
- `kwargs` = `{n: v if isinstance(v, Callable) else set(v) for n, v in kwargs.items()}` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.operations.search.search.Query`

<a id="hotpot-cheminfo-operations-search-search-substructure"></a>
### `Substructure`

定义 owner `hotpot.cheminfo.operations.search.search.Substructure`；基类 无。

- `__init__(self)`
- `__repr__(self)`
- `from_mol(cls, mol: Molecule, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None) -> 'Substructure'` [classmethod]
- `from_smarts(cls, smarts: str, *, semantics: SmartsSemantics=SmartsSemantics.FULL_GRAPH) -> 'Substructure'` [classmethod]
- `from_smiles(cls, smiles: str, addition_atom_attr: dict[int, dict[str, set]]=None, addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]]=None)` [classmethod]
- `add_atom(self, atom_query: Union[Atom, QueryAtom])`
- `add_bond(self, atom1: Union[int, QueryAtom], atom2: Union[int, QueryAtom], **bond_attrs)`
- `query_atoms` = `[]` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `query_bonds` = `[]` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-search-search-searcher"></a>
### `Searcher`

定义 owner `hotpot.cheminfo.operations.search.search.Searcher`；基类 无。

- `__init__(self, substructure: 'Substructure')`
- `has_match(self, mol: Molecule) -> bool`
- `iter_mappings(self, mol: Molecule, max_matches: int=None) -> Iterator[MappingProxyType]`
- `search(self, mol: Molecule, max_matches: int=None) -> 'Hits'`
- `substructure` = `substructure` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-search-search-hits"></a>
### `Hits`

定义 owner `hotpot.cheminfo.operations.search.search.Hits`；基类 无。

- `__init__(self, sub, mol, mapping_source, get_hit: bool=True, max_matches: int=None)`
- `__iter__(self)`
- `__getitem__(self, item: int) -> Union[frozenset[int], 'Hit']`
- `__len__(self)`
- `__bool__(self)`
- `__contains__(self, item: Union[Iterable[int], 'Hit'])`
- `hits` [只读]: `list['Hit']`
- `sub` = `sub` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `mol` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `mapping_source` = `mapping_source` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `get_hit` = `get_hit` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `max_matches` = `max_matches` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `truncated` = `mappings.truncated` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-operations-search-search-hit"></a>
### `Hit`

定义 owner `hotpot.cheminfo.operations.search.search.Hit`；基类 无。

- `__init__(self, mol, sub, atom_indices, mappings)`
- `mapped_bonds(self, mapping_index: int=0)`
- `mapped_atom_indices(self, query_idx: int) -> tuple[int, ...]`
- `mapped_atoms(self, query_idx: int) -> tuple[Atom, ...]`
- `induced_bonds` [只读]
- `mappings` [只读]
- `mol` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `sub` = `sub` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `atom_indices` = `frozenset(atom_indices)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `atoms` = `[self.mol.atoms[i] for i in sorted(self.atom_indices)]` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `bonds` = `self.mapped_bonds()` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-operations-search-semantics"></a>
## `hotpot.cheminfo.operations.search.semantics`

`hotpot/cheminfo/operations/search/semantics.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-search-semantics-rings-for"></a>
### `rings_for`

```python
rings_for(semantics, atom)
```

<a id="hotpot-cheminfo-operations-search-semantics-bonds-for"></a>
### `bonds_for`

```python
bonds_for(semantics, atom)
```

<a id="hotpot-cheminfo-operations-search-semantics-degree"></a>
### `degree`

```python
degree(semantics, atom)
```

<a id="hotpot-cheminfo-operations-search-semantics-connectivity"></a>
### `connectivity`

```python
connectivity(semantics, atom)
```

<a id="hotpot-cheminfo-operations-search-semantics-valence"></a>
### `valence`

```python
valence(semantics, atom)
```


<a id="hotpot-cheminfo-operations-search-smarts"></a>
## `hotpot.cheminfo.operations.search.smarts`

`hotpot/cheminfo/operations/search/smarts.py`；显式 __all__: 无


<a id="hotpot-cheminfo-operations-search-smarts-parse-bracket-atom"></a>
### `parse_bracket_atom`

```python
parse_bracket_atom(expr_text: str, *, semantics: SmartsSemantics=SmartsSemantics.FULL_GRAPH) -> Dict[str, object]
```

<a id="hotpot-cheminfo-operations-search-smarts-substructure-from-smarts"></a>
### `substructure_from_smarts`

```python
substructure_from_smarts(smarts: str, *, semantics: SmartsSemantics=SmartsSemantics.FULL_GRAPH) -> Substructure
```


<a id="hotpot-cheminfo-operations-solvent-properties"></a>
## `hotpot.cheminfo.operations.solvent_properties`

`hotpot/cheminfo/operations/solvent_properties.py`；显式 __all__: ('get_solvent_dataframe', 'get_medium_dataframe', 'query_solvent', 'query_medium')


<a id="hotpot-cheminfo-operations-solvent-properties-get-medium-dataframe"></a>
### `get_medium_dataframe`

```python
get_medium_dataframe(sub_sheet: Literal['clean', 'fill']='clean')
```

<a id="hotpot-cheminfo-operations-solvent-properties-get-solvent-dataframe"></a>
### `get_solvent_dataframe`

```python
get_solvent_dataframe(sub_sheet: Literal['clean', 'total']='clean')
```

<a id="hotpot-cheminfo-operations-solvent-properties-query-medium"></a>
### `query_medium`

```python
query_medium(list_id: list[Union[str, int]], id_type: Literal['CASs', 'Cid']='CASs', sub_sheet: Literal['clean', 'fill']='clean') -> pd.DataFrame
```

<a id="hotpot-cheminfo-operations-solvent-properties-query-solvent"></a>
### `query_solvent`

```python
query_solvent(list_id: list[Union[str, int]], id_type: Literal['CASs', 'Cid']='CASs', sub_sheet: Literal['clean', 'total']='clean') -> pd.DataFrame
```


<a id="hotpot-cheminfo-operations-stereochemistry"></a>
## `hotpot.cheminfo.operations.stereochemistry`

`hotpot/cheminfo/operations/stereochemistry.py`；显式 __all__: ()


<a id="hotpot-cheminfo-operations-stereochemistry-validate-stereochemistry"></a>
### `validate_stereochemistry`

```python
validate_stereochemistry(mol, *, coordinates: NDArray[np.float64] | None=None) -> StereoValidationReport
```

<a id="hotpot-cheminfo-operations-stereochemistry-assign-stereochemistry"></a>
### `assign_stereochemistry`

```python
assign_stereochemistry(mol, *, coordinates: NDArray[np.float64] | None=None) -> StereoAssignmentReport
```

<a id="hotpot-cheminfo-operations-stereochemistry-align-stereochemistry"></a>
### `align_stereochemistry`

```python
align_stereochemistry(reference, target, atom_mapping: Mapping[int, int]) -> StereoAlignmentReport
```


<a id="hotpot-cheminfo-operations-thermo"></a>
## `hotpot.cheminfo.operations.thermo`

`hotpot/cheminfo/operations/thermo.py`；显式 __all__: ('Thermo',)


<a id="hotpot-cheminfo-operations-thermo-thermo"></a>
### `Thermo`

定义 owner `hotpot.cheminfo.operations.thermo.Thermo`；基类 无。

- `__dir__(self)`
- `__getattr__(self, item)`
- `__init__(self, mol, temp: float=298.15, pressure: float=101325)`
- `get_chem(self, ids: Union[str, Iterable[str]]) -> 'pd.DataFrame'`
- `mol_to_thermo(self, mol)`
- `properties` = <长声明值；见 `hotpot/cheminfo/operations/thermo.py:27`> [类声明值；可重绑定不构成事实写许可]
- `mol` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `temp` = `temp` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `pressure` = `pressure` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-operations-topology"></a>
## `hotpot.cheminfo.operations.topology`

`hotpot/cheminfo/operations/topology.py`；显式 __all__: ()



<a id="hotpot-cheminfo-operations-topology-queries"></a>
## `hotpot.cheminfo.operations.topology_queries`

`hotpot/cheminfo/operations/topology_queries.py`；显式 __all__: ()


<a id="hotpot-cheminfo-operations-topology-queries-atom-idx-pair-to-bond-idx"></a>
### `atom_idx_pair_to_bond_idx`

```python
atom_idx_pair_to_bond_idx(mol: 'Molecule', *atom_idx_pair: tuple[int, int]) -> list[int]
```


<a id="hotpot-cheminfo-operations-visualization"></a>
## `hotpot.cheminfo.operations.visualization`

`hotpot/cheminfo/operations/visualization.py`；显式 __all__: ('draw_single_mol', 'draw_grid')


<a id="hotpot-cheminfo-operations-visualization-draw-single-mol"></a>
### `draw_single_mol`

```python
draw_single_mol(mol_in: Union[str, 'Molecule'], save_path: Optional[str]=None, mol_size: tuple[int, int]=(600, 600), output_format: Optional[DrawOutputFormat]=None, font_size: int=24, font: str='Arial', fontweight: str='bold', colorful_atom: bool=False, atom_color_palette: Optional[dict]=None, legend: str='', sanitize: bool=False, atom_colors=None, bond_colors=None, atom_hl_values: np.ndarray=None, cmap_name: str='coolwarm', vmin: Optional[float]=None, vmax: Optional[float]=None, threshold: float=0.75, colorbar: bool=False, cb_orientation: Literal['vertical', 'horizontal']='vertical')
```

<a id="hotpot-cheminfo-operations-visualization-draw-grid"></a>
### `draw_grid`

```python
draw_grid(list_mols: Iterable[Union[str, 'Molecule']], save_path: Optional[str]=None, mol_size: tuple[int, int]=(300, 300), output_format: Optional[DrawOutputFormat]=None, font_size: int=20, font: str='Arial', fontweight: str='bold', colorful_atom: bool=True, atom_color_palette: Optional[dict]=None, sanitize: bool=False, n_cols: Optional[int]=None, legends: Optional[list[str]]=None, atom_colors: Optional[list[dict]]=None, bond_colors: Optional[list[dict]]=None, list_atom_values: np.ndarray=None, cmap_name: str='coolwarm', vmin: Optional[float]=None, vmax: Optional[float]=None, threshold: float=0.75, colorbar: bool=False, cb_orientation: Literal['vertical', 'horizontal']='horizontal')
```

模块值/类型别名（声明存在不构成事实写许可）：

- `DrawOutputFormat` = `Literal['svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp']`


<a id="hotpot-cheminfo-pubchem"></a>
## `hotpot.cheminfo.pubchem`

`hotpot/cheminfo/pubchem.py`；显式 __all__: ('get_compound', 'get_compounds_cid', 'smi_to_cid', 'smi_to_cas', 'smi_to_name', 'cid_to_smi', 'name_to_smi', 'cid_to_cas', 'pubchem_service')


<a id="hotpot-cheminfo-pubchem-get-compound"></a>
### `get_compound`

```python
get_compound(*args, **kwargs)
```

<a id="hotpot-cheminfo-pubchem-get-compounds-cid"></a>
### `get_compounds_cid`

```python
get_compounds_cid(identifier: str, id_type: str='name')
```

<a id="hotpot-cheminfo-pubchem-smi-to-cas"></a>
### `smi_to_cas`

```python
smi_to_cas(smiles: str) -> Optional[str]
```

<a id="hotpot-cheminfo-pubchem-cid-to-cas"></a>
### `cid_to_cas`

```python
cid_to_cas(cid: int) -> Optional[str]
```

<a id="hotpot-cheminfo-pubchem-smi-to-name"></a>
### `smi_to_name`

```python
smi_to_name(smiles: str) -> Optional[str]
```

<a id="hotpot-cheminfo-pubchem-smi-to-cid"></a>
### `smi_to_cid`

```python
smi_to_cid(smiles: str)
```

<a id="hotpot-cheminfo-pubchem-cid-to-smi"></a>
### `cid_to_smi`

```python
cid_to_smi(cid)
```

<a id="hotpot-cheminfo-pubchem-name-to-smi"></a>
### `name_to_smi`

```python
name_to_smi(name: str)
```

<a id="hotpot-cheminfo-pubchem-pubchemservice"></a>
### `PubChemService`

定义 owner `hotpot.cheminfo.pubchem.PubChemService`；基类 无。

- `__init__(self, verbose: bool=False)`
- `get_compound(identifier: str, id_type: str='name') -> Optional[pcp.Compound]` [staticmethod]
- `name_to_smi(self, name: str) -> Optional[str]`
- `smi_to_cas(self, smiles: str) -> Optional[str]`
- `smi_to_name(self, smiles: str) -> Optional[str]`
- `smi_to_cid(self, smiles: str) -> Optional[int]`
- `cid_to_smi(self, cid: int) -> Optional[str]`
- `cid_to_cas(self, cid: int) -> Optional[str]`
- `cid_to_name(self, cid: int) -> Optional[str]`
- `name_to_cid(self, name: str) -> Optional[int]`
- `convert(self, identifier: str, from_type: str, to_type: str) -> Optional[str]`
- `cas_regex` = `re.compile('^\\d{2,7}-\\d{2}-\\d$')` [类声明值；可重绑定不构成事实写许可]
- `verbose` = `verbose` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

模块值/类型别名（声明存在不构成事实写许可）：

- `cas_regex` = `re.compile('^\\d{2,7}-\\d{2}-\\d')`
- `pubchem_service` = `PubChemService()`


<a id="hotpot-cheminfo-rdworks"></a>
## `hotpot.cheminfo.rdworks`

`hotpot/cheminfo/rdworks.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rdworks-load-fp"></a>
### `load_fp`

```python
load_fp(list_mols: list[Chem.Mol], generator: Literal['Morgen', 'RdKit', 'AtomPair', 'Torison']='Morgen', gen_kw: dict=None, sparse: bool=True, counts: bool=True)
```

<a id="hotpot-cheminfo-rdworks-merge-fps"></a>
### `merge_fps`

```python
merge_fps(list_fp: list[cDS.ULongSparseIntVect])
```

<a id="hotpot-cheminfo-rdworks-fps-to-array"></a>
### `fps_to_array`

```python
fps_to_array(fps: list, to_bit: bool=False)
```

<a id="hotpot-cheminfo-rdworks-assign-fp-values-atoms"></a>
### `assign_fp_values_atoms`

```python
assign_fp_values_atoms(mol: Chem.Mol, scores_values: np.ndarray, list_fp: list[int], fp_map: dict[int, list[tuple[int, int]]], mode: str='sum', scale_to_unit: bool=True)
```
