# 结构事实、集合、帧与 stereo：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot`](#hotpot)
- [`hotpot.__version__`](#hotpot---version--)
- [`hotpot.cheminfo`](#hotpot-cheminfo)
- [`hotpot.cheminfo.stereo`](#hotpot-cheminfo-stereo)
- [`hotpot.cheminfo.structure`](#hotpot-cheminfo-structure)
- [`hotpot.cheminfo.structure._atom_table`](#hotpot-cheminfo-structure--atom-table)
- [`hotpot.cheminfo.structure._builder`](#hotpot-cheminfo-structure--builder)
- [`hotpot.cheminfo.structure._cif_facts`](#hotpot-cheminfo-structure--cif-facts)
- [`hotpot.cheminfo.structure._commit`](#hotpot-cheminfo-structure--commit)
- [`hotpot.cheminfo.structure._crystal`](#hotpot-cheminfo-structure--crystal)
- [`hotpot.cheminfo.structure._crystal_codec`](#hotpot-cheminfo-structure--crystal-codec)
- [`hotpot.cheminfo.structure._declarations`](#hotpot-cheminfo-structure--declarations)
- [`hotpot.cheminfo.structure._dependencies`](#hotpot-cheminfo-structure--dependencies)
- [`hotpot.cheminfo.structure._fields`](#hotpot-cheminfo-structure--fields)
- [`hotpot.cheminfo.structure._identity`](#hotpot-cheminfo-structure--identity)
- [`hotpot.cheminfo.structure._mutation`](#hotpot-cheminfo-structure--mutation)
- [`hotpot.cheminfo.structure._results`](#hotpot-cheminfo-structure--results)
- [`hotpot.cheminfo.structure._selection`](#hotpot-cheminfo-structure--selection)
- [`hotpot.cheminfo.structure._source_identity`](#hotpot-cheminfo-structure--source-identity)
- [`hotpot.cheminfo.structure._state`](#hotpot-cheminfo-structure--state)
- [`hotpot.cheminfo.structure._stereo`](#hotpot-cheminfo-structure--stereo)
- [`hotpot.cheminfo.structure._stereo_codec`](#hotpot-cheminfo-structure--stereo-codec)
- [`hotpot.cheminfo.structure._topology`](#hotpot-cheminfo-structure--topology)
- [`hotpot.cheminfo.structure.atom`](#hotpot-cheminfo-structure-atom)
- [`hotpot.cheminfo.structure.bond`](#hotpot-cheminfo-structure-bond)
- [`hotpot.cheminfo.structure.bond_kind`](#hotpot-cheminfo-structure-bond-kind)
- [`hotpot.cheminfo.structure.bundle`](#hotpot-cheminfo-structure-bundle)
- [`hotpot.cheminfo.structure.conformer`](#hotpot-cheminfo-structure-conformer)
- [`hotpot.cheminfo.structure.constraints`](#hotpot-cheminfo-structure-constraints)
- [`hotpot.cheminfo.structure.crystal`](#hotpot-cheminfo-structure-crystal)
- [`hotpot.cheminfo.structure.errors`](#hotpot-cheminfo-structure-errors)
- [`hotpot.cheminfo.structure.forcefields`](#hotpot-cheminfo-structure-forcefields)
- [`hotpot.cheminfo.structure.molecule`](#hotpot-cheminfo-structure-molecule)
- [`hotpot.cheminfo.structure.snapshots`](#hotpot-cheminfo-structure-snapshots)
- [`hotpot.cheminfo.structure.stereo`](#hotpot-cheminfo-structure-stereo)
- [`hotpot.cheminfo.structure.substructure`](#hotpot-cheminfo-structure-substructure)
- [`hotpot.cheminfo.structure.symmetry`](#hotpot-cheminfo-structure-symmetry)

<a id="hotpot"></a>
## `hotpot`

`hotpot/__init__.py`；显式 __all__: ('Molecule', 'Atom', 'Bond', 'BondKind', 'Crystal', 'SpaceGroup', 'SymmUnit', 'MolReader', 'MolWriter', 'read_mol', 'MolBundle', 'to_pyg_dataset', 'is_molecule_input', 'to_hotpot_mol', 'Fragment', 'EdgeShoulder', 'AtomLink', 'AlkylGraft', 'BondAdding', 'AtomReplace', 'RingWedge', 'AssembleFactory', 'Searcher', 'Substructure', 'Hit', 'Hits', 'Query', 'QueryAtom', 'QueryBond', 'SmartsSemantics', 'SmartsSyntaxError', 'UnsupportedSmartsError', 'version')

- 导出 `Molecule` → `hotpot.cheminfo.structure.molecule.Molecule`
- 导出 `Atom` → `hotpot.cheminfo.structure.atom.Atom`
- 导出 `Bond` → `hotpot.cheminfo.structure.bond.Bond`
- 导出 `BondKind` → `hotpot.cheminfo.structure.bond_kind.BondKind`
- 导出 `Crystal` → `hotpot.cheminfo.structure.crystal.Crystal`
- 导出 `SpaceGroup` → `hotpot.cheminfo.structure.symmetry.SpaceGroup`
- 导出 `SymmUnit` → `hotpot.cheminfo.structure.symmetry.SymmUnit`
- 导出 `MolReader` → `hotpot.cheminfo.operations.io.MolReader`
- 导出 `MolWriter` → `hotpot.cheminfo.operations.io.MolWriter`
- 导出 `read_mol` → `hotpot.cheminfo.operations.io.read_mol`；`read_mol(src, fmt=None, **kwargs)`
- 导出 `MolBundle` → `hotpot.cheminfo.structure.bundle.MolBundle`
- 导出 `to_pyg_dataset` → `hotpot.cheminfo.operations.collections.to_pyg_dataset`；`to_pyg_dataset(list_mol: tp.Iterable[Molecule], dataset_root: os.PathLike)`
- 导出 `is_molecule_input` → `hotpot.cheminfo.operations.conversion.is_molecule_input`；`is_molecule_input(value) -> 'bool'`
- 导出 `to_hotpot_mol` → `hotpot.cheminfo.operations.conversion.to_hotpot_mol`；`to_hotpot_mol(value, *, fmt=None, **kwargs) -> 'Molecule'`
- 导出 `Fragment` → `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- 导出 `EdgeShoulder` → `hotpot.cheminfo.operations.assembly.assembler.EdgeShoulder`
- 导出 `AtomLink` → `hotpot.cheminfo.operations.assembly.assembler.AtomLink`
- 导出 `AlkylGraft` → `hotpot.cheminfo.operations.assembly.assembler.AlkylGraft`
- 导出 `BondAdding` → `hotpot.cheminfo.operations.assembly.assembler.BondAdding`
- 导出 `AtomReplace` → `hotpot.cheminfo.operations.assembly.assembler.AtomReplace`
- 导出 `RingWedge` → `hotpot.cheminfo.operations.assembly.assembler.RingWedge`
- 导出 `AssembleFactory` → `hotpot.cheminfo.operations.assembly.assembler.AssembleFactory`
- 导出 `Searcher` → `hotpot.cheminfo.operations.search.search.Searcher`
- 导出 `Substructure` → `hotpot.cheminfo.operations.search.search.Substructure`
- 导出 `Hit` → `hotpot.cheminfo.operations.search.search.Hit`
- 导出 `Hits` → `hotpot.cheminfo.operations.search.search.Hits`
- 导出 `Query` → `hotpot.cheminfo.operations.search.search.Query`
- 导出 `QueryAtom` → `hotpot.cheminfo.operations.search.search.QueryAtom`
- 导出 `QueryBond` → `hotpot.cheminfo.operations.search.search.QueryBond`
- 导出 `SmartsSemantics` → `hotpot.cheminfo.structure.substructure.SmartsSemantics`
- 导出 `SmartsSyntaxError` → `hotpot.cheminfo.rules.chemistry.substructure.errors.SmartsSyntaxError`
- 导出 `UnsupportedSmartsError` → `hotpot.cheminfo.rules.chemistry.substructure.errors.UnsupportedSmartsError`

<a id="hotpot-version"></a>
### `version`

```python
version()
```


<a id="hotpot---version--"></a>
## `hotpot.__version__`

`hotpot/__version__.py`；显式 __all__: 无



<a id="hotpot-cheminfo"></a>
## `hotpot.cheminfo`

`hotpot/cheminfo/__init__.py`；显式 __all__: ('Molecule', 'Atom', 'Bond', 'BondKind', 'Crystal', 'SpaceGroup', 'SymmUnit', 'MolReader', 'MolWriter', 'read_mol', 'MolBundle', 'to_pyg_dataset', 'is_molecule_input', 'to_hotpot_mol', 'Fragment', 'EdgeShoulder', 'AtomLink', 'AlkylGraft', 'BondAdding', 'AtomReplace', 'RingWedge', 'AssembleFactory', 'Searcher', 'Substructure', 'Hit', 'Hits', 'Query', 'QueryAtom', 'QueryBond', 'SmartsSemantics', 'SmartsSyntaxError', 'UnsupportedSmartsError')

- 导出 `Molecule` → `hotpot.cheminfo.structure.molecule.Molecule`
- 导出 `Atom` → `hotpot.cheminfo.structure.atom.Atom`
- 导出 `Bond` → `hotpot.cheminfo.structure.bond.Bond`
- 导出 `BondKind` → `hotpot.cheminfo.structure.bond_kind.BondKind`
- 导出 `Crystal` → `hotpot.cheminfo.structure.crystal.Crystal`
- 导出 `SpaceGroup` → `hotpot.cheminfo.structure.symmetry.SpaceGroup`
- 导出 `SymmUnit` → `hotpot.cheminfo.structure.symmetry.SymmUnit`
- 导出 `MolReader` → `hotpot.cheminfo.operations.io.MolReader`
- 导出 `MolWriter` → `hotpot.cheminfo.operations.io.MolWriter`
- 导出 `read_mol` → `hotpot.cheminfo.operations.io.read_mol`；`read_mol(src, fmt=None, **kwargs)`
- 导出 `MolBundle` → `hotpot.cheminfo.structure.bundle.MolBundle`
- 导出 `to_pyg_dataset` → `hotpot.cheminfo.operations.collections.to_pyg_dataset`；`to_pyg_dataset(list_mol: tp.Iterable[Molecule], dataset_root: os.PathLike)`
- 导出 `is_molecule_input` → `hotpot.cheminfo.operations.conversion.is_molecule_input`；`is_molecule_input(value) -> 'bool'`
- 导出 `to_hotpot_mol` → `hotpot.cheminfo.operations.conversion.to_hotpot_mol`；`to_hotpot_mol(value, *, fmt=None, **kwargs) -> 'Molecule'`
- 导出 `Fragment` → `hotpot.cheminfo.operations.assembly.fragment.Fragment`
- 导出 `EdgeShoulder` → `hotpot.cheminfo.operations.assembly.assembler.EdgeShoulder`
- 导出 `AtomLink` → `hotpot.cheminfo.operations.assembly.assembler.AtomLink`
- 导出 `AlkylGraft` → `hotpot.cheminfo.operations.assembly.assembler.AlkylGraft`
- 导出 `BondAdding` → `hotpot.cheminfo.operations.assembly.assembler.BondAdding`
- 导出 `AtomReplace` → `hotpot.cheminfo.operations.assembly.assembler.AtomReplace`
- 导出 `RingWedge` → `hotpot.cheminfo.operations.assembly.assembler.RingWedge`
- 导出 `AssembleFactory` → `hotpot.cheminfo.operations.assembly.assembler.AssembleFactory`
- 导出 `Searcher` → `hotpot.cheminfo.operations.search.search.Searcher`
- 导出 `Substructure` → `hotpot.cheminfo.operations.search.search.Substructure`
- 导出 `Hit` → `hotpot.cheminfo.operations.search.search.Hit`
- 导出 `Hits` → `hotpot.cheminfo.operations.search.search.Hits`
- 导出 `Query` → `hotpot.cheminfo.operations.search.search.Query`
- 导出 `QueryAtom` → `hotpot.cheminfo.operations.search.search.QueryAtom`
- 导出 `QueryBond` → `hotpot.cheminfo.operations.search.search.QueryBond`
- 导出 `SmartsSemantics` → `hotpot.cheminfo.structure.substructure.SmartsSemantics`
- 导出 `SmartsSyntaxError` → `hotpot.cheminfo.rules.chemistry.substructure.errors.SmartsSyntaxError`
- 导出 `UnsupportedSmartsError` → `hotpot.cheminfo.rules.chemistry.substructure.errors.UnsupportedSmartsError`


<a id="hotpot-cheminfo-stereo"></a>
## `hotpot.cheminfo.stereo`

`hotpot/cheminfo/stereo.py`；显式 __all__: ('AtomTransportRecord', 'BondTransportRecord', 'HydrogenTransportState', 'StereoAlignmentReport', 'StereoAnchor', 'StereoAnchorKind', 'StereoAssignmentReport', 'StereoCarrier', 'StereoCarrierKind', 'StereoDerivedDescriptor', 'StereoGroupKind', 'StereoGroupRef', 'StereoIssueCode', 'StereoPayload', 'StereoRecord', 'StereoType', 'StereoValidationIssue', 'StereoValidationReport', 'align_stereochemistry', 'assign_stereochemistry', 'get_stereo_payload', 'set_stereo_payload', 'validate_stereochemistry')

- 导出 `AtomTransportRecord` → `hotpot.cheminfo.structure.stereo.AtomTransportRecord`
- 导出 `BondTransportRecord` → `hotpot.cheminfo.structure.stereo.BondTransportRecord`
- 导出 `HydrogenTransportState` → `hotpot.cheminfo.structure.stereo.HydrogenTransportState`
- 导出 `StereoAlignmentReport` → `hotpot.cheminfo.structure.stereo.StereoAlignmentReport`
- 导出 `StereoAnchor` → `hotpot.cheminfo.structure.stereo.StereoAnchor`
- 导出 `StereoAnchorKind` → `hotpot.cheminfo.structure.stereo.StereoAnchorKind`
- 导出 `StereoAssignmentReport` → `hotpot.cheminfo.structure.stereo.StereoAssignmentReport`
- 导出 `StereoCarrier` → `hotpot.cheminfo.structure.stereo.StereoCarrier`
- 导出 `StereoCarrierKind` → `hotpot.cheminfo.structure.stereo.StereoCarrierKind`
- 导出 `StereoDerivedDescriptor` → `hotpot.cheminfo.structure.stereo.StereoDerivedDescriptor`
- 导出 `StereoGroupKind` → `hotpot.cheminfo.structure.stereo.StereoGroupKind`
- 导出 `StereoGroupRef` → `hotpot.cheminfo.structure.stereo.StereoGroupRef`
- 导出 `StereoIssueCode` → `hotpot.cheminfo.structure.stereo.StereoIssueCode`
- 导出 `StereoPayload` → `hotpot.cheminfo.structure.stereo.StereoPayload`
- 导出 `StereoRecord` → `hotpot.cheminfo.structure.stereo.StereoRecord`
- 导出 `StereoType` → `hotpot.cheminfo.structure.stereo.StereoType`
- 导出 `StereoValidationIssue` → `hotpot.cheminfo.structure.stereo.StereoValidationIssue`
- 导出 `StereoValidationReport` → `hotpot.cheminfo.structure.stereo.StereoValidationReport`

<a id="hotpot-cheminfo-stereo-get-stereo-payload"></a>
### `get_stereo_payload`

```python
get_stereo_payload(mol: Molecule) -> StereoPayload
```

<a id="hotpot-cheminfo-stereo-set-stereo-payload"></a>
### `set_stereo_payload`

```python
set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None
```

<a id="hotpot-cheminfo-stereo-validate-stereochemistry"></a>
### `validate_stereochemistry`

```python
validate_stereochemistry(mol: Molecule, *, coordinates: NDArray[np.float64] | None=None) -> StereoValidationReport
```

<a id="hotpot-cheminfo-stereo-assign-stereochemistry"></a>
### `assign_stereochemistry`

```python
assign_stereochemistry(mol: Molecule, *, coordinates: NDArray[np.float64] | None=None) -> StereoAssignmentReport
```

<a id="hotpot-cheminfo-stereo-align-stereochemistry"></a>
### `align_stereochemistry`

```python
align_stereochemistry(reference: Molecule, target: Molecule, atom_mapping: Mapping[int, int]) -> StereoAlignmentReport
```


<a id="hotpot-cheminfo-structure"></a>
## `hotpot.cheminfo.structure`

`hotpot/cheminfo/structure/__init__.py`；显式 __all__: ('Molecule', 'Atom', 'Bond', 'BondKind', 'Crystal', 'SpaceGroup', 'SymmUnit')

- 导出 `Molecule` → `hotpot.cheminfo.structure.molecule.Molecule`
- 导出 `Atom` → `hotpot.cheminfo.structure.atom.Atom`
- 导出 `Bond` → `hotpot.cheminfo.structure.bond.Bond`
- 导出 `BondKind` → `hotpot.cheminfo.structure.bond_kind.BondKind`
- 导出 `Crystal` → `hotpot.cheminfo.structure.crystal.Crystal`
- 导出 `SpaceGroup` → `hotpot.cheminfo.structure.symmetry.SpaceGroup`
- 导出 `SymmUnit` → `hotpot.cheminfo.structure.symmetry.SymmUnit`


<a id="hotpot-cheminfo-structure--atom-table"></a>
## `hotpot.cheminfo.structure._atom_table`

`hotpot/cheminfo/structure/_atom_table.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--builder"></a>
## `hotpot.cheminfo.structure._builder`

`hotpot/cheminfo/structure/_builder.py`；显式 __all__: ()


<a id="hotpot-cheminfo-structure--builder-moleculerecord"></a>
### `MoleculeRecord`

定义 owner `hotpot.cheminfo.structure._builder.MoleculeRecord`；基类 无。

```python
MoleculeRecord(atom_count: int, atom_values: bytes, atom_presence: bytes, atom_sources: bytes, atom_frac_valences: tuple, bond_endpoints: tuple, bond_values: bytes, bond_kinds: tuple, bond_directions: tuple, bond_sources: tuple, bond_metadata: tuple, charge: tuple, spin_mult: tuple, cell: tuple | None, stereo_payload: StereoPayload | None, source_frames: tuple, conformer_results: tuple, frame_selection: tuple | None, angle_constraints: tuple, torsion_constraints: tuple, identifier: tuple, pair_wiberg_bond_orders: tuple = (), bond_aromatic: tuple[bool, ...] = (), partial_charge_result: tuple[PartialChargeSet, tuple[str, ...]] | None = None)
```

- `__reduce__(self)`
- `__deepcopy__(self, memo)`
- `atom_count: int` [只读值字段]
- `atom_values: bytes` [只读值字段]
- `atom_presence: bytes` [只读值字段]
- `atom_sources: bytes` [只读值字段]
- `atom_frac_valences: tuple` [只读值字段]
- `bond_endpoints: tuple` [只读值字段]
- `bond_values: bytes` [只读值字段]
- `bond_kinds: tuple` [只读值字段]
- `bond_directions: tuple` [只读值字段]
- `bond_sources: tuple` [只读值字段]
- `bond_metadata: tuple` [只读值字段]
- `charge: tuple` [只读值字段]
- `spin_mult: tuple` [只读值字段]
- `cell: tuple | None` [只读值字段]
- `stereo_payload: StereoPayload | None` [只读值字段]
- `source_frames: tuple` [只读值字段]
- `conformer_results: tuple` [只读值字段]
- `frame_selection: tuple | None` [只读值字段]
- `angle_constraints: tuple` [只读值字段]
- `torsion_constraints: tuple` [只读值字段]
- `identifier: tuple` [只读值字段]
- `pair_wiberg_bond_orders: tuple` = `()` [只读值字段]
- `bond_aromatic: tuple[bool, ...]` = `()` [只读值字段]
- `partial_charge_result: tuple[PartialChargeSet, tuple[str, ...]] | None` = `None` [只读值字段]


<a id="hotpot-cheminfo-structure--cif-facts"></a>
## `hotpot.cheminfo.structure._cif_facts`

`hotpot/cheminfo/structure/_cif_facts.py`；显式 __all__: 无



<a id="hotpot-cheminfo-structure--commit"></a>
## `hotpot.cheminfo.structure._commit`

`hotpot/cheminfo/structure/_commit.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--crystal"></a>
## `hotpot.cheminfo.structure._crystal`

`hotpot/cheminfo/structure/_crystal.py`；显式 __all__: 无



<a id="hotpot-cheminfo-structure--crystal-codec"></a>
## `hotpot.cheminfo.structure._crystal_codec`

`hotpot/cheminfo/structure/_crystal_codec.py`；显式 __all__: 无



<a id="hotpot-cheminfo-structure--declarations"></a>
## `hotpot.cheminfo.structure._declarations`

`hotpot/cheminfo/structure/_declarations.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--dependencies"></a>
## `hotpot.cheminfo.structure._dependencies`

`hotpot/cheminfo/structure/_dependencies.py`；显式 __all__: ()


模块值/类型别名（声明存在不构成事实写许可）：

- `GROUPS` = `('VALENCE', 'IMPLICIT_H', 'AROMATIC_REPRESENTATION', 'FORMAL_CHARGE_COVERAGE')`
- `GROUP_INDEX` = `MappingProxyType({group: index for index, group in enumerate(GROUPS)})`
- `GROUP_FIELDS` = `('valence', 'implicit_hydrogens', 'is_aromatic', 'formal_charge')`
- `GROUP_UPSTREAM` = `MappingProxyType({'VALENCE': (), 'IMPLICIT_H': ('VALENCE',), 'AROMATIC_REPRESENTATION': ('VALENCE', 'IMPLICIT_H'), 'FORMAL_CHARGE_COVERAGE': ()})`
- `GROUP_SCOPE` = `MappingProxyType({'VALENCE': 'neighbourhood', 'IMPLICIT_H': 'neighbourhood', 'AROMATIC_REPRESENTATION': 'component', 'FORMAL_CHARGE_COVERAGE': 'molecule'})`
- `GROUP_READ_POLICY` = `MappingProxyType({'VALENCE': 'pure_local_projection_without_value_storage', 'IMPLICIT_H': 'prepare_source_aware_hydrogens', 'AROMATIC_REPRESENTATION': 'prepare_coupled_aromatic_representation', 'FORMAL_CHARGE_COVERAGE': 'check_explicit_coverage_without_redistribution'})`


<a id="hotpot-cheminfo-structure--fields"></a>
## `hotpot.cheminfo.structure._fields`

`hotpot/cheminfo/structure/_fields.py`；显式 __all__: ()


<a id="hotpot-cheminfo-structure--fields-molblock"></a>
### `MolBlock`

定义 owner `hotpot.cheminfo.structure._fields.MolBlock`；基类 无。

- `__repr__(self)`
- `__copy__(self)`
- `__dir__(self)`
- `__getattr__(self, item)`
- `__setattr__(self, key, value)`
- `attrs_enumerator` [只读]: `tuple`
- `attr_dict` [只读]: `dict`
- `in_ring` [只读]
- `in_organic` [只读]: `bool`
- `rings` [只读]

模块值/类型别名（声明存在不构成事实写许可）：

- `ATOM_SCHEMA` = `<长模块值；见 hotpot/cheminfo/structure/_fields.py:13>`
- `ATOM_FIELDS` = `tuple((name for name, _ in ATOM_SCHEMA))`
- `TABLE_FIELDS` = `ATOM_FIELDS[:16]`
- `CONSTRAINT_FIELDS` = `ATOM_FIELDS[16:]`
- `TABLE_PRESENCE_ALL` = `(1 << len(TABLE_FIELDS)) - 1`
- `FIELD_INDEX` = `MappingProxyType({name: index for index, name in enumerate(ATOM_FIELDS)})`
- `FIELD_CODECS` = `MappingProxyType(dict(ATOM_SCHEMA))`
- `FIELD_PRESENCE_BITS` = `tuple((1 << index for index in range(len(ATOM_FIELDS))))`
- `ATOM_PRESENCE_ALL` = `(1 << len(ATOM_FIELDS)) - 1`
- `ATOM_SOURCE_CODES` = `<长模块值；见 hotpot/cheminfo/structure/_fields.py:42>`
- `COORDINATE_FIELDS` = `('x', 'y', 'z')`
- `DERIVED_FIELDS` = `frozenset({'n', 's', 'p', 'd', 'f', 'g', 'valence', 'partial_charge'})`
- `FACT_FIELDS` = `tuple((name for name in ATOM_FIELDS if name not in DERIVED_FIELDS))`
- `ATOM_SCALAR_READONLY` = `frozenset({*DERIVED_FIELDS, 'is_aromatic', 'electron_configuration'})`
- `SCALAR_FIELDS` = `tuple((name for name in TABLE_FIELDS if name not in DERIVED_FIELDS and name not in COORDINATE_FIELDS))`
- `FIELD_ADDRESSES` = `tuple((None if name in DERIVED_FIELDS else (1, COORDINATE_FIELDS.index(name)) if name in COORDINATE_FIELDS else (0, SCALAR_FIELDS.index(name)) for name in TABLE_FIELDS))`
- `ALIAS_TARGETS` = `MappingProxyType({'symbol': ('atomic_number',), 'coordinates': COORDINATE_FIELDS, 'constraint': ('x_constraint', 'y_constraint', 'z_constraint')})`
- `ATOM_BATCH_WRITABLE` = `frozenset({'atomic_number', 'symbol', 'formal_charge', 'implicit_hydrogens', 'id', 'coordinates', *COORDINATE_FIELDS})`
- `ATOM_BATCH_READONLY` = `frozenset({*ATOM_FIELDS, *ALIAS_TARGETS, 'attrs', 'attrs_enumerator', 'electron_configuration', 'idx', 'mol', 'mca', 'isotope'}) - ATOM_BATCH_WRITABLE`


<a id="hotpot-cheminfo-structure--identity"></a>
## `hotpot.cheminfo.structure._identity`

`hotpot/cheminfo/structure/_identity.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--mutation"></a>
## `hotpot.cheminfo.structure._mutation`

`hotpot/cheminfo/structure/_mutation.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--results"></a>
## `hotpot.cheminfo.structure._results`

`hotpot/cheminfo/structure/_results.py`；显式 __all__: ()


<a id="hotpot-cheminfo-structure--results-partialchargeset"></a>
### `PartialChargeSet`

定义 owner `hotpot.cheminfo.structure._results.PartialChargeSet`；基类 无。

```python
PartialChargeSet(atom_uids: tuple[int, ...], values: tuple[float | None, ...], method: str, method_version: str | None, parameters: tuple[tuple[str, str | None], ...], units: str, provenance: tuple[int, ...])
```

- `__deepcopy__(self, memo)`
- `atom_uids: tuple[int, ...]` [只读值字段]
- `values: tuple[float | None, ...]` [只读值字段]
- `method: str` [只读值字段]
- `method_version: str | None` [只读值字段]
- `parameters: tuple[tuple[str, str | None], ...]` [只读值字段]
- `units: str` [只读值字段]
- `provenance: tuple[int, ...]` [只读值字段]


<a id="hotpot-cheminfo-structure--selection"></a>
## `hotpot.cheminfo.structure._selection`

`hotpot/cheminfo/structure/_selection.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--source-identity"></a>
## `hotpot.cheminfo.structure._source_identity`

`hotpot/cheminfo/structure/_source_identity.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--state"></a>
## `hotpot.cheminfo.structure._state`

`hotpot/cheminfo/structure/_state.py`；显式 __all__: 无



<a id="hotpot-cheminfo-structure--stereo"></a>
## `hotpot.cheminfo.structure._stereo`

`hotpot/cheminfo/structure/_stereo.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure--stereo-codec"></a>
## `hotpot.cheminfo.structure._stereo_codec`

`hotpot/cheminfo/structure/_stereo_codec.py`；显式 __all__: 无



<a id="hotpot-cheminfo-structure--topology"></a>
## `hotpot.cheminfo.structure._topology`

`hotpot/cheminfo/structure/_topology.py`；显式 __all__: 无



<a id="hotpot-cheminfo-structure-atom"></a>
## `hotpot.cheminfo.structure.atom`

`hotpot/cheminfo/structure/atom.py`；显式 __all__: 无


<a id="hotpot-cheminfo-structure-atom-atom"></a>
### `Atom`

定义 owner `hotpot.cheminfo.structure.atom.Atom`；基类 `MolBlock`。

- `__init__(self, mol: Molecule=None, **kwargs)`
- `__reduce_ex__(self, protocol)`
- `__setstate__(self, state)`
- `add_atom(self, atom: Union[str, int, 'Atom']=1, bond_order: int=1, atom_attrs: dict=None, bond_attrs: dict=None)`
- `random_point_on_sphere(radius: float=1.0, rng: Optional[np.random.Generator]=None)` [staticmethod]
- `add_hydrogen(self, num: int=None) -> list['Atom']`
- `calc_implicit_hydrogens(self)`
- `link_with(self, other: 'Atom')`
- `calc_electron_config(self) -> (int, int, list[int])`
- `get_formal_charge(self) -> cython.int`
- `get_valence(self) -> cython.int`
- `set_valence_to_default(self)`
- `isotope` [读/写]（native setter）
- `attrs` [只读]
- `period` [只读]: `int`
- `group` [只读]
- `polar_hydrogen_site` [只读]: `bool`
- `bonds` [只读]: `list['Bond']`
- `constraint` [读/写]: `bool`；setter `constraint(self, value: bool)`
- `coordinates` [读/写]；setter `coordinates(self, value: types.ArrayLike)`
- `exact_mass` [只读]
- `electronegativity` [只读]: `Optional[float]`
- `explicit_hydrogens` [只读]: `int`
- `covalent_radius` [只读]: `float`
- `density` [只读]: `float`
- `hyb` [只读]: `int`
- `idx` [只读]: `int`
- `mca` [只读]: `float`
- `is_error_electron_configure` [只读]: `bool`
- `is_hydrogen` [只读]: `bool`
- `is_polar_hydrogen` [只读]: `bool`
- `is_lanthanide` [只读]
- `is_actinide` [只读]
- `is_noble_gases` [只读]
- `is_halogens` [只读]
- `is_metal` [只读]
- `label` [只读]: `str`
- `mass` [只读]
- `mol` [只读]: `Molecule`
- `molecule` [只读]
- `neigh_idx` [只读]: `np.ndarray`
- `hydrogens` [只读]
- `h_count` [只读]
- `connectivity` [只读]
- `neighbours` [只读]: `list['Atom']`
- `heavy_neighbours` [只读]: `list['Atom']`
- `missing_electrons_element` [只读]
- `missing_electrons` [只读]
- `open_shell_electrons` [只读]: `int`
- `electron_configuration` [只读]: `np.ndarray[int]`
- `l` [只读]: `int`
- `n` [只读]: `int`
- `oxidation_state` [只读]: `int`
- `sum_bond_orders` [只读]: `int`
- `sum_heavy_cov_orders` [只读]: `int`
- `sum_covalent_orders` [只读]: `int`
- `symbol` [读/写]: `str`；setter `symbol(self, value: str)`
- `vector` [只读]: `np.ndarray`
- `atomic_number` [读/写]: `int`（native setter）（动态投影）
- `s` [只读]: `int`（动态投影）
- `p` [只读]: `int`（动态投影）
- `d` [只读]: `int`（动态投影）
- `f` [只读]: `int`（动态投影）
- `g` [只读]: `int`（动态投影）
- `formal_charge` [读/写]: `int`（native setter）（动态投影）
- `partial_charge` [只读]: `float | None`（动态投影）
- `is_aromatic` [只读]: `bool`（动态投影）
- `x` [读/写]: `float`（native setter）（动态投影）
- `y` [读/写]: `float`（native setter）（动态投影）
- `z` [读/写]: `float`（native setter）（动态投影）
- `valence` [只读]: `int`（动态投影）
- `implicit_hydrogens` [读/写]: `int`（native setter）（动态投影）
- `id` [读/写]: `int`（native setter）（动态投影）
- `x_constraint` [读/写]: `bool`（native setter）（动态投影）
- `y_constraint` [读/写]: `bool`（native setter）（动态投影）
- `z_constraint` [读/写]: `bool`（native setter）（动态投影）
- `__setattr__(self, key, value)`
- `frac_valence` [可写实例字段；非自动批准事实写许可]
- `__repr__(self)` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `__copy__(self)` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `__dir__(self)` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `__getattr__(self, item)` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `attrs_enumerator` [只读]: `tuple` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `attr_dict` [只读]: `dict` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `in_ring` [只读] ← `hotpot.cheminfo.structure._fields.MolBlock`
- `in_organic` [只读]: `bool` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `rings` [只读] ← `hotpot.cheminfo.structure._fields.MolBlock`


<a id="hotpot-cheminfo-structure-bond"></a>
## `hotpot.cheminfo.structure.bond`

`hotpot/cheminfo/structure/bond.py`；显式 __all__: 无


<a id="hotpot-cheminfo-structure-bond-bond"></a>
### `Bond`

定义 owner `hotpot.cheminfo.structure.bond.Bond`；基类 `AtomSeq`, `MolBlock`。

- `__init__(self, atom1: Atom, atom2: Atom, **kwargs)`
- `__reduce_ex__(self, protocol)`
- `__setstate__(self, state)`
- `__repr__(self)`
- `__eq__(self, other)`
- `__hash__(self)`
- `__getattr__(self, item)`
- `another_end(self, this_end: Atom)`
- `bond_segment_distance(self, other_bond: 'Bond') -> float`
- `attrs` [只读]
- `atom1` [只读]: `Atom`
- `atom2` [只读]: `Atom`
- `bond_kind` [只读]: `BondKind`
- `bond_direction` [只读]: `Optional[str]`
- `bond_source` [只读]: `Optional[str]`
- `bond_source_metadata` [只读]: `dict`
- `attr_dict` [只读]: `dict`
- `bond_segment` [只读]: `geometry.Segment`
- `idx` [只读]: `int`
- `ideal_bond_length` [只读]: `float`
- `length` [只读]: `float`
- `label` [只读]
- `is_aromatic` [只读]: `bool`
- `is_covalent` [只读]: `bool`
- `is_heavy_covalent` [只读]: `bool`
- `is_metal_ligand_bond` [只读]: `bool`
- `rotatable` [只读]: `bool`
- `bond_order` [读/写]: `float`
- `constraint` [读/写]: `bool`
- `id` [读/写]: `int`
- `__setattr__(self, name, value)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__len__(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__contains__(self, item: Union[Atom, 'Bond'])` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `to_mol(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms` [只读]: `tuple[Atom, ...]` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms_indices` [只读] ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bonds` [只读]: `list['Bond']` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `mol` [只读]: `Molecule` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `has_metal` [只读] ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atomN` [只读]: `Atom` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bondN` [只读]: `Bond` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `aNidx` [只读]: `int` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__copy__(self)` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `__dir__(self)` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `attrs_enumerator` [只读]: `tuple` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `in_ring` [只读] ← `hotpot.cheminfo.structure._fields.MolBlock`
- `in_organic` [只读]: `bool` ← `hotpot.cheminfo.structure._fields.MolBlock`
- `rings` [只读] ← `hotpot.cheminfo.structure._fields.MolBlock`

模块值/类型别名（声明存在不构成事实写许可）：

- `geometry` = `_geometry_namespace()`


<a id="hotpot-cheminfo-structure-bond-kind"></a>
## `hotpot.cheminfo.structure.bond_kind`

`hotpot/cheminfo/structure/bond_kind.py`；显式 __all__: ('BondKind',)


<a id="hotpot-cheminfo-structure-bond-kind-bondkind"></a>
### `BondKind`

定义 owner `hotpot.cheminfo.structure.bond_kind.BondKind`；基类 `str`, `Enum`。

- `SINGLE` = `'single'` [枚举常量]
- `DOUBLE` = `'double'` [枚举常量]
- `TRIPLE` = `'triple'` [枚举常量]
- `AROMATIC` = `'aromatic'` [枚举常量]
- `ZERO` = `'zero'` [枚举常量]
- `DATIVE` = `'dative'` [枚举常量]
- `UNKNOWN` = `'unknown'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.


<a id="hotpot-cheminfo-structure-bundle"></a>
## `hotpot.cheminfo.structure.bundle`

`hotpot/cheminfo/structure/bundle.py`；显式 __all__: ('MolBundle',)


<a id="hotpot-cheminfo-structure-bundle-molbundle"></a>
### `MolBundle`

定义 owner `hotpot.cheminfo.structure.bundle.MolBundle`；基类 无。

- `__init__(self, list_mol: tp.Iterable[Molecule])`
- `to_pyg_dataset(self, dataset_root: os.PathLike)`
- `list_mol` = `list(list_mol)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-structure-conformer"></a>
## `hotpot.cheminfo.structure.conformer`

`hotpot/cheminfo/structure/conformer.py`；显式 __all__: ()


<a id="hotpot-cheminfo-structure-conformer-conformerbatch"></a>
### `ConformerBatch`

定义 owner `hotpot.cheminfo.structure.conformer.ConformerBatch`；基类 无。

```python
ConformerBatch(coordinates: NDArray[np.float64], ordinals: tuple[int, ...])
```

- `__reduce__(self)`
- `coordinates: NDArray[np.float64]` [只读值字段]
- `ordinals: tuple[int, ...]` [只读值字段]

<a id="hotpot-cheminfo-structure-conformer-energyforceresult"></a>
### `EnergyForceResult`

定义 owner `hotpot.cheminfo.structure.conformer.EnergyForceResult`；基类 无。

```python
EnergyForceResult(atom_uids: tuple[int, ...], values: tuple, frame_indices: tuple, method: str, method_version: str | None, parameters: tuple, units: tuple, provenance: tuple)
```

- `__reduce__(self)`
- `__deepcopy__(self, memo)`
- `atom_uids: tuple[int, ...]` [只读值字段]
- `values: tuple` [只读值字段]
- `frame_indices: tuple` [只读值字段]
- `method: str` [只读值字段]
- `method_version: str | None` [只读值字段]
- `parameters: tuple` [只读值字段]
- `units: tuple` [只读值字段]
- `provenance: tuple` [只读值字段]

<a id="hotpot-cheminfo-structure-conformer-coordinateframe"></a>
### `CoordinateFrame`

定义 owner `hotpot.cheminfo.structure.conformer.CoordinateFrame`；基类 无。

```python
CoordinateFrame(atom_uids: tuple[int, ...], coordinates: np.ndarray, observations: EnergyForceResult | None, observation_index: int | None, provenance: tuple)
```

- `__reduce__(self)`
- `__deepcopy__(self, memo)`
- `atom_uids: tuple[int, ...]` [只读值字段]
- `coordinates: np.ndarray` [只读值字段]
- `observations: EnergyForceResult | None` [只读值字段]
- `observation_index: int | None` [只读值字段]
- `provenance: tuple` [只读值字段]

<a id="hotpot-cheminfo-structure-conformer-conformerensembleresult"></a>
### `ConformerEnsembleResult`

定义 owner `hotpot.cheminfo.structure.conformer.ConformerEnsembleResult`；基类 无。

```python
ConformerEnsembleResult(atom_uids: tuple[int, ...], batch: ConformerBatch, observations: EnergyForceResult | None, input_source: object)
```

- `__reduce__(self)`
- `__deepcopy__(self, memo)`
- `atom_uids: tuple[int, ...]` [只读值字段]
- `batch: ConformerBatch` [只读值字段]
- `observations: EnergyForceResult | None` [只读值字段]
- `input_source: object` [只读值字段]

<a id="hotpot-cheminfo-structure-conformer-conformers"></a>
### `Conformers`

定义 owner `hotpot.cheminfo.structure.conformer.Conformers`；基类 无。

- `__init__(self)`
- `__getstate__(self)`
- `__setstate__(self, state)`
- `__len__(self)`
- `__getitem__(self, idx)`
- `__iter__(self)`
- `index_attr(self, name, i)`
- `add(self, coords: _types.ArrayLike, energy: Union[_types.ArrayLike, float]=None)`
- `clear(self)`
- `coordinates(self, i)`
- `energy(self, i)`
- `gibbs(self, i)`
- `partial_charge(self, i)`
- `force(self, i)`


<a id="hotpot-cheminfo-structure-constraints"></a>
## `hotpot.cheminfo.structure.constraints`

`hotpot/cheminfo/structure/constraints.py`；显式 __all__: ()



<a id="hotpot-cheminfo-structure-crystal"></a>
## `hotpot.cheminfo.structure.crystal`

`hotpot/cheminfo/structure/crystal.py`；显式 __all__: 无


<a id="hotpot-cheminfo-structure-crystal-crystal"></a>
### `Crystal`

定义 owner `hotpot.cheminfo.structure.crystal.Crystal`；基类 无。

- `__init__(self, a, b, c, alpha, beta, gamma, *, molecule=None, space_group=None, site_relations=None, bond_images=None)`
- `from_matrix(cls, matrix, *, molecule=None, space_group=None, site_relations=None, bond_images=None)` [classmethod]
- `from_cif(cls, source, *, block=None)` [classmethod]
- `to_cif(self)`
- `to_pyg(self)`
- `from_pyg(cls, data, *, field_map=None)` [classmethod]
- `set_cell(self, matrix, *, preserve='cartesian')`
- `set_space_group(self, space_group, *, site_relations=None)`
- `set_site_relations(self, relations)`
- `to_fractional(self, cartesian)`
- `to_cartesian(self, fractional)`
- `operate(self, operation)`
- `unit_cell(self)`
- `unit_molecules(self)`
- `expand(self, repeats)`
- `same_molecule_candidates(self, *, tolerance)`
- `molecular_symmetry(self, molecule=None, *, tolerance)`
- `linking_number(self, first_cycle, second_cycle, *, tolerance)`
- `geometry_findings(self, *, coincidence_tolerance, close_contact_cutoff, ring_scope='full_graph', max_ring_size=None)`
- `copy(self)`
- `__reduce_ex__(self, protocol)`
- `molecule` [只读]
- `matrix` [只读]
- `volume` [只读]
- `space_group` [只读]
- `symm_units` [只读]
- `site_relations` [只读]
- `bond_images` [只读]
- `representation` [只读]
- `image_map` [只读]
- `a` [读/写]: `float`（动态投影）
- `b` [读/写]: `float`（动态投影）
- `c` [读/写]: `float`（动态投影）
- `alpha` [读/写]: `float`（动态投影）
- `beta` [读/写]: `float`（动态投影）
- `gamma` [读/写]: `float`（动态投影）


<a id="hotpot-cheminfo-structure-errors"></a>
## `hotpot.cheminfo.structure.errors`

`hotpot/cheminfo/structure/errors.py`；显式 __all__: 无


<a id="hotpot-cheminfo-structure-errors-notinsamemolecule"></a>
### `NotInSameMolecule`

定义 owner `hotpot.cheminfo.structure.errors.NotInSameMolecule`；基类 `Exception`。

- `__init__(self, obj1, obj2)`
- `obj1` = `obj1` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `obj2` = `obj2` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `message` = `f'The {type(obj1).__name__} and {type(obj2).__name__} must belong to the same Molecule'` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-cheminfo-structure-errors-objnotinmolecule"></a>
### `ObjNotInMolecule`

定义 owner `hotpot.cheminfo.structure.errors.ObjNotInMolecule`；基类 `Exception`。

- `__init__(self, obj, mol)`
- `obj` = `obj` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `mol` = `mol` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `message` = `f'The {type(obj).__name__} is not a current member of the Molecule'` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-structure-forcefields"></a>
## `hotpot.cheminfo.structure.forcefields`

`hotpot/cheminfo/structure/forcefields.py`；显式 __all__: ('AtomIdentity', 'BondTopologyRevision', 'ForceFieldFrame', 'CoordinationEnvironment', 'AtomTopologySignature', 'BondTopology', 'TopologyReference', 'TrajectoryStart', 'TrajectoryStage', 'TrajectoryEvent', 'RingFrameEvidence', 'CoordinationFrameEvidence', 'OptimizationFrameEvidence', 'FrameEvidence')


<a id="hotpot-cheminfo-structure-forcefields-atomtopologysignature"></a>
### `AtomTopologySignature`

定义 owner `hotpot.cheminfo.structure.forcefields.AtomTopologySignature`；基类 无。

```python
AtomTopologySignature(index: int, atom_id: int, atomic_number: int, formal_charge: int)
```

- `index: int` [只读值字段]
- `atom_id: int` [只读值字段]
- `atomic_number: int` [只读值字段]
- `formal_charge: int` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-bondtopology"></a>
### `BondTopology`

定义 owner `hotpot.cheminfo.structure.forcefields.BondTopology`；基类 无。

```python
BondTopology(atom_indices: Tuple[int, int], bond_order: float, bond_kind: str)
```

- `from_bond(cls, bond: 'Bond') -> 'BondTopology'` [classmethod]
- `atom_indices: Tuple[int, int]` [只读值字段]
- `bond_order: float` [只读值字段]
- `bond_kind: str` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-topologyreference"></a>
### `TopologyReference`

定义 owner `hotpot.cheminfo.structure.forcefields.TopologyReference`；基类 无。

```python
TopologyReference(atoms: Tuple[AtomTopologySignature, ...], bonds: Tuple[BondTopology, ...], allow_added_hydrogens: bool = True)
```

- `atoms: Tuple[AtomTopologySignature, ...]` [只读值字段]
- `bonds: Tuple[BondTopology, ...]` [只读值字段]
- `allow_added_hydrogens: bool` = `True` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-trajectorystart"></a>
### `TrajectoryStart`

定义 owner `hotpot.cheminfo.structure.forcefields.TrajectoryStart`；基类 `str`, `Enum`。

- `LIGAND_BUILD` = `'ligand_build'` [枚举常量]
- `COORDINATION_RESTORATION` = `'coordination_restoration'` [枚举常量]
- `COMPLEX_UNTANGLING` = `'complex_untangling'` [枚举常量]
- `FINAL_OPTIMIZATION` = `'final_optimization'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-forcefields-trajectorystage"></a>
### `TrajectoryStage`

定义 owner `hotpot.cheminfo.structure.forcefields.TrajectoryStage`；基类 `str`, `Enum`。

- `LIGAND_BUILD` = `'ligand_build'` [枚举常量]
- `COORDINATION_RESTORATION` = `'coordination_restoration'` [枚举常量]
- `COMPLEX_UNTANGLING` = `'complex_untangling'` [枚举常量]
- `FINAL_OPTIMIZATION` = `'final_optimization'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-forcefields-trajectoryevent"></a>
### `TrajectoryEvent`

定义 owner `hotpot.cheminfo.structure.forcefields.TrajectoryEvent`；基类 `str`, `Enum`。

- `INITIAL` = `'initial'` [枚举常量]
- `BUILD_COMPLETE` = `'build_complete'` [枚举常量]
- `WARMUP_COMPLETE` = `'warmup_complete'` [枚举常量]
- `COORDINATION_READY` = `'coordination_ready'` [枚举常量]
- `BOND_TRIAL` = `'bond_trial'` [枚举常量]
- `BOND_ACCEPTED` = `'bond_accepted'` [枚举常量]
- `BOND_REJECTED` = `'bond_rejected'` [枚举常量]
- `BOND_ROLLBACK` = `'bond_rollback'` [枚举常量]
- `BOND_FORCED` = `'bond_forced'` [枚举常量]
- `METAL_RELOCATION_TRIAL` = `'metal_relocation_trial'` [枚举常量]
- `METAL_RELOCATED` = `'metal_relocated'` [枚举常量]
- `METAL_RELOCATION_FAILED` = `'metal_relocation_failed'` [枚举常量]
- `TOPOLOGY_CHECKPOINT` = `'topology_checkpoint'` [枚举常量]
- `RING_OPENED` = `'ring_opened'` [枚举常量]
- `PERTURBED` = `'perturbed'` [枚举常量]
- `OPTIMIZED` = `'optimized'` [枚举常量]
- `RING_CLOSED` = `'ring_closed'` [枚举常量]
- `SETTLED` = `'settled'` [枚举常量]
- `ROLLED_BACK` = `'rolled_back'` [枚举常量]
- `EPOCH_COMPLETE` = `'epoch_complete'` [枚举常量]
- `TERMINAL` = `'terminal'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-forcefields-ringframeevidence"></a>
### `RingFrameEvidence`

定义 owner `hotpot.cheminfo.structure.forcefields.RingFrameEvidence`；基类 无。

```python
RingFrameEvidence(confirmed_piercing_count: int, uncertain_relation_count: Optional[int] = 0, ring_scope: Optional[Literal['full_graph', 'ligand_skeleton']] = None, max_ring_size: Optional[int] = None, selected_ring_count: Optional[int] = None, excluded_ring_count: Optional[int] = None, candidate_pair_count: Optional[int] = None, aabb_separated_pair_count: Optional[int] = None, exact_pair_count: Optional[int] = None, does_not_pierce_pair_count: Optional[int] = None, scan_complete: Optional[bool] = None)
```

- `confirmed_piercing_count: int` [只读值字段]
- `uncertain_relation_count: Optional[int]` = `0` [只读值字段]
- `ring_scope: Optional[Literal['full_graph', 'ligand_skeleton']]` = `None` [只读值字段]
- `max_ring_size: Optional[int]` = `None` [只读值字段]
- `selected_ring_count: Optional[int]` = `None` [只读值字段]
- `excluded_ring_count: Optional[int]` = `None` [只读值字段]
- `candidate_pair_count: Optional[int]` = `None` [只读值字段]
- `aabb_separated_pair_count: Optional[int]` = `None` [只读值字段]
- `exact_pair_count: Optional[int]` = `None` [只读值字段]
- `does_not_pierce_pair_count: Optional[int]` = `None` [只读值字段]
- `scan_complete: Optional[bool]` = `None` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-coordinationframeevidence"></a>
### `CoordinationFrameEvidence`

定义 owner `hotpot.cheminfo.structure.forcefields.CoordinationFrameEvidence`；基类 无。

```python
CoordinationFrameEvidence(bond_atom_indices: Optional[Tuple[int, int]], accepted: Optional[bool], pending_bond_count: int = 0, forced: bool = False, piercing_relation_count: int = 0, undetermined_relation_count: int = 0, excluded_ring_count: int = 0, metal_atom_index: Optional[int] = None, relocation_status: Optional[Literal['relocated', 'infeasible']] = None, relocation_candidates_evaluated: int = 0, safe_donor_atom_indices: Tuple[int, ...] = (), minimum_normalized_clearance: Optional[float] = None, coordination_distance_deviation: Optional[float] = None)
```

- `bond_atom_indices: Optional[Tuple[int, int]]` [只读值字段]
- `accepted: Optional[bool]` [只读值字段]
- `pending_bond_count: int` = `0` [只读值字段]
- `forced: bool` = `False` [只读值字段]
- `piercing_relation_count: int` = `0` [只读值字段]
- `undetermined_relation_count: int` = `0` [只读值字段]
- `excluded_ring_count: int` = `0` [只读值字段]
- `metal_atom_index: Optional[int]` = `None` [只读值字段]
- `relocation_status: Optional[Literal['relocated', 'infeasible']]` = `None` [只读值字段]
- `relocation_candidates_evaluated: int` = `0` [只读值字段]
- `safe_donor_atom_indices: Tuple[int, ...]` = `()` [只读值字段]
- `minimum_normalized_clearance: Optional[float]` = `None` [只读值字段]
- `coordination_distance_deviation: Optional[float]` = `None` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-optimizationframeevidence"></a>
### `OptimizationFrameEvidence`

定义 owner `hotpot.cheminfo.structure.forcefields.OptimizationFrameEvidence`；基类 无。

```python
OptimizationFrameEvidence(converged: bool, exploded: bool, finite_coordinates: bool, finite_energy: bool, finite_gradients: bool, rms_gradient_kj_mol_angstrom: Optional[float] = None, max_gradient_kj_mol_angstrom: Optional[float] = None, energy_change_kj_mol: Optional[float] = None, max_displacement_angstrom: Optional[float] = None)
```

- `converged: bool` [只读值字段]
- `exploded: bool` [只读值字段]
- `finite_coordinates: bool` [只读值字段]
- `finite_energy: bool` [只读值字段]
- `finite_gradients: bool` [只读值字段]
- `rms_gradient_kj_mol_angstrom: Optional[float]` = `None` [只读值字段]
- `max_gradient_kj_mol_angstrom: Optional[float]` = `None` [只读值字段]
- `energy_change_kj_mol: Optional[float]` = `None` [只读值字段]
- `max_displacement_angstrom: Optional[float]` = `None` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-atomidentity"></a>
### `AtomIdentity`

定义 owner `hotpot.cheminfo.structure.forcefields.AtomIdentity`；基类 无。

```python
AtomIdentity(index: int, atom_id: int, atomic_number: int, formal_charge: int, symbol: str)
```

- `from_atom(cls, atom: 'Atom', *, index: Optional[int]=None) -> 'AtomIdentity'` [classmethod]
- `index: int` [只读值字段]
- `atom_id: int` [只读值字段]
- `atomic_number: int` [只读值字段]
- `formal_charge: int` [只读值字段]
- `symbol: str` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-bondtopologyrevision"></a>
### `BondTopologyRevision`

定义 owner `hotpot.cheminfo.structure.forcefields.BondTopologyRevision`；基类 无。

```python
BondTopologyRevision(index: int, bonds: Tuple[BondTopology, ...])
```

- `index: int` [只读值字段]
- `bonds: Tuple[BondTopology, ...]` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-forcefieldframe"></a>
### `ForceFieldFrame`

定义 owner `hotpot.cheminfo.structure.forcefields.ForceFieldFrame`；基类 无。

```python
ForceFieldFrame(index: int, stage: TrajectoryStage, event: TrajectoryEvent, coordinate_revision: int, topology_revision: int, energy_kj_mol: Optional[float] = None, component_index: Optional[int] = None, attempt: Optional[int] = None, step: Optional[int] = None, evidence: Optional[FrameEvidence] = None)
```

- `index: int` [只读值字段]
- `stage: TrajectoryStage` [只读值字段]
- `event: TrajectoryEvent` [只读值字段]
- `coordinate_revision: int` [只读值字段]
- `topology_revision: int` [只读值字段]
- `energy_kj_mol: Optional[float]` = `None` [只读值字段]
- `component_index: Optional[int]` = `None` [只读值字段]
- `attempt: Optional[int]` = `None` [只读值字段]
- `step: Optional[int]` = `None` [只读值字段]
- `evidence: Optional[FrameEvidence]` = `None` [只读值字段]

<a id="hotpot-cheminfo-structure-forcefields-coordinationenvironment"></a>
### `CoordinationEnvironment`

定义 owner `hotpot.cheminfo.structure.forcefields.CoordinationEnvironment`；基类 无。

```python
CoordinationEnvironment(metal_idx: int, donor_indices: Tuple[int, ...], coordination_number: int, metal_atomic_number: int, metal_formal_charge: int, donor_atomic_numbers: Tuple[int, ...], chelate_groups: Tuple[Tuple[int, ...], ...])
```

- `metal_idx: int` [只读值字段]
- `donor_indices: Tuple[int, ...]` [只读值字段]
- `coordination_number: int` [只读值字段]
- `metal_atomic_number: int` [只读值字段]
- `metal_formal_charge: int` [只读值字段]
- `donor_atomic_numbers: Tuple[int, ...]` [只读值字段]
- `chelate_groups: Tuple[Tuple[int, ...], ...]` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `FrameEvidence` = `Union[RingFrameEvidence, CoordinationFrameEvidence, OptimizationFrameEvidence]`


<a id="hotpot-cheminfo-structure-molecule"></a>
## `hotpot.cheminfo.structure.molecule`

`hotpot/cheminfo/structure/molecule.py`；显式 __all__: 无


<a id="hotpot-cheminfo-structure-molecule-molecule"></a>
### `Molecule`

定义 owner `hotpot.cheminfo.structure.molecule.Molecule`；基类 无。

- `__init__(self)`
- `__dir__(self)`
- `__repr__(self)`
- `__add__(self, other)`
- `__copy__(self)`
- `__getstate__(self)`
- `__setstate__(self, state)`
- `__reduce_ex__(self, protocol)`
- `updata_atoms(self, *, atoms=None, **columns) -> None`
- `copy(self)`
- `create_crystal(self, a, b, c, alpha, beta, gamma, *, space_group=None, site_relations=None, bond_images=None)`
- `read_one(src, fmt=None, **kwargs)` [staticmethod]
- `calc_mol_default_charge(self)`
- `simple_paths(self, source=None, target=None, cutoff=None)`
- `add_atom(self, atom)`
- `build_all_pair_links(self, metal, threshold: Optional[float]=None, greedy=True, *, first_threshold: Optional[float]=None)`
- `auto_pair_metal(self, metal, threshold: Optional[float]=None, greedy=True, probability: bool=False, *, first_threshold: Optional[float]=None) -> Union['Molecule', tuple['Molecule', float]]`
- `replace_atom(self, original_atom_idx: int, new_atom: Union[int, str, 'Atom'])`
- `add_bond(self, atom1: Union[int, 'Atom'], atom2: Union[int, 'Atom'], bond_order=1.0, **kwargs)`
- `add_bonds(self, idx_order: Iterable[tuple[int, int, float]], list_kw: Optional[list[dict]]=None) -> list['Bond']`
- `add_component(self, component: 'Molecule')`
- `add_hydrogens(self, rm_polar_hs: bool=True, rng: Optional[np.random.Generator]=None)`
- `get_thermo(self, temp: float=298.15, pressure: float=101325.0) -> 'Thermo'`
- `clear_constraints(self) -> None`
- `assign_bond_order(self)`
- `bond(self, a1idx: int, a2idx: int)`
- `build3d(self, forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']]='UFF', *, algorithm: Literal['steepest', 'conjugate']='conjugate', epochs: int=100, steps_per_epoch: int=100, add_hydrogens: bool=True, quality_level: Literal['off', 'basic', 'standard', 'strict']='standard', quality_thresholds: Optional['ff.StructureAcceptanceThresholds']=None, seed: Optional[int]=None, timeout: float=1000.0, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional['ff.OptimizationStoppingCriteria']=None, convergence_level: 'ff.ConvergenceLevel'=ff.DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional['ff.TrajectoryStart']=None, trajectory_path: Optional['ff.TrajectoryPath']=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5, candidate_count: Optional[int]=None, max_attempts: int=50, candidate_warmup_steps: int=500, candidate_score_steps: int=1000, best_candidate_refine_steps: int=3000, ligand_untangling_attempts: int=20, coordination_restoration_attempts: int=20, coordination_relaxation_steps: int=100, complex_untangling_attempts: int=30)`
- `update_mol_charge(self)`
- `determine_rings_aromatic(self)`
- `conformer_load(self, i: int)`
- `conformer_add(self, coords: Optional[types.ArrayLike]=None, energy: Optional[Union[types.ArrayLike, float]]=None)`
- `conformer_clear(self)`
- `conformer_get(self, idx: Union[int, slice]) -> dict`
- `optimize(self, forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']]=None, *, algorithm: Literal['steepest', 'conjugate']='conjugate', epochs: int=100, steps_per_epoch: int=100, complex_untangling_attempts: int=30, add_hydrogens: bool=True, quality_level: Literal['off', 'basic', 'standard', 'strict']='standard', quality_thresholds: Optional['ff.StructureAcceptanceThresholds']=None, seed: Optional[int]=None, perturb_interval: Optional[int]=None, perturb_sigma: float=0.5, stopping_criteria: Optional['ff.OptimizationStoppingCriteria']=None, convergence_level: 'ff.ConvergenceLevel'=ff.DEFAULT_CONVERGENCE_LEVEL, save_movie: bool=False, trajectory_start: Optional['ff.TrajectoryStart']=None, trajectory_path: Optional['ff.TrajectoryPath']=None, increasing_vdw: bool=False, vdw_cutoff_start: float=0.0, vdw_cutoff_end: float=12.5)`
- `calc_atom_valence(self, assign_aromatic: Optional[bool]=None)`
- `calc_implicit_hydrogens(self)`
- `constraint_bonds_angles(self, exclude_metal_bonds: bool=True)`
- `create_atom(self, **kwargs) -> 'Atom'`
- `get_partial_charge(self, model: PartialChargeModel='qeq', model_args: Optional[str]=None) -> tuple[float, ...]`
- `assign_partial_charge(self, model: PartialChargeModel='qeq', model_args: Optional[str]=None) -> tuple[float, ...]`
- `graph_spectral(self, norm: Literal['infinite', 'min', 'l1', 'l2']='l2')`
- `bond_ring_piercing_state(self, *, ring_scope: Literal['full_graph', 'ligand_skeleton']='full_graph', max_ring_size: int=_DEFAULT_BOND_RING_MAXIMUM_SIZE) -> 'geometry.PiercingState'`
- `bond_ring_relations(self, *, ring_scope: Literal['full_graph', 'ligand_skeleton']='full_graph', max_ring_size: int=_DEFAULT_BOND_RING_MAXIMUM_SIZE) -> 'geometry_adapter.BondRingScanReport[Ring, Bond]'`
- `link_atoms(self, assign_bond_order: bool=True)`
- `refresh_atom_id(self)`
- `remove_atom(self, atom: Union[int, 'Atom']) -> None`
- `remove_atoms(self, atoms: Iterable['Atom']) -> None`
- `remove_bond(self, bond: 'Bond') -> None`
- `remove_bonds(self, bonds: Iterable['Bond']) -> None`
- `force_remove_polar_hydrogens(self)`
- `remove_hydrogens(self)`
- `remove_metals(self)`
- `search_substructure(self, smarts: str, *, semantics='full_graph')`
- `set_default_valence(self)`
- `similarity(self, other: 'Molecule', method: str='spectrum', norm: Literal['infinite', 'min', 'l1', 'l2']='l2')`
- `longest_path(self)`
- `canonical_tree(self)`
- `shortest_path_index(self, atom1, atom2) -> list[int]`
- `shortest_path(self, atom1, atom2) -> list['Atom']`
- `shortest_paths_indices(self, atom1, atom2)`
- `rings_for_scope(self, ring_scope: Literal['full_graph', 'ligand_skeleton'], *, max_size: Optional[int]=None, max_cycles: Optional[int]=graph_algorithms.DEFAULT_RELEVANT_CYCLE_LIMIT) -> list['Ring']`
- `to_pyg(self)`
- `from_pyg(cls, data, *, field_map=None)` [classmethod]
- `to_obmol(self) -> 'ob.OBMol'`
- `to_rdmol(self)`
- `to_pybel_mol(self) -> 'pb.Molecule'`
- `translation(self, vector: types.ArrayLike)`
- `write(self, filename=None, fmt: Optional[str]=None, overwrite=False, write_single: bool=False, ob_opt: dict=None, **kwargs)`
- `charge` [读/写]；setter `charge(self, value)`
- `spin_mult` [读/写]；setter `spin_mult(self, value)`
- `crystal` [只读]
- `hydrogens` [只读]: `list['Atom']`
- `sum_explicit_hydrogens` [只读]: `int`
- `sum_implicit_hydrogens` [只读]: `int`
- `has_hydrogens` [只读]: `bool`
- `cid` [只读]: `Optional[int]`
- `name` [只读]: `str`
- `cas` [只读]
- `atom_pairs` [只读]: `'AtomPairs'`
- `angles` [只读]: `list['Angle']`
- `torsions` [只读]: `list['Torsion']`
- `atom_matrix` [只读]: `np.ndarray`
- `atoms` [只读]
- `c_bonds` [只读]: `list['Bond']`
- `bonds` [只读]: `list['Bond']`
- `sum_atoms_charge` [只读]: `int`
- `default_unpaired_electrons` [只读]
- `default_spin_mult` [只读]: `int`
- `conformers` [只读]: `'Conformers'`
- `conformers_number` [只读]: `int`
- `components` [只读]: `list['Molecule']`
- `coordinates` [读/写]: `np.ndarray`；独立 `(N,3)` float64 Cartesian/Angstrom 快照，唯一事实在 Atom table；setter `coordinates(self, value: np.ndarray) -> None` 经既有原子提交。
- `fractional_coordinates` [读/写]: `np.ndarray`；独立 `(N,3)` float64、无量纲且不 wrap；仅通过当前配对 Crystal 的 `to_fractional`/`to_cartesian` 转换；setter `fractional_coordinates(self, value: np.ndarray) -> None` 转回 Cartesian 后走 `coordinates` 原子提交。无 Crystal 或读取缺少完整 Cartesian 事实时抛出 `ValueError` (`CRYSTAL-CONTEXT-INTEGRITY`)；不新增坐标存储、缓存或持久化字段。
- `atom_pairwise_index` [只读]: `np.ndarray`
- `pair_dist` [只读]: `np.ndarray`
- `dist_matrix` [只读]: `np.ndarray`
- `element_counts` [只读]
- `formula` [只读]: `str`
- `is_disorder` [只读]
- `has_3d` [只读]
- `has_metal` [只读]: `bool`
- `heavy_atoms` [只读]: `list['Atom']`
- `mca_sites` [只读]: `dict['Atom', float]`
- `is_organic` [只读]: `bool`
- `is_full_halogenated` [只读]: `bool`
- `link_matrix` [只读]: `np.ndarray`
- `metals` [只读]: `list['Atom']`
- `atom_id_dict` [只读]
- `bond_id_dict` [只读]
- `InChi` [只读]: `str`
- `smiles` [只读]: `str`
- `kekulize_smiles` [只读]: `str`
- `rings` [只读]: `list['Ring']`
- `cycle_basis` [只读]: `list[tuple[int, ...]]`
- `all_cycle_basis` [只读]: `list[list[tuple[int, ...]]]`
- `basis_rings` [只读]: `list['Ring']`
- `all_basis_rings` [只读]: `list[list['Ring']]`
- `aromatic_joint_rings` [只读]: `list['JointRing']`
- `rings_small` [只读]: `list['Ring']`
- `ligand_rings` [只读]: `list['Ring']`
- `weight` [只读]
- `energy` [读/写]（动态投影）
- `partial_charge` [读/写]（动态投影）
- `gibbs` [读/写]（动态投影）
- `force` [读/写]（动态投影）
- `zero_point` [读/写]（动态投影）
- `thermo` [读/写]（动态投影）
- `capacity` [读/写]（动态投影）
- `properties` = `{}` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `identifier` [可写实例字段；非自动批准事实写许可]

模块值/类型别名（声明存在不构成事实写许可）：

- `graph_algorithms` = `_bind_declaration_contracts()`
- `ff` = `_bind_declaration_contracts()`
- `geometry` = `_bind_declaration_contracts()`
- `geometry_adapter` = `_bind_declaration_contracts()`
- `PartialChargeModel` = `Literal['eem', 'eem2015ha', 'eem2015hm', 'eem2015hn', 'eem2015ba', 'eem2015bm', 'eem2015bn', 'gasteiger', 'mmff94', 'qeq', 'qtpie', 'eqeq', 'fromfile', 'none']`
- `OPENBABEL_PARTIAL_CHARGE_MODELS` = `('eem', 'eem2015ha', 'eem2015hm', 'eem2015hn', 'eem2015ba', 'eem2015bm', 'eem2015bn', 'gasteiger', 'mmff94', 'qeq', 'qtpie', 'eqeq', 'fromfile', 'none')`


<a id="hotpot-cheminfo-structure-snapshots"></a>
## `hotpot.cheminfo.structure.snapshots`

`hotpot/cheminfo/structure/snapshots.py`；显式 __all__: ()


<a id="hotpot-cheminfo-structure-snapshots-molecularfactsidentity"></a>
### `MolecularFactsIdentity`

定义 owner `hotpot.cheminfo.structure.snapshots.MolecularFactsIdentity`；基类 无。

```python
MolecularFactsIdentity(chemical_identity_digest: 'str')
```

- `chemical_identity_digest: 'str'` [只读值字段]

<a id="hotpot-cheminfo-structure-snapshots-componentidentity"></a>
### `ComponentIdentity`

定义 owner `hotpot.cheminfo.structure.snapshots.ComponentIdentity`；基类 无。

```python
ComponentIdentity(identity: 'str', atom_indices: 'Tuple[int, ...]')
```

- `identity: 'str'` [只读值字段]
- `atom_indices: 'Tuple[int, ...]'` [只读值字段]

<a id="hotpot-cheminfo-structure-snapshots-hydrogeninventory"></a>
### `HydrogenInventory`

定义 owner `hotpot.cheminfo.structure.snapshots.HydrogenInventory`；基类 无。

```python
HydrogenInventory(explicit_counts: 'Tuple[Union[int, None], ...]', implicit_counts: 'Tuple[Union[int, None], ...]')
```

- `explicit_counts: 'Tuple[Union[int, None], ...]'` [只读值字段]
- `implicit_counts: 'Tuple[Union[int, None], ...]'` [只读值字段]

<a id="hotpot-cheminfo-structure-snapshots-semanticbond"></a>
### `SemanticBond`

定义 owner `hotpot.cheminfo.structure.snapshots.SemanticBond`；基类 无。

```python
SemanticBond(atom_indices: 'Tuple[int, int]', order: 'Union[float, None]', aromatic: 'bool', kind: 'SemanticBondKind')
```

- `atom_indices: 'Tuple[int, int]'` [只读值字段]
- `order: 'Union[float, None]'` [只读值字段]
- `aromatic: 'bool'` [只读值字段]
- `kind: 'SemanticBondKind'` [只读值字段]

<a id="hotpot-cheminfo-structure-snapshots-molecularfacts"></a>
### `MolecularFacts`

定义 owner `hotpot.cheminfo.structure.snapshots.MolecularFacts`；基类 无。

```python
MolecularFacts(identity: 'MolecularFactsIdentity', geometry_digest: 'Union[str, None]', atomic_numbers: 'Tuple[int, ...]', mass_numbers: 'Tuple[Union[int, None], ...]', atom_map: 'Tuple[int, ...]', source_components: 'Tuple[ComponentIdentity, ...]', covalent_components: 'Tuple[ComponentIdentity, ...]', hydrogen_inventory: 'HydrogenInventory', semantic_bonds: 'Tuple[SemanticBond, ...]', coordinates: 'Union[Tuple[Tuple[float, float, float], ...], None]', nuclear_charge_sum: 'int')
```

- `identity: 'MolecularFactsIdentity'` [只读值字段]
- `geometry_digest: 'Union[str, None]'` [只读值字段]
- `atomic_numbers: 'Tuple[int, ...]'` [只读值字段]
- `mass_numbers: 'Tuple[Union[int, None], ...]'` [只读值字段]
- `atom_map: 'Tuple[int, ...]'` [只读值字段]
- `source_components: 'Tuple[ComponentIdentity, ...]'` [只读值字段]
- `covalent_components: 'Tuple[ComponentIdentity, ...]'` [只读值字段]
- `hydrogen_inventory: 'HydrogenInventory'` [只读值字段]
- `semantic_bonds: 'Tuple[SemanticBond, ...]'` [只读值字段]
- `coordinates: 'Union[Tuple[Tuple[float, float, float], ...], None]'` [只读值字段]
- `nuclear_charge_sum: 'int'` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `SemanticBondKind` = `Literal['covalent', 'coordination', 'unknown']`


<a id="hotpot-cheminfo-structure-stereo"></a>
## `hotpot.cheminfo.structure.stereo`

`hotpot/cheminfo/structure/stereo.py`；显式 __all__: ()


<a id="hotpot-cheminfo-structure-stereo-stereoanchorkind"></a>
### `StereoAnchorKind`

定义 owner `hotpot.cheminfo.structure.stereo.StereoAnchorKind`；基类 `Enum`。

- `ATOM` = `'atom'` [枚举常量]
- `BOND` = `'bond'` [枚举常量]
- `AXIS` = `'axis'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-stereo-stereocarrierkind"></a>
### `StereoCarrierKind`

定义 owner `hotpot.cheminfo.structure.stereo.StereoCarrierKind`；基类 `Enum`。

- `ATOM` = `'atom'` [枚举常量]
- `IMPLICIT_H` = `'implicit_h'` [枚举常量]
- `LONE_PAIR` = `'lone_pair'` [枚举常量]
- `VACANCY` = `'vacancy'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-stereo-stereogroupkind"></a>
### `StereoGroupKind`

定义 owner `hotpot.cheminfo.structure.stereo.StereoGroupKind`；基类 `Enum`。

- `ABS` = `'abs'` [枚举常量]
- `AND` = `'and'` [枚举常量]
- `OR` = `'or'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-stereo-stereogroupref"></a>
### `StereoGroupRef`

定义 owner `hotpot.cheminfo.structure.stereo.StereoGroupRef`；基类 无。

```python
StereoGroupRef(scope: str, kind: StereoGroupKind, id: int)
```

- `scope: str` [只读值字段]
- `kind: StereoGroupKind` [只读值字段]
- `id: int` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereotype"></a>
### `StereoType`

定义 owner `hotpot.cheminfo.structure.stereo.StereoType`；基类 `Enum`。

- `TETRAHEDRAL` = `'tetrahedral'` [枚举常量]
- `DOUBLE_BOND` = `'double_bond'` [枚举常量]
- `ALLENE` = `'allene'` [枚举常量]
- `ATROPISOMERIC` = `'atropisomeric'` [枚举常量]
- `SQUARE_PLANAR` = `'square_planar'` [枚举常量]
- `TRIGONAL_BIPYRAMIDAL` = `'trigonal_bipyramidal'` [枚举常量]
- `SQUARE_PYRAMIDAL` = `'square_pyramidal'` [枚举常量]
- `OCTAHEDRAL` = `'octahedral'` [枚举常量]
- `UNKNOWN` = `'unknown'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-stereo-stereoissuecode"></a>
### `StereoIssueCode`

定义 owner `hotpot.cheminfo.structure.stereo.StereoIssueCode`；基类 `Enum`。

- `INVALID_ANCHOR` = `'invalid_anchor'` [枚举常量]
- `INVALID_CARRIER` = `'invalid_carrier'` [枚举常量]
- `INVALID_DESCRIPTOR` = `'invalid_descriptor'` [枚举常量]
- `NON_STEREOGENIC` = `'non_stereogenic'` [枚举常量]
- `DEGENERATE_GEOMETRY` = `'degenerate_geometry'` [枚举常量]
- `COORDINATE_CONTRADICTION` = `'coordinate_contradiction'` [枚举常量]
- `INCOMPLETE_MAPPING` = `'incomplete_mapping'` [枚举常量]
- `UNSUPPORTED` = `'unsupported'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-structure-stereo-stereocarrier"></a>
### `StereoCarrier`

定义 owner `hotpot.cheminfo.structure.stereo.StereoCarrier`；基类 无。

```python
StereoCarrier(kind: StereoCarrierKind, atom: int | None = None)
```

- `kind: StereoCarrierKind` [只读值字段]
- `atom: int | None` = `None` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereoanchor"></a>
### `StereoAnchor`

定义 owner `hotpot.cheminfo.structure.stereo.StereoAnchor`；基类 无。

```python
StereoAnchor(kind: StereoAnchorKind, atom_indices: tuple[int, ...])
```

- `kind: StereoAnchorKind` [只读值字段]
- `atom_indices: tuple[int, ...]` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereorecord"></a>
### `StereoRecord`

定义 owner `hotpot.cheminfo.structure.stereo.StereoRecord`；基类 无。

```python
StereoRecord(anchor: StereoAnchor, carriers: tuple[StereoCarrier, ...], descriptor: str, group: StereoGroupRef | None, source_namespace: str, source_code: str)
```

- `anchor: StereoAnchor` [只读值字段]
- `carriers: tuple[StereoCarrier, ...]` [只读值字段]
- `descriptor: str` [只读值字段]
- `group: StereoGroupRef | None` [只读值字段]
- `source_namespace: str` [只读值字段]
- `source_code: str` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-hydrogentransportstate"></a>
### `HydrogenTransportState`

定义 owner `hotpot.cheminfo.structure.stereo.HydrogenTransportState`；基类 无。

```python
HydrogenTransportState(implicit_hydrogens: int | None, bracket_explicit_hydrogens: int | None, no_implicit: bool | None)
```

- `implicit_hydrogens: int | None` [只读值字段]
- `bracket_explicit_hydrogens: int | None` [只读值字段]
- `no_implicit: bool | None` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-atomtransportrecord"></a>
### `AtomTransportRecord`

定义 owner `hotpot.cheminfo.structure.stereo.AtomTransportRecord`；基类 无。

```python
AtomTransportRecord(atom: int, isotope: int | None, radical_electrons: int | None, current_hydrogens: HydrogenTransportState | None, source_hydrogens: HydrogenTransportState | None, source_namespace: str | None)
```

- `atom: int` [只读值字段]
- `isotope: int | None` [只读值字段]
- `radical_electrons: int | None` [只读值字段]
- `current_hydrogens: HydrogenTransportState | None` [只读值字段]
- `source_hydrogens: HydrogenTransportState | None` [只读值字段]
- `source_namespace: str | None` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-bondtransportrecord"></a>
### `BondTransportRecord`

定义 owner `hotpot.cheminfo.structure.stereo.BondTransportRecord`；基类 无。

```python
BondTransportRecord(atom_indices: tuple[int, int], display_direction: str | None)
```

- `atom_indices: tuple[int, int]` [只读值字段]
- `display_direction: str | None` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereopayload"></a>
### `StereoPayload`

定义 owner `hotpot.cheminfo.structure.stereo.StereoPayload`；基类 无。

```python
StereoPayload(schema_version: int, records: tuple[StereoRecord, ...], atom_facts: tuple[AtomTransportRecord, ...], bond_facts: tuple[BondTransportRecord, ...])
```

- `schema_version: int` [只读值字段]
- `records: tuple[StereoRecord, ...]` [只读值字段]
- `atom_facts: tuple[AtomTransportRecord, ...]` [只读值字段]
- `bond_facts: tuple[BondTransportRecord, ...]` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereovalidationissue"></a>
### `StereoValidationIssue`

定义 owner `hotpot.cheminfo.structure.stereo.StereoValidationIssue`；基类 无。

```python
StereoValidationIssue(code: StereoIssueCode, message: str, record_index: int | None = None, atom_indices: tuple[int, ...] = ())
```

- `code: StereoIssueCode` [只读值字段]
- `message: str` [只读值字段]
- `record_index: int | None` = `None` [只读值字段]
- `atom_indices: tuple[int, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereovalidationreport"></a>
### `StereoValidationReport`

定义 owner `hotpot.cheminfo.structure.stereo.StereoValidationReport`；基类 无。

```python
StereoValidationReport(is_valid: bool, issues: tuple[StereoValidationIssue, ...])
```

- `is_valid: bool` [只读值字段]
- `issues: tuple[StereoValidationIssue, ...]` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereoderiveddescriptor"></a>
### `StereoDerivedDescriptor`

定义 owner `hotpot.cheminfo.structure.stereo.StereoDerivedDescriptor`；基类 无。

```python
StereoDerivedDescriptor(record_index: int, stereo_type: StereoType, label: str, system: str, is_absolute: bool)
```

- `record_index: int` [只读值字段]
- `stereo_type: StereoType` [只读值字段]
- `label: str` [只读值字段]
- `system: str` [只读值字段]
- `is_absolute: bool` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereoassignmentreport"></a>
### `StereoAssignmentReport`

定义 owner `hotpot.cheminfo.structure.stereo.StereoAssignmentReport`；基类 无。

```python
StereoAssignmentReport(payload: StereoPayload, unresolved: tuple[StereoValidationIssue, ...], derived: tuple[StereoDerivedDescriptor, ...] = ())
```

- `payload: StereoPayload` [只读值字段]
- `unresolved: tuple[StereoValidationIssue, ...]` [只读值字段]
- `derived: tuple[StereoDerivedDescriptor, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-structure-stereo-stereoalignmentreport"></a>
### `StereoAlignmentReport`

定义 owner `hotpot.cheminfo.structure.stereo.StereoAlignmentReport`；基类 无。

```python
StereoAlignmentReport(is_equivalent: bool, aligned_payload: StereoPayload, issues: tuple[StereoValidationIssue, ...])
```

- `is_equivalent: bool` [只读值字段]
- `aligned_payload: StereoPayload` [只读值字段]
- `issues: tuple[StereoValidationIssue, ...]` [只读值字段]


<a id="hotpot-cheminfo-structure-substructure"></a>
## `hotpot.cheminfo.structure.substructure`

`hotpot/cheminfo/structure/substructure.py`；显式 __all__: 无


<a id="hotpot-cheminfo-structure-substructure-smartssemantics"></a>
### `SmartsSemantics`

定义 owner `hotpot.cheminfo.structure.substructure.SmartsSemantics`；基类 `str`, `Enum`。

- `rings_for(self, atom)`
- `bonds_for(self, atom)`
- `degree(self, atom) -> int`
- `connectivity(self, atom) -> int`
- `valence(self, atom) -> float`
- `FULL_GRAPH` = `'full_graph'` [枚举常量]
- `LIGAND_SKELETON` = `'ligand_skeleton'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.


<a id="hotpot-cheminfo-structure-symmetry"></a>
## `hotpot.cheminfo.structure.symmetry`

`hotpot/cheminfo/structure/symmetry.py`；显式 __all__: ('SpaceGroup', 'SymmUnit')


<a id="hotpot-cheminfo-structure-symmetry-spacegroup"></a>
### `SpaceGroup`

定义 owner `hotpot.cheminfo.structure.symmetry.SpaceGroup`；基类 无。

- `__init__(self, number: int)`
- `__setattr__(self, name, value)`
- `__delattr__(self, name)`
- `from_operations(cls, operators, *, reference_number=None, to_reference=None)` [classmethod]
- `identity(self)`
- `lift(self, index: int, image=(0, 0, 0))`
- `contains(self, operation)`
- `__contains__(self, operation)`
- `change_setting(self, P, p)`
- `__eq__(self, other)`
- `__hash__(self)`
- `__repr__(self)`
- `__reduce__(self)`
- `__copy__(self)`
- `__deepcopy__(self, memo)`
- `number` [只读]
- `hermann_mauguin` [只读]
- `hall` [只读]
- `setting` [只读]
- `crystal_system` [只读]
- `point_group` [只读]
- `symm_units` [只读]
- `centering_translations` [只读]
- `translation_basis` [只读]
- `point_rotations` [只读]
- `cell_order` [只读]
- `point_group_order` [只读]

<a id="hotpot-cheminfo-structure-symmetry-symmunit"></a>
### `SymmUnit`

定义 owner `hotpot.cheminfo.structure.symmetry.SymmUnit`；基类 `ABC`。

- `from_affine(cls, R, t, *, context)` [classmethod]
- `from_xyz(cls, symbol: str, *, context)` [classmethod]
- `identity(cls, *, context)` [classmethod]
- `operate(self, fractional)`
- `compose(self, other)`
- `__matmul__(self, other)`
- `inverse(self)`
- `__invert__(self)`
- `power(self, n: int)`
- `__pow__(self, n)`
- `reduce_to_cell(self)`
- `__eq__(self, other)`
- `__hash__(self)`
- `__repr__(self)`
- `__reduce__(self)`
- `__copy__(self)`
- `__deepcopy__(self, memo)`
- `R` [只读]
- `t` [只读]
- `context` [只读]
- `xyz` [只读]
- `kind` [只读]
