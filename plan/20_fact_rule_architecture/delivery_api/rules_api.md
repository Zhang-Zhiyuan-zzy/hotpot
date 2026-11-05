# 科学规则、数学与只读合约：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot.cheminfo.reference`](#hotpot-cheminfo-reference)
- [`hotpot.cheminfo.reference.elements`](#hotpot-cheminfo-reference-elements)
- [`hotpot.cheminfo.reference.properties`](#hotpot-cheminfo-reference-properties)
- [`hotpot.cheminfo.reference.space_groups`](#hotpot-cheminfo-reference-space-groups)
- [`hotpot.cheminfo.rules`](#hotpot-cheminfo-rules)
- [`hotpot.cheminfo.rules.chemistry`](#hotpot-cheminfo-rules-chemistry)
- [`hotpot.cheminfo.rules.chemistry.bonding`](#hotpot-cheminfo-rules-chemistry-bonding)
- [`hotpot.cheminfo.rules.chemistry.bonding.aromaticity`](#hotpot-cheminfo-rules-chemistry-bonding-aromaticity)
- [`hotpot.cheminfo.rules.chemistry.bonding.bond_order`](#hotpot-cheminfo-rules-chemistry-bonding-bond-order)
- [`hotpot.cheminfo.rules.chemistry.bonding.formal_charge`](#hotpot-cheminfo-rules-chemistry-bonding-formal-charge)
- [`hotpot.cheminfo.rules.chemistry.bonding.hydrogen`](#hotpot-cheminfo-rules-chemistry-bonding-hydrogen)
- [`hotpot.cheminfo.rules.chemistry.bonding.kekule`](#hotpot-cheminfo-rules-chemistry-bonding-kekule)
- [`hotpot.cheminfo.rules.chemistry.bonding.valence`](#hotpot-cheminfo-rules-chemistry-bonding-valence)
- [`hotpot.cheminfo.rules.chemistry.crystal_identification`](#hotpot-cheminfo-rules-chemistry-crystal-identification)
- [`hotpot.cheminfo.rules.chemistry.crystal_matching`](#hotpot-cheminfo-rules-chemistry-crystal-matching)
- [`hotpot.cheminfo.rules.chemistry.descriptors`](#hotpot-cheminfo-rules-chemistry-descriptors)
- [`hotpot.cheminfo.rules.chemistry.descriptors.composition`](#hotpot-cheminfo-rules-chemistry-descriptors-composition)
- [`hotpot.cheminfo.rules.chemistry.descriptors.formula`](#hotpot-cheminfo-rules-chemistry-descriptors-formula)
- [`hotpot.cheminfo.rules.chemistry.descriptors.geometry`](#hotpot-cheminfo-rules-chemistry-descriptors-geometry)
- [`hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum`](#hotpot-cheminfo-rules-chemistry-descriptors-graph-spectrum)
- [`hotpot.cheminfo.rules.chemistry.descriptors.rings`](#hotpot-cheminfo-rules-chemistry-descriptors-rings)
- [`hotpot.cheminfo.rules.chemistry.elements`](#hotpot-cheminfo-rules-chemistry-elements)
- [`hotpot.cheminfo.rules.chemistry.mca_sites`](#hotpot-cheminfo-rules-chemistry-mca-sites)
- [`hotpot.cheminfo.rules.chemistry.scientific_state`](#hotpot-cheminfo-rules-chemistry-scientific-state)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.charge`](#hotpot-cheminfo-rules-chemistry-scientific-state-charge)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts`](#hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.charge.lewis`](#hotpot-cheminfo-rules-chemistry-scientific-state-charge-lewis)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.charge.net_charge`](#hotpot-cheminfo-rules-chemistry-scientific-state-charge-net-charge)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.composition`](#hotpot-cheminfo-rules-chemistry-scientific-state-composition)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.contracts`](#hotpot-cheminfo-rules-chemistry-scientific-state-contracts)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state`](#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts`](#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.inference`](#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-inference)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.neutral_atom`](#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-neutral-atom)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.identities`](#hotpot-cheminfo-rules-chemistry-scientific-state-identities)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.legality`](#hotpot-cheminfo-rules-chemistry-scientific-state-legality)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state`](#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts`](#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.coordination_counting`](#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-coordination-counting)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.inference`](#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-inference)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.registry`](#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-registry)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.americium_curium_iii`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-americium-curium-iii)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.anionic_x_ligands`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-anionic-x-ligands)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.base`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-base)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.f_block_hund`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-f-block-hund)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.ionic_configuration_spin`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-ionic-configuration-spin)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.lanthanide_iii`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-lanthanide-iii)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.main_group_closed_shell`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-main-group-closed-shell)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.monoatomic_ions`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-monoatomic-ions)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.neutral_l_donors`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-neutral-l-donors)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-oxidation-priors)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.thorium_iv`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-thorium-iv)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.trusted_history`](#hotpot-cheminfo-rules-chemistry-scientific-state-providers-trusted-history)
- [`hotpot.cheminfo.rules.chemistry.scientific_state.request`](#hotpot-cheminfo-rules-chemistry-scientific-state-request)
- [`hotpot.cheminfo.rules.chemistry.stereochemistry`](#hotpot-cheminfo-rules-chemistry-stereochemistry)
- [`hotpot.cheminfo.rules.chemistry.stereochemistry.algebra`](#hotpot-cheminfo-rules-chemistry-stereochemistry-algebra)
- [`hotpot.cheminfo.rules.chemistry.stereochemistry.alignment`](#hotpot-cheminfo-rules-chemistry-stereochemistry-alignment)
- [`hotpot.cheminfo.rules.chemistry.stereochemistry.assignment`](#hotpot-cheminfo-rules-chemistry-stereochemistry-assignment)
- [`hotpot.cheminfo.rules.chemistry.stereochemistry.coordinates`](#hotpot-cheminfo-rules-chemistry-stereochemistry-coordinates)
- [`hotpot.cheminfo.rules.chemistry.stereochemistry.graph`](#hotpot-cheminfo-rules-chemistry-stereochemistry-graph)
- [`hotpot.cheminfo.rules.chemistry.stereochemistry.validation`](#hotpot-cheminfo-rules-chemistry-stereochemistry-validation)
- [`hotpot.cheminfo.rules.chemistry.substructure`](#hotpot-cheminfo-rules-chemistry-substructure)
- [`hotpot.cheminfo.rules.chemistry.substructure._smarts_syntax`](#hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax)
- [`hotpot.cheminfo.rules.chemistry.substructure.errors`](#hotpot-cheminfo-rules-chemistry-substructure-errors)
- [`hotpot.cheminfo.rules.chemistry.substructure.predicates`](#hotpot-cheminfo-rules-chemistry-substructure-predicates)
- [`hotpot.cheminfo.rules.chemistry.substructure.semantics`](#hotpot-cheminfo-rules-chemistry-substructure-semantics)
- [`hotpot.cheminfo.rules.chemistry.substructure.smarts`](#hotpot-cheminfo-rules-chemistry-substructure-smarts)
- [`hotpot.cheminfo.rules.mathematics`](#hotpot-cheminfo-rules-mathematics)
- [`hotpot.cheminfo.rules.mathematics.cell`](#hotpot-cheminfo-rules-mathematics-cell)
- [`hotpot.cheminfo.rules.mathematics.geometry`](#hotpot-cheminfo-rules-mathematics-geometry)
- [`hotpot.cheminfo.rules.mathematics.geometry.linking`](#hotpot-cheminfo-rules-mathematics-geometry-linking)
- [`hotpot.cheminfo.rules.mathematics.geometry.native`](#hotpot-cheminfo-rules-mathematics-geometry-native)
- [`hotpot.cheminfo.rules.mathematics.geometry.object`](#hotpot-cheminfo-rules-mathematics-geometry-object)
- [`hotpot.cheminfo.rules.mathematics.geometry.relation`](#hotpot-cheminfo-rules-mathematics-geometry-relation)
- [`hotpot.cheminfo.rules.mathematics.geometry.settings`](#hotpot-cheminfo-rules-mathematics-geometry-settings)
- [`hotpot.cheminfo.rules.mathematics.graph`](#hotpot-cheminfo-rules-mathematics-graph)
- [`hotpot.cheminfo.rules.mathematics.graph.components`](#hotpot-cheminfo-rules-mathematics-graph-components)
- [`hotpot.cheminfo.rules.mathematics.graph.cycles`](#hotpot-cheminfo-rules-mathematics-graph-cycles)
- [`hotpot.cheminfo.rules.mathematics.graph.matrix`](#hotpot-cheminfo-rules-mathematics-graph-matrix)
- [`hotpot.cheminfo.rules.mathematics.graph.paths`](#hotpot-cheminfo-rules-mathematics-graph-paths)
- [`hotpot.cheminfo.rules.mathematics.graph.traversal`](#hotpot-cheminfo-rules-mathematics-graph-traversal)
- [`hotpot.cheminfo.rules.mathematics.graph_matching`](#hotpot-cheminfo-rules-mathematics-graph-matching)
- [`hotpot.cheminfo.rules.mathematics.internal_coordinates`](#hotpot-cheminfo-rules-mathematics-internal-coordinates)
- [`hotpot.cheminfo.rules.mathematics.periodic`](#hotpot-cheminfo-rules-mathematics-periodic)
- [`hotpot.cheminfo.rules.mathematics.sampling`](#hotpot-cheminfo-rules-mathematics-sampling)
- [`hotpot.cheminfo.rules.mathematics.symmetry`](#hotpot-cheminfo-rules-mathematics-symmetry)
- [`hotpot.cheminfo.rules.mathematics.topology`](#hotpot-cheminfo-rules-mathematics-topology)
- [`hotpot.cheminfo.rules.mathematics.view_geometry`](#hotpot-cheminfo-rules-mathematics-view-geometry)
- [`hotpot.cheminfo.rules.physics`](#hotpot-cheminfo-rules-physics)
- [`hotpot.cheminfo.rules.physics.forcefields`](#hotpot-cheminfo-rules-physics-forcefields)
- [`hotpot.cheminfo.rules.physics.forcefields.acceptance`](#hotpot-cheminfo-rules-physics-forcefields-acceptance)
- [`hotpot.cheminfo.rules.physics.forcefields.contracts`](#hotpot-cheminfo-rules-physics-forcefields-contracts)
- [`hotpot.cheminfo.rules.physics.forcefields.coordinates`](#hotpot-cheminfo-rules-physics-forcefields-coordinates)
- [`hotpot.cheminfo.rules.physics.forcefields.settings`](#hotpot-cheminfo-rules-physics-forcefields-settings)

<a id="hotpot-cheminfo-reference"></a>
## `hotpot.cheminfo.reference`

`hotpot/cheminfo/reference/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-reference-elements"></a>
## `hotpot.cheminfo.reference.elements`

`hotpot/cheminfo/reference/elements.py`；显式 __all__: ()



<a id="hotpot-cheminfo-reference-properties"></a>
## `hotpot.cheminfo.reference.properties`

`hotpot/cheminfo/reference/properties.py`；显式 __all__: ()



<a id="hotpot-cheminfo-reference-space-groups"></a>
## `hotpot.cheminfo.reference.space_groups`

`hotpot/cheminfo/reference/space_groups.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules"></a>
## `hotpot.cheminfo.rules`

`hotpot/cheminfo/rules/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry"></a>
## `hotpot.cheminfo.rules.chemistry`

`hotpot/cheminfo/rules/chemistry/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-chemistry-bonding"></a>
## `hotpot.cheminfo.rules.chemistry.bonding`

`hotpot/cheminfo/rules/chemistry/bonding/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-bonding-aromaticity"></a>
## `hotpot.cheminfo.rules.chemistry.bonding.aromaticity`

`hotpot/cheminfo/rules/chemistry/bonding/aromaticity.py`；显式 __all__: ()


模块值/类型别名（声明存在不构成事实写许可）：

- `AROMATIC_PLANARITY_RELATIVE_TOLERANCE` = `0.03`


<a id="hotpot-cheminfo-rules-chemistry-bonding-bond-order"></a>
## `hotpot.cheminfo.rules.chemistry.bonding.bond_order`

`hotpot/cheminfo/rules/chemistry/bonding/bond_order.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-bonding-formal-charge"></a>
## `hotpot.cheminfo.rules.chemistry.bonding.formal_charge`

`hotpot/cheminfo/rules/chemistry/bonding/formal_charge.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-bonding-hydrogen"></a>
## `hotpot.cheminfo.rules.chemistry.bonding.hydrogen`

`hotpot/cheminfo/rules/chemistry/bonding/hydrogen.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-bonding-kekule"></a>
## `hotpot.cheminfo.rules.chemistry.bonding.kekule`

`hotpot/cheminfo/rules/chemistry/bonding/kekule.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-bonding-valence"></a>
## `hotpot.cheminfo.rules.chemistry.bonding.valence`

`hotpot/cheminfo/rules/chemistry/bonding/valence.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-crystal-identification"></a>
## `hotpot.cheminfo.rules.chemistry.crystal_identification`

`hotpot/cheminfo/rules/chemistry/crystal_identification.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-crystal-matching"></a>
## `hotpot.cheminfo.rules.chemistry.crystal_matching`

`hotpot/cheminfo/rules/chemistry/crystal_matching.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-descriptors"></a>
## `hotpot.cheminfo.rules.chemistry.descriptors`

`hotpot/cheminfo/rules/chemistry/descriptors/__init__.py`；显式 __all__: ('GraphSpectrum', 'atoms_electron_configurations', 'calc_electron_config', 'calc_spectrum')

- 导出 `GraphSpectrum` → `hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum.GraphSpectrum`
- 导出 `atoms_electron_configurations` → `hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum.atoms_electron_configurations`；`atoms_electron_configurations(atomic_numbers: np.ndarray, length: int=4) -> np.ndarray`
- 导出 `calc_electron_config` → `hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum.calc_electron_config`；`calc_electron_config(atomic_number: int, length: int=4) -> (int, list)`
- 导出 `calc_spectrum` → `hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum.calc_spectrum`；`calc_spectrum(adj: types.ArrayLike, atomic_numbers: types.ArrayLike, length: int=4) -> np.ndarray`


<a id="hotpot-cheminfo-rules-chemistry-descriptors-composition"></a>
## `hotpot.cheminfo.rules.chemistry.descriptors.composition`

`hotpot/cheminfo/rules/chemistry/descriptors/composition.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-descriptors-formula"></a>
## `hotpot.cheminfo.rules.chemistry.descriptors.formula`

`hotpot/cheminfo/rules/chemistry/descriptors/formula.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-descriptors-geometry"></a>
## `hotpot.cheminfo.rules.chemistry.descriptors.geometry`

`hotpot/cheminfo/rules/chemistry/descriptors/geometry.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-descriptors-graph-spectrum"></a>
## `hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum`

`hotpot/cheminfo/rules/chemistry/descriptors/graph_spectrum.py`；显式 __all__: ('calc_electron_config', 'atoms_electron_configurations', 'calc_spectrum', 'GraphSpectrum')


<a id="hotpot-cheminfo-rules-chemistry-descriptors-graph-spectrum-calc-electron-config"></a>
### `calc_electron_config`

```python
calc_electron_config(atomic_number: int, length: int=4) -> (int, list)
```

<a id="hotpot-cheminfo-rules-chemistry-descriptors-graph-spectrum-atoms-electron-configurations"></a>
### `atoms_electron_configurations`

```python
atoms_electron_configurations(atomic_numbers: np.ndarray, length: int=4) -> np.ndarray
```

<a id="hotpot-cheminfo-rules-chemistry-descriptors-graph-spectrum-calc-spectrum"></a>
### `calc_spectrum`

```python
calc_spectrum(adj: types.ArrayLike, atomic_numbers: types.ArrayLike, length: int=4) -> np.ndarray
```

<a id="hotpot-cheminfo-rules-chemistry-descriptors-graph-spectrum-graphspectrum"></a>
### `GraphSpectrum`

定义 owner `hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum.GraphSpectrum`；基类 无。

- `__init__(self, spectrum: np.ndarray, norm: Literal['infinite', 'min', 'l1', 'l2']='l2')`
- `__or__(self, other: 'GraphSpectrum')`
- `similarity(self, other: 'GraphSpectrum')`
- `from_adj_atoms(cls, adj: np.ndarray, atomic_numbers: np.ndarray, length: int=4, norm: Literal['infinite', 'min', 'l1', 'l2']='l2') -> 'GraphSpectrum'` [classmethod]
- `vectors` [只读]
- `width` [只读]: `int`
- `spectrum` = `spectrum` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `norm` = `norm` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-rules-chemistry-descriptors-rings"></a>
## `hotpot.cheminfo.rules.chemistry.descriptors.rings`

`hotpot/cheminfo/rules/chemistry/descriptors/rings.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-elements"></a>
## `hotpot.cheminfo.rules.chemistry.elements`

`hotpot/cheminfo/rules/chemistry/elements.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-mca-sites"></a>
## `hotpot.cheminfo.rules.chemistry.mca_sites`

`hotpot/cheminfo/rules/chemistry/mca_sites/__init__.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rules-chemistry-mca-sites-detectedsite"></a>
### `DetectedSite`

定义 owner `hotpot.cheminfo.rules.chemistry.mca_sites.DetectedSite`；基类 无。

```python
DetectedSite(atom_index: int, site_type: str)
```

- `atom_index: int` [只读值字段]
- `site_type: str` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-mca-sites-name-sites"></a>
### `name_sites`

```python
name_sites(rule_sources, eligibility) -> tuple[DetectedSite, ...]
```

<a id="hotpot-cheminfo-rules-chemistry-mca-sites-validate-target-domain"></a>
### `validate_target_domain`

```python
validate_target_domain(atomic_numbers: tuple[int, ...], atom_formal_charges: tuple[int, ...], formal_charge: int, position: int, allow_charged: bool=False) -> None
```

模块值/类型别名（声明存在不构成事实写许可）：

- `NUCLEOPHILE_RULES` = `<长模块值；见 hotpot/cheminfo/rules/chemistry/mca_sites/__init__.py:12>`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state`

`hotpot/cheminfo/rules/chemistry/scientific_state/__init__.py`；显式 __all__: ('ScientificQuantity', 'ScientificAdmissibility', 'ResolutionStatus', 'CoverageStatus', 'StateValueAuthority', 'ScientificFinding', 'CandidateProvenance', 'ProviderValue', 'CandidateBudget', 'DEFAULT_CANDIDATE_BUDGET', 'ScientificOperationRequirements')

- 导出 `ScientificQuantity` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificQuantity`
- 导出 `ScientificAdmissibility` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificAdmissibility`
- 导出 `ResolutionStatus` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ResolutionStatus`
- 导出 `CoverageStatus` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.CoverageStatus`
- 导出 `StateValueAuthority` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.StateValueAuthority`
- 导出 `ScientificFinding` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificFinding`
- 导出 `CandidateProvenance` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.CandidateProvenance`
- 导出 `ProviderValue` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ProviderValue`
- 导出 `CandidateBudget` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.CandidateBudget`
- 导出 `DEFAULT_CANDIDATE_BUDGET` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.DEFAULT_CANDIDATE_BUDGET`
- 导出 `ScientificOperationRequirements` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificOperationRequirements`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.charge`

`hotpot/cheminfo/rules/chemistry/scientific_state/charge/__init__.py`；显式 __all__: ('NetChargeCandidateEvidence', 'AtomFormalChargeEvidence', 'ComponentChargeEvidence', 'NetChargeResult', 'LewisFormalChargeCandidate', 'LewisFormalChargeResult')

- 导出 `NetChargeCandidateEvidence` → `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.NetChargeCandidateEvidence`
- 导出 `AtomFormalChargeEvidence` → `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.AtomFormalChargeEvidence`
- 导出 `ComponentChargeEvidence` → `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.ComponentChargeEvidence`
- 导出 `NetChargeResult` → `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.NetChargeResult`
- 导出 `LewisFormalChargeCandidate` → `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.LewisFormalChargeCandidate`
- 导出 `LewisFormalChargeResult` → `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.LewisFormalChargeResult`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts`

`hotpot/cheminfo/rules/chemistry/scientific_state/charge/contracts.py`；显式 __all__: ('NetChargeCandidateEvidence', 'AtomFormalChargeEvidence', 'ComponentChargeEvidence', 'NetChargeResult', 'LewisFormalChargeCandidate', 'LewisFormalChargeResult')


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts-netchargecandidateevidence"></a>
### `NetChargeCandidateEvidence`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.NetChargeCandidateEvidence`；基类 无。

```python
NetChargeCandidateEvidence(provenance: CandidateProvenance, charge: int)
```

- `provenance: CandidateProvenance` [只读值字段]
- `charge: int` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts-atomformalchargeevidence"></a>
### `AtomFormalChargeEvidence`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.AtomFormalChargeEvidence`；基类 无。

```python
AtomFormalChargeEvidence(provenance: CandidateProvenance, charges: Tuple[Optional[int], ...], coverage: CoverageStatus)
```

- `provenance: CandidateProvenance` [只读值字段]
- `charges: Tuple[Optional[int], ...]` [只读值字段]
- `coverage: CoverageStatus` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts-componentchargeevidence"></a>
### `ComponentChargeEvidence`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.ComponentChargeEvidence`；基类 无。

```python
ComponentChargeEvidence(provenance: CandidateProvenance, component_charges: Tuple[Tuple[str, int], ...], coverage: CoverageStatus)
```

- `provenance: CandidateProvenance` [只读值字段]
- `component_charges: Tuple[Tuple[str, int], ...]` [只读值字段]
- `coverage: CoverageStatus` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts-netchargeresult"></a>
### `NetChargeResult`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.NetChargeResult`；基类 无。

```python
NetChargeResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[NetChargeCandidateEvidence, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
```

- `identity: str` [只读值字段]
- `facts_identity: MolecularFactsIdentity` [只读值字段]
- `candidates: Tuple[NetChargeCandidateEvidence, ...]` [只读值字段]
- `admissibility: ScientificAdmissibility` [只读值字段]
- `resolution: ResolutionStatus` [只读值字段]
- `findings: Tuple[ScientificFinding, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts-lewisformalchargecandidate"></a>
### `LewisFormalChargeCandidate`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.LewisFormalChargeCandidate`；基类 无。

```python
LewisFormalChargeCandidate(provenance: CandidateProvenance, net_charge_candidate_identity: str, atom_formal_charges: Tuple[int, ...], bond_orders: Tuple[float, ...], nonbonding_electrons: Tuple[int, ...], radical_electrons: Tuple[int, ...], component_charges: Tuple[Tuple[str, int], ...])
```

- `provenance: CandidateProvenance` [只读值字段]
- `net_charge_candidate_identity: str` [只读值字段]
- `atom_formal_charges: Tuple[int, ...]` [只读值字段]
- `bond_orders: Tuple[float, ...]` [只读值字段]
- `nonbonding_electrons: Tuple[int, ...]` [只读值字段]
- `radical_electrons: Tuple[int, ...]` [只读值字段]
- `component_charges: Tuple[Tuple[str, int], ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts-lewisformalchargeresult"></a>
### `LewisFormalChargeResult`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts.LewisFormalChargeResult`；基类 无。

```python
LewisFormalChargeResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[LewisFormalChargeCandidate, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
```

- `identity: str` [只读值字段]
- `facts_identity: MolecularFactsIdentity` [只读值字段]
- `candidates: Tuple[LewisFormalChargeCandidate, ...]` [只读值字段]
- `admissibility: ScientificAdmissibility` [只读值字段]
- `resolution: ResolutionStatus` [只读值字段]
- `findings: Tuple[ScientificFinding, ...]` [只读值字段]


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-lewis"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.charge.lewis`

`hotpot/cheminfo/rules/chemistry/scientific_state/charge/lewis.py`；显式 __all__: ('solve_main_group_lewis',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-lewis-solve-main-group-lewis"></a>
### `solve_main_group_lewis`

```python
solve_main_group_lewis(facts: MolecularFacts, net_charge: NetChargeResult, *, candidate_budget: CandidateBudget=DEFAULT_CANDIDATE_BUDGET) -> Tuple[LewisFormalChargeResult, ...]
```


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-net-charge"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.charge.net_charge`

`hotpot/cheminfo/rules/chemistry/scientific_state/charge/net_charge.py`；显式 __all__: ('resolve_net_charge',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-charge-net-charge-resolve-net-charge"></a>
### `resolve_net_charge`

```python
resolve_net_charge(facts: MolecularFacts, request: StateRequest=StateRequest(), *, provided: Tuple[NetChargeResult, ...]=()) -> Tuple[NetChargeResult, ...]
```


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.composition`

`hotpot/cheminfo/rules/chemistry/scientific_state/composition.py`；显式 __all__: ('CandidateDomainScope', 'MaterializationStatus', 'CandidateSearchStatus', 'StateValueOrigin', 'ScientificStateIdentity', 'FactorKey', 'ConstraintValue', 'CandidateConstraint', 'FrontierBound', 'CandidateDomainSummary', 'RejectedPathSummary', 'BranchAssignment', 'StateValueRef', 'DerivedStateValue', 'StateCandidate', 'FactorCandidateValue', 'FactorDomainContribution', 'CandidateFactor', 'ProposedCandidateValue', 'ProposedCandidateFactor', 'ConstraintGuard', 'GuardedCandidateConstraint', 'CandidateProposal', 'CoupledScientificStateCandidate', 'ScientificStateCandidateSpace', 'ScientificResolutionResult')


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-candidatedomainscope"></a>
### `CandidateDomainScope`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateDomainScope`；基类 `Enum`。

- `EXHAUSTIVE` = `'exhaustive'` [枚举常量]
- `DOMAIN_BOUNDED` = `'domain_bounded'` [枚举常量]
- `OPEN` = `'open'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-materializationstatus"></a>
### `MaterializationStatus`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.MaterializationStatus`；基类 `Enum`。

- `COMPLETE` = `'complete'` [枚举常量]
- `RESOURCE_TRUNCATED` = `'resource_truncated'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-candidatesearchstatus"></a>
### `CandidateSearchStatus`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateSearchStatus`；基类 `Enum`。

- `NOT_EVALUATED` = `'not_evaluated'` [枚举常量]
- `DECISION_CERTIFIED` = `'decision_certified'` [枚举常量]
- `TRUNCATED_BUT_DECISION_CERTIFIED` = `'truncated_but_decision_certified'` [枚举常量]
- `NO_CERTIFIED_DECISION` = `'no_certified_decision'` [枚举常量]
- `RESOURCE_LIMIT_NO_CERTIFIED_DECISION` = `'resource_limit_no_certified_decision'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-statevalueorigin"></a>
### `StateValueOrigin`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.StateValueOrigin`；基类 `Enum`。

- `RESULT` = `'result'` [枚举常量]
- `DERIVED_CONSTRAINT` = `'derived_constraint'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-scientificstateidentity"></a>
### `ScientificStateIdentity`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ScientificStateIdentity`；基类 无。

```python
ScientificStateIdentity(digest: str)
```

- `digest: str` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-factorkey"></a>
### `FactorKey`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FactorKey`；基类 无。

```python
FactorKey(quantity: ScientificQuantity, subject_identity: str)
```

- `quantity: ScientificQuantity` [只读值字段]
- `subject_identity: str` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-candidateconstraint"></a>
### `CandidateConstraint`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateConstraint`；基类 无。

```python
CandidateConstraint(identity: str, constraint_type: str, schema_version: str, factor_keys: Tuple[FactorKey, ...], parameters: Tuple[Tuple[str, ConstraintValue], ...])
```

- `identity: str` [只读值字段]
- `constraint_type: str` [只读值字段]
- `schema_version: str` [只读值字段]
- `factor_keys: Tuple[FactorKey, ...]` [只读值字段]
- `parameters: Tuple[Tuple[str, ConstraintValue], ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-frontierbound"></a>
### `FrontierBound`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FrontierBound`；基类 无。

```python
FrontierBound(branch_identity: str, schema_version: str, optimistic_key: Tuple[str, ...])
```

- `branch_identity: str` [只读值字段]
- `schema_version: str` [只读值字段]
- `optimistic_key: Tuple[str, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-candidatedomainsummary"></a>
### `CandidateDomainSummary`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateDomainSummary`；基类 无。

```python
CandidateDomainSummary(remaining_count_upper_bound: Optional[int], continuation_identity: Optional[str], frontier_bound: Optional[FrontierBound])
```

- `remaining_count_upper_bound: Optional[int]` [只读值字段]
- `continuation_identity: Optional[str]` [只读值字段]
- `frontier_bound: Optional[FrontierBound]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-rejectedpathsummary"></a>
### `RejectedPathSummary`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.RejectedPathSummary`；基类 无。

```python
RejectedPathSummary(candidate_identities: Tuple[str, ...], constraint_identity: str, reason_codes: Tuple[str, ...])
```

- `candidate_identities: Tuple[str, ...]` [只读值字段]
- `constraint_identity: str` [只读值字段]
- `reason_codes: Tuple[str, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-statevalueref"></a>
### `StateValueRef`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.StateValueRef`；基类 无。

```python
StateValueRef(quantity: ScientificQuantity, value: int, origin: StateValueOrigin, origin_identity: str)
```

- `quantity: ScientificQuantity` [只读值字段]
- `value: int` [只读值字段]
- `origin: StateValueOrigin` [只读值字段]
- `origin_identity: str` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-derivedstatevalue"></a>
### `DerivedStateValue`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.DerivedStateValue`；基类 无。

```python
DerivedStateValue(provenance: CandidateProvenance, quantity: ScientificQuantity, value: int, equation_id: str, parent_candidate_ids: Tuple[str, ...], counting_convention: CountingConvention)
```

- `provenance: CandidateProvenance` [只读值字段]
- `quantity: ScientificQuantity` [只读值字段]
- `value: int` [只读值字段]
- `equation_id: str` [只读值字段]
- `parent_candidate_ids: Tuple[str, ...]` [只读值字段]
- `counting_convention: CountingConvention` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-branchassignment"></a>
### `BranchAssignment`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.BranchAssignment`；基类 无。

```python
BranchAssignment(group_id: str, option_id: str)
```

- `group_id: str` [只读值字段]
- `option_id: str` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-factorcandidatevalue"></a>
### `FactorCandidateValue`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FactorCandidateValue`；基类 无。

```python
FactorCandidateValue(candidate: StateCandidate, source_proposal_ids: Tuple[str, ...], branch_assignments: Tuple[BranchAssignment, ...])
```

- `candidate: StateCandidate` [只读值字段]
- `source_proposal_ids: Tuple[str, ...]` [只读值字段]
- `branch_assignments: Tuple[BranchAssignment, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-factordomaincontribution"></a>
### `FactorDomainContribution`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.FactorDomainContribution`；基类 无。

```python
FactorDomainContribution(proposal_id: str, domain_scope: CandidateDomainScope, materialization_status: MaterializationStatus, unexpanded_domain_summary: Optional[CandidateDomainSummary])
```

- `proposal_id: str` [只读值字段]
- `domain_scope: CandidateDomainScope` [只读值字段]
- `materialization_status: MaterializationStatus` [只读值字段]
- `unexpanded_domain_summary: Optional[CandidateDomainSummary]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-candidatefactor"></a>
### `CandidateFactor`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateFactor`；基类 无。

```python
CandidateFactor(identity: str, factor_key: FactorKey, domain_definition_id: str, values: Tuple[FactorCandidateValue, ...], domain_contributions: Tuple[FactorDomainContribution, ...], unexpanded_domain_summary: Optional[CandidateDomainSummary], domain_scope: CandidateDomainScope, materialization_status: MaterializationStatus)
```

- `identity: str` [只读值字段]
- `factor_key: FactorKey` [只读值字段]
- `domain_definition_id: str` [只读值字段]
- `values: Tuple[FactorCandidateValue, ...]` [只读值字段]
- `domain_contributions: Tuple[FactorDomainContribution, ...]` [只读值字段]
- `unexpanded_domain_summary: Optional[CandidateDomainSummary]` [只读值字段]
- `domain_scope: CandidateDomainScope` [只读值字段]
- `materialization_status: MaterializationStatus` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-proposedcandidatevalue"></a>
### `ProposedCandidateValue`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ProposedCandidateValue`；基类 无。

```python
ProposedCandidateValue(candidate: StateCandidate, branch_assignments: Tuple[BranchAssignment, ...] = ())
```

- `candidate: StateCandidate` [只读值字段]
- `branch_assignments: Tuple[BranchAssignment, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-proposedcandidatefactor"></a>
### `ProposedCandidateFactor`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ProposedCandidateFactor`；基类 无。

```python
ProposedCandidateFactor(factor_key: FactorKey, domain_definition_id: str, materialized_candidates: Tuple[ProposedCandidateValue, ...], domain_scope: CandidateDomainScope, materialization_status: MaterializationStatus, unexpanded_domain_summary: Optional[CandidateDomainSummary])
```

- `factor_key: FactorKey` [只读值字段]
- `domain_definition_id: str` [只读值字段]
- `materialized_candidates: Tuple[ProposedCandidateValue, ...]` [只读值字段]
- `domain_scope: CandidateDomainScope` [只读值字段]
- `materialization_status: MaterializationStatus` [只读值字段]
- `unexpanded_domain_summary: Optional[CandidateDomainSummary]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-constraintguard"></a>
### `ConstraintGuard`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ConstraintGuard`；基类 无。

```python
ConstraintGuard(all_candidate_ids: Tuple[str, ...] = (), branch_assignments: Tuple[BranchAssignment, ...] = ())
```

- `all_candidate_ids: Tuple[str, ...]` = `()` [只读值字段]
- `branch_assignments: Tuple[BranchAssignment, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-guardedcandidateconstraint"></a>
### `GuardedCandidateConstraint`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.GuardedCandidateConstraint`；基类 无。

```python
GuardedCandidateConstraint(guard: ConstraintGuard, constraint: CandidateConstraint)
```

- `guard: ConstraintGuard` [只读值字段]
- `constraint: CandidateConstraint` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-candidateproposal"></a>
### `CandidateProposal`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CandidateProposal`；基类 无。

```python
CandidateProposal(proposal_id: str, factors: Tuple[ProposedCandidateFactor, ...], constraints: Tuple[GuardedCandidateConstraint, ...])
```

- `proposal_id: str` [只读值字段]
- `factors: Tuple[ProposedCandidateFactor, ...]` [只读值字段]
- `constraints: Tuple[GuardedCandidateConstraint, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-coupledscientificstatecandidate"></a>
### `CoupledScientificStateCandidate`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.CoupledScientificStateCandidate`；基类 无。

```python
CoupledScientificStateCandidate(identity: ScientificStateIdentity, net_charge: StateValueRef, component_assignment_ids: Tuple[str, ...], coordination_assignment_ids: Tuple[str, ...], oxidation_state_assignment_id: Optional[str], electronic_state_result_id: str, electronic_state_candidate_identity: Optional[str], derived_values: Tuple[DerivedStateValue, ...], evidence_ids: Tuple[str, ...], assumption_ids: Tuple[str, ...])
```

- `identity: ScientificStateIdentity` [只读值字段]
- `net_charge: StateValueRef` [只读值字段]
- `component_assignment_ids: Tuple[str, ...]` [只读值字段]
- `coordination_assignment_ids: Tuple[str, ...]` [只读值字段]
- `oxidation_state_assignment_id: Optional[str]` [只读值字段]
- `electronic_state_result_id: str` [只读值字段]
- `electronic_state_candidate_identity: Optional[str]` [只读值字段]
- `derived_values: Tuple[DerivedStateValue, ...]` [只读值字段]
- `evidence_ids: Tuple[str, ...]` [只读值字段]
- `assumption_ids: Tuple[str, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-scientificstatecandidatespace"></a>
### `ScientificStateCandidateSpace`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ScientificStateCandidateSpace`；基类 无。

```python
ScientificStateCandidateSpace(factors: Tuple[CandidateFactor, ...], constraints: Tuple[CandidateConstraint, ...], candidate_budget: CandidateBudget, candidate_budget_artifact_id: str, frontier_bound_schema_version: str, expanded_paths: Tuple[CoupledScientificStateCandidate, ...], rejected_path_summaries: Tuple[RejectedPathSummary, ...], frontier: Tuple[FrontierBound, ...], search_status: CandidateSearchStatus)
```

- `factors: Tuple[CandidateFactor, ...]` [只读值字段]
- `constraints: Tuple[CandidateConstraint, ...]` [只读值字段]
- `candidate_budget: CandidateBudget` [只读值字段]
- `candidate_budget_artifact_id: str` [只读值字段]
- `frontier_bound_schema_version: str` [只读值字段]
- `expanded_paths: Tuple[CoupledScientificStateCandidate, ...]` [只读值字段]
- `rejected_path_summaries: Tuple[RejectedPathSummary, ...]` [只读值字段]
- `frontier: Tuple[FrontierBound, ...]` [只读值字段]
- `search_status: CandidateSearchStatus` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-composition-scientificresolutionresult"></a>
### `ScientificResolutionResult`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.composition.ScientificResolutionResult`；基类 无。

```python
ScientificResolutionResult(status: ResolutionStatus, exhaustive: bool, equivalent_state_identities: Tuple[ScientificStateIdentity, ...])
```

- `status: ResolutionStatus` [只读值字段]
- `exhaustive: bool` [只读值字段]
- `equivalent_state_identities: Tuple[ScientificStateIdentity, ...]` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `ConstraintValue` = `Union[None, bool, int, float, str, Tuple['ConstraintValue', ...]]`
- `StateCandidate` = `Union[NetChargeCandidateEvidence, LewisFormalChargeCandidate, CoordinationCountingCandidate, OxidationStateCandidateEvidence, SpinContributionCandidate, ElectronicStateCandidate, DerivedStateValue]`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.contracts`

`hotpot/cheminfo/rules/chemistry/scientific_state/contracts.py`；显式 __all__: ('ScientificQuantity', 'ScientificAdmissibility', 'ResolutionStatus', 'CoverageStatus', 'StateValueAuthority', 'ScientificFinding', 'CandidateProvenance', 'ProviderValue', 'CandidateBudget', 'DEFAULT_CANDIDATE_BUDGET', 'ScientificOperationRequirements')


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-scientificquantity"></a>
### `ScientificQuantity`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificQuantity`；基类 `Enum`。

- `NET_CHARGE` = `'net_charge'` [枚举常量]
- `LEWIS_FORMAL_CHARGE` = `'lewis_formal_charge'` [枚举常量]
- `COORDINATION_COUNTING` = `'coordination_counting'` [枚举常量]
- `OXIDATION_STATE` = `'oxidation_state'` [枚举常量]
- `ELECTRONIC_STATE` = `'electronic_state'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-scientificadmissibility"></a>
### `ScientificAdmissibility`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificAdmissibility`；基类 `Enum`。

- `ADMISSIBLE` = `'admissible'` [枚举常量]
- `INADMISSIBLE` = `'inadmissible'` [枚举常量]
- `NOT_EVALUATED` = `'not_evaluated'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-resolutionstatus"></a>
### `ResolutionStatus`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ResolutionStatus`；基类 `Enum`。

- `UNIQUE` = `'unique'` [枚举常量]
- `AMBIGUOUS` = `'ambiguous'` [枚举常量]
- `UNRESOLVED` = `'unresolved'` [枚举常量]
- `OUT_OF_DOMAIN` = `'out_of_domain'` [枚举常量]
- `CONFLICT` = `'conflict'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-coveragestatus"></a>
### `CoverageStatus`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.CoverageStatus`；基类 `Enum`。

- `COMPLETE` = `'complete'` [枚举常量]
- `PARTIAL` = `'partial'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-statevalueauthority"></a>
### `StateValueAuthority`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.StateValueAuthority`；基类 `Enum`。

- `VERIFIED_EXPLICIT` = `'verified_explicit'` [枚举常量]
- `VERIFIED_SCIENTIFIC_SOURCE` = `'verified_scientific_source'` [枚举常量]
- `EXACT_IDENTITY_TRUSTED_SOURCE` = `'exact_identity_trusted_source'` [枚举常量]
- `DOMAIN_RULE` = `'domain_rule'` [枚举常量]
- `DERIVED_CONSTRAINT` = `'derived_constraint'` [枚举常量]
- `UNRESOLVED` = `'unresolved'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-scientificfinding"></a>
### `ScientificFinding`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificFinding`；基类 无。

```python
ScientificFinding(code: str, message: str, related_identities: Tuple[str, ...] = ())
```

- `code: str` [只读值字段]
- `message: str` [只读值字段]
- `related_identities: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-candidateprovenance"></a>
### `CandidateProvenance`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.CandidateProvenance`；基类 无。

```python
CandidateProvenance(identity: str, authority: StateValueAuthority, source_identity: str, evidence_ids: Tuple[str, ...] = (), assumption_ids: Tuple[str, ...] = ())
```

- `identity: str` [只读值字段]
- `authority: StateValueAuthority` [只读值字段]
- `source_identity: str` [只读值字段]
- `evidence_ids: Tuple[str, ...]` = `()` [只读值字段]
- `assumption_ids: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-candidatebudget"></a>
### `CandidateBudget`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.CandidateBudget`；基类 无。

```python
CandidateBudget(max_materialized_values_per_factor: int, max_expanded_paths: int, max_frontier_paths: int)
```

- `max_materialized_values_per_factor: int` [只读值字段]
- `max_expanded_paths: int` [只读值字段]
- `max_frontier_paths: int` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-contracts-scientificoperationrequirements"></a>
### `ScientificOperationRequirements`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ScientificOperationRequirements`；基类 无。

```python
ScientificOperationRequirements(operation_id: str, required_resolved_quantities: frozenset[ScientificQuantity], allowed_unresolved_quantities: frozenset[ScientificQuantity])
```

- `operation_id: str` [只读值字段]
- `required_resolved_quantities: frozenset[ScientificQuantity]` [只读值字段]
- `allowed_unresolved_quantities: frozenset[ScientificQuantity]` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `ProviderValue` = `Union[None, bool, int, float, str, Tuple['ProviderValue', ...]]`
- `DEFAULT_CANDIDATE_BUDGET` = `CandidateBudget(max_materialized_values_per_factor=64, max_expanded_paths=4096, max_frontier_paths=1024)`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state`

`hotpot/cheminfo/rules/chemistry/scientific_state/electronic_state/__init__.py`；显式 __all__: ('ElectronicStateCandidateEvidence', 'SpinContributionScope', 'SpinContributionCandidate', 'ElectronicStateCandidate', 'ElectronicStateInferenceResult', 'NeutralAtomReference')

- 导出 `ElectronicStateCandidateEvidence` → `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.ElectronicStateCandidateEvidence`
- 导出 `SpinContributionScope` → `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.SpinContributionScope`
- 导出 `SpinContributionCandidate` → `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.SpinContributionCandidate`
- 导出 `ElectronicStateCandidate` → `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.ElectronicStateCandidate`
- 导出 `ElectronicStateInferenceResult` → `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.ElectronicStateInferenceResult`
- 导出 `NeutralAtomReference` → `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.NeutralAtomReference`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts`

`hotpot/cheminfo/rules/chemistry/scientific_state/electronic_state/contracts.py`；显式 __all__: ('ElectronicStateCandidateEvidence', 'SpinContributionScope', 'SpinContributionCandidate', 'ElectronicStateCandidate', 'ElectronicStateInferenceResult', 'NeutralAtomReference')


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts-spincontributionscope"></a>
### `SpinContributionScope`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.SpinContributionScope`；基类 `Enum`。

- `MOLECULAR` = `'molecular'` [枚举常量]
- `COMPONENT` = `'component'` [枚举常量]
- `ATOM` = `'atom'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts-electronicstatecandidateevidence"></a>
### `ElectronicStateCandidateEvidence`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.ElectronicStateCandidateEvidence`；基类 无。

```python
ElectronicStateCandidateEvidence(provenance: CandidateProvenance, two_s: int, multiplicity: Optional[int], pure_spin: bool)
```

- `provenance: CandidateProvenance` [只读值字段]
- `two_s: int` [只读值字段]
- `multiplicity: Optional[int]` [只读值字段]
- `pure_spin: bool` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts-spincontributioncandidate"></a>
### `SpinContributionCandidate`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.SpinContributionCandidate`；基类 无。

```python
SpinContributionCandidate(provenance: CandidateProvenance, scope: SpinContributionScope, subject_identity: str, parent_candidate_ids: Tuple[str, ...], two_s: int, pure_spin: bool)
```

- `provenance: CandidateProvenance` [只读值字段]
- `scope: SpinContributionScope` [只读值字段]
- `subject_identity: str` [只读值字段]
- `parent_candidate_ids: Tuple[str, ...]` [只读值字段]
- `two_s: int` [只读值字段]
- `pure_spin: bool` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts-electronicstatecandidate"></a>
### `ElectronicStateCandidate`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.ElectronicStateCandidate`；基类 无。

```python
ElectronicStateCandidate(provenance: CandidateProvenance, net_charge_candidate_identity: str, oxidation_state_candidate_identity: Optional[str], electron_count: int, two_s: int, multiplicity: Optional[int], pure_spin: bool)
```

- `provenance: CandidateProvenance` [只读值字段]
- `net_charge_candidate_identity: str` [只读值字段]
- `oxidation_state_candidate_identity: Optional[str]` [只读值字段]
- `electron_count: int` [只读值字段]
- `two_s: int` [只读值字段]
- `multiplicity: Optional[int]` [只读值字段]
- `pure_spin: bool` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts-electronicstateinferenceresult"></a>
### `ElectronicStateInferenceResult`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.ElectronicStateInferenceResult`；基类 无。

```python
ElectronicStateInferenceResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[ElectronicStateCandidate, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
```

- `identity: str` [只读值字段]
- `facts_identity: MolecularFactsIdentity` [只读值字段]
- `candidates: Tuple[ElectronicStateCandidate, ...]` [只读值字段]
- `admissibility: ScientificAdmissibility` [只读值字段]
- `resolution: ResolutionStatus` [只读值字段]
- `findings: Tuple[ScientificFinding, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts-neutralatomreference"></a>
### `NeutralAtomReference`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts.NeutralAtomReference`；基类 无。

```python
NeutralAtomReference(atomic_number: int, electron_count: int, ground_configuration: str, two_s: int, source_identity: str, registry_version: str, content_digest: str)
```

- `atomic_number: int` [只读值字段]
- `electron_count: int` [只读值字段]
- `ground_configuration: str` [只读值字段]
- `two_s: int` [只读值字段]
- `source_identity: str` [只读值字段]
- `registry_version: str` [只读值字段]
- `content_digest: str` [只读值字段]


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-inference"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.inference`

`hotpot/cheminfo/rules/chemistry/scientific_state/electronic_state/inference.py`；显式 __all__: ('infer_electronic_states',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-inference-infer-electronic-states"></a>
### `infer_electronic_states`

```python
infer_electronic_states(facts: MolecularFacts, request: StateRequest=StateRequest(), *, net_charge_results: Tuple[NetChargeResult, ...], oxidation_state_results: Tuple[OxidationStateInferenceResult, ...]=(), provided: Tuple[ElectronicStateInferenceResult, ...]=()) -> Tuple[ElectronicStateInferenceResult, ...]
```


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-neutral-atom"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.neutral_atom`

`hotpot/cheminfo/rules/chemistry/scientific_state/electronic_state/neutral_atom.py`；显式 __all__: ('neutral_atom_reference',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-neutral-atom-neutral-atom-reference"></a>
### `neutral_atom_reference`

```python
neutral_atom_reference(atomic_number: int) -> Optional[NeutralAtomReference]
```


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-identities"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.identities`

`hotpot/cheminfo/rules/chemistry/scientific_state/identities.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-chemistry-scientific-state-legality"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.legality`

`hotpot/cheminfo/rules/chemistry/scientific_state/legality.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state`

`hotpot/cheminfo/rules/chemistry/scientific_state/oxidation_state/__init__.py`；显式 __all__: ('CountingConvention', 'OxidationStateOverride', 'CoordinationFragmentAssignment', 'CoordinationCountingCandidate', 'CoordinationElectronCountingResult', 'OxidationStateCandidateEvidence', 'OxidationStateInferenceResult', 'ElementOxidationStateDomain')

- 导出 `CountingConvention` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CountingConvention`
- 导出 `OxidationStateOverride` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.OxidationStateOverride`
- 导出 `CoordinationFragmentAssignment` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CoordinationFragmentAssignment`
- 导出 `CoordinationCountingCandidate` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CoordinationCountingCandidate`
- 导出 `CoordinationElectronCountingResult` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CoordinationElectronCountingResult`
- 导出 `OxidationStateCandidateEvidence` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.OxidationStateCandidateEvidence`
- 导出 `OxidationStateInferenceResult` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.OxidationStateInferenceResult`
- 导出 `ElementOxidationStateDomain` → `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.ElementOxidationStateDomain`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts`

`hotpot/cheminfo/rules/chemistry/scientific_state/oxidation_state/contracts.py`；显式 __all__: ('CountingConvention', 'OxidationStateOverride', 'CoordinationFragmentAssignment', 'CoordinationCountingCandidate', 'CoordinationElectronCountingResult', 'OxidationStateCandidateEvidence', 'OxidationStateInferenceResult', 'ElementOxidationStateDomain')


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-countingconvention"></a>
### `CountingConvention`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CountingConvention`；基类 `Enum`。

- `IONIC` = `'ionic'` [枚举常量]
- `DONOR_RETURN` = `'donor_return'` [枚举常量]
- `CBC` = `'cbc'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-oxidationstateoverride"></a>
### `OxidationStateOverride`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.OxidationStateOverride`；基类 无。

```python
OxidationStateOverride(atom_index: int, oxidation_state: int)
```

- `atom_index: int` [只读值字段]
- `oxidation_state: int` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-coordinationfragmentassignment"></a>
### `CoordinationFragmentAssignment`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CoordinationFragmentAssignment`；基类 无。

```python
CoordinationFragmentAssignment(component_identity: str, ionic_bookkeeping_charge: Optional[int], lxz_class: Optional[str], donated_electrons: Optional[int])
```

- `component_identity: str` [只读值字段]
- `ionic_bookkeeping_charge: Optional[int]` [只读值字段]
- `lxz_class: Optional[str]` [只读值字段]
- `donated_electrons: Optional[int]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-coordinationcountingcandidate"></a>
### `CoordinationCountingCandidate`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CoordinationCountingCandidate`；基类 无。

```python
CoordinationCountingCandidate(provenance: CandidateProvenance, convention: CountingConvention, fragments: Tuple[CoordinationFragmentAssignment, ...])
```

- `provenance: CandidateProvenance` [只读值字段]
- `convention: CountingConvention` [只读值字段]
- `fragments: Tuple[CoordinationFragmentAssignment, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-coordinationelectroncountingresult"></a>
### `CoordinationElectronCountingResult`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.CoordinationElectronCountingResult`；基类 无。

```python
CoordinationElectronCountingResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[CoordinationCountingCandidate, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
```

- `identity: str` [只读值字段]
- `facts_identity: MolecularFactsIdentity` [只读值字段]
- `candidates: Tuple[CoordinationCountingCandidate, ...]` [只读值字段]
- `admissibility: ScientificAdmissibility` [只读值字段]
- `resolution: ResolutionStatus` [只读值字段]
- `findings: Tuple[ScientificFinding, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-oxidationstatecandidateevidence"></a>
### `OxidationStateCandidateEvidence`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.OxidationStateCandidateEvidence`；基类 无。

```python
OxidationStateCandidateEvidence(provenance: CandidateProvenance, convention: CountingConvention, atom_states: Tuple[Optional[int], ...], coverage: CoverageStatus, coordination_counting_candidate_identity: Optional[str] = None)
```

- `provenance: CandidateProvenance` [只读值字段]
- `convention: CountingConvention` [只读值字段]
- `atom_states: Tuple[Optional[int], ...]` [只读值字段]
- `coverage: CoverageStatus` [只读值字段]
- `coordination_counting_candidate_identity: Optional[str]` = `None` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-oxidationstateinferenceresult"></a>
### `OxidationStateInferenceResult`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.OxidationStateInferenceResult`；基类 无。

```python
OxidationStateInferenceResult(identity: str, facts_identity: MolecularFactsIdentity, candidates: Tuple[OxidationStateCandidateEvidence, ...], admissibility: ScientificAdmissibility, resolution: ResolutionStatus, findings: Tuple[ScientificFinding, ...])
```

- `identity: str` [只读值字段]
- `facts_identity: MolecularFactsIdentity` [只读值字段]
- `candidates: Tuple[OxidationStateCandidateEvidence, ...]` [只读值字段]
- `admissibility: ScientificAdmissibility` [只读值字段]
- `resolution: ResolutionStatus` [只读值字段]
- `findings: Tuple[ScientificFinding, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts-elementoxidationstatedomain"></a>
### `ElementOxidationStateDomain`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts.ElementOxidationStateDomain`；基类 无。

```python
ElementOxidationStateDomain(atomic_number: int, states: Tuple[int, ...], source_identity: str, registry_version: str, content_digest: str, common_states: Tuple[int, ...] = (), important_states: Tuple[int, ...] = (), default_state: Optional[int] = None, default_basis: str = "")
```

- `atomic_number: int` [只读值字段]
- `states: Tuple[int, ...]` [只读值字段]
- `source_identity: str` [只读值字段]
- `registry_version: str` [只读值字段]
- `content_digest: str` [只读值字段]
- `common_states: Tuple[int, ...]` = `()` [只读值字段；来源报告的常见氧化态子集]
- `important_states: Tuple[int, ...]` = `()` [只读值字段；来源特别强调的常见态子集]
- `default_state: Optional[int]` = `None` [只读值字段；具名的运行默认值，不代表任意环境下的真实氧化态]
- `default_basis: str` = `""` [只读值字段；来源选择或建模择优规则]


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-coordination-counting"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.coordination_counting`

`hotpot/cheminfo/rules/chemistry/scientific_state/oxidation_state/coordination_counting.py`；显式 __all__: ('infer_coordination_electron_counting',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-coordination-counting-infer-coordination-electron-counting"></a>
### `infer_coordination_electron_counting`

```python
infer_coordination_electron_counting(facts: MolecularFacts, *, lewis_results: Tuple[LewisFormalChargeResult, ...]=(), conventions: Tuple[CountingConvention, ...]=()) -> Tuple[CoordinationElectronCountingResult, ...]
```


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-inference"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.inference`

`hotpot/cheminfo/rules/chemistry/scientific_state/oxidation_state/inference.py`；显式 __all__: ('infer_oxidation_states',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-inference-infer-oxidation-states"></a>
### `infer_oxidation_states`

```python
infer_oxidation_states(facts: MolecularFacts, request: StateRequest=StateRequest(), *, counting_results: Tuple[CoordinationElectronCountingResult, ...]=(), provided: Tuple[OxidationStateInferenceResult, ...]=()) -> Tuple[OxidationStateInferenceResult, ...]
```


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-registry"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.registry`

`hotpot/cheminfo/rules/chemistry/scientific_state/oxidation_state/registry.py`；显式 __all__: ('element_oxidation_state_domain',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-registry-element-oxidation-state-domain"></a>
### `element_oxidation_state_domain`

```python
element_oxidation_state_domain(atomic_number: int) -> ElementOxidationStateDomain
```

`states` 保留不排序优先级的可能态域；`important_states ⊆ common_states ⊆ states`。
118 个元素均有具名的运行默认值：优先来源唯一强调的状态，多值时采用明确记录的
最小非零绝对值选择；来源缺失时的元素态 0 是建模参考，不能冒充来源报道的常见态。
Nh 的可能态域仍为空。实际求解先应用用户输入和适用的专门化学规则，再使用这些默认值。


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/__init__.py`；显式 __all__: ('RuleKind', 'RulePhase', 'ResultType', 'RuleValue', 'RuleDependency', 'RuleSpecification', 'RuleToggle', 'EvidenceItem', 'RankAdjustment', 'RuleApplicability', 'RuleContext', 'RuleOutcome', 'RuleTrace')

- 导出 `RuleKind` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleKind`
- 导出 `RulePhase` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RulePhase`
- 导出 `ResultType` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.ResultType`
- 导出 `RuleValue` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ProviderValue`
- 导出 `RuleDependency` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleDependency`
- 导出 `RuleSpecification` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleSpecification`
- 导出 `RuleToggle` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleToggle`
- 导出 `EvidenceItem` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.EvidenceItem`
- 导出 `RankAdjustment` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RankAdjustment`
- 导出 `RuleApplicability` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleApplicability`
- 导出 `RuleContext` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleContext`
- 导出 `RuleOutcome` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleOutcome`
- 导出 `RuleTrace` → `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleTrace`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-americium-curium-iii"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.americium_curium_iii`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/americium_curium_iii.py`；显式 __all__: ('AmericiumCuriumIIIPriorRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-americium-curium-iii-americiumcuriumiiipriorrule"></a>
### `AmericiumCuriumIIIPriorRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.americium_curium_iii.AmericiumCuriumIIIPriorRule`；基类 `_OxidationStatePriorRule`。

- `__init__(self) -> None`
- `applicable(self, context: RuleContext) -> RuleApplicability` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `specification` = `RuleSpecification(rule_id=artifact.rule_id, rule_version=artifact.rule_version, kind=RuleKind.EMPIRICAL, phase=RulePhase.OXIDATION_STATE_CANDIDATE, requires=(), consumes=(ResultType.COORDINATION_COUNTING_RESULT,), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE, ResultType.RANK_ADJUSTMENT), applicability_schema=artifact.applicability_schema)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-anionic-x-ligands"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.anionic_x_ligands`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/anionic_x_ligands.py`；显式 __all__: ('AnionicXLigandRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-anionic-x-ligands-anionicxligandrule"></a>
### `AnionicXLigandRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.anionic_x_ligands.AnionicXLigandRule`；基类 `InferenceRule`。

- `__init__(self, source_atom_formal_charges=None)`；可选绑定已有不可变来源电荷证据，无参数调用保留。完整或部分来源覆盖时，不以供体局部特征覆盖该组分的已提供电荷。
- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification` = `RuleSpecification(rule_id=_RULE_ID, rule_version=_RULE_VERSION, kind=RuleKind.EMPIRICAL, phase=RulePhase.COORDINATION_COUNTING, requires=(), consumes=(), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE), applicability_schema=_DOMAIN_DEFINITION)` [类声明值；可重绑定不构成事实写许可]
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-base"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/base.py`；显式 __all__: ('InferenceRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-base-inferencerule"></a>
### `InferenceRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`；基类 `ABC`。

- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome`
- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification: RuleSpecification` [字段声明；写权限依具体合同]


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/contracts.py`；显式 __all__: ('RuleKind', 'RulePhase', 'ResultType', 'RuleValue', 'RuleDependency', 'RuleSpecification', 'RuleToggle', 'EvidenceItem', 'RankAdjustment', 'RuleApplicability', 'RuleContext', 'RuleOutcome', 'RuleTrace')

- 导出 `RuleValue` → `hotpot.cheminfo.rules.chemistry.scientific_state.contracts.ProviderValue`

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-rulekind"></a>
### `RuleKind`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleKind`；基类 `Enum`。

- `EMPIRICAL` = `'empirical'` [枚举常量]
- `AI` = `'ai'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-rulephase"></a>
### `RulePhase`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RulePhase`；基类 `Enum`。

- `SOURCE_EVIDENCE` = `'source_evidence'` [枚举常量]
- `CHARGE_EVIDENCE` = `'charge_evidence'` [枚举常量]
- `COORDINATION_COUNTING` = `'coordination_counting'` [枚举常量]
- `REDOX_CANDIDATE` = `'redox_candidate'` [枚举常量]
- `OXIDATION_STATE_CANDIDATE` = `'oxidation_state_candidate'` [枚举常量]
- `OXIDATION_STATE_RANKING` = `'oxidation_state_ranking'` [枚举常量]
- `ELECTRONIC_STATE_CANDIDATE` = `'electronic_state_candidate'` [枚举常量]
- `ELECTRONIC_STATE_RANKING` = `'electronic_state_ranking'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-resulttype"></a>
### `ResultType`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.ResultType`；基类 `Enum`。

- `NET_CHARGE_RESULT` = `'net_charge_result'` [枚举常量]
- `LEWIS_FORMAL_CHARGE_RESULT` = `'lewis_formal_charge_result'` [枚举常量]
- `COORDINATION_COUNTING_RESULT` = `'coordination_counting_result'` [枚举常量]
- `OXIDATION_STATE_RESULT` = `'oxidation_state_result'` [枚举常量]
- `ELECTRONIC_STATE_RESULT` = `'electronic_state_result'` [枚举常量]
- `SPIN_CONTRIBUTION` = `'spin_contribution'` [枚举常量]
- `CANDIDATE_PROPOSAL` = `'candidate_proposal'` [枚举常量]
- `EVIDENCE` = `'evidence'` [枚举常量]
- `RANK_ADJUSTMENT` = `'rank_adjustment'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-ruledependency"></a>
### `RuleDependency`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleDependency`；基类 无。

```python
RuleDependency(rule_id: str, version_specifier: str)
```

- `rule_id: str` [只读值字段]
- `version_specifier: str` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-rulespecification"></a>
### `RuleSpecification`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleSpecification`；基类 无。

```python
RuleSpecification(rule_id: str, rule_version: str, kind: RuleKind, phase: RulePhase, requires: Tuple[RuleDependency, ...], consumes: Tuple[ResultType, ...], produces: Tuple[ResultType, ...], applicability_schema: str)
```

- `rule_id: str` [只读值字段]
- `rule_version: str` [只读值字段]
- `kind: RuleKind` [只读值字段]
- `phase: RulePhase` [只读值字段]
- `requires: Tuple[RuleDependency, ...]` [只读值字段]
- `consumes: Tuple[ResultType, ...]` [只读值字段]
- `produces: Tuple[ResultType, ...]` [只读值字段]
- `applicability_schema: str` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-ruletoggle"></a>
### `RuleToggle`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleToggle`；基类 无。

```python
RuleToggle(rule_id: str, enabled: bool, configuration: Tuple[Tuple[str, RuleValue], ...] = ())
```

- `rule_id: str` [只读值字段]
- `enabled: bool` [只读值字段]
- `configuration: Tuple[Tuple[str, RuleValue], ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-evidenceitem"></a>
### `EvidenceItem`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.EvidenceItem`；基类 无。

```python
EvidenceItem(identity: str, authority: StateValueAuthority, source_identity: str, claim: str, locator: Optional[str] = None)
```

- `identity: str` [只读值字段]
- `authority: StateValueAuthority` [只读值字段]
- `source_identity: str` [只读值字段]
- `claim: str` [只读值字段]
- `locator: Optional[str]` = `None` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-rankadjustment"></a>
### `RankAdjustment`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RankAdjustment`；基类 无。

```python
RankAdjustment(target_identity: str, rank_dimension: str, value: float, reason: str, assumption_ids: Tuple[str, ...] = ())
```

- `target_identity: str` [只读值字段]
- `rank_dimension: str` [只读值字段]
- `value: float` [只读值字段]
- `reason: str` [只读值字段]
- `assumption_ids: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-ruleapplicability"></a>
### `RuleApplicability`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleApplicability`；基类 无。

```python
RuleApplicability(matches: bool, reason_codes: Tuple[str, ...] = ())
```

- `matches: bool` [只读值字段]
- `reason_codes: Tuple[str, ...]` = `()` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-rulecontext"></a>
### `RuleContext`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleContext`；基类 无。

```python
RuleContext(facts: MolecularFacts, phase: RulePhase, requirements: ScientificOperationRequirements, candidate_space: Optional[ScientificStateCandidateSpace], prior_outcomes: Tuple['RuleOutcome', ...], candidate_budget: CandidateBudget)
```

- `facts: MolecularFacts` [只读值字段]
- `phase: RulePhase` [只读值字段]
- `requirements: ScientificOperationRequirements` [只读值字段]
- `candidate_space: Optional[ScientificStateCandidateSpace]` [只读值字段]
- `prior_outcomes: Tuple['RuleOutcome', ...]` [只读值字段]
- `candidate_budget: CandidateBudget` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-ruleoutcome"></a>
### `RuleOutcome`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleOutcome`；基类 无。

```python
RuleOutcome(rule_id: str, rule_version: str, applicability: RuleApplicability, candidate_proposal: Optional[CandidateProposal], evidence: Tuple[EvidenceItem, ...], rank_adjustments: Tuple[RankAdjustment, ...], assumptions: Tuple[str, ...], warnings: Tuple[str, ...])
```

- `rule_id: str` [只读值字段]
- `rule_version: str` [只读值字段]
- `applicability: RuleApplicability` [只读值字段]
- `candidate_proposal: Optional[CandidateProposal]` [只读值字段]
- `evidence: Tuple[EvidenceItem, ...]` [只读值字段]
- `rank_adjustments: Tuple[RankAdjustment, ...]` [只读值字段]
- `assumptions: Tuple[str, ...]` [只读值字段]
- `warnings: Tuple[str, ...]` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts-ruletrace"></a>
### `RuleTrace`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts.RuleTrace`；基类 无。

```python
RuleTrace(phase: RulePhase, outcome: RuleOutcome)
```

- `phase: RulePhase` [只读值字段]
- `outcome: RuleOutcome` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `RuleValue` = `ProviderValue`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-f-block-hund"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.f_block_hund`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/f_block_hund.py`；显式 __all__: ('FBlockHundPriorRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-f-block-hund-fblockhundpriorrule"></a>
### `FBlockHundPriorRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.f_block_hund.FBlockHundPriorRule`；基类 `InferenceRule`。

- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification` = <长声明值；见 `hotpot/cheminfo/rules/chemistry/scientific_state/providers/f_block_hund.py:161`> [类声明值；可重绑定不构成事实写许可]
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-ionic-configuration-spin"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.ionic_configuration_spin`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/ionic_configuration_spin.py`；显式 __all__: ('IonicConfigurationSpinRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-ionic-configuration-spin-ionicconfigurationspinrule"></a>
### `IonicConfigurationSpinRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.ionic_configuration_spin.IonicConfigurationSpinRule`；基类 `InferenceRule`。

- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification` = `RuleSpecification(rule_id=_RULE_ID, rule_version=_RULE_VERSION, kind=RuleKind.EMPIRICAL, phase=RulePhase.ELECTRONIC_STATE_CANDIDATE, requires=(), consumes=(ResultType.OXIDATION_STATE_RESULT,), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE), applicability_schema=_DOMAIN_DEFINITION)` [类声明值；可重绑定不构成事实写许可]
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-lanthanide-iii"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.lanthanide_iii`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/lanthanide_iii.py`；显式 __all__: ('LanthanideIIIPriorRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-lanthanide-iii-lanthanideiiipriorrule"></a>
### `LanthanideIIIPriorRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.lanthanide_iii.LanthanideIIIPriorRule`；基类 `_OxidationStatePriorRule`。

- `__init__(self) -> None`
- `applicable(self, context: RuleContext) -> RuleApplicability` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `specification` = `RuleSpecification(rule_id=artifact.rule_id, rule_version=artifact.rule_version, kind=RuleKind.EMPIRICAL, phase=RulePhase.OXIDATION_STATE_CANDIDATE, requires=(), consumes=(ResultType.COORDINATION_COUNTING_RESULT,), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE, ResultType.RANK_ADJUSTMENT), applicability_schema=artifact.applicability_schema)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-main-group-closed-shell"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.main_group_closed_shell`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/main_group_closed_shell.py`；显式 __all__: ('MainGroupClosedShellRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-main-group-closed-shell-maingroupclosedshellrule"></a>
### `MainGroupClosedShellRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.main_group_closed_shell.MainGroupClosedShellRule`；基类 `InferenceRule`。

- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification` = <长声明值；见 `hotpot/cheminfo/rules/chemistry/scientific_state/providers/main_group_closed_shell.py:301`> [类声明值；可重绑定不构成事实写许可]
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-monoatomic-ions"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.monoatomic_ions`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/monoatomic_ions.py`；显式 __all__: ('MonoatomicIonRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-monoatomic-ions-monoatomicionrule"></a>
### `MonoatomicIonRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.monoatomic_ions.MonoatomicIonRule`；基类 `InferenceRule`。

- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification` = `RuleSpecification(rule_id=_RULE_ID, rule_version=_RULE_VERSION, kind=RuleKind.EMPIRICAL, phase=RulePhase.CHARGE_EVIDENCE, requires=(), consumes=(), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE), applicability_schema=_DOMAIN_DEFINITION)` [类声明值；可重绑定不构成事实写许可]
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-neutral-l-donors"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.neutral_l_donors`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/neutral_l_donors.py`；显式 __all__: ('NeutralLDonorRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-neutral-l-donors-neutralldonorrule"></a>
### `NeutralLDonorRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.neutral_l_donors.NeutralLDonorRule`；基类 `InferenceRule`。

- `__init__(self, source_atom_formal_charges=None)`；可选绑定已有不可变来源电荷证据，无参数调用保留。完整或部分来源覆盖时，不以供体局部特征覆盖该组分的已提供电荷。
- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification` = `RuleSpecification(rule_id=_RULE_ID, rule_version=_RULE_VERSION, kind=RuleKind.EMPIRICAL, phase=RulePhase.COORDINATION_COUNTING, requires=(), consumes=(), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE), applicability_schema=_DOMAIN_DEFINITION)` [类声明值；可重绑定不构成事实写许可]
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-oxidation-priors"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/oxidation_priors.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-thorium-iv"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.thorium_iv`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/thorium_iv.py`；显式 __all__: ('ThoriumIVPriorRule',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-thorium-iv-thoriumivpriorrule"></a>
### `ThoriumIVPriorRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.thorium_iv.ThoriumIVPriorRule`；基类 `_OxidationStatePriorRule`。

- `__init__(self) -> None`
- `applicable(self, context: RuleContext) -> RuleApplicability` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `specification` = `RuleSpecification(rule_id=artifact.rule_id, rule_version=artifact.rule_version, kind=RuleKind.EMPIRICAL, phase=RulePhase.OXIDATION_STATE_CANDIDATE, requires=(), consumes=(ResultType.COORDINATION_COUNTING_RESULT,), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE, ResultType.RANK_ADJUSTMENT), applicability_schema=artifact.applicability_schema)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema） ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors._OxidationStatePriorRule`
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-trusted-history"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.providers.trusted_history`

`hotpot/cheminfo/rules/chemistry/scientific_state/providers/trusted_history.py`；显式 __all__: ('TrustedScientificStateRecord', 'TrustedHistoryRule')


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-trusted-history-trustedscientificstaterecord"></a>
### `TrustedScientificStateRecord`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.trusted_history.TrustedScientificStateRecord`；基类 无。

```python
TrustedScientificStateRecord(identity: str, facts_identity_digest: str, atom_map: Tuple[int, ...], source_identity: str, source_digest: str, authority: StateValueAuthority, net_charge: Optional[int] = None, oxidation_states: Tuple[Optional[int], ...] = (), counting_convention: CountingConvention = CountingConvention.IONIC, two_s: Optional[int] = None, multiplicity: Optional[int] = None, pure_spin: bool = True, assumption_ids: Tuple[str, ...] = (), locator: Optional[str] = None)
```

- `identity: str` [只读值字段]
- `facts_identity_digest: str` [只读值字段]
- `atom_map: Tuple[int, ...]` [只读值字段]
- `source_identity: str` [只读值字段]
- `source_digest: str` [只读值字段]
- `authority: StateValueAuthority` [只读值字段]
- `net_charge: Optional[int]` = `None` [只读值字段]
- `oxidation_states: Tuple[Optional[int], ...]` = `()` [只读值字段]
- `counting_convention: CountingConvention` = `CountingConvention.IONIC` [只读值字段]
- `two_s: Optional[int]` = `None` [只读值字段]
- `multiplicity: Optional[int]` = `None` [只读值字段]
- `pure_spin: bool` = `True` [只读值字段]
- `assumption_ids: Tuple[str, ...]` = `()` [只读值字段]
- `locator: Optional[str]` = `None` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-scientific-state-providers-trusted-history-trustedhistoryrule"></a>
### `TrustedHistoryRule`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.providers.trusted_history.TrustedHistoryRule`；基类 `InferenceRule`。

- `__init__(self, records: Tuple[TrustedScientificStateRecord, ...]=()) -> None`
- `applicable(self, context: RuleContext) -> RuleApplicability`
- `action(self, context: RuleContext, configuration: Tuple[Tuple[str, RuleValue], ...]) -> RuleOutcome`
- `specification` = `RuleSpecification(rule_id=_RULE_ID, rule_version=_RULE_VERSION, kind=RuleKind.EMPIRICAL, phase=RulePhase.SOURCE_EVIDENCE, requires=(), consumes=(), produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE, ResultType.RANK_ADJUSTMENT), applicability_schema=_DOMAIN_DEFINITION)` [类声明值；可重绑定不构成事实写许可]
- `evaluate(self, toggle: RuleToggle, context: RuleContext) -> RuleOutcome` ← `hotpot.cheminfo.rules.chemistry.scientific_state.providers.base.InferenceRule`


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-request"></a>
## `hotpot.cheminfo.rules.chemistry.scientific_state.request`

`hotpot/cheminfo/rules/chemistry/scientific_state/request.py`；显式 __all__: ('StateRequest',)


<a id="hotpot-cheminfo-rules-chemistry-scientific-state-request-staterequest"></a>
### `StateRequest`

定义 owner `hotpot.cheminfo.rules.chemistry.scientific_state.request.StateRequest`；基类 无。

```python
StateRequest(net_charge: Optional[int] = None, atom_formal_charges: Optional[AtomFormalChargeEvidence] = None, component_charges: Optional[ComponentChargeEvidence] = None, oxidation_overrides: Tuple[OxidationStateOverride, ...] = (), two_s: Optional[int] = None, multiplicity: Optional[int] = None, charge_candidates: Tuple[NetChargeCandidateEvidence, ...] = (), oxidation_candidates: Tuple[OxidationStateCandidateEvidence, ...] = (), electronic_candidates: Tuple[ElectronicStateCandidateEvidence, ...] = ())
```

- `net_charge: Optional[int]` = `None` [只读值字段]
- `atom_formal_charges: Optional[AtomFormalChargeEvidence]` = `None` [只读值字段]
- `component_charges: Optional[ComponentChargeEvidence]` = `None` [只读值字段]
- `oxidation_overrides: Tuple[OxidationStateOverride, ...]` = `()` [只读值字段]
- `two_s: Optional[int]` = `None` [只读值字段]
- `multiplicity: Optional[int]` = `None` [只读值字段]
- `charge_candidates: Tuple[NetChargeCandidateEvidence, ...]` = `()` [只读值字段]
- `oxidation_candidates: Tuple[OxidationStateCandidateEvidence, ...]` = `()` [只读值字段]
- `electronic_candidates: Tuple[ElectronicStateCandidateEvidence, ...]` = `()` [只读值字段]


<a id="hotpot-cheminfo-rules-chemistry-stereochemistry"></a>
## `hotpot.cheminfo.rules.chemistry.stereochemistry`

`hotpot/cheminfo/rules/chemistry/stereochemistry/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-chemistry-stereochemistry-algebra"></a>
## `hotpot.cheminfo.rules.chemistry.stereochemistry.algebra`

`hotpot/cheminfo/rules/chemistry/stereochemistry/algebra.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-stereochemistry-alignment"></a>
## `hotpot.cheminfo.rules.chemistry.stereochemistry.alignment`

`hotpot/cheminfo/rules/chemistry/stereochemistry/alignment.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-stereochemistry-assignment"></a>
## `hotpot.cheminfo.rules.chemistry.stereochemistry.assignment`

`hotpot/cheminfo/rules/chemistry/stereochemistry/assignment.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-stereochemistry-coordinates"></a>
## `hotpot.cheminfo.rules.chemistry.stereochemistry.coordinates`

`hotpot/cheminfo/rules/chemistry/stereochemistry/coordinates.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-stereochemistry-graph"></a>
## `hotpot.cheminfo.rules.chemistry.stereochemistry.graph`

`hotpot/cheminfo/rules/chemistry/stereochemistry/graph.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-stereochemistry-validation"></a>
## `hotpot.cheminfo.rules.chemistry.stereochemistry.validation`

`hotpot/cheminfo/rules/chemistry/stereochemistry/validation.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-chemistry-substructure"></a>
## `hotpot.cheminfo.rules.chemistry.substructure`

`hotpot/cheminfo/rules/chemistry/substructure/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax"></a>
## `hotpot.cheminfo.rules.chemistry.substructure._smarts_syntax`

`hotpot/cheminfo/rules/chemistry/substructure/_smarts_syntax.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax-tokentype"></a>
### `TokenType`

定义 owner `hotpot.cheminfo.rules.chemistry.substructure._smarts_syntax.TokenType`；基类 `str`, `Enum`。

- `ATOM` = `'ATOM'` [枚举常量]
- `BRACKET` = `'BRACKET'` [枚举常量]
- `BOND` = `'BOND'` [枚举常量]
- `BRANCH_L` = `'BRANCH_L'` [枚举常量]
- `BRANCH_R` = `'BRANCH_R'` [枚举常量]
- `RING` = `'RING'` [枚举常量]
- `DOT` = `'DOT'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax-token"></a>
### `Token`

定义 owner `hotpot.cheminfo.rules.chemistry.substructure._smarts_syntax.Token`；基类 无。

```python
Token(type: TokenType, text: str, start: int, end: int)
```

- `type: TokenType` [只读值字段]
- `text: str` [只读值字段]
- `start: int` [只读值字段]
- `end: int` [只读值字段]

<a id="hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax-atomic-number"></a>
### `atomic_number`

```python
atomic_number(symbol: str) -> int
```

<a id="hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax-tokenize-with-spans"></a>
### `tokenize_with_spans`

```python
tokenize_with_spans(smarts: str) -> List[Token]
```

<a id="hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax-tokenize"></a>
### `tokenize`

```python
tokenize(smarts: str) -> List[Tuple[TokenType, str]]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `BOND_CHARS: Set[str]` = `{'-', '=', '#', ':', '/', '\\', '~'}`
- `AROMATIC_LOWER: Set[str]` = `{'b', 'c', 'n', 'o', 'p', 's', 'se', 'as'}`
- `ANY_ATOM_TOKEN` = `'*'`


<a id="hotpot-cheminfo-rules-chemistry-substructure-errors"></a>
## `hotpot.cheminfo.rules.chemistry.substructure.errors`

`hotpot/cheminfo/rules/chemistry/substructure/errors.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rules-chemistry-substructure-errors-smartssyntaxerror"></a>
### `SmartsSyntaxError`

定义 owner `hotpot.cheminfo.rules.chemistry.substructure.errors.SmartsSyntaxError`；基类 `ValueError`。


<a id="hotpot-cheminfo-rules-chemistry-substructure-errors-unsupportedsmartserror"></a>
### `UnsupportedSmartsError`

定义 owner `hotpot.cheminfo.rules.chemistry.substructure.errors.UnsupportedSmartsError`；基类 `NotImplementedError`。



<a id="hotpot-cheminfo-rules-chemistry-substructure-predicates"></a>
## `hotpot.cheminfo.rules.chemistry.substructure.predicates`

`hotpot/cheminfo/rules/chemistry/substructure/predicates.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rules-chemistry-substructure-predicates-match-atom-expression"></a>
### `match_atom_expression`

```python
match_atom_expression(expression, values)
```

<a id="hotpot-cheminfo-rules-chemistry-substructure-predicates-match-bond-expression"></a>
### `match_bond_expression`

```python
match_bond_expression(expression, values)
```


<a id="hotpot-cheminfo-rules-chemistry-substructure-semantics"></a>
## `hotpot.cheminfo.rules.chemistry.substructure.semantics`

`hotpot/cheminfo/rules/chemistry/substructure/semantics.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rules-chemistry-substructure-semantics-resolve-smarts-semantics"></a>
### `resolve_smarts_semantics`

```python
resolve_smarts_semantics(value) -> SmartsSemantics
```

<a id="hotpot-cheminfo-rules-chemistry-substructure-semantics-degree"></a>
### `degree`

```python
degree(bond_count)
```

<a id="hotpot-cheminfo-rules-chemistry-substructure-semantics-connectivity"></a>
### `connectivity`

```python
connectivity(bond_count, implicit_hydrogens)
```

<a id="hotpot-cheminfo-rules-chemistry-substructure-semantics-valence"></a>
### `valence`

```python
valence(bond_order_sum, implicit_hydrogens)
```

<a id="hotpot-cheminfo-rules-chemistry-substructure-semantics-ligand-valence"></a>
### `ligand_valence`

```python
ligand_valence(bond_orders, implicit_hydrogens)
```


<a id="hotpot-cheminfo-rules-chemistry-substructure-smarts"></a>
## `hotpot.cheminfo.rules.chemistry.substructure.smarts`

`hotpot/cheminfo/rules/chemistry/substructure/smarts.py`；显式 __all__: 无


<a id="hotpot-cheminfo-rules-chemistry-substructure-smarts-bondorder"></a>
### `BondOrder`

定义 owner `hotpot.cheminfo.rules.chemistry.substructure.smarts.BondOrder`；基类 `int`, `Enum`。

- `SINGLE` = `1` [枚举常量]
- `DOUBLE` = `2` [枚举常量]
- `TRIPLE` = `3` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-substructure-smarts-aromaticflag"></a>
### `AromaticFlag`

定义 owner `hotpot.cheminfo.rules.chemistry.substructure.smarts.AromaticFlag`；基类 `Enum`。

- `AROMATIC` = `'arom'` [枚举常量]
- `NON_AROMATIC` = `'non_arom'` [枚举常量]
- `UNSET` = `'unset'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-chemistry-substructure-smarts-parse-smarts"></a>
### `parse_smarts`

```python
parse_smarts(smarts: str, *, semantics: SmartsSemantics=SmartsSemantics.FULL_GRAPH)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `METAL_TOKEN` = `'M'`
- `LANTHANIDE_TOKEN` = `'Ln'`
- `ACTINIDE_TOKEN` = `'An'`
- `PERIOD_PREFIX` = `'NP'`
- `GROUP_PREFIX` = `'NG'`
- `ANY_ATOM_TOKEN` = `'*'`
- `SINGLE_BOND_TOKEN` = `'-'`
- `DOUBLE_BOND_TOKEN` = `'='`
- `TRIPLE_BOND_TOKEN` = `'#'`
- `AROMATIC_BOND_TOKEN` = `':'`
- `ANY_BOND_TOKEN` = `'~'`
- `UP_BOND_TOKEN` = `'/'`
- `DOWN_BOND_TOKEN` = `'\\'`


<a id="hotpot-cheminfo-rules-mathematics"></a>
## `hotpot.cheminfo.rules.mathematics`

`hotpot/cheminfo/rules/mathematics/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-mathematics-cell"></a>
## `hotpot.cheminfo.rules.mathematics.cell`

`hotpot/cheminfo/rules/mathematics/cell.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-mathematics-geometry"></a>
## `hotpot.cheminfo.rules.mathematics.geometry`

`hotpot/cheminfo/rules/mathematics/geometry/__init__.py`；显式 __all__: ('NumericToleranceSettings', 'SurfaceEnumerationSettings', 'GeometrySettings', 'DEFAULT_GEOMETRY_SETTINGS', 'Point', 'Line', 'Segment', 'Plane', 'Triangle', 'Cycle', 'PlanarityKind', 'LineRelationKind', 'PointCycleLocation', 'SurfaceEmbeddingState', 'SurfaceSegmentState', 'PiercingState', 'SegmentCycleFeature', 'SegmentCycleIndeterminacy', 'CycleSurfaceModel', 'PlanarityMeasurement', 'LineRelation', 'PointPairDistance', 'ClosestCycleEdge', 'SurfaceFamilyEvidence', 'SegmentCycleRelation', 'SegmentCycleScreening', 'measure_planarity', 'determine_line_relation', 'line_distance', 'point_segment_distance', 'segment_segment_distance', 'point_pair_distances', 'find_point_pairs_below_distance', 'locate_point_in_planar_cycle', 'iter_segment_cycle_relations', 'iter_segment_cycle_screenings', 'determine_segment_cycle_relation', 'closest_cycle_edge')

- 导出 `NumericToleranceSettings` → `hotpot.cheminfo.rules.mathematics.geometry.settings.NumericToleranceSettings`
- 导出 `SurfaceEnumerationSettings` → `hotpot.cheminfo.rules.mathematics.geometry.settings.SurfaceEnumerationSettings`
- 导出 `GeometrySettings` → `hotpot.cheminfo.rules.mathematics.geometry.settings.GeometrySettings`
- 导出 `DEFAULT_GEOMETRY_SETTINGS` → `hotpot.cheminfo.rules.mathematics.geometry.settings.DEFAULT_GEOMETRY_SETTINGS`
- 导出 `Point` → `hotpot.cheminfo.rules.mathematics.geometry.object.Point`
- 导出 `Line` → `hotpot.cheminfo.rules.mathematics.geometry.object.Line`
- 导出 `Segment` → `hotpot.cheminfo.rules.mathematics.geometry.object.Segment`
- 导出 `Plane` → `hotpot.cheminfo.rules.mathematics.geometry.object.Plane`
- 导出 `Triangle` → `hotpot.cheminfo.rules.mathematics.geometry.object.Triangle`
- 导出 `Cycle` → `hotpot.cheminfo.rules.mathematics.geometry.object.Cycle`
- 导出 `PlanarityKind` → `hotpot.cheminfo.rules.mathematics.geometry.relation.PlanarityKind`
- 导出 `LineRelationKind` → `hotpot.cheminfo.rules.mathematics.geometry.relation.LineRelationKind`
- 导出 `PointCycleLocation` → `hotpot.cheminfo.rules.mathematics.geometry.relation.PointCycleLocation`
- 导出 `SurfaceEmbeddingState` → `hotpot.cheminfo.rules.mathematics.geometry.relation.SurfaceEmbeddingState`
- 导出 `SurfaceSegmentState` → `hotpot.cheminfo.rules.mathematics.geometry.relation.SurfaceSegmentState`
- 导出 `PiercingState` → `hotpot.cheminfo.rules.mathematics.geometry.relation.PiercingState`
- 导出 `SegmentCycleFeature` → `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleFeature`
- 导出 `SegmentCycleIndeterminacy` → `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleIndeterminacy`
- 导出 `CycleSurfaceModel` → `hotpot.cheminfo.rules.mathematics.geometry.relation.CycleSurfaceModel`
- 导出 `PlanarityMeasurement` → `hotpot.cheminfo.rules.mathematics.geometry.relation.PlanarityMeasurement`
- 导出 `LineRelation` → `hotpot.cheminfo.rules.mathematics.geometry.relation.LineRelation`
- 导出 `PointPairDistance` → `hotpot.cheminfo.rules.mathematics.geometry.relation.PointPairDistance`
- 导出 `ClosestCycleEdge` → `hotpot.cheminfo.rules.mathematics.geometry.relation.ClosestCycleEdge`
- 导出 `SurfaceFamilyEvidence` → `hotpot.cheminfo.rules.mathematics.geometry.relation.SurfaceFamilyEvidence`
- 导出 `SegmentCycleRelation` → `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleRelation`
- 导出 `SegmentCycleScreening` → `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleScreening`
- 导出 `measure_planarity` → `hotpot.cheminfo.rules.mathematics.geometry.relation.measure_planarity`；`measure_planarity(cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PlanarityMeasurement`
- 导出 `determine_line_relation` → `hotpot.cheminfo.rules.mathematics.geometry.relation.determine_line_relation`；`determine_line_relation(first: Line, second: Line, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> LineRelation`
- 导出 `line_distance` → `hotpot.cheminfo.rules.mathematics.geometry.relation.line_distance`；`line_distance(first: Line, second: Line, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float`
- 导出 `point_segment_distance` → `hotpot.cheminfo.rules.mathematics.geometry.relation.point_segment_distance`；`point_segment_distance(point: Point, segment: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float`
- 导出 `segment_segment_distance` → `hotpot.cheminfo.rules.mathematics.geometry.relation.segment_segment_distance`；`segment_segment_distance(first: Segment, second: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float`
- 导出 `point_pair_distances` → `hotpot.cheminfo.rules.mathematics.geometry.relation.point_pair_distances`；`point_pair_distances(points: Sequence[Point]) -> Tuple[PointPairDistance, ...]`
- 导出 `find_point_pairs_below_distance` → `hotpot.cheminfo.rules.mathematics.geometry.relation.find_point_pairs_below_distance`；`find_point_pairs_below_distance(points: Sequence[Point], threshold: float) -> Tuple[PointPairDistance, ...]`
- 导出 `locate_point_in_planar_cycle` → `hotpot.cheminfo.rules.mathematics.geometry.relation.locate_point_in_planar_cycle`；`locate_point_in_planar_cycle(point: Point, cycle: Cycle, plane: Plane, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PointCycleLocation`
- 导出 `iter_segment_cycle_relations` → `hotpot.cheminfo.rules.mathematics.geometry.relation.iter_segment_cycle_relations`；`iter_segment_cycle_relations(segments: Iterable[Segment], cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Iterator[SegmentCycleRelation]`
- 导出 `iter_segment_cycle_screenings` → `hotpot.cheminfo.rules.mathematics.geometry.relation.iter_segment_cycle_screenings`；`iter_segment_cycle_screenings(segments: Iterable[Segment], cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Iterator[SegmentCycleScreening]`
- 导出 `determine_segment_cycle_relation` → `hotpot.cheminfo.rules.mathematics.geometry.relation.determine_segment_cycle_relation`；`determine_segment_cycle_relation(segment: Segment, cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> SegmentCycleRelation`
- 导出 `closest_cycle_edge` → `hotpot.cheminfo.rules.mathematics.geometry.relation.closest_cycle_edge`；`closest_cycle_edge(cycle: Cycle, segment: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Optional[ClosestCycleEdge]`


<a id="hotpot-cheminfo-rules-mathematics-geometry-linking"></a>
## `hotpot.cheminfo.rules.mathematics.geometry.linking`

`hotpot/cheminfo/rules/mathematics/geometry/linking.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-mathematics-geometry-native"></a>
## `hotpot.cheminfo.rules.mathematics.geometry.native`

`hotpot/cheminfo/rules/mathematics/geometry/native.py`；显式 __all__: ('NumericTolerances', 'PointSegmentMeasurement', 'SegmentSegmentMeasurement', 'measure_planarity', 'locate_point_in_planar_cycle', 'closest_cycle_edge', 'determine_line_relation', 'line_distance', 'point_segment_measurement', 'point_segment_distance', 'segment_segment_measurement', 'segment_segment_distance', 'point_pair_distances', 'find_point_pairs_below_distance', 'aabb_bounds', 'aabb_separation_mask', 'segment_aabb_separation_mask', 'aabb_candidate_pairs')

- 导出 `NumericTolerances` → `hotpot.cheminfo.rules.mathematics.geometry._geometry_native.NumericTolerances`
- 导出 `PointSegmentMeasurement` → `hotpot.cheminfo.rules.mathematics.geometry._geometry_native.PointSegmentMeasurement`
- 导出 `SegmentSegmentMeasurement` → `hotpot.cheminfo.rules.mathematics.geometry._geometry_native.SegmentSegmentMeasurement`

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-measure-planarity"></a>
### `measure_planarity`

```python
measure_planarity(cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PlanarityMeasurement
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-prepare-planar-cycle"></a>
### `prepare_planar_cycle`

```python
prepare_planar_cycle(cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PreparedPlanarCycle
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-locate-point-in-planar-cycle"></a>
### `locate_point_in_planar_cycle`

```python
locate_point_in_planar_cycle(point: Point, cycle: Cycle, plane: Plane, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PointCycleLocation
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-closest-cycle-edge"></a>
### `closest_cycle_edge`

```python
closest_cycle_edge(cycle: Cycle, segment: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Optional[ClosestCycleEdge]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-determine-planar-segment-cycle-relation"></a>
### `determine_planar_segment_cycle_relation`

```python
determine_planar_segment_cycle_relation(segment: Segment, cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> SegmentCycleRelation
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-planar-segment-cycle-relations"></a>
### `planar_segment_cycle_relations`

```python
planar_segment_cycle_relations(segments: Iterable[Segment], cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Tuple[SegmentCycleRelation, ...]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-planar-segment-cycle-screenings"></a>
### `planar_segment_cycle_screenings`

```python
planar_segment_cycle_screenings(segments: Iterable[Segment], cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Tuple[SegmentCycleScreening, ...]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-determine-line-relation"></a>
### `determine_line_relation`

```python
determine_line_relation(first: Line, second: Line, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> LineRelation
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-line-distance"></a>
### `line_distance`

```python
line_distance(first: Line, second: Line, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-point-segment-measurement"></a>
### `point_segment_measurement`

```python
point_segment_measurement(point: Point, segment: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PointSegmentMeasurement
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-point-segment-distance"></a>
### `point_segment_distance`

```python
point_segment_distance(point: Point, segment: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-segment-segment-measurement"></a>
### `segment_segment_measurement`

```python
segment_segment_measurement(first: Segment, second: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> SegmentSegmentMeasurement
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-segment-segment-distance"></a>
### `segment_segment_distance`

```python
segment_segment_distance(first: Segment, second: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-point-pair-distances"></a>
### `point_pair_distances`

```python
point_pair_distances(points: Sequence[Point]) -> Tuple[PointPairDistance, ...]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-find-point-pairs-below-distance"></a>
### `find_point_pairs_below_distance`

```python
find_point_pairs_below_distance(points: Sequence[Point], threshold: float) -> Tuple[PointPairDistance, ...]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-aabb-bounds"></a>
### `aabb_bounds`

```python
aabb_bounds(coordinates: Iterable[Coordinates]) -> np.ndarray
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-aabb-separation-mask"></a>
### `aabb_separation_mask`

```python
aabb_separation_mask(first_bounds: BoundsInput, second_bounds: BoundsInput, padding: Union[float, Sequence[float], np.ndarray]) -> np.ndarray
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-segment-aabb-separation-mask"></a>
### `segment_aabb_separation_mask`

```python
segment_aabb_separation_mask(segments: Iterable[Segment], target_bounds: BoundsInput, padding: Union[float, Sequence[float], np.ndarray]) -> np.ndarray
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-native-aabb-candidate-pairs"></a>
### `aabb_candidate_pairs`

```python
aabb_candidate_pairs(first_bounds: BoundsInput, second_bounds: BoundsInput, padding: float) -> np.ndarray
```

模块值/类型别名（声明存在不构成事实写许可）：

- `NumericTolerances` = `_native.NumericTolerances`
- `PointSegmentMeasurement` = `_native.PointSegmentMeasurement`
- `SegmentSegmentMeasurement` = `_native.SegmentSegmentMeasurement`
- `PreparedPlanarCycle` = `_native.PreparedPlanarCycle`
- `SurfaceEnumerationLimits` = `_native.SurfaceEnumerationLimits`
- `PreparedNonplanarSurfaceFamily` = `_native.PreparedNonplanarSurfaceFamily`
- `PreparedCycle` = `_native.PreparedCycle`
- `PreparedCycleBatch` = `_native.PreparedCycleBatch`
- `SegmentCycleBatch` = `_native.SegmentCycleBatch`
- `DetailLevel` = `_native.DetailLevel`
- `Coordinates` = `Union[Sequence[float], Point]`
- `BoundsInput` = `Union[np.ndarray, Sequence[Sequence[float]]]`
- `ArrayValues` = `Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], Sequence[Sequence[Sequence[float]]]]`


<a id="hotpot-cheminfo-rules-mathematics-geometry-object"></a>
## `hotpot.cheminfo.rules.mathematics.geometry.object`

`hotpot/cheminfo/rules/mathematics/geometry/object.py`；显式 __all__: ('Point', 'Line', 'Segment', 'Plane', 'Triangle', 'Cycle')


<a id="hotpot-cheminfo-rules-mathematics-geometry-object-point"></a>
### `Point`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.object.Point`；基类 无。

- `__init__(self, coordinates: Iterable[float]) -> None`
- `from_coordinates(cls, coordinates: Iterable[float]) -> 'Point'` [classmethod]
- `__iter__(self) -> Iterator[float]`
- `x` [只读]: `float`
- `y` [只读]: `float`
- `z` [只读]: `float`
- `coordinates: Coordinates3` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-object-line"></a>
### `Line`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.object.Line`；基类 无。

- `__init__(self, origin: PointInput, direction: Iterable[float]) -> None`
- `from_points(cls, first: PointInput, second: PointInput) -> 'Line'` [classmethod]
- `origin: Point` [只读值字段]
- `direction: Coordinates3` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-object-segment"></a>
### `Segment`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.object.Segment`；基类 无。

- `__init__(self, start: PointInput, end: PointInput) -> None`
- `direction` [只读]: `Coordinates3`
- `length` [只读]: `float`
- `start: Point` [只读值字段]
- `end: Point` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-object-plane"></a>
### `Plane`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.object.Plane`；基类 无。

- `__init__(self, point: PointInput, normal: Iterable[float]) -> None`
- `point: Point` [只读值字段]
- `normal: Coordinates3` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-object-triangle"></a>
### `Triangle`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.object.Triangle`；基类 无。

- `__init__(self, first: PointInput, second: PointInput, third: PointInput) -> None`
- `vertices` [只读]: `Tuple[Point, Point, Point]`
- `edges` [只读]: `Tuple[Segment, Segment, Segment]`
- `first: Point` [只读值字段]
- `second: Point` [只读值字段]
- `third: Point` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-object-cycle"></a>
### `Cycle`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.object.Cycle`；基类 无。

- `__init__(self, vertices: Iterable[PointInput]) -> None`
- `__len__(self) -> int`
- `__iter__(self) -> Iterator[Point]`
- `edges` [只读]: `Tuple[Segment, ...]`
- `vertices: Tuple[Point, ...]` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `Coordinates3` = `Tuple[float, float, float]`
- `PointInput` = `Union['Point', Iterable[float]]`


<a id="hotpot-cheminfo-rules-mathematics-geometry-relation"></a>
## `hotpot.cheminfo.rules.mathematics.geometry.relation`

`hotpot/cheminfo/rules/mathematics/geometry/relation.py`；显式 __all__: ('PlanarityKind', 'LineRelationKind', 'PointCycleLocation', 'SurfaceEmbeddingState', 'SurfaceSegmentState', 'PiercingState', 'SegmentCycleFeature', 'SegmentCycleIndeterminacy', 'CycleSurfaceModel', 'PlanarityMeasurement', 'LineRelation', 'PointPairDistance', 'ClosestCycleEdge', 'SurfaceFamilyEvidence', 'SegmentCycleRelation', 'SegmentCycleScreening', 'measure_planarity', 'determine_line_relation', 'line_distance', 'point_segment_distance', 'segment_segment_distance', 'point_pair_distances', 'find_point_pairs_below_distance', 'locate_point_in_planar_cycle', 'iter_segment_cycle_relations', 'iter_segment_cycle_screenings', 'determine_segment_cycle_relation', 'closest_cycle_edge')


<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-planaritykind"></a>
### `PlanarityKind`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.PlanarityKind`；基类 `Enum`。

- `PLANAR` = `'planar'` [枚举常量]
- `NONPLANAR` = `'nonplanar'` [枚举常量]
- `DEGENERATE` = `'degenerate'` [枚举常量]
- `UNDETERMINED` = `'undetermined'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-linerelationkind"></a>
### `LineRelationKind`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.LineRelationKind`；基类 `Enum`。

- `INTERSECTING` = `'intersecting'` [枚举常量]
- `PARALLEL` = `'parallel'` [枚举常量]
- `COINCIDENT` = `'coincident'` [枚举常量]
- `SKEW` = `'skew'` [枚举常量]
- `DEGENERATE` = `'degenerate'` [枚举常量]
- `UNDETERMINED` = `'undetermined'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-pointcyclelocation"></a>
### `PointCycleLocation`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.PointCycleLocation`；基类 `Enum`。

- `INTERIOR` = `'interior'` [枚举常量]
- `BOUNDARY` = `'boundary'` [枚举常量]
- `EXTERIOR` = `'exterior'` [枚举常量]
- `UNDETERMINED` = `'undetermined'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-surfaceembeddingstate"></a>
### `SurfaceEmbeddingState`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.SurfaceEmbeddingState`；基类 `Enum`。

- `EMBEDDED` = `'embedded'` [枚举常量]
- `PROVEN_NON_EMBEDDED` = `'proven_non_embedded'` [枚举常量]
- `CONSTRUCTION_UNDETERMINED` = `'construction_undetermined'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-surfacesegmentstate"></a>
### `SurfaceSegmentState`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.SurfaceSegmentState`；基类 `Enum`。

- `INTERSECTING` = `'intersecting'` [枚举常量]
- `NON_PIERCING` = `'non_piercing'` [枚举常量]
- `EVALUATION_UNDETERMINED` = `'evaluation_undetermined'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-piercingstate"></a>
### `PiercingState`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.PiercingState`；基类 `Enum`。

- `PIERCES` = `'pierces'` [枚举常量]
- `DOES_NOT_PIERCE` = `'does_not_pierce'` [枚举常量]
- `UNDETERMINED` = `'undetermined'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-segmentcyclefeature"></a>
### `SegmentCycleFeature`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleFeature`；基类 `Enum`。

- `TRANSVERSE_INTERIOR` = `'transverse_interior'` [枚举常量]
- `LINE_EXTENSION_INTERIOR` = `'line_extension_interior'` [枚举常量]
- `CYCLE_EDGE_CONTACT` = `'cycle_edge_contact'` [枚举常量]
- `CYCLE_VERTEX_CONTACT` = `'cycle_vertex_contact'` [枚举常量]
- `SEGMENT_ENDPOINT_CONTACT` = `'segment_endpoint_contact'` [枚举常量]
- `COPLANAR_CONTACT` = `'coplanar_contact'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-segmentcycleindeterminacy"></a>
### `SegmentCycleIndeterminacy`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleIndeterminacy`；基类 `Enum`。

- `NONFINITE_INPUT` = `'nonfinite_input'` [枚举常量]
- `NUMERIC_BAND` = `'numeric_band'` [枚举常量]
- `TOLERANCE_DOMAIN` = `'tolerance_domain'` [枚举常量]
- `DEGENERATE_CYCLE` = `'degenerate_cycle'` [枚举常量]
- `DEGENERATE_SEGMENT` = `'degenerate_segment'` [枚举常量]
- `DEGENERATE_TRIANGLE` = `'degenerate_triangle'` [枚举常量]
- `SELF_INTERSECTION` = `'self_intersection'` [枚举常量]
- `SURFACE_DISAGREEMENT` = `'surface_disagreement'` [枚举常量]
- `INCOMPLETE_SURFACE_FAMILY` = `'incomplete_surface_family'` [枚举常量]
- `SURFACE_CONSTRUCTION` = `'surface_construction'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-cyclesurfacemodel"></a>
### `CycleSurfaceModel`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.CycleSurfaceModel`；基类 `Enum`。

- `PLANAR_POLYGON` = `'planar_polygon'` [枚举常量]
- `VERTEX_TRIANGULATION_FAMILY` = `'vertex_triangulation_family'` [枚举常量]

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-planaritymeasurement"></a>
### `PlanarityMeasurement`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.PlanarityMeasurement`；基类 无。

```python
PlanarityMeasurement(kind: PlanarityKind, centroid: Point, normal: Optional[Tuple[float, float, float]], singular_values: Tuple[float, float, float], maximum_deviation: float, rms_deviation: float, length_scale: float, length_tolerance: float)
```

- `kind: PlanarityKind` [只读值字段]
- `centroid: Point` [只读值字段]
- `normal: Optional[Tuple[float, float, float]]` [只读值字段]
- `singular_values: Tuple[float, float, float]` [只读值字段]
- `maximum_deviation: float` [只读值字段]
- `rms_deviation: float` [只读值字段]
- `length_scale: float` [只读值字段]
- `length_tolerance: float` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-linerelation"></a>
### `LineRelation`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.LineRelation`；基类 无。

```python
LineRelation(kind: LineRelationKind, distance: Optional[float], parallel_measure: float)
```

- `kind: LineRelationKind` [只读值字段]
- `distance: Optional[float]` [只读值字段]
- `parallel_measure: float` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-pointpairdistance"></a>
### `PointPairDistance`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.PointPairDistance`；基类 无。

```python
PointPairDistance(first_index: int, second_index: int, distance: float)
```

- `first_index: int` [只读值字段]
- `second_index: int` [只读值字段]
- `distance: float` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-closestcycleedge"></a>
### `ClosestCycleEdge`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.ClosestCycleEdge`；基类 无。

```python
ClosestCycleEdge(edge_index: int, edge: Segment, distance: float)
```

- `edge_index: int` [只读值字段]
- `edge: Segment` [只读值字段]
- `distance: float` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-surfacefamilyevidence"></a>
### `SurfaceFamilyEvidence`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.SurfaceFamilyEvidence`；基类 无。

```python
SurfaceFamilyEvidence(enumeration_complete: bool, enumerated_surface_count: int, embedded_surface_count: int, proven_non_embedded_surface_count: int, construction_undetermined_count: int, intersecting_surface_count: int, non_piercing_surface_count: int, evaluation_undetermined_count: int, segment_triangle_tests_used: int, triangle_pair_tests_used: int)
```

- `enumeration_complete: bool` [只读值字段]
- `enumerated_surface_count: int` [只读值字段]
- `embedded_surface_count: int` [只读值字段]
- `proven_non_embedded_surface_count: int` [只读值字段]
- `construction_undetermined_count: int` [只读值字段]
- `intersecting_surface_count: int` [只读值字段]
- `non_piercing_surface_count: int` [只读值字段]
- `evaluation_undetermined_count: int` [只读值字段]
- `segment_triangle_tests_used: int` [只读值字段]
- `triangle_pair_tests_used: int` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-segmentcyclerelation"></a>
### `SegmentCycleRelation`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleRelation`；基类 无。

```python
SegmentCycleRelation(state: PiercingState, features: FrozenSet[SegmentCycleFeature], indeterminacy_causes: FrozenSet[SegmentCycleIndeterminacy], surface_model: Optional[CycleSurfaceModel], intersection_points: Tuple[Point, ...], closest_boundary_edge: Optional[ClosestCycleEdge], surface_evidence: SurfaceFamilyEvidence, settings: GeometrySettings)
```

- `state: PiercingState` [只读值字段]
- `features: FrozenSet[SegmentCycleFeature]` [只读值字段]
- `indeterminacy_causes: FrozenSet[SegmentCycleIndeterminacy]` [只读值字段]
- `surface_model: Optional[CycleSurfaceModel]` [只读值字段]
- `intersection_points: Tuple[Point, ...]` [只读值字段]
- `closest_boundary_edge: Optional[ClosestCycleEdge]` [只读值字段]
- `surface_evidence: SurfaceFamilyEvidence` [只读值字段]
- `settings: GeometrySettings` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-segmentcyclescreening"></a>
### `SegmentCycleScreening`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.relation.SegmentCycleScreening`；基类 无。

```python
SegmentCycleScreening(state: PiercingState, relation: Optional[SegmentCycleRelation], aabb_separated: bool, surface_complete: bool)
```

- `state: PiercingState` [只读值字段]
- `relation: Optional[SegmentCycleRelation]` [只读值字段]
- `aabb_separated: bool` [只读值字段]
- `surface_complete: bool` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-measure-planarity"></a>
### `measure_planarity`

```python
measure_planarity(cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PlanarityMeasurement
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-determine-line-relation"></a>
### `determine_line_relation`

```python
determine_line_relation(first: Line, second: Line, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> LineRelation
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-line-distance"></a>
### `line_distance`

```python
line_distance(first: Line, second: Line, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-point-segment-distance"></a>
### `point_segment_distance`

```python
point_segment_distance(point: Point, segment: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-segment-segment-distance"></a>
### `segment_segment_distance`

```python
segment_segment_distance(first: Segment, second: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> float
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-point-pair-distances"></a>
### `point_pair_distances`

```python
point_pair_distances(points: Sequence[Point]) -> Tuple[PointPairDistance, ...]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-find-point-pairs-below-distance"></a>
### `find_point_pairs_below_distance`

```python
find_point_pairs_below_distance(points: Sequence[Point], threshold: float) -> Tuple[PointPairDistance, ...]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-locate-point-in-planar-cycle"></a>
### `locate_point_in_planar_cycle`

```python
locate_point_in_planar_cycle(point: Point, cycle: Cycle, plane: Plane, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> PointCycleLocation
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-iter-segment-cycle-screenings"></a>
### `iter_segment_cycle_screenings`

```python
iter_segment_cycle_screenings(segments: Iterable[Segment], cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Iterator[SegmentCycleScreening]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-iter-segment-cycle-relations"></a>
### `iter_segment_cycle_relations`

```python
iter_segment_cycle_relations(segments: Iterable[Segment], cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Iterator[SegmentCycleRelation]
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-determine-segment-cycle-relation"></a>
### `determine_segment_cycle_relation`

```python
determine_segment_cycle_relation(segment: Segment, cycle: Cycle, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> SegmentCycleRelation
```

<a id="hotpot-cheminfo-rules-mathematics-geometry-relation-closest-cycle-edge"></a>
### `closest_cycle_edge`

```python
closest_cycle_edge(cycle: Cycle, segment: Segment, settings: GeometrySettings=DEFAULT_GEOMETRY_SETTINGS) -> Optional[ClosestCycleEdge]
```


<a id="hotpot-cheminfo-rules-mathematics-geometry-settings"></a>
## `hotpot.cheminfo.rules.mathematics.geometry.settings`

`hotpot/cheminfo/rules/mathematics/geometry/settings.py`；显式 __all__: ('NumericToleranceSettings', 'SurfaceEnumerationSettings', 'GeometrySettings', 'DEFAULT_GEOMETRY_SETTINGS')


<a id="hotpot-cheminfo-rules-mathematics-geometry-settings-numerictolerancesettings"></a>
### `NumericToleranceSettings`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.settings.NumericToleranceSettings`；基类 无。

```python
NumericToleranceSettings(absolute_length: float = 1e-08, relative_length: float = 1e-10, parameter: float = 1e-10, machine_epsilon_factor: float = 64.0, predicate_guard_factor: float = 4.0, planarity_factor: float = 1.0, winding_residual: float = 1e-10, intersection_merge_factor: float = 4.0, aabb_padding_factor: float = 4.0)
```

- `absolute_length: float` = `1e-08` [只读值字段]
- `relative_length: float` = `1e-10` [只读值字段]
- `parameter: float` = `1e-10` [只读值字段]
- `machine_epsilon_factor: float` = `64.0` [只读值字段]
- `predicate_guard_factor: float` = `4.0` [只读值字段]
- `planarity_factor: float` = `1.0` [只读值字段]
- `winding_residual: float` = `1e-10` [只读值字段]
- `intersection_merge_factor: float` = `4.0` [只读值字段]
- `aabb_padding_factor: float` = `4.0` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-settings-surfaceenumerationsettings"></a>
### `SurfaceEnumerationSettings`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.settings.SurfaceEnumerationSettings`；基类 无。

```python
SurfaceEnumerationSettings(maximum_cycle_vertices: int = 8, maximum_surface_count: int = 132, maximum_segment_triangle_tests: int = 792, maximum_triangle_pair_tests: int = 1980)
```

- `maximum_cycle_vertices: int` = `8` [只读值字段]
- `maximum_surface_count: int` = `132` [只读值字段]
- `maximum_segment_triangle_tests: int` = `792` [只读值字段]
- `maximum_triangle_pair_tests: int` = `1980` [只读值字段]

<a id="hotpot-cheminfo-rules-mathematics-geometry-settings-geometrysettings"></a>
### `GeometrySettings`

定义 owner `hotpot.cheminfo.rules.mathematics.geometry.settings.GeometrySettings`；基类 无。

```python
GeometrySettings(tolerance: NumericToleranceSettings = field(default_factory=NumericToleranceSettings), surface: SurfaceEnumerationSettings = field(default_factory=SurfaceEnumerationSettings))
```

- `tolerance: NumericToleranceSettings` = `field(default_factory=NumericToleranceSettings)` [只读值字段]
- `surface: SurfaceEnumerationSettings` = `field(default_factory=SurfaceEnumerationSettings)` [只读值字段]

模块值/类型别名（声明存在不构成事实写许可）：

- `DEFAULT_GEOMETRY_SETTINGS` = `GeometrySettings()`


<a id="hotpot-cheminfo-rules-mathematics-graph"></a>
## `hotpot.cheminfo.rules.mathematics.graph`

`hotpot/cheminfo/rules/mathematics/graph/__init__.py`；显式 __all__: ('linkmat2adj', 'adj2laplacian', 'graph_dfs_path', 'DEFAULT_RELEVANT_CYCLE_LIMIT', 'RelevantCycleLimitExceeded', 'relevant_cycles')

- 导出 `linkmat2adj` → `hotpot.cheminfo.rules.mathematics.graph.matrix.linkmat2adj`；`linkmat2adj(note_num: int, linkmat: np.ndarray) -> np.ndarray`
- 导出 `adj2laplacian` → `hotpot.cheminfo.rules.mathematics.graph.matrix.adj2laplacian`；`adj2laplacian(adj: np.ndarray, norm: bool=True) -> np.ndarray`
- 导出 `graph_dfs_path` → `hotpot.cheminfo.rules.mathematics.graph.traversal.graph_dfs_path`；`graph_dfs_path(node_count: int, edges: Iterable[tuple[int, int]], start_node: int=None, scope_nodes: Container=None, min_deep: int=None, max_deep: int=None) -> Optional[list[int]]`
- 导出 `DEFAULT_RELEVANT_CYCLE_LIMIT` → `hotpot.cheminfo.rules.mathematics.graph.cycles.DEFAULT_RELEVANT_CYCLE_LIMIT`
- 导出 `RelevantCycleLimitExceeded` → `hotpot.cheminfo.rules.mathematics.graph.cycles.RelevantCycleLimitExceeded`
- 导出 `relevant_cycles` → `hotpot.cheminfo.rules.mathematics.graph.cycles.relevant_cycles`；`relevant_cycles(edges: Iterable[EdgeInput], *, max_size: Optional[int]=None, max_cycles: Optional[int]=DEFAULT_RELEVANT_CYCLE_LIMIT) -> tuple[NodeCycle, ...]`


<a id="hotpot-cheminfo-rules-mathematics-graph-components"></a>
## `hotpot.cheminfo.rules.mathematics.graph.components`

`hotpot/cheminfo/rules/mathematics/graph/components.py`；显式 __all__: ('connected_components',)


<a id="hotpot-cheminfo-rules-mathematics-graph-components-connected-components"></a>
### `connected_components`

```python
connected_components(node_count: int, edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, ...], ...]
```


<a id="hotpot-cheminfo-rules-mathematics-graph-cycles"></a>
## `hotpot.cheminfo.rules.mathematics.graph.cycles`

`hotpot/cheminfo/rules/mathematics/graph/cycles.py`；显式 __all__: ('DEFAULT_RELEVANT_CYCLE_LIMIT', 'RelevantCycleLimitExceeded', 'relevant_cycles')


<a id="hotpot-cheminfo-rules-mathematics-graph-cycles-relevantcyclelimitexceeded"></a>
### `RelevantCycleLimitExceeded`

定义 owner `hotpot.cheminfo.rules.mathematics.graph.cycles.RelevantCycleLimitExceeded`；基类 `RuntimeError`。


<a id="hotpot-cheminfo-rules-mathematics-graph-cycles-relevant-cycles"></a>
### `relevant_cycles`

```python
relevant_cycles(edges: Iterable[EdgeInput], *, max_size: Optional[int]=None, max_cycles: Optional[int]=DEFAULT_RELEVANT_CYCLE_LIMIT) -> tuple[NodeCycle, ...]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `NodeIndex` = `int`
- `EdgeInput` = `Sequence[int]`
- `DenseEdge` = `tuple[int, int]`
- `NodeCycle` = `tuple[NodeIndex, ...]`
- `DEFAULT_RELEVANT_CYCLE_LIMIT` = `10000`


<a id="hotpot-cheminfo-rules-mathematics-graph-matrix"></a>
## `hotpot.cheminfo.rules.mathematics.graph.matrix`

`hotpot/cheminfo/rules/mathematics/graph/matrix.py`；显式 __all__: ('linkmat2adj', 'adj2laplacian')


<a id="hotpot-cheminfo-rules-mathematics-graph-matrix-linkmat2adj"></a>
### `linkmat2adj`

```python
linkmat2adj(note_num: int, linkmat: np.ndarray) -> np.ndarray
```

<a id="hotpot-cheminfo-rules-mathematics-graph-matrix-adj2laplacian"></a>
### `adj2laplacian`

```python
adj2laplacian(adj: np.ndarray, norm: bool=True) -> np.ndarray
```


<a id="hotpot-cheminfo-rules-mathematics-graph-paths"></a>
## `hotpot.cheminfo.rules.mathematics.graph.paths`

`hotpot/cheminfo/rules/mathematics/graph/paths.py`；显式 __all__: ('all_simple_paths', 'depth_first_path', 'longest_simple_path', 'shortest_path')


<a id="hotpot-cheminfo-rules-mathematics-graph-paths-shortest-path"></a>
### `shortest_path`

```python
shortest_path(node_count: int, edges: Iterable[tuple[int, int]], source: int, target: int) -> list[int]
```

<a id="hotpot-cheminfo-rules-mathematics-graph-paths-all-simple-paths"></a>
### `all_simple_paths`

```python
all_simple_paths(node_count: int, edges: Iterable[tuple[int, int]], source: int, target: int, cutoff: int | None=None) -> list[list[int]]
```

<a id="hotpot-cheminfo-rules-mathematics-graph-paths-longest-simple-path"></a>
### `longest_simple_path`

```python
longest_simple_path(node_count: int, edges: Iterable[tuple[int, int]], source: int | None=None, target: int | None=None) -> list[int]
```

<a id="hotpot-cheminfo-rules-mathematics-graph-paths-depth-first-path"></a>
### `depth_first_path`

```python
depth_first_path(node_count: int, edges: Iterable[tuple[int, int]], start_node: int=0, scope_nodes: Container[int] | None=None, min_depth: int | None=None, max_depth: int | None=None) -> list[int] | None
```


<a id="hotpot-cheminfo-rules-mathematics-graph-traversal"></a>
## `hotpot.cheminfo.rules.mathematics.graph.traversal`

`hotpot/cheminfo/rules/mathematics/graph/traversal.py`；显式 __all__: ('graph_dfs_path',)


<a id="hotpot-cheminfo-rules-mathematics-graph-traversal-graph-dfs-path"></a>
### `graph_dfs_path`

```python
graph_dfs_path(node_count: int, edges: Iterable[tuple[int, int]], start_node: int=None, scope_nodes: Container=None, min_deep: int=None, max_deep: int=None) -> Optional[list[int]]
```


<a id="hotpot-cheminfo-rules-mathematics-graph-matching"></a>
## `hotpot.cheminfo.rules.mathematics.graph_matching`

`hotpot/cheminfo/rules/mathematics/graph_matching/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-mathematics-internal-coordinates"></a>
## `hotpot.cheminfo.rules.mathematics.internal_coordinates`

`hotpot/cheminfo/rules/mathematics/internal_coordinates.py`；显式 __all__: ('InternalCoordinates',)


<a id="hotpot-cheminfo-rules-mathematics-internal-coordinates-internalcoordinates"></a>
### `InternalCoordinates`

定义 owner `hotpot.cheminfo.rules.mathematics.internal_coordinates.InternalCoordinates`；基类 无。

- `__init__(self, coordinates: np.ndarray, link_matrix: np.ndarray)`
- `calc_zmat_idx(link_matrix: np.ndarray, node_count: Optional[int]=None) -> np.ndarray` [staticmethod]
- `calc_zmat(link_matrix: np.ndarray, coords: np.ndarray) -> np.ndarray` [staticmethod]
- `calc_angle(a: types.ArrayLike, b: types.ArrayLike, c: types.ArrayLike)` [staticmethod]
- `calc_dehedral(a: types.ArrayLike, b: types.ArrayLike, c: types.ArrayLike, d: types.ArrayLike, return_degree: bool=True)` [staticmethod]
- `coordinates` = `coordinates` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）
- `link_matrix` = `link_matrix` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）


<a id="hotpot-cheminfo-rules-mathematics-periodic"></a>
## `hotpot.cheminfo.rules.mathematics.periodic`

`hotpot/cheminfo/rules/mathematics/periodic.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-mathematics-sampling"></a>
## `hotpot.cheminfo.rules.mathematics.sampling`

`hotpot/cheminfo/rules/mathematics/sampling.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-mathematics-symmetry"></a>
## `hotpot.cheminfo.rules.mathematics.symmetry`

`hotpot/cheminfo/rules/mathematics/symmetry.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-mathematics-topology"></a>
## `hotpot.cheminfo.rules.mathematics.topology`

`hotpot/cheminfo/rules/mathematics/topology.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-mathematics-view-geometry"></a>
## `hotpot.cheminfo.rules.mathematics.view_geometry`

`hotpot/cheminfo/rules/mathematics/view_geometry.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-physics"></a>
## `hotpot.cheminfo.rules.physics`

`hotpot/cheminfo/rules/physics/__init__.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-physics-forcefields"></a>
## `hotpot.cheminfo.rules.physics.forcefields`

`hotpot/cheminfo/rules/physics/forcefields/__init__.py`；显式 __all__: ('OptimizationStoppingCriteria',)

- 导出 `OptimizationStoppingCriteria` → `hotpot.cheminfo.rules.physics.forcefields.contracts.OptimizationStoppingCriteria`


<a id="hotpot-cheminfo-rules-physics-forcefields-acceptance"></a>
## `hotpot.cheminfo.rules.physics.forcefields.acceptance`

`hotpot/cheminfo/rules/physics/forcefields/acceptance.py`；显式 __all__: 无



<a id="hotpot-cheminfo-rules-physics-forcefields-contracts"></a>
## `hotpot.cheminfo.rules.physics.forcefields.contracts`

`hotpot/cheminfo/rules/physics/forcefields/contracts.py`；显式 __all__: ('OptimizationStoppingCriteria', 'AcceptanceCheck', 'StructureAcceptanceThresholds', 'ForceFieldAcceptanceEvidence', 'ForceFieldDiagnosticValue')


<a id="hotpot-cheminfo-rules-physics-forcefields-contracts-optimizationstoppingcriteria"></a>
### `OptimizationStoppingCriteria`

定义 owner `hotpot.cheminfo.rules.physics.forcefields.contracts.OptimizationStoppingCriteria`；基类 无。

```python
OptimizationStoppingCriteria(window: int = 5, maximum_energy_change_kj_mol: float = 0.0001, maximum_atom_displacement_angstrom: float = 0.0001, maximum_rms_gradient_kj_mol_angstrom: float = 1.0, maximum_gradient_kj_mol_angstrom: float = 5.0)
```

- `window: int` = `5` [只读值字段]
- `maximum_energy_change_kj_mol: float` = `0.0001` [只读值字段]
- `maximum_atom_displacement_angstrom: float` = `0.0001` [只读值字段]
- `maximum_rms_gradient_kj_mol_angstrom: float` = `1.0` [只读值字段]
- `maximum_gradient_kj_mol_angstrom: float` = `5.0` [只读值字段]

<a id="hotpot-cheminfo-rules-physics-forcefields-contracts-acceptancecheck"></a>
### `AcceptanceCheck`

定义 owner `hotpot.cheminfo.rules.physics.forcefields.contracts.AcceptanceCheck`；基类 无。

```python
AcceptanceCheck(name: str, passed: bool, severity: Literal['info', 'warning', 'error'] = 'error', measured: ForceFieldDiagnosticValue = None, threshold: ForceFieldDiagnosticValue = None, atom_indices: Tuple[int, ...] = (), bond_indices: Tuple[int, ...] = (), message: str = '')
```

- `name: str` [只读值字段]
- `passed: bool` [只读值字段]
- `severity: Literal['info', 'warning', 'error']` = `'error'` [只读值字段]
- `measured: ForceFieldDiagnosticValue` = `None` [只读值字段]
- `threshold: ForceFieldDiagnosticValue` = `None` [只读值字段]
- `atom_indices: Tuple[int, ...]` = `()` [只读值字段]
- `bond_indices: Tuple[int, ...]` = `()` [只读值字段]
- `message: str` = `''` [只读值字段]

<a id="hotpot-cheminfo-rules-physics-forcefields-contracts-structureacceptancethresholds"></a>
### `StructureAcceptanceThresholds`

定义 owner `hotpot.cheminfo.rules.physics.forcefields.contracts.StructureAcceptanceThresholds`；基类 无。

```python
StructureAcceptanceThresholds(overlap_tolerance: float = 0.001, basic_minimum_distance: float = 0.4, standard_minimum_distance: float = 0.5, standard_covalent_radius_scale: float = 0.55, maximum_bond_distance: float = 30.0, covalent_bond_ratio: Tuple[float, float] = (0.65, 1.45), metal_ligand_bond_ratio: Tuple[float, float] = (0.65, 1.6), strict_rms_gradient: float = 1.0, strict_max_gradient: float = 5.0, strict_energy_change: float = 0.0001, strict_max_displacement: float = 0.0001, strict_stability_window: int = 5)
```

- `overlap_tolerance: float` = `0.001` [只读值字段]
- `basic_minimum_distance: float` = `0.4` [只读值字段]
- `standard_minimum_distance: float` = `0.5` [只读值字段]
- `standard_covalent_radius_scale: float` = `0.55` [只读值字段]
- `maximum_bond_distance: float` = `30.0` [只读值字段]
- `covalent_bond_ratio: Tuple[float, float]` = `(0.65, 1.45)` [只读值字段]
- `metal_ligand_bond_ratio: Tuple[float, float]` = `(0.65, 1.6)` [只读值字段]
- `strict_rms_gradient: float` = `1.0` [只读值字段]
- `strict_max_gradient: float` = `5.0` [只读值字段]
- `strict_energy_change: float` = `0.0001` [只读值字段]
- `strict_max_displacement: float` = `0.0001` [只读值字段]
- `strict_stability_window: int` = `5` [只读值字段]

<a id="hotpot-cheminfo-rules-physics-forcefields-contracts-forcefieldacceptanceevidence"></a>
### `ForceFieldAcceptanceEvidence`

定义 owner `hotpot.cheminfo.rules.physics.forcefields.contracts.ForceFieldAcceptanceEvidence`；基类 `TypedDict`。

- `setup_succeeded: bool` [字段声明；写权限依具体合同]
- `converged: bool` [字段声明；写权限依具体合同]
- `epochs_completed: int` [字段声明；写权限依具体合同]
- `segment_epochs_completed: int` [字段声明；写权限依具体合同]
- `final_energy: float` [字段声明；写权限依具体合同]
- `energy_unit: str` [字段声明；写权限依具体合同]
- `rms_gradient: float` [字段声明；写权限依具体合同]
- `max_gradient: float` [字段声明；写权限依具体合同]
- `exploded: bool` [字段声明；写权限依具体合同]
- `energy_changes: Sequence[float]` [字段声明；写权限依具体合同]
- `max_displacements: Sequence[float]` [字段声明；写权限依具体合同]

模块值/类型别名（声明存在不构成事实写许可）：

- `AcceptanceLevel` = `Literal['off', 'basic', 'standard', 'strict']`
- `ForceFieldStage` = `Literal['candidate', 'final']`
- `ForceFieldDiagnosticValue` = `Union[None, bool, int, float, str, AtomTopologySignature, BondTopology, Tuple['ForceFieldDiagnosticValue', ...], Mapping[str, 'ForceFieldDiagnosticValue']]`


<a id="hotpot-cheminfo-rules-physics-forcefields-coordinates"></a>
## `hotpot.cheminfo.rules.physics.forcefields.coordinates`

`hotpot/cheminfo/rules/physics/forcefields/coordinates.py`；显式 __all__: ()



<a id="hotpot-cheminfo-rules-physics-forcefields-settings"></a>
## `hotpot.cheminfo.rules.physics.forcefields.settings`

`hotpot/cheminfo/rules/physics/forcefields/settings.py`；显式 __all__: ()
