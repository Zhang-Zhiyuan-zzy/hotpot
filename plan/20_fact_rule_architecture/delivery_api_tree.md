# Plan20 当前完整交付 API 树

源码身份：`2c2ef889abb6adcceba043b606da2e718dbea109`，分支 `experiment/plan20-kekulize-aromatic-protocol`。日期 `2026-11-05`。
本树覆盖当前 chemistry 内核、其保留工具封装、MCA/CBond/FF、xTB、pipeline 与命令层。每个模块链接到完整签名、属性权限、字段、别名和本地继承成员。
现有非 Plan20 插件/研究及 dataset 辅助层另见 [辅助模块/导出索引](delivery_api/auxiliary_module_index.md)；其完整静态成员参考见 [辅助模块及类入口](delivery_api/auxiliary_index.md)，不声称已重新验收其科学或可选依赖运行行为。
主题全签名：[结构](delivery_api/structure_api.md) · [视图](delivery_api/views_api.md) · [规则](delivery_api/rules_api.md) · [操作](delivery_api/operations_api.md) · [adapter](delivery_api/adapters_api.md) · [wrapper](delivery_api/wrappers_api.md) · [workflow](delivery_api/workflows_api.md) · [CLI Python](delivery_api/cli_api.md)。补充：[动态权限](delivery_api/dynamic_permissions.md) · [CLI 参数](delivery_api/cli_inventory.md) · [native 声明](delivery_api/native_contracts.md)。原始机器目录、提取脚本与身份清单见 [按需读取的原始库存索引](delivery_api/README.md#raw-inventory)。

## 常用化学类型入口

### [`hotpot.cheminfo.structure.molecule.Molecule`](delivery_api/structure_api.md#hotpot-cheminfo-structure-molecule-molecule)

- 属性/字段：`charge`（读写）, `spin_mult`（读写）, `crystal`（只读）, `hydrogens`（只读）, `sum_explicit_hydrogens`（只读）, `sum_implicit_hydrogens`（只读）, `has_hydrogens`（只读）, `cid`（只读）, `name`（只读）, `cas`（只读）, `atom_pairs`（只读）, `angles`（只读）, `torsions`（只读）, `atom_matrix`（只读）, `atoms`（只读）, `c_bonds`（只读）, `bonds`（只读）, `sum_atoms_charge`（只读）, `default_unpaired_electrons`（只读）, `default_spin_mult`（只读）, `conformers`（只读）, `conformers_number`（只读）, `components`（只读）, `coordinates`（读写）, `atom_pairwise_index`（只读）, `pair_dist`（只读）, `dist_matrix`（只读）, `element_counts`（只读）, `formula`（只读）, `is_disorder`（只读）, `has_3d`（只读）, `has_metal`（只读）, `heavy_atoms`（只读）, `mca_sites`（只读）, `is_organic`（只读）, `is_full_halogenated`（只读）, `link_matrix`（只读）, `metals`（只读）, `atom_id_dict`（只读）, `bond_id_dict`（只读）, `InChi`（只读）, `smiles`（只读）, `kekulize_smiles`（只读）, `rings`（只读）, `cycle_basis`（只读）, `all_cycle_basis`（只读）, `basis_rings`（只读）, `all_basis_rings`（只读）, `aromatic_joint_rings`（只读）, `rings_small`（只读）, `ligand_rings`（只读）, `weight`（只读）, `energy`（读写）, `partial_charge`（读写）, `gibbs`（读写）, `force`（读写）, `zero_point`（读写）, `thermo`（读写）, `capacity`（读写）, `properties`（读写）, `identifier`（读写）
- 方法/协议：`__init__`, `__dir__`, `__repr__`, `__add__`, `__copy__`, `__getstate__`, `__setstate__`, `__reduce_ex__`, `updata_atoms`, `copy`, `create_crystal`, `read_one`, `calc_mol_default_charge`, `simple_paths`, `add_atom`, `build_all_pair_links`, `auto_pair_metal`, `replace_atom`, `add_bond`, `add_bonds`, `add_component`, `add_hydrogens`, `get_thermo`, `clear_constraints`, `assign_bond_order`, `bond`, `build3d`, `update_mol_charge`, `determine_rings_aromatic`, `conformer_load`, `conformer_add`, `conformer_clear`, `conformer_get`, `optimize`, `calc_atom_valence`, `calc_implicit_hydrogens`, `constraint_bonds_angles`, `create_atom`, `get_partial_charge`, `assign_partial_charge`, `graph_spectral`, `bond_ring_piercing_state`, `bond_ring_relations`, `link_atoms`, `refresh_atom_id`, `remove_atom`, `remove_atoms`, `remove_bond`, `remove_bonds`, `force_remove_polar_hydrogens`, `remove_hydrogens`, `remove_metals`, `search_substructure`, `set_default_valence`, `similarity`, `longest_path`, `canonical_tree`, `shortest_path_index`, `shortest_path`, `shortest_paths_indices`, `rings_for_scope`, `to_pyg`, `from_pyg`, `to_obmol`, `to_rdmol`, `to_pybel_mol`, `translation`, `write`

### [`hotpot.cheminfo.structure.atom.Atom`](delivery_api/structure_api.md#hotpot-cheminfo-structure-atom-atom)

- 属性/字段：`isotope`（读写）, `attrs`（只读）, `period`（只读）, `group`（只读）, `polar_hydrogen_site`（只读）, `bonds`（只读）, `constraint`（读写）, `coordinates`（读写）, `exact_mass`（只读）, `electronegativity`（只读）, `explicit_hydrogens`（只读）, `covalent_radius`（只读）, `density`（只读）, `hyb`（只读）, `idx`（只读）, `mca`（只读）, `is_error_electron_configure`（只读）, `is_hydrogen`（只读）, `is_polar_hydrogen`（只读）, `is_lanthanide`（只读）, `is_actinide`（只读）, `is_noble_gases`（只读）, `is_halogens`（只读）, `is_metal`（只读）, `label`（只读）, `mass`（只读）, `mol`（只读）, `molecule`（只读）, `neigh_idx`（只读）, `hydrogens`（只读）, `h_count`（只读）, `connectivity`（只读）, `neighbours`（只读）, `heavy_neighbours`（只读）, `missing_electrons_element`（只读）, `missing_electrons`（只读）, `open_shell_electrons`（只读）, `electron_configuration`（只读）, `l`（只读）, `n`（只读）, `oxidation_state`（只读）, `sum_bond_orders`（只读）, `sum_heavy_cov_orders`（只读）, `sum_covalent_orders`（只读）, `symbol`（读写）, `vector`（只读）, `atomic_number`（读写）, `s`（只读）, `p`（只读）, `d`（只读）, `f`（只读）, `g`（只读）, `formal_charge`（读写）, `partial_charge`（只读）, `is_aromatic`（只读）, `x`（读写）, `y`（读写）, `z`（读写）, `valence`（只读）, `implicit_hydrogens`（读写）, `id`（读写）, `x_constraint`（读写）, `y_constraint`（读写）, `z_constraint`（读写）, `frac_valence`（读写）, `attrs_enumerator`（只读）, `attr_dict`（只读）, `in_ring`（只读）, `in_organic`（只读）, `rings`（只读）
- 方法/协议：`__init__`, `__reduce_ex__`, `__setstate__`, `add_atom`, `random_point_on_sphere`, `add_hydrogen`, `calc_implicit_hydrogens`, `link_with`, `calc_electron_config`, `get_formal_charge`, `get_valence`, `set_valence_to_default`, `__setattr__`, `__repr__`, `__copy__`, `__dir__`, `__getattr__`

### [`hotpot.cheminfo.structure.bond.Bond`](delivery_api/structure_api.md#hotpot-cheminfo-structure-bond-bond)

- 属性/字段：`attrs`（只读）, `atom1`（只读）, `atom2`（只读）, `bond_kind`（只读）, `bond_direction`（只读）, `bond_source`（只读）, `bond_source_metadata`（只读）, `attr_dict`（只读）, `bond_segment`（只读）, `idx`（只读）, `ideal_bond_length`（只读）, `length`（只读）, `label`（只读）, `is_aromatic`（只读）, `is_covalent`（只读）, `is_heavy_covalent`（只读）, `is_metal_ligand_bond`（只读）, `rotatable`（只读）, `bond_order`（读写）, `constraint`（读写）, `id`（读写）, `atoms`（只读）, `atoms_indices`（只读）, `bonds`（只读）, `mol`（只读）, `has_metal`（只读）, `atomN`（只读）, `bondN`（只读）, `aNidx`（只读）, `attrs_enumerator`（只读）, `in_ring`（只读）, `in_organic`（只读）, `rings`（只读）
- 方法/协议：`__init__`, `__reduce_ex__`, `__setstate__`, `__repr__`, `__eq__`, `__hash__`, `__getattr__`, `another_end`, `bond_segment_distance`, `__setattr__`, `__len__`, `__contains__`, `to_mol`, `__copy__`, `__dir__`

### [`hotpot.cheminfo.structure.bond_kind.BondKind`](delivery_api/structure_api.md#hotpot-cheminfo-structure-bond-kind-bondkind)

- 属性/字段：`SINGLE`（只读）, `DOUBLE`（只读）, `TRIPLE`（只读）, `AROMATIC`（只读）, `ZERO`（只读）, `DATIVE`（只读）, `UNKNOWN`（只读）

### [`hotpot.cheminfo.structure.crystal.Crystal`](delivery_api/structure_api.md#hotpot-cheminfo-structure-crystal-crystal)

- 属性/字段：`molecule`（只读）, `matrix`（只读）, `volume`（只读）, `space_group`（只读）, `symm_units`（只读）, `site_relations`（只读）, `bond_images`（只读）, `representation`（只读）, `image_map`（只读）, `a`（读写）, `b`（读写）, `c`（读写）, `alpha`（读写）, `beta`（读写）, `gamma`（读写）
- 方法/协议：`__init__`, `from_matrix`, `from_cif`, `to_cif`, `to_pyg`, `from_pyg`, `set_cell`, `set_space_group`, `set_site_relations`, `to_fractional`, `to_cartesian`, `operate`, `unit_cell`, `unit_molecules`, `expand`, `same_molecule_candidates`, `molecular_symmetry`, `linking_number`, `geometry_findings`, `copy`, `__reduce_ex__`

### [`hotpot.cheminfo.structure.symmetry.SpaceGroup`](delivery_api/structure_api.md#hotpot-cheminfo-structure-symmetry-spacegroup)

- 属性/字段：`number`（只读）, `hermann_mauguin`（只读）, `hall`（只读）, `setting`（只读）, `crystal_system`（只读）, `point_group`（只读）, `symm_units`（只读）, `centering_translations`（只读）, `translation_basis`（只读）, `point_rotations`（只读）, `cell_order`（只读）, `point_group_order`（只读）
- 方法/协议：`__init__`, `__setattr__`, `__delattr__`, `from_operations`, `identity`, `lift`, `contains`, `__contains__`, `change_setting`, `__eq__`, `__hash__`, `__repr__`, `__reduce__`, `__copy__`, `__deepcopy__`

### [`hotpot.cheminfo.structure.symmetry.SymmUnit`](delivery_api/structure_api.md#hotpot-cheminfo-structure-symmetry-symmunit)

- 属性/字段：`R`（只读）, `t`（只读）, `context`（只读）, `xyz`（只读）, `kind`（只读）
- 方法/协议：`from_affine`, `from_xyz`, `identity`, `operate`, `compose`, `__matmul__`, `inverse`, `__invert__`, `power`, `__pow__`, `reduce_to_cell`, `__eq__`, `__hash__`, `__repr__`, `__reduce__`, `__copy__`, `__deepcopy__`

### [`hotpot.cheminfo.structure.bundle.MolBundle`](delivery_api/structure_api.md#hotpot-cheminfo-structure-bundle-molbundle)

- 属性/字段：`list_mol`（读写）
- 方法/协议：`__init__`, `to_pyg_dataset`

### [`hotpot.cheminfo.structure.conformer.Conformers`](delivery_api/structure_api.md#hotpot-cheminfo-structure-conformer-conformers)

- 方法/协议：`__init__`, `__getstate__`, `__setstate__`, `__len__`, `__getitem__`, `__iter__`, `index_attr`, `add`, `clear`, `coordinates`, `energy`, `gibbs`, `partial_charge`, `force`

### [`hotpot.cheminfo.structure.conformer.ConformerBatch`](delivery_api/structure_api.md#hotpot-cheminfo-structure-conformer-conformerbatch)

- 属性/字段：`coordinates`（只读）, `ordinals`（只读）
- 方法/协议：`__reduce__`

### [`hotpot.cheminfo.structure.conformer.EnergyForceResult`](delivery_api/structure_api.md#hotpot-cheminfo-structure-conformer-energyforceresult)

- 属性/字段：`atom_uids`（只读）, `values`（只读）, `frame_indices`（只读）, `method`（只读）, `method_version`（只读）, `parameters`（只读）, `units`（只读）, `provenance`（只读）
- 方法/协议：`__reduce__`, `__deepcopy__`

### [`hotpot.cheminfo.structure.conformer.CoordinateFrame`](delivery_api/structure_api.md#hotpot-cheminfo-structure-conformer-coordinateframe)

- 属性/字段：`atom_uids`（只读）, `coordinates`（只读）, `observations`（只读）, `observation_index`（只读）, `provenance`（只读）
- 方法/协议：`__reduce__`, `__deepcopy__`

### [`hotpot.cheminfo.structure.conformer.ConformerEnsembleResult`](delivery_api/structure_api.md#hotpot-cheminfo-structure-conformer-conformerensembleresult)

- 属性/字段：`atom_uids`（只读）, `batch`（只读）, `observations`（只读）, `input_source`（只读）
- 方法/协议：`__reduce__`, `__deepcopy__`

### [`hotpot.cheminfo.views.sequence.AtomSeq`](delivery_api/views_api.md#hotpot-cheminfo-views-sequence-atomseq)

- 属性/字段：`atoms`（只读）, `atoms_indices`（只读）, `bonds`（只读）, `mol`（只读）, `has_metal`（只读）, `atomN`（只读）, `bondN`（只读）, `aNidx`（只读）
- 方法/协议：`__setattr__`, `__reduce_ex__`, `__init__`, `__repr__`, `__getattr__`, `__len__`, `__contains__`, `__eq__`, `to_mol`

### [`hotpot.cheminfo.views.geometry.Angle`](delivery_api/views_api.md#hotpot-cheminfo-views-geometry-angle)

- 属性/字段：`degrees`（只读）, `constraint`（读写）, `atoms`（只读）, `atoms_indices`（只读）, `bonds`（只读）, `mol`（只读）, `has_metal`（只读）, `atomN`（只读）, `bondN`（只读）, `aNidx`（只读）
- 方法/协议：`__init__`, `__setattr__`, `__reduce_ex__`, `__repr__`, `__getattr__`, `__len__`, `__contains__`, `__eq__`, `to_mol`

### [`hotpot.cheminfo.views.geometry.Torsion`](delivery_api/views_api.md#hotpot-cheminfo-views-geometry-torsion)

- 属性/字段：`degrees`（只读）, `rotatable`（只读）, `constraint`（读写）, `atoms`（只读）, `atoms_indices`（只读）, `bonds`（只读）, `mol`（只读）, `has_metal`（只读）, `atomN`（只读）, `bondN`（只读）, `aNidx`（只读）
- 方法/协议：`__init__`, `__setattr__`, `__reduce_ex__`, `__repr__`, `__getattr__`, `__len__`, `__contains__`, `__eq__`, `to_mol`

### [`hotpot.cheminfo.views.topology.Ring`](delivery_api/views_api.md#hotpot-cheminfo-views-topology-ring)

- 属性/字段：`is_aromatic`（只读）, `is_disorder`（只读）, `has_metal`（只读）, `has_3d`（只读）, `geometry_cycle`（只读）, `atoms`（只读）, `atoms_indices`（只读）, `bonds`（只读）, `mol`（只读）, `atomN`（只读）, `bondN`（只读）, `aNidx`（只读）
- 方法/协议：`__init__`, `to_mol`, `to_pair_edge`, `joint_with`, `joint_ring`, `next_atom`, `relation_to_bond`, `closest_edge_to_bond`, `__setattr__`, `__reduce_ex__`, `__repr__`, `__getattr__`, `__len__`, `__contains__`, `__eq__`

### [`hotpot.cheminfo.views.topology.JointRing`](delivery_api/views_api.md#hotpot-cheminfo-views-topology-jointring)

- 属性/字段：`rings`（只读）, `atoms`（只读）, `bonds`（只读）, `mol`（只读）, `empty`（只读）
- 方法/协议：`__setattr__`, `__reduce_ex__`, `__init__`, `__contains__`, `index`, `index_bond`, `atom_neigh_atom`, `atom_neigh_bond`, `bond_neigh_bond`, `check_kekulize`

### [`hotpot.cheminfo.views.pairs.AtomPair`](delivery_api/views_api.md#hotpot-cheminfo-views-pairs-atompair)

- 属性/字段：`atoms`（只读）, `atom1`（只读）, `atom2`（只读）, `mol`（只读）, `wiberg_bond_order`（读写）, `attrs`（只读）, `shortest_path`（只读）, `length_shortest_path`（只读）, `distance`（只读）, `attr_names`（只读）
- 方法/协议：`__init__`, `__reduce_ex__`, `__repr__`, `__eq__`, `__contains__`

### [`hotpot.cheminfo.views.pairs.AtomPairs`](delivery_api/views_api.md#hotpot-cheminfo-views-pairs-atompairs)

- 属性/字段：`mol`（只读）, `pair_distance`（只读）, `idx_matrix`（只读）
- 方法/协议：`__init__`, `__reduce_ex__`, `__len__`, `__iter__`, `__getitem__`, `getdefault`, `copy`, `update_pairs`, `pairs_shortest_path`, `get`, `items`, `keys`, `values`, `__contains__`, `__eq__`, `__ne__`

## 完整交付模块树

标记：`__all__` 表示模块显式导出；“直接模块定义”可调用性不等于新增批准公开承诺。

- [`hotpot`](delivery_api/structure_api.md#hotpot) — __all__: `Molecule`, `Atom`, `Bond`, `BondKind`, `Crystal`, `SpaceGroup`, `SymmUnit`, `MolReader`, `MolWriter`, `read_mol`, `MolBundle`, `to_pyg_dataset`, `is_molecule_input`, `to_hotpot_mol`, `Fragment`, `EdgeShoulder`, `AtomLink`, `AlkylGraft`, `BondAdding`, `AtomReplace`, `RingWedge`, `AssembleFactory`, `Searcher`, `Substructure`, `Hit`, `Hits`, `Query`, `QueryAtom`, `QueryBond`, `SmartsSemantics`, `SmartsSyntaxError`, `UnsupportedSmartsError`, `version`
- [`hotpot.__main__`](delivery_api/cli_api.md#hotpot---main--) — 直接模块定义（私有实现模块；非新增公开路径）: `is_running_in_foreground`, `show_version`, `build_parser`, `run`, `main`
- [`hotpot.__version__`](delivery_api/structure_api.md#hotpot---version--) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
- [`hotpot._cli`](delivery_api/cli_api.md#hotpot--cli) — __all__（私有实现模块；非新增公开路径）: `MarkdownDocumentationAction`
- [`hotpot.cheminfo`](delivery_api/structure_api.md#hotpot-cheminfo) — __all__: `Molecule`, `Atom`, `Bond`, `BondKind`, `Crystal`, `SpaceGroup`, `SymmUnit`, `MolReader`, `MolWriter`, `read_mol`, `MolBundle`, `to_pyg_dataset`, `is_molecule_input`, `to_hotpot_mol`, `Fragment`, `EdgeShoulder`, `AtomLink`, `AlkylGraft`, `BondAdding`, `AtomReplace`, `RingWedge`, `AssembleFactory`, `Searcher`, `Substructure`, `Hit`, `Hits`, `Query`, `QueryAtom`, `QueryBond`, `SmartsSemantics`, `SmartsSyntaxError`, `UnsupportedSmartsError`
  - [`hotpot.cheminfo.adapters`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters._pyg_crystal`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters--pyg-crystal) — 直接模块定义（私有实现模块；非新增公开路径）: `encode_cell`, `decode_cell`, `foreign_cell`
    - [`hotpot.cheminfo.adapters._pyg_molecule`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters--pyg-molecule) — 直接模块定义（私有实现模块；非新增公开路径）: `capture_details`, `encode`, `decode`
    - [`hotpot.cheminfo.adapters.calculation`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-calculation) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.calculation.contracts`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-calculation-contracts) — __all__: `BackendIdentity`, `BackendContext`
    - [`hotpot.cheminfo.adapters.cif`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-cif) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.conversion`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-conversion) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.inference`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.inference.all_atom_model`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-all-atom-model) — 直接模块定义: `UnsupportedAllAtomOptimization`, `materialize_all_atom_model`
      - [`hotpot.cheminfo.adapters.inference.artifacts`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-artifacts) — 直接模块定义: `read_json`, `sha256`, `bundle_sha256`, `is_file`, `external_files`, `download_snapshot`
      - [`hotpot.cheminfo.adapters.inference.cbond`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-cbond) — 直接模块定义: `padding_rings`, `resolve_device`, `create_sessions`, `embed_graph`, `predict`
      - [`hotpot.cheminfo.adapters.inference.device`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-device) — 直接模块定义: `GPUDeviceMemory`, `parse_nvidia_smi_memory`, `gpu_memory_bytes`, `gpu_total_memory_bytes`, `gpu_total_memories_bytes`
      - [`hotpot.cheminfo.adapters.inference.features`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-features) — 直接模块定义: `mol_to_unimolv2`, `collate_molecule_rows`
      - [`hotpot.cheminfo.adapters.inference.runtime`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-runtime) — 直接模块定义: `cuda_provider_available`, `MCARuntime`, `create_cuda_runtime_from_artifact`
      - [`hotpot.cheminfo.adapters.inference.serialization`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-serialization) — 直接模块定义: `copy_feature_molecule`, `encode_molecule`, `decode_molecule`
      - [`hotpot.cheminfo.adapters.inference.system`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-inference-system) — 直接模块定义: `current_cpu_affinity`, `pin_current_process`, `pin_worker_from_queue`
    - [`hotpot.cheminfo.adapters.io`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-io) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.io.base`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-io-base) — 直接模块定义: `IoBase`
      - [`hotpot.cheminfo.adapters.io.formats`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-io-formats) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.adapters.io.formats.g16`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-io-formats-g16) — 直接模块定义: `from_gjf`
        - [`hotpot.cheminfo.adapters.io.formats.openbabel`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-io-formats-openbabel) — 直接模块定义: `get_ob_conversion`, `write_obmol_to_string`
        - [`hotpot.cheminfo.adapters.io.formats.pubchem`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-io-formats-pubchem) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.native.forcefields`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-native-forcefields) — 直接模块定义: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.adapters.native.forcefields.contracts`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-native-forcefields-contracts) — __all__: `ComplexSessionInput`
        - [`hotpot.cheminfo.adapters.native.forcefields.native`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-native-forcefields-native) — __all__: `ComplexOptimizationOptions`, `ConvergenceLevel`, `DEFAULT_CONVERGENCE_LEVEL`, `CoordinationStageOptions`, `FrameDetail`, `MetalPlacementOptions`, `RingScreeningOptions`, `StructureSnapshot`, `assess_metal_position`, `create_coordination_session`, `create_optimization_session`, `optimize_complex`, `place_metal`, `place_metals`, `restore_coordination`, `run_complex_workflow`, `run_complex_workflow_from_input`, `set_coordination_active_mask`, `set_ligand_bond_active_mask`, `snapshot_structure`, `update_structure_coordinates`
        - [`hotpot.cheminfo.adapters.native.forcefields.reports`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-native-forcefields-reports) — __all__: `ComplexOptimizationResult`, `ComplexWorkflowResult`, `CoordinationStageResult`, `NativeBondRingFinding`, `NativeRingCheckpointReport`, `NativeRingGraphScope`, `NativeStageStatus`, `NativeTopologyRevision`, `NativeTrajectoryBatch`, `unpack_native_topology_revisions`
      - [`hotpot.cheminfo.adapters.native.molecule_buffers`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-native-molecule-buffers) — 直接模块定义: `NativeBondKindCode`, `NativeMoleculeField`, `NativeStereoAnchorKindCode`, `NativeStereoCarrierKindCode`, `NativeStereoGroupKindCode`, `NativeAtomTransportFieldCode`, `NativeHydrogenTransportFieldCode`, `NativeUtf8Table`, `NativeStereoRecordBuffers`, `NativeAtomTransportBuffers`, `NativeBondTransportBuffers`, `NativeMoleculeBuffers`
      - [`hotpot.cheminfo.adapters.native.substructure`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-native-substructure) — 直接模块定义: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.adapters.native.substructure.matching`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-native-substructure-matching) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.openbabel`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-openbabel) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.openbabel._runtime`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-openbabel--runtime) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.openbabel.conversion`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-openbabel-conversion) — 直接模块定义: `extract_obatom_coordinate`, `extract_obmol_coordinates`, `set_obmol_coordinates`, `to_arrays`
      - [`hotpot.cheminfo.adapters.openbabel.crystal`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-openbabel-crystal) — 直接模块定义: `to_obcell`
      - [`hotpot.cheminfo.adapters.openbabel.forcefields`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-openbabel-forcefields) — __all__: `build`, `single_optimize`, `optimize`
      - [`hotpot.cheminfo.adapters.openbabel.perception`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-openbabel-perception) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.openbabel.stereo_transport`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-openbabel-stereo-transport) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.pyg`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-pyg) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.rdkit`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-rdkit) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.rdkit.conformers`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-rdkit-conformers) — 直接模块定义: `ensure_3d_conformer`
      - [`hotpot.cheminfo.adapters.rdkit.conversion`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-rdkit-conversion) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.rdkit.sampling`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-rdkit-sampling) — 直接模块定义: `generate`
      - [`hotpot.cheminfo.adapters.rdkit.stereo_transport`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-rdkit-stereo-transport) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.structure`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.structure.crystal`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-crystal) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.structure.forcefields`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-forcefields) — __all__: `capture_topology`, `collect_coordination_environments`
      - [`hotpot.cheminfo.adapters.structure.geometry`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-geometry) — __all__: `PairScope`, `RingScope`, `AtomGeometry`, `AtomPairTarget`, `BondGeometry`, `RingGeometry`, `BondRingTarget`, `AtomPairDistance`, `BondRingFinding`, `RingEdgeDistance`, `BondRingScanReport`, `BondRingScreeningReport`, `BondRingScreeningPlan`, `BondRingFrameWorkspace`, `point_from_atom`, `segment_from_bond`, `cycle_from_ring`, `iter_atom_geometries`, `iter_atom_pair_targets`, `iter_ring_geometries`, `iter_bond_ring_targets`, `measure_atom_pair_distances`, `determine_bond_ring_relation`, `iter_bond_ring_findings`, `scan_bond_ring_relations`, `prepare_bond_ring_screening_plan`, `prepare_bond_ring_frame`, `screen_bond_ring_workspace`, `screen_segments_against_ring_workspace`, `screen_bonds_against_rings`, `screen_bond_ring_relations`, `determine_bond_ring_piercing_state`
      - [`hotpot.cheminfo.adapters.structure.inference`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-inference) — 直接模块定义: `mca_molecule_record`, `direct_edge_to_indirect`, `extract_potentials_cbonds`, `extract_atom_attrs`, `extract_bond_attrs`, `extract_atom_pairs`, `extract_ring_attrs`
      - [`hotpot.cheminfo.adapters.structure.scientific_state`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-scientific-state) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.structure.stereochemistry`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-stereochemistry) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.structure.substructure`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-substructure) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.structure.substructure_query`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-structure-substructure-query) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.thermo`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-thermo) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.thermo.chemical`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-thermo-chemical) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.adapters.visualization`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-visualization) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.adapters.visualization.draw`](delivery_api/adapters_api.md#hotpot-cheminfo-adapters-visualization-draw) — __all__: `compute_atom_bond_colors`, `create_svg_colorbar`, `merge_colorbar_to_svg`
  - [`hotpot.cheminfo.elements`](delivery_api/operations_api.md#hotpot-cheminfo-elements) — __all__: `elements`, `element_properties`, `PropertiesSheet`
  - [`hotpot.cheminfo.obWrappers`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers) — __all__: `BondKindCode`, `BuildReport`, `CoordinateChange`, `ConvergenceLevel`, `DEFAULT_CONVERGENCE_LEVEL`, `HybridizationChange`, `OptimizationFrame`, `OptimizationCheckReport`, `OptimizationFailure`, `OptimizationReport`, `RuleApplication`, `RuleDescriptor`, `RuleExecutionReport`, `RuleStage`, `SingleOptimizationReport`, `available_rules`, `build`, `check_optimization_state`, `inspect_rules`, `optimize`, `single_optimize`
    - [`hotpot.cheminfo.obWrappers.builder`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-builder) — __all__: `build`
    - [`hotpot.cheminfo.obWrappers.checks`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-checks) — __all__: `check_optimization_state`
    - [`hotpot.cheminfo.obWrappers.contracts`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-contracts) — __all__: `BuildReport`, `BondKindCode`, `CoordinateChange`, `ConvergenceLevel`, `DEFAULT_CONVERGENCE_LEVEL`, `HybridizationChange`, `OptimizationFrame`, `OptimizationCheckReport`, `OptimizationFailure`, `OptimizationReport`, `RuleApplication`, `RuleDescriptor`, `RuleExecutionReport`, `RuleStage`, `SingleOptimizationReport`
    - [`hotpot.cheminfo.obWrappers.forcefield`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-forcefield) — __all__: `optimize`, `single_optimize`
    - [`hotpot.cheminfo.obWrappers.native`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-native) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.obWrappers.operation`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-operation) — __all__: `single_optimize`
    - [`hotpot.cheminfo.obWrappers.packing`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-packing) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.obWrappers.registry`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-registry) — __all__: `available_rules`, `inspect_rules`
    - [`hotpot.cheminfo.obWrappers.reports`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-reports) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.obWrappers.settings`](delivery_api/wrappers_api.md#hotpot-cheminfo-obwrappers-settings) — __all__: `TORSION_REPAIR_ANGLE_RADIANS`, `TORSION_SINGULARITY_THRESHOLD`
  - [`hotpot.cheminfo.operations`](delivery_api/operations_api.md#hotpot-cheminfo-operations) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations._aromatic_protocol`](delivery_api/operations_api.md#hotpot-cheminfo-operations--aromatic-protocol) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations._carrier_signatures`](delivery_api/operations_api.md#hotpot-cheminfo-operations--carrier-signatures) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.assembly`](delivery_api/operations_api.md#hotpot-cheminfo-operations-assembly) — __all__: `Fragment`, `EdgeShoulder`, `AtomLink`, `AlkylGraft`, `BondAdding`, `AtomReplace`, `RingWedge`, `AssembleFactory`, `alkyl_generator`, `atom_link_atom_action`, `shoulder_bond_action`, `bond_order_add`, `atom_replace`, `ring_wedge`
      - [`hotpot.cheminfo.operations.assembly.action_func`](delivery_api/operations_api.md#hotpot-cheminfo-operations-assembly-action-func) — 直接模块定义: `actions_register`, `atom_link_atom_action`, `bond_order_add`, `atom_replace`, `shoulder_bond_action`, `ring_wedge`
      - [`hotpot.cheminfo.operations.assembly.assembler`](delivery_api/operations_api.md#hotpot-cheminfo-operations-assembly-assembler) — __all__: `EdgeShoulder`, `AtomLink`, `AlkylGraft`, `BondAdding`, `AtomReplace`, `RingWedge`, `AssembleFactory`
      - [`hotpot.cheminfo.operations.assembly.fragment`](delivery_api/operations_api.md#hotpot-cheminfo-operations-assembly-fragment) — 直接模块定义: `Fragment`
    - [`hotpot.cheminfo.operations.calculation`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.calculation.scientific_state`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state) — __all__: `extract_molecular_facts`, `resolve_scientific_state`, `project_state`, `resolve_and_project_state`, `MolecularFacts`, `StateRequest`, `StatePolicy`, `ScientificOperationRequirements`, `ScientificStateBundle`, `BackendStateRequest`, `BackendProjectionResult`, `ScientificStateWorkflowResult`
        - [`hotpot.cheminfo.operations.calculation.scientific_state._rule_serialization`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state--rule-serialization) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.composition`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition) — __all__: `compose_scientific_state_space`, `CandidateDomainScope`, `MaterializationStatus`, `CandidateSearchStatus`, `StateValueOrigin`, `ScientificStateIdentity`, `FactorKey`, `ConstraintValue`, `CandidateConstraint`, `FrontierBound`, `CandidateDomainSummary`, `RejectedPathSummary`, `BranchAssignment`, `StateValueRef`, `DerivedStateValue`, `StateCandidate`, `FactorCandidateValue`, `FactorDomainContribution`, `CandidateFactor`, `ProposedCandidateValue`, `ProposedCandidateFactor`, `ConstraintGuard`, `GuardedCandidateConstraint`, `CandidateProposal`, `CoupledScientificStateCandidate`, `ScientificStateCandidateSpace`, `ScientificResolutionResult`
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition._artifacts`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition--artifacts) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition._candidate_space_serialization`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition--candidate-space-serialization) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition._decision_snapshot`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition--decision-snapshot) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition._serialization`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition--serialization) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.candidate_space`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-candidate-space) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.compose`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-compose) — __all__: `compose_scientific_state_space`
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.constraints`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-constraints) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.coupling`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-coupling) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.decision_search`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-decision-search) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.derive`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-derive) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.expand`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-expand) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.resolution`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-resolution) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.operations.calculation.scientific_state.composition.validation`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-composition-validation) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.contracts`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-contracts) — __all__: `StatePolicy`, `ProviderToggle`, `ProviderSet`, `ActivatedRule`, `RuleSet`
        - [`hotpot.cheminfo.operations.calculation.scientific_state.defaults`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-defaults) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.execution_selection`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-execution-selection) — __all__: `project_state`, `resolve_and_project_state`, `BackendStateValues`, `BackendStateRequest`, `BackendDecisionSource`, `BackendExecutability`, `BackendProjectionFinding`, `BackendExecutionDecision`, `BackendExecutionEvidence`, `ExecutionPolicySet`, `BackendStateProjection`, `BackendProjectionStatus`, `BackendProjectionResult`, `ScientificStateWorkflowResult`
        - [`hotpot.cheminfo.operations.calculation.scientific_state.facts`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-facts) — __all__: `extract_molecular_facts`
        - [`hotpot.cheminfo.operations.calculation.scientific_state.manifest`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-manifest) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.operation_requirements`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-operation-requirements) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.profile_decision`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-profile-decision) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.profile_eligibility`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-profile-eligibility) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.profiles`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-profiles) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.rule_controller`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-rule-controller) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.rule_registry`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-rule-registry) — __all__: `InferenceRuleRegistry`
        - [`hotpot.cheminfo.operations.calculation.scientific_state.rule_validation`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-rule-validation) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.calculation.scientific_state.state_resolver`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-state-resolver) — __all__: `resolve_scientific_state`, `ProvidedScientificResults`, `ScientificStateBundle`
        - [`hotpot.cheminfo.operations.calculation.scientific_state.state_selection`](delivery_api/operations_api.md#hotpot-cheminfo-operations-calculation-scientific-state-state-selection) — __all__: `StateDecisionAction`, `StateDecisionSource`, `DecisionCertificationStatus`, `StaticEligibilityFinding`, `DecisionReason`, `StaticProfileEligibility`, `DecisionCertifiability`, `ExecutionStateDecision`
    - [`hotpot.cheminfo.operations.chemistry`](delivery_api/operations_api.md#hotpot-cheminfo-operations-chemistry) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.collections`](delivery_api/operations_api.md#hotpot-cheminfo-operations-collections) — __all__: `to_pyg_dataset`
    - [`hotpot.cheminfo.operations.conformers`](delivery_api/operations_api.md#hotpot-cheminfo-operations-conformers) — __all__: `ConformerBatch`, `ConformerSampler`, `ConformerSamplingError`, `ConformerSamplingRequest`, `OpenBabelConformerAdapter`, `RDKitConformerAdapter`
      - [`hotpot.cheminfo.operations.conformers.contracts`](delivery_api/operations_api.md#hotpot-cheminfo-operations-conformers-contracts) — __all__: `ConformerBatch`, `ConformerSamplingError`, `ConformerSamplingRequest`
      - [`hotpot.cheminfo.operations.conformers.openbabel`](delivery_api/operations_api.md#hotpot-cheminfo-operations-conformers-openbabel) — 直接模块定义: `OpenBabelConformerAdapter`
      - [`hotpot.cheminfo.operations.conformers.rdkit`](delivery_api/operations_api.md#hotpot-cheminfo-operations-conformers-rdkit) — 直接模块定义: `RDKitConformerAdapter`
      - [`hotpot.cheminfo.operations.conformers.sampler`](delivery_api/operations_api.md#hotpot-cheminfo-operations-conformers-sampler) — __all__: `ConformerSampler`
    - [`hotpot.cheminfo.operations.conversion`](delivery_api/operations_api.md#hotpot-cheminfo-operations-conversion) — __all__: `is_molecule_input`, `to_hotpot_mol`, `mol2obmol`, `obmol2mol`, `to_rdmol`, `from_rdmol`, `write_by_pybel`
    - [`hotpot.cheminfo.operations.crystal`](delivery_api/operations_api.md#hotpot-cheminfo-operations-crystal) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.crystal_generation`](delivery_api/operations_api.md#hotpot-cheminfo-operations-crystal-generation) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.crystal_geometry`](delivery_api/operations_api.md#hotpot-cheminfo-operations-crystal-geometry) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.crystal_io`](delivery_api/operations_api.md#hotpot-cheminfo-operations-crystal-io) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.crystal_matching`](delivery_api/operations_api.md#hotpot-cheminfo-operations-crystal-matching) — 直接模块定义: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.descriptors`](delivery_api/operations_api.md#hotpot-cheminfo-operations-descriptors) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.forcefields`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields) — __all__: `TrajectoryPath`, `TrajectoryStart`, `TrajectoryStage`, `TrajectoryEvent`, `AtomIdentity`, `BondTopology`, `BondTopologyRevision`, `RingFrameEvidence`, `CoordinationFrameEvidence`, `OptimizationFrameEvidence`, `FrameEvidence`, `ForceFieldFrame`, `ForceFieldTrajectory`, `ForceFieldTrajectoryArchive`, `OptimizationAlgorithm`, `OptimizationRoute`, `OptimizationAttemptStatus`, `OptimizationAttemptReport`, `OptimizationRoutingReport`, `ConvergenceLevel`, `DEFAULT_CONVERGENCE_LEVEL`, `OptimizationStoppingCriteria`, `TerminationReason`, `ForceFieldDiagnosticValue`, `ForceFieldRunReport`, `Build3DReport`, `CandidateRejection`, `RingUntanglingReport`, `CoordinationBondRestorationReport`, `ComplexBuildDiagnostics`, `ForceFieldWorkflowReport`, `BuildAndOptimizeReport`, `ComplexBuildReport`, `ForceFieldSetupReport`, `ForceFieldSetupStage`, `ForceFieldWorkflowStage`, `AcceptanceCheck`, `StructureAcceptanceThresholds`, `ForceFieldAcceptanceEvidence`, `AtomTopologySignature`, `TopologyReference`, `ForceFieldValidationReport`, `CoordinationEnvironment`, `ForceFieldError`, `ForceFieldSetupError`, `BuildWorkerError`, `BuildTimeoutError`, `ComplexBuildError`, `ComplexBuildWarning`, `ComplexBuildWorkerError`, `ComplexBuildTimeoutError`, `GeometryQualityError`, `GeometryQualityWarning`, `capture_topology`, `evaluate_structure_acceptance`, `is_structure_accepted`, `perturb`, `collect_coordination_environments`, `build3d`, `optimize`, `build_complex3d`, `optimize_complex`, `complexes_build`, `build_and_optimize`, `auto_optimize`
      - [`hotpot.cheminfo.operations.forcefields.acceptance`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-acceptance) — __all__: `evaluate_structure_acceptance`, `is_structure_accepted`
      - [`hotpot.cheminfo.operations.forcefields.attempts`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-attempts) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.forcefields.auto`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-auto) — __all__: `auto_optimize`
      - [`hotpot.cheminfo.operations.forcefields.backend`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-backend) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.forcefields.contracts`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-contracts) — __all__: `TrajectoryPath`, `OptimizationAlgorithm`, `ConvergenceLevel`, `DEFAULT_CONVERGENCE_LEVEL`, `TerminationReason`, `OptimizationRoute`, `OptimizationAttemptStatus`, `OptimizationAttemptReport`, `OptimizationRoutingReport`, `ForceFieldDiagnosticValue`, `ForceFieldRunReport`, `Build3DReport`, `CandidateRejection`, `RingUntanglingReport`, `CoordinationBondRestorationReport`, `ComplexBuildDiagnostics`, `ForceFieldWorkflowReport`, `BuildAndOptimizeReport`, `ComplexBuildReport`, `ForceFieldSetupReport`, `ForceFieldSetupStage`, `ForceFieldWorkflowStage`, `AcceptanceCheck`, `StructureAcceptanceThresholds`, `ForceFieldAcceptanceEvidence`, `ForceFieldValidationReport`, `CoordinationEnvironment`, `ForceFieldError`, `ForceFieldSetupError`, `BuildWorkerError`, `BuildTimeoutError`, `ComplexBuildError`, `ComplexBuildWarning`, `ComplexBuildWorkerError`, `ComplexBuildTimeoutError`, `GeometryQualityError`, `GeometryQualityWarning`
      - [`hotpot.cheminfo.operations.forcefields.coordinates`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-coordinates) — __all__: `perturb`
      - [`hotpot.cheminfo.operations.forcefields.coordination`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-coordination) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.forcefields.ff`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-ff) — __all__: `TrajectoryPath`, `TrajectoryStart`, `TrajectoryStage`, `TrajectoryEvent`, `AtomIdentity`, `BondTopology`, `BondTopologyRevision`, `RingFrameEvidence`, `CoordinationFrameEvidence`, `OptimizationFrameEvidence`, `FrameEvidence`, `ForceFieldFrame`, `ForceFieldTrajectory`, `ForceFieldTrajectoryArchive`, `OptimizationAlgorithm`, `OptimizationRoute`, `OptimizationAttemptStatus`, `OptimizationAttemptReport`, `OptimizationRoutingReport`, `ConvergenceLevel`, `DEFAULT_CONVERGENCE_LEVEL`, `OptimizationStoppingCriteria`, `TerminationReason`, `ForceFieldDiagnosticValue`, `ForceFieldRunReport`, `Build3DReport`, `CandidateRejection`, `RingUntanglingReport`, `CoordinationBondRestorationReport`, `ComplexBuildDiagnostics`, `ForceFieldWorkflowReport`, `BuildAndOptimizeReport`, `ComplexBuildReport`, `ForceFieldSetupReport`, `ForceFieldSetupStage`, `ForceFieldWorkflowStage`, `AcceptanceCheck`, `StructureAcceptanceThresholds`, `ForceFieldAcceptanceEvidence`, `AtomTopologySignature`, `TopologyReference`, `ForceFieldValidationReport`, `CoordinationEnvironment`, `ForceFieldError`, `ForceFieldSetupError`, `BuildWorkerError`, `BuildTimeoutError`, `ComplexBuildError`, `ComplexBuildWarning`, `ComplexBuildWorkerError`, `ComplexBuildTimeoutError`, `GeometryQualityError`, `GeometryQualityWarning`, `capture_topology`, `evaluate_structure_acceptance`, `is_structure_accepted`, `perturb`, `collect_coordination_environments`, `build3d`, `optimize`, `build_complex3d`, `optimize_complex`, `complexes_build`, `build_and_optimize`, `auto_optimize`
      - [`hotpot.cheminfo.operations.forcefields.ligand`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-ligand) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.forcefields.native_adapters`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-native-adapters) — __all__: `NATIVE_WARNING_MESSAGES`, `NativeOptimizationPerturbationStreams`, `NativePerturbationStreams`, `apply_native_selected_structure`, `coordination_restoration_report`, `forcefield_run_report`, `ingest_native_trajectory`, `native_coordination_offsets`, `native_optimization_offsets`, `native_perturbation_streams`, `native_warning_messages`, `ring_untangling_report`
      - [`hotpot.cheminfo.operations.forcefields.optimizer`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-optimizer) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.forcefields.repair`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-repair) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.forcefields.stage`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-stage) — __all__: `ForceFieldRouteOperations`, `ForceFieldRouteOptions`, `STAGE`, `execute_forcefield_route`, `get_stage`, `json_value`
      - [`hotpot.cheminfo.operations.forcefields.trajectory`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-trajectory) — __all__: `TrajectoryStart`, `TrajectoryStage`, `TrajectoryEvent`, `AtomIdentity`, `BondTopology`, `BondTopologyRevision`, `RingFrameEvidence`, `CoordinationFrameEvidence`, `OptimizationFrameEvidence`, `FrameEvidence`, `ForceFieldFrame`, `ForceFieldTrajectory`, `ForceFieldTrajectoryArchive`
      - [`hotpot.cheminfo.operations.forcefields.workers`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-workers) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.forcefields.workflows`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-workflows) — __all__: `build3d`, `optimize`, `build_complex3d`, `optimize_complex`, `complexes_build`, `build_and_optimize`
      - [`hotpot.cheminfo.operations.forcefields.working_copy`](delivery_api/operations_api.md#hotpot-cheminfo-operations-forcefields-working-copy) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.geometry`](delivery_api/operations_api.md#hotpot-cheminfo-operations-geometry) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.inference`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.inference.artifacts`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-artifacts) — __all__: `ModelArtifact`, `ModelArtifactUnavailable`, `ModelSource`, `load_manifest`, `verify_artifact`
        - [`hotpot.cheminfo.operations.inference.artifacts.errors`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-artifacts-errors) — 直接模块定义: `ModelArtifactUnavailable`
        - [`hotpot.cheminfo.operations.inference.artifacts.manifest`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-artifacts-manifest) — 直接模块定义: `load_manifest`, `sha256`, `bundle_sha256`, `artifact_is_complete`, `verify_artifact`
        - [`hotpot.cheminfo.operations.inference.artifacts.registry`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-artifacts-registry) — 直接模块定义: `ArtifactSpec`
        - [`hotpot.cheminfo.operations.inference.artifacts.resolver`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-artifacts-resolver) — 直接模块定义: `ModelSource`, `ModelArtifact`
      - [`hotpot.cheminfo.operations.inference.cbond`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-cbond) — __all__: `CBondRuntime`
        - [`hotpot.cheminfo.operations.inference.cbond.apply`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-cbond-apply) — 直接模块定义: `CBondStep`, `CBondPathResult`, `CBondStructureResult`, `get_cbond_runtime`, `extract_cbond_inputs`, `get_graph_cbond_inputs`, `get_cbond_inputs_model`, `pred_xg`, `pred_cb_value`, `cbond_prediction`, `signmod_with_offset`, `init_metal_ligand_pair`, `auto_build_cbond`, `build_one_cbond`, `build_all_possible_cbond`
        - [`hotpot.cheminfo.operations.inference.cbond.constants`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-cbond-constants) — 直接模块定义: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.operations.inference.cbond.runtime`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-cbond-runtime) — 直接模块定义: `CBondRuntime`
        - [`hotpot.cheminfo.operations.inference.cbond.stage`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-cbond-stage) — __all__: `STAGE`, `get_stage`
      - [`hotpot.cheminfo.operations.inference.features`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-features) — 直接模块定义: `extract_potentials_cbonds`, `extract_atom_attrs`, `extract_bond_attrs`, `extract_atom_pairs`, `extract_ring_attrs`
      - [`hotpot.cheminfo.operations.inference.mca`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca) — __all__: `mca`, `AtomPrediction`, `MCACpuPredictor`, `MCAGpuOutOfMemoryError`, `MCAGpuPredictor`, `MoleculePrediction`, `SitePrediction`, `predict_mca`, `predict_mca_cpu`, `predict_mca_gpu`
        - [`hotpot.cheminfo.operations.inference.mca.api`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-api) — __all__: `MCACpuPredictor`, `MCAGpuPredictor`, `predict_mca`, `predict_mca_cpu`, `predict_mca_gpu`
        - [`hotpot.cheminfo.operations.inference.mca.attachment`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-attachment) — __all__: `mca`
        - [`hotpot.cheminfo.operations.inference.mca.batching`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-batching) — 直接模块定义: `AtomRow`, `LogicalBatch`, `LogicalBatchPlanner`, `scatter_batch_predictions`
        - [`hotpot.cheminfo.operations.inference.mca.cpu_backend`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-cpu-backend) — 直接模块定义: `MCACpuPredictor`
        - [`hotpot.cheminfo.operations.inference.mca.gpu_backend`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-gpu-backend) — 直接模块定义: `MCAGpuPredictor`
        - [`hotpot.cheminfo.operations.inference.mca.gpu_consumer`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-gpu-consumer) — 直接模块定义: `GpuConsumerConfig`, `GpuBatchRequest`, `GpuConsumerReady`, `GpuBatchResult`, `GpuConsumerFailure`, `gpu_consumer_main`
        - [`hotpot.cheminfo.operations.inference.mca.gpu_memory`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-gpu-memory) — 直接模块定义: `automatic_gpu_batch_size`, `validate_gpu_batch_size`, `resolve_gpu_batch_size`, `next_lower_gpu_batch_size`, `MCAGpuOutOfMemoryError`, `is_cuda_out_of_memory`
        - [`hotpot.cheminfo.operations.inference.mca.model_store`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-model-store) — 直接模块定义: `ModelStore`, `ResolvedModelArtifact`, `resolved_model_artifact`, `create_cpu_runtime`, `create_cuda_runtime`
        - [`hotpot.cheminfo.operations.inference.mca.pipeline`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-pipeline) — __all__: `FeatureDict`, `PreparedMolecule`, `SiteRecord`, `DEFAULT_CONFORMER_SEED`, `build_features`, `build_prediction`, `detect_sites`, `is_single_molecule_input`, `prepare_molecule`
        - [`hotpot.cheminfo.operations.inference.mca.resources`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-resources) — 直接模块定义: `InsufficientCPUAffinityError`, `InsufficientCPUCountError`, `CPUWorkerAssignment`, `GPUCPUAssignment`, `GPUResourcePlan`, `resolve_gpu_indices`, `resolve_cpu_ids`, `validate_gpu_resource_request`, `build_gpu_resource_plan`, `gpu_window_atom_budget`
        - [`hotpot.cheminfo.operations.inference.mca.results`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-results) — 直接模块定义: `AtomPrediction`, `SitePrediction`, `MoleculePrediction`, `AtomDTO`, `BondDTO`, `MoleculeDTO`
        - [`hotpot.cheminfo.operations.inference.mca.serialization`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-serialization) — __all__: `MCA_MOLECULE_DTO_VERSION`, `AtomDTO`, `BondDTO`, `MoleculeDTO`, `hotpot_mol_from_dto`, `normalize_molecule`, `rdmol_from_dto`, `validate_dto`
        - [`hotpot.cheminfo.operations.inference.mca.site_detection`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-site-detection) — 直接模块定义: `find_nucleophilic_sites`
        - [`hotpot.cheminfo.operations.inference.mca.workers`](delivery_api/workflows_api.md#hotpot-cheminfo-operations-inference-mca-workers) — __all__: `CPUPreprocessResult`, `CPUPreprocessTask`, `FeatureDict`, `FeatureResult`, `FeatureTask`, `SiteDetectionResult`, `SiteDetectionTask`, `build_features_worker`, `cpu_preprocess_worker`, `detect_sites_worker`
    - [`hotpot.cheminfo.operations.io`](delivery_api/operations_api.md#hotpot-cheminfo-operations-io) — __all__: `MolReader`, `MolWriter`, `read_mol`
    - [`hotpot.cheminfo.operations.native_packing`](delivery_api/operations_api.md#hotpot-cheminfo-operations-native-packing) — 直接模块定义: `pack_molecule`
    - [`hotpot.cheminfo.operations.search`](delivery_api/operations_api.md#hotpot-cheminfo-operations-search) — __all__: `Query`, `QueryAtom`, `QueryBond`, `Substructure`, `Searcher`, `Hits`, `Hit`, `SmartsSyntaxError`, `UnsupportedSmartsError`, `SmartsSemantics`
      - [`hotpot.cheminfo.operations.search.native`](delivery_api/operations_api.md#hotpot-cheminfo-operations-search-native) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.operations.search.predicates`](delivery_api/operations_api.md#hotpot-cheminfo-operations-search-predicates) — 直接模块定义: `match_atom_expression`, `match_bond_expression`
      - [`hotpot.cheminfo.operations.search.search`](delivery_api/operations_api.md#hotpot-cheminfo-operations-search-search) — __all__: `Query`, `QueryAtom`, `QueryBond`, `Substructure`, `Searcher`, `Hits`, `Hit`
      - [`hotpot.cheminfo.operations.search.semantics`](delivery_api/operations_api.md#hotpot-cheminfo-operations-search-semantics) — 直接模块定义: `rings_for`, `bonds_for`, `degree`, `connectivity`, `valence`
      - [`hotpot.cheminfo.operations.search.smarts`](delivery_api/operations_api.md#hotpot-cheminfo-operations-search-smarts) — 直接模块定义: `parse_bracket_atom`, `substructure_from_smarts`
    - [`hotpot.cheminfo.operations.solvent_properties`](delivery_api/operations_api.md#hotpot-cheminfo-operations-solvent-properties) — __all__: `get_solvent_dataframe`, `get_medium_dataframe`, `query_solvent`, `query_medium`
    - [`hotpot.cheminfo.operations.stereochemistry`](delivery_api/operations_api.md#hotpot-cheminfo-operations-stereochemistry) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.thermo`](delivery_api/operations_api.md#hotpot-cheminfo-operations-thermo) — __all__: `Thermo`
    - [`hotpot.cheminfo.operations.topology`](delivery_api/operations_api.md#hotpot-cheminfo-operations-topology) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.topology_queries`](delivery_api/operations_api.md#hotpot-cheminfo-operations-topology-queries) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.operations.visualization`](delivery_api/operations_api.md#hotpot-cheminfo-operations-visualization) — __all__: `draw_single_mol`, `draw_grid`
  - [`hotpot.cheminfo.pubchem`](delivery_api/operations_api.md#hotpot-cheminfo-pubchem) — __all__: `get_compound`, `get_compounds_cid`, `smi_to_cid`, `smi_to_cas`, `smi_to_name`, `cid_to_smi`, `name_to_smi`, `cid_to_cas`, `pubchem_service`
  - [`hotpot.cheminfo.rdWrappers`](delivery_api/wrappers_api.md#hotpot-cheminfo-rdwrappers) — __all__: `ConformerBatch`, `ConformerRecord`, `CoordinationEmbeddingConstraints`, `ETKDGOptions`, `ETKDGSampler`, `GenerationStatus`, `RuntimeInfo`, `generate_conformers`, `runtime_info`
    - [`hotpot.cheminfo.rdWrappers.contracts`](delivery_api/wrappers_api.md#hotpot-cheminfo-rdwrappers-contracts) — __all__: `ConformerBatch`, `ConformerRecord`, `CoordinationEmbeddingConstraints`, `ETKDGOptions`, `GenerationStatus`, `RuntimeInfo`
    - [`hotpot.cheminfo.rdWrappers.native`](delivery_api/wrappers_api.md#hotpot-cheminfo-rdwrappers-native) — 直接模块定义: `generate`, `runtime_info`
    - [`hotpot.cheminfo.rdWrappers.sampler`](delivery_api/wrappers_api.md#hotpot-cheminfo-rdwrappers-sampler) — 直接模块定义: `ETKDGSampler`, `generate_conformers`, `runtime_info`
  - [`hotpot.cheminfo.rdworks`](delivery_api/operations_api.md#hotpot-cheminfo-rdworks) — 直接模块定义: `load_fp`, `merge_fps`, `fps_to_array`, `assign_fp_values_atoms`
  - [`hotpot.cheminfo.reference`](delivery_api/rules_api.md#hotpot-cheminfo-reference) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.reference.elements`](delivery_api/rules_api.md#hotpot-cheminfo-reference-elements) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.reference.properties`](delivery_api/rules_api.md#hotpot-cheminfo-reference-properties) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.reference.space_groups`](delivery_api/rules_api.md#hotpot-cheminfo-reference-space-groups) — __all__: 空；仅命名空间/私有实现
  - [`hotpot.cheminfo.rules`](delivery_api/rules_api.md#hotpot-cheminfo-rules) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.rules.chemistry`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.chemistry.bonding`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-bonding) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.bonding.aromaticity`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-bonding-aromaticity) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.bonding.bond_order`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-bonding-bond-order) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.bonding.formal_charge`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-bonding-formal-charge) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.bonding.hydrogen`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-bonding-hydrogen) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.bonding.kekule`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-bonding-kekule) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.bonding.valence`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-bonding-valence) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.chemistry.crystal_identification`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-crystal-identification) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.chemistry.crystal_matching`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-crystal-matching) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.chemistry.descriptors`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-descriptors) — __all__: `GraphSpectrum`, `atoms_electron_configurations`, `calc_electron_config`, `calc_spectrum`
        - [`hotpot.cheminfo.rules.chemistry.descriptors.composition`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-descriptors-composition) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.descriptors.formula`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-descriptors-formula) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.descriptors.geometry`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-descriptors-geometry) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-descriptors-graph-spectrum) — __all__: `calc_electron_config`, `atoms_electron_configurations`, `calc_spectrum`, `GraphSpectrum`
        - [`hotpot.cheminfo.rules.chemistry.descriptors.rings`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-descriptors-rings) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.chemistry.elements`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-elements) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.chemistry.mca_sites`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-mca-sites) — 直接模块定义: `DetectedSite`, `name_sites`, `validate_target_domain`
      - [`hotpot.cheminfo.rules.chemistry.scientific_state`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state) — __all__: `ScientificQuantity`, `ScientificAdmissibility`, `ResolutionStatus`, `CoverageStatus`, `StateValueAuthority`, `ScientificFinding`, `CandidateProvenance`, `ProviderValue`, `CandidateBudget`, `DEFAULT_CANDIDATE_BUDGET`, `ScientificOperationRequirements`
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.charge`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-charge) — __all__: `NetChargeCandidateEvidence`, `AtomFormalChargeEvidence`, `ComponentChargeEvidence`, `NetChargeResult`, `LewisFormalChargeCandidate`, `LewisFormalChargeResult`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-charge-contracts) — __all__: `NetChargeCandidateEvidence`, `AtomFormalChargeEvidence`, `ComponentChargeEvidence`, `NetChargeResult`, `LewisFormalChargeCandidate`, `LewisFormalChargeResult`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.charge.lewis`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-charge-lewis) — __all__: `solve_main_group_lewis`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.charge.net_charge`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-charge-net-charge) — __all__: `resolve_net_charge`
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.composition`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-composition) — __all__: `CandidateDomainScope`, `MaterializationStatus`, `CandidateSearchStatus`, `StateValueOrigin`, `ScientificStateIdentity`, `FactorKey`, `ConstraintValue`, `CandidateConstraint`, `FrontierBound`, `CandidateDomainSummary`, `RejectedPathSummary`, `BranchAssignment`, `StateValueRef`, `DerivedStateValue`, `StateCandidate`, `FactorCandidateValue`, `FactorDomainContribution`, `CandidateFactor`, `ProposedCandidateValue`, `ProposedCandidateFactor`, `ConstraintGuard`, `GuardedCandidateConstraint`, `CandidateProposal`, `CoupledScientificStateCandidate`, `ScientificStateCandidateSpace`, `ScientificResolutionResult`
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.contracts`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-contracts) — __all__: `ScientificQuantity`, `ScientificAdmissibility`, `ResolutionStatus`, `CoverageStatus`, `StateValueAuthority`, `ScientificFinding`, `CandidateProvenance`, `ProviderValue`, `CandidateBudget`, `DEFAULT_CANDIDATE_BUDGET`, `ScientificOperationRequirements`
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state) — __all__: `ElectronicStateCandidateEvidence`, `SpinContributionScope`, `SpinContributionCandidate`, `ElectronicStateCandidate`, `ElectronicStateInferenceResult`, `NeutralAtomReference`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-contracts) — __all__: `ElectronicStateCandidateEvidence`, `SpinContributionScope`, `SpinContributionCandidate`, `ElectronicStateCandidate`, `ElectronicStateInferenceResult`, `NeutralAtomReference`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.inference`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-inference) — __all__: `infer_electronic_states`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.neutral_atom`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-electronic-state-neutral-atom) — __all__: `neutral_atom_reference`
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.identities`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-identities) — 直接模块定义: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.legality`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-legality) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state) — __all__: `CountingConvention`, `OxidationStateOverride`, `CoordinationFragmentAssignment`, `CoordinationCountingCandidate`, `CoordinationElectronCountingResult`, `OxidationStateCandidateEvidence`, `OxidationStateInferenceResult`, `ElementOxidationStateDomain`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-contracts) — __all__: `CountingConvention`, `OxidationStateOverride`, `CoordinationFragmentAssignment`, `CoordinationCountingCandidate`, `CoordinationElectronCountingResult`, `OxidationStateCandidateEvidence`, `OxidationStateInferenceResult`, `ElementOxidationStateDomain`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.coordination_counting`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-coordination-counting) — __all__: `infer_coordination_electron_counting`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.inference`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-inference) — __all__: `infer_oxidation_states`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.registry`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-oxidation-state-registry) — __all__: `element_oxidation_state_domain`
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers) — __all__: `RuleKind`, `RulePhase`, `ResultType`, `RuleValue`, `RuleDependency`, `RuleSpecification`, `RuleToggle`, `EvidenceItem`, `RankAdjustment`, `RuleApplicability`, `RuleContext`, `RuleOutcome`, `RuleTrace`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.americium_curium_iii`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-americium-curium-iii) — __all__: `AmericiumCuriumIIIPriorRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.anionic_x_ligands`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-anionic-x-ligands) — __all__: `AnionicXLigandRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.base`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-base) — __all__: `InferenceRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-contracts) — __all__: `RuleKind`, `RulePhase`, `ResultType`, `RuleValue`, `RuleDependency`, `RuleSpecification`, `RuleToggle`, `EvidenceItem`, `RankAdjustment`, `RuleApplicability`, `RuleContext`, `RuleOutcome`, `RuleTrace`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.f_block_hund`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-f-block-hund) — __all__: `FBlockHundPriorRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.ionic_configuration_spin`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-ionic-configuration-spin) — __all__: `IonicConfigurationSpinRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.lanthanide_iii`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-lanthanide-iii) — __all__: `LanthanideIIIPriorRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.main_group_closed_shell`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-main-group-closed-shell) — __all__: `MainGroupClosedShellRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.monoatomic_ions`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-monoatomic-ions) — __all__: `MonoatomicIonRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.neutral_l_donors`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-neutral-l-donors) — __all__: `NeutralLDonorRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.oxidation_priors`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-oxidation-priors) — __all__: 空；仅命名空间/私有实现
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.thorium_iv`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-thorium-iv) — __all__: `ThoriumIVPriorRule`
          - [`hotpot.cheminfo.rules.chemistry.scientific_state.providers.trusted_history`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-providers-trusted-history) — __all__: `TrustedScientificStateRecord`, `TrustedHistoryRule`
        - [`hotpot.cheminfo.rules.chemistry.scientific_state.request`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-scientific-state-request) — __all__: `StateRequest`
      - [`hotpot.cheminfo.rules.chemistry.stereochemistry`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-stereochemistry) — 直接模块定义: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.stereochemistry.algebra`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-stereochemistry-algebra) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.stereochemistry.alignment`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-stereochemistry-alignment) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.stereochemistry.assignment`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-stereochemistry-assignment) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.stereochemistry.coordinates`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-stereochemistry-coordinates) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.stereochemistry.graph`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-stereochemistry-graph) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.stereochemistry.validation`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-stereochemistry-validation) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.chemistry.substructure`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-substructure) — 直接模块定义: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.chemistry.substructure._smarts_syntax`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-substructure--smarts-syntax) — 直接模块定义（私有实现模块；非新增公开路径）: `TokenType`, `Token`, `atomic_number`, `tokenize_with_spans`, `tokenize`
        - [`hotpot.cheminfo.rules.chemistry.substructure.errors`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-substructure-errors) — 直接模块定义: `SmartsSyntaxError`, `UnsupportedSmartsError`
        - [`hotpot.cheminfo.rules.chemistry.substructure.predicates`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-substructure-predicates) — 直接模块定义: `match_atom_expression`, `match_bond_expression`
        - [`hotpot.cheminfo.rules.chemistry.substructure.semantics`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-substructure-semantics) — 直接模块定义: `resolve_smarts_semantics`, `degree`, `connectivity`, `valence`, `ligand_valence`
        - [`hotpot.cheminfo.rules.chemistry.substructure.smarts`](delivery_api/rules_api.md#hotpot-cheminfo-rules-chemistry-substructure-smarts) — 直接模块定义: `BondOrder`, `AromaticFlag`, `parse_smarts`
    - [`hotpot.cheminfo.rules.mathematics`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.mathematics.cell`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-cell) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.mathematics.geometry`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-geometry) — __all__: `NumericToleranceSettings`, `SurfaceEnumerationSettings`, `GeometrySettings`, `DEFAULT_GEOMETRY_SETTINGS`, `Point`, `Line`, `Segment`, `Plane`, `Triangle`, `Cycle`, `PlanarityKind`, `LineRelationKind`, `PointCycleLocation`, `SurfaceEmbeddingState`, `SurfaceSegmentState`, `PiercingState`, `SegmentCycleFeature`, `SegmentCycleIndeterminacy`, `CycleSurfaceModel`, `PlanarityMeasurement`, `LineRelation`, `PointPairDistance`, `ClosestCycleEdge`, `SurfaceFamilyEvidence`, `SegmentCycleRelation`, `SegmentCycleScreening`, `measure_planarity`, `determine_line_relation`, `line_distance`, `point_segment_distance`, `segment_segment_distance`, `point_pair_distances`, `find_point_pairs_below_distance`, `locate_point_in_planar_cycle`, `iter_segment_cycle_relations`, `iter_segment_cycle_screenings`, `determine_segment_cycle_relation`, `closest_cycle_edge`
        - [`hotpot.cheminfo.rules.mathematics.geometry.linking`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-geometry-linking) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.mathematics.geometry.native`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-geometry-native) — __all__: `NumericTolerances`, `PointSegmentMeasurement`, `SegmentSegmentMeasurement`, `measure_planarity`, `locate_point_in_planar_cycle`, `closest_cycle_edge`, `determine_line_relation`, `line_distance`, `point_segment_measurement`, `point_segment_distance`, `segment_segment_measurement`, `segment_segment_distance`, `point_pair_distances`, `find_point_pairs_below_distance`, `aabb_bounds`, `aabb_separation_mask`, `segment_aabb_separation_mask`, `aabb_candidate_pairs`
        - [`hotpot.cheminfo.rules.mathematics.geometry.object`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-geometry-object) — __all__: `Point`, `Line`, `Segment`, `Plane`, `Triangle`, `Cycle`
        - [`hotpot.cheminfo.rules.mathematics.geometry.relation`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-geometry-relation) — __all__: `PlanarityKind`, `LineRelationKind`, `PointCycleLocation`, `SurfaceEmbeddingState`, `SurfaceSegmentState`, `PiercingState`, `SegmentCycleFeature`, `SegmentCycleIndeterminacy`, `CycleSurfaceModel`, `PlanarityMeasurement`, `LineRelation`, `PointPairDistance`, `ClosestCycleEdge`, `SurfaceFamilyEvidence`, `SegmentCycleRelation`, `SegmentCycleScreening`, `measure_planarity`, `determine_line_relation`, `line_distance`, `point_segment_distance`, `segment_segment_distance`, `point_pair_distances`, `find_point_pairs_below_distance`, `locate_point_in_planar_cycle`, `iter_segment_cycle_relations`, `iter_segment_cycle_screenings`, `determine_segment_cycle_relation`, `closest_cycle_edge`
        - [`hotpot.cheminfo.rules.mathematics.geometry.settings`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-geometry-settings) — __all__: `NumericToleranceSettings`, `SurfaceEnumerationSettings`, `GeometrySettings`, `DEFAULT_GEOMETRY_SETTINGS`
      - [`hotpot.cheminfo.rules.mathematics.graph`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-graph) — __all__: `linkmat2adj`, `adj2laplacian`, `graph_dfs_path`, `DEFAULT_RELEVANT_CYCLE_LIMIT`, `RelevantCycleLimitExceeded`, `relevant_cycles`
        - [`hotpot.cheminfo.rules.mathematics.graph.components`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-graph-components) — __all__: `connected_components`
        - [`hotpot.cheminfo.rules.mathematics.graph.cycles`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-graph-cycles) — __all__: `DEFAULT_RELEVANT_CYCLE_LIMIT`, `RelevantCycleLimitExceeded`, `relevant_cycles`
        - [`hotpot.cheminfo.rules.mathematics.graph.matrix`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-graph-matrix) — __all__: `linkmat2adj`, `adj2laplacian`
        - [`hotpot.cheminfo.rules.mathematics.graph.paths`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-graph-paths) — __all__: `all_simple_paths`, `depth_first_path`, `longest_simple_path`, `shortest_path`
        - [`hotpot.cheminfo.rules.mathematics.graph.traversal`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-graph-traversal) — __all__: `graph_dfs_path`
      - [`hotpot.cheminfo.rules.mathematics.graph_matching`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-graph-matching) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.mathematics.internal_coordinates`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-internal-coordinates) — __all__: `InternalCoordinates`
      - [`hotpot.cheminfo.rules.mathematics.periodic`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-periodic) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.mathematics.sampling`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-sampling) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.mathematics.symmetry`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-symmetry) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.mathematics.topology`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-topology) — __all__: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.mathematics.view_geometry`](delivery_api/rules_api.md#hotpot-cheminfo-rules-mathematics-view-geometry) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.rules.physics`](delivery_api/rules_api.md#hotpot-cheminfo-rules-physics) — 直接模块定义: 空；仅命名空间/私有实现
      - [`hotpot.cheminfo.rules.physics.forcefields`](delivery_api/rules_api.md#hotpot-cheminfo-rules-physics-forcefields) — __all__: `OptimizationStoppingCriteria`
        - [`hotpot.cheminfo.rules.physics.forcefields.acceptance`](delivery_api/rules_api.md#hotpot-cheminfo-rules-physics-forcefields-acceptance) — 直接模块定义: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.physics.forcefields.contracts`](delivery_api/rules_api.md#hotpot-cheminfo-rules-physics-forcefields-contracts) — __all__: `OptimizationStoppingCriteria`, `AcceptanceCheck`, `StructureAcceptanceThresholds`, `ForceFieldAcceptanceEvidence`, `ForceFieldDiagnosticValue`
        - [`hotpot.cheminfo.rules.physics.forcefields.coordinates`](delivery_api/rules_api.md#hotpot-cheminfo-rules-physics-forcefields-coordinates) — __all__: 空；仅命名空间/私有实现
        - [`hotpot.cheminfo.rules.physics.forcefields.settings`](delivery_api/rules_api.md#hotpot-cheminfo-rules-physics-forcefields-settings) — __all__: 空；仅命名空间/私有实现
  - [`hotpot.cheminfo.stereo`](delivery_api/structure_api.md#hotpot-cheminfo-stereo) — __all__: `AtomTransportRecord`, `BondTransportRecord`, `HydrogenTransportState`, `StereoAlignmentReport`, `StereoAnchor`, `StereoAnchorKind`, `StereoAssignmentReport`, `StereoCarrier`, `StereoCarrierKind`, `StereoDerivedDescriptor`, `StereoGroupKind`, `StereoGroupRef`, `StereoIssueCode`, `StereoPayload`, `StereoRecord`, `StereoType`, `StereoValidationIssue`, `StereoValidationReport`, `align_stereochemistry`, `assign_stereochemistry`, `get_stereo_payload`, `set_stereo_payload`, `validate_stereochemistry`
  - [`hotpot.cheminfo.structure`](delivery_api/structure_api.md#hotpot-cheminfo-structure) — __all__: `Molecule`, `Atom`, `Bond`, `BondKind`, `Crystal`, `SpaceGroup`, `SymmUnit`
    - [`hotpot.cheminfo.structure._atom_table`](delivery_api/structure_api.md#hotpot-cheminfo-structure--atom-table) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._builder`](delivery_api/structure_api.md#hotpot-cheminfo-structure--builder) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._cif_facts`](delivery_api/structure_api.md#hotpot-cheminfo-structure--cif-facts) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._commit`](delivery_api/structure_api.md#hotpot-cheminfo-structure--commit) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._crystal`](delivery_api/structure_api.md#hotpot-cheminfo-structure--crystal) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._crystal_codec`](delivery_api/structure_api.md#hotpot-cheminfo-structure--crystal-codec) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._declarations`](delivery_api/structure_api.md#hotpot-cheminfo-structure--declarations) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._dependencies`](delivery_api/structure_api.md#hotpot-cheminfo-structure--dependencies) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._fields`](delivery_api/structure_api.md#hotpot-cheminfo-structure--fields) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._identity`](delivery_api/structure_api.md#hotpot-cheminfo-structure--identity) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._mutation`](delivery_api/structure_api.md#hotpot-cheminfo-structure--mutation) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._results`](delivery_api/structure_api.md#hotpot-cheminfo-structure--results) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._selection`](delivery_api/structure_api.md#hotpot-cheminfo-structure--selection) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._source_identity`](delivery_api/structure_api.md#hotpot-cheminfo-structure--source-identity) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._state`](delivery_api/structure_api.md#hotpot-cheminfo-structure--state) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._stereo`](delivery_api/structure_api.md#hotpot-cheminfo-structure--stereo) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._stereo_codec`](delivery_api/structure_api.md#hotpot-cheminfo-structure--stereo-codec) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure._topology`](delivery_api/structure_api.md#hotpot-cheminfo-structure--topology) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure.atom`](delivery_api/structure_api.md#hotpot-cheminfo-structure-atom) — 直接模块定义: `Atom`
    - [`hotpot.cheminfo.structure.bond`](delivery_api/structure_api.md#hotpot-cheminfo-structure-bond) — 直接模块定义: `Bond`
    - [`hotpot.cheminfo.structure.bond_kind`](delivery_api/structure_api.md#hotpot-cheminfo-structure-bond-kind) — __all__: `BondKind`
    - [`hotpot.cheminfo.structure.bundle`](delivery_api/structure_api.md#hotpot-cheminfo-structure-bundle) — __all__: `MolBundle`
    - [`hotpot.cheminfo.structure.conformer`](delivery_api/structure_api.md#hotpot-cheminfo-structure-conformer) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure.constraints`](delivery_api/structure_api.md#hotpot-cheminfo-structure-constraints) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure.crystal`](delivery_api/structure_api.md#hotpot-cheminfo-structure-crystal) — 直接模块定义: `Crystal`
    - [`hotpot.cheminfo.structure.errors`](delivery_api/structure_api.md#hotpot-cheminfo-structure-errors) — 直接模块定义: `NotInSameMolecule`, `ObjNotInMolecule`
    - [`hotpot.cheminfo.structure.forcefields`](delivery_api/structure_api.md#hotpot-cheminfo-structure-forcefields) — __all__: `AtomIdentity`, `BondTopologyRevision`, `ForceFieldFrame`, `CoordinationEnvironment`, `AtomTopologySignature`, `BondTopology`, `TopologyReference`, `TrajectoryStart`, `TrajectoryStage`, `TrajectoryEvent`, `RingFrameEvidence`, `CoordinationFrameEvidence`, `OptimizationFrameEvidence`, `FrameEvidence`
    - [`hotpot.cheminfo.structure.molecule`](delivery_api/structure_api.md#hotpot-cheminfo-structure-molecule) — 直接模块定义: `Molecule`
    - [`hotpot.cheminfo.structure.snapshots`](delivery_api/structure_api.md#hotpot-cheminfo-structure-snapshots) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure.stereo`](delivery_api/structure_api.md#hotpot-cheminfo-structure-stereo) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.structure.substructure`](delivery_api/structure_api.md#hotpot-cheminfo-structure-substructure) — 直接模块定义: `SmartsSemantics`
    - [`hotpot.cheminfo.structure.symmetry`](delivery_api/structure_api.md#hotpot-cheminfo-structure-symmetry) — __all__: `SpaceGroup`, `SymmUnit`
  - [`hotpot.cheminfo.views`](delivery_api/views_api.md#hotpot-cheminfo-views) — __all__: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.views._source`](delivery_api/views_api.md#hotpot-cheminfo-views--source) — __all__（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
    - [`hotpot.cheminfo.views.geometry`](delivery_api/views_api.md#hotpot-cheminfo-views-geometry) — 直接模块定义: `Angle`, `Torsion`
    - [`hotpot.cheminfo.views.pairs`](delivery_api/views_api.md#hotpot-cheminfo-views-pairs) — 直接模块定义: `AtomPair`, `AtomPairs`
    - [`hotpot.cheminfo.views.sequence`](delivery_api/views_api.md#hotpot-cheminfo-views-sequence) — 直接模块定义: `AtomSeq`
    - [`hotpot.cheminfo.views.topology`](delivery_api/views_api.md#hotpot-cheminfo-views-topology) — 直接模块定义: `JointRing`, `Ring`
- [`hotpot.commands`](delivery_api/cli_api.md#hotpot-commands) — 直接模块定义: 空；仅命名空间/私有实现
  - [`hotpot.commands.cbond`](delivery_api/cli_api.md#hotpot-commands-cbond) — 直接模块定义: `load_cli_documentation`, `add_arguments`, `format_single_result`, `format_all_results`, `run`, `build_parser`, `main`
  - [`hotpot.commands.convert`](delivery_api/cli_api.md#hotpot-commands-convert) — 直接模块定义: `add_arguments`, `convert`, `run`, `ConversionFailure`, `ConversionWorkerResult`, `ConversionBatchError`, `convert_sdf_to_smiles`, `convert_smiles_to_3dmol`, `convert_g16log_to_gjf`
  - [`hotpot.commands.ff`](delivery_api/cli_api.md#hotpot-commands-ff) — __all__: `add_arguments`, `build_parser`, `load_cli_documentation`, `main`, `run`
  - [`hotpot.commands.mca`](delivery_api/cli_api.md#hotpot-commands-mca) — 直接模块定义: `load_cli_documentation`, `add_arguments`, `save_plot`, `run`, `build_parser`, `main`
  - [`hotpot.commands.ml_train`](delivery_api/cli_api.md#hotpot-commands-ml-train) — 直接模块定义: `add_arguments`, `train`
  - [`hotpot.commands.models`](delivery_api/cli_api.md#hotpot-commands-models) — 直接模块定义: `add_arguments`, `run`
  - [`hotpot.commands.optimize`](delivery_api/cli_api.md#hotpot-commands-optimize) — 直接模块定义: `read_excel`, `add_arguments`, `optimize`, `example_cof_params`
- [`hotpot.pipeline`](delivery_api/workflows_api.md#hotpot-pipeline) — __all__: `Artifact`, `DuplicateStageError`, `JSONScalar`, `JSONValue`, `MolecularPayload`, `MolecularRecord`, `MolecularStage`, `PipelineDefinitionError`, `PipelineExecutionError`, `PipelineRunResult`, `PipelineStatus`, `PreparedMolecularStage`, `StageContext`, `StageExecutionError`, `StageResult`, `StageSpec`, `StageStatus`, `UnknownStageError`, `artifact_from_file`, `builtin_stage_names`, `get_stage`, `register_stage`, `run_pipeline`, `stage_import_path`, `payload_sha256`
  - [`hotpot.pipeline._sample_control`](delivery_api/workflows_api.md#hotpot-pipeline--sample-control) — 直接模块定义（私有实现模块；非新增公开路径）: 空；仅命名空间/私有实现
  - [`hotpot.pipeline.artifacts`](delivery_api/workflows_api.md#hotpot-pipeline-artifacts) — __all__: `artifact_from_file`, `payload_sha256`
  - [`hotpot.pipeline.cli`](delivery_api/workflows_api.md#hotpot-pipeline-cli) — __all__: `add_arguments`, `build_parser`, `load_cli_documentation`, `load_json_workflow`, `main`, `parse_inline_stages`, `run`
  - [`hotpot.pipeline.contracts`](delivery_api/workflows_api.md#hotpot-pipeline-contracts) — __all__: `Artifact`, `JSONScalar`, `JSONValue`, `MolecularPayload`, `MolecularRecord`, `MolecularStage`, `PipelineDefinitionError`, `PipelineExecutionError`, `PipelineRunResult`, `PipelineStatus`, `PreparedMolecularStage`, `StageContext`, `StageExecutionError`, `StageResult`, `StageSpec`, `StageStatus`
  - [`hotpot.pipeline.registry`](delivery_api/workflows_api.md#hotpot-pipeline-registry) — __all__: `DuplicateStageError`, `UnknownStageError`, `builtin_stage_names`, `get_stage`, `register_stage`, `stage_import_path`
  - [`hotpot.pipeline.runner`](delivery_api/workflows_api.md#hotpot-pipeline-runner) — __all__: `run_pipeline`
  - [`hotpot.pipeline.state_recovery`](delivery_api/workflows_api.md#hotpot-pipeline-state-recovery) — __all__: `decide_state_recovery`, `RecoveryAction`, `RecoveryPolicySet`, `RecoveryBudget`, `DEFAULT_RECOVERY_BUDGET`, `RecoveryDecision`
  - [`hotpot.plugins.xtb`](delivery_api/workflows_api.md#hotpot-plugins-xtb) — __all__: `GFNXTBMethod`, `XTBApplicabilityError`, `XTBArtifact`, `XTBBackendInfo`, `XTBError`, `XTBExecutableError`, `XTBGeometry`, `XTBExecutionError`, `XTBInputError`, `XTBMethod`, `XTBParsedResult`, `XTBRequest`, `XTBResultError`, `XTBRunReport`, `XTBStreamError`, `XTBStreamMetadata`, `XTBStreamProvenance`, `XTBStreamRecord`, `XTBTask`, `commit_xtb_coordinates`, `molecule_to_xtb_geometry`, `metadata_from_report`, `parse_xtb_artifacts`, `probe_xtb_backend`, `prepare_xtb_input`, `run_xtb`, `run_gfn_xtb`, `run_gfnff`, `read_sdf_records`, `validate_element_support`, `write_sdf_records`, `XTBBackendContext`
    - [`hotpot.plugins.xtb.adapter`](delivery_api/workflows_api.md#hotpot-plugins-xtb-adapter) — __all__: `commit_xtb_coordinates`, `molecule_to_xtb_geometry`, `parse_xtb_artifacts`, `prepare_xtb_input`
    - [`hotpot.plugins.xtb.backend`](delivery_api/workflows_api.md#hotpot-plugins-xtb-backend) — __all__: `probe_xtb_backend`
    - [`hotpot.plugins.xtb.capabilities`](delivery_api/workflows_api.md#hotpot-plugins-xtb-capabilities) — __all__: `validate_element_support`
    - [`hotpot.plugins.xtb.cli`](delivery_api/workflows_api.md#hotpot-plugins-xtb-cli) — __all__: `add_arguments`, `build_parser`, `load_cli_documentation`, `main`, `run`
    - [`hotpot.plugins.xtb.contracts`](delivery_api/workflows_api.md#hotpot-plugins-xtb-contracts) — __all__: `GFNXTBMethod`, `XTBApplicabilityError`, `XTBArtifact`, `XTBBackendInfo`, `XTBError`, `XTBExecutableError`, `XTBExecutionError`, `XTBInputError`, `XTBGeometry`, `XTBMethod`, `XTBParsedResult`, `XTBRequest`, `XTBResultError`, `XTBRunReport`, `XTBTask`
    - [`hotpot.plugins.xtb.runner`](delivery_api/workflows_api.md#hotpot-plugins-xtb-runner) — __all__: `run_xtb`
    - [`hotpot.plugins.xtb.stage`](delivery_api/workflows_api.md#hotpot-plugins-xtb-stage) — __all__: `STAGE`, `get_stage`
    - [`hotpot.plugins.xtb.state_projection`](delivery_api/workflows_api.md#hotpot-plugins-xtb-state-projection) — __all__: `XTBBackendContext`
    - [`hotpot.plugins.xtb.state_resolution`](delivery_api/workflows_api.md#hotpot-plugins-xtb-state-resolution) — 直接模块定义: `ResolvedXTBState`, `XTBStateResolutionError`, `resolve_xtb_state`
    - [`hotpot.plugins.xtb.stream`](delivery_api/workflows_api.md#hotpot-plugins-xtb-stream) — __all__: `XTBStreamError`, `XTBStreamMetadata`, `XTBStreamProvenance`, `XTBStreamRecord`, `metadata_from_report`, `read_sdf_records`, `write_sdf_records`
    - [`hotpot.plugins.xtb.workflow`](delivery_api/workflows_api.md#hotpot-plugins-xtb-workflow) — __all__: `run_gfn_xtb`, `run_gfnff`

## 补充边界

- `SymmUnit` 的实际实例统一为私有 `_AffineSymmUnit`；公开 concrete kind 为 `identity`, `translation`, `inversion`, `rotation`, `screw`, `reflection`, `glide`, `rotoinversion`，不存在另行公开的八个子类。
- 约束通过 `Atom.constraint/x_constraint/y_constraint/z_constraint`、`Bond.constraint`、`Angle.constraint`、`Torsion.constraint` 以及 Molecule 批量操作访问；`_ConstraintSet` 保持私有。
- 通用图转换仅 `Molecule/Crystal.to_pyg()` 与 `from_pyg(..., field_map=None)`，当前 `hotpot.molecule/2`；`to_pyg_dataset` 为集合导出。无 `to_pyg_data` 兼容入口。
- 原生入口由 `.pyi` 逐条列出，名称含下划线者标记为实现边界；不将 native 所有可调用符号都宣称为用户 API。
- 私有结构/结果 owner 的完整机器声明保留于 JSON；不因为实现声明名存在便升级为公开合同。


## 辅助完整成员参考

- [辅助模块及类入口](delivery_api/auxiliary_index.md)：172 个模块的精确模块/类锚点。
- [dataset/ds 与工具](delivery_api/auxiliary_data_utils_api.md)：54 个模块。
- [ComplexFormer](delivery_api/auxiliary_complexformer_api.md)：48 个模块。
- [其他插件](delivery_api/auxiliary_plugins_api.md)：70 个模块。

这些静态声明与主交付的 354 个模块合计覆盖当前 526 个 Python 模块。运行时第三方成员、可选依赖可用性和科学验证范围依各自说明。
