# 拓扑与几何视图：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot.cheminfo.views`](#hotpot-cheminfo-views)
- [`hotpot.cheminfo.views._source`](#hotpot-cheminfo-views--source)
- [`hotpot.cheminfo.views.geometry`](#hotpot-cheminfo-views-geometry)
- [`hotpot.cheminfo.views.pairs`](#hotpot-cheminfo-views-pairs)
- [`hotpot.cheminfo.views.sequence`](#hotpot-cheminfo-views-sequence)
- [`hotpot.cheminfo.views.topology`](#hotpot-cheminfo-views-topology)

<a id="hotpot-cheminfo-views"></a>
## `hotpot.cheminfo.views`

`hotpot/cheminfo/views/__init__.py`；显式 __all__: ()



<a id="hotpot-cheminfo-views--source"></a>
## `hotpot.cheminfo.views._source`

`hotpot/cheminfo/views/_source.py`；显式 __all__: ()



<a id="hotpot-cheminfo-views-geometry"></a>
## `hotpot.cheminfo.views.geometry`

`hotpot/cheminfo/views/geometry.py`；显式 __all__: 无


<a id="hotpot-cheminfo-views-geometry-angle"></a>
### `Angle`

定义 owner `hotpot.cheminfo.views.geometry.Angle`；基类 `AtomSeq`。

- `__init__(self, a1: Atom, a2: Atom, a3: Atom)`
- `degrees` [只读]: `float`
- `constraint` [读/写]: `bool`（动态投影）
- `__setattr__(self, name, value)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__reduce_ex__(self, protocol)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__repr__(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__getattr__(self, item)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__len__(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__contains__(self, item: Union[Atom, 'Bond'])` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__eq__(self, other)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `to_mol(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms` [只读]: `tuple[Atom, ...]` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms_indices` [只读] ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bonds` [只读]: `list['Bond']` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `mol` [只读]: `Molecule` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `has_metal` [只读] ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atomN` [只读]: `Atom` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bondN` [只读]: `Bond` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `aNidx` [只读]: `int` ← `hotpot.cheminfo.views.sequence.AtomSeq`

<a id="hotpot-cheminfo-views-geometry-torsion"></a>
### `Torsion`

定义 owner `hotpot.cheminfo.views.geometry.Torsion`；基类 `AtomSeq`。

- `__init__(self, a: Atom, b: Atom, c: Atom, d: Atom)`
- `degrees` [只读]: `float`
- `rotatable` [只读]: `bool`
- `constraint` [读/写]: `bool`（动态投影）
- `__setattr__(self, name, value)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__reduce_ex__(self, protocol)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__repr__(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__getattr__(self, item)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__len__(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__contains__(self, item: Union[Atom, 'Bond'])` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__eq__(self, other)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `to_mol(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms` [只读]: `tuple[Atom, ...]` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms_indices` [只读] ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bonds` [只读]: `list['Bond']` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `mol` [只读]: `Molecule` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `has_metal` [只读] ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atomN` [只读]: `Atom` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bondN` [只读]: `Bond` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `aNidx` [只读]: `int` ← `hotpot.cheminfo.views.sequence.AtomSeq`


<a id="hotpot-cheminfo-views-pairs"></a>
## `hotpot.cheminfo.views.pairs`

`hotpot/cheminfo/views/pairs.py`；显式 __all__: 无


<a id="hotpot-cheminfo-views-pairs-atompair"></a>
### `AtomPair`

定义 owner `hotpot.cheminfo.views.pairs.AtomPair`；基类 无。

- `__init__(self, atom1: Atom, atom2: Atom)`
- `__reduce_ex__(self, protocol)`
- `__repr__(self)`
- `__eq__(self, other)`
- `__contains__(self, atom)`
- `atoms` [只读]
- `atom1` [只读]
- `atom2` [只读]
- `mol` [只读]
- `wiberg_bond_order` [读/写]；setter `wiberg_bond_order(self, value)`
- `attrs` [只读]
- `shortest_path` [只读]: `list['Atom']`
- `length_shortest_path` [只读]: `int`
- `distance` [只读]
- `attr_names` = `('wiberg_bond_order', 'length_shortest_path')` [类声明值；可重绑定不构成事实写许可]

<a id="hotpot-cheminfo-views-pairs-atompairs"></a>
### `AtomPairs`

定义 owner `hotpot.cheminfo.views.pairs.AtomPairs`；基类 `Mapping`。

- `__init__(self, *args, mol, **kwargs)`
- `__reduce_ex__(self, protocol)`
- `__len__(self)`
- `__iter__(self)`
- `__getitem__(self, atoms)`
- `getdefault(self, atom_pair)`
- `copy(self)`
- `update_pairs(self)`
- `pairs_shortest_path(self) -> dict`
- `mol` [只读]
- `pair_distance` [只读]
- `idx_matrix` [只读]: `np.ndarray`
- `get(self, key, default=None)` ← `collections.abc.Mapping`
- `items(self)` ← `collections.abc.Mapping`
- `keys(self)` ← `collections.abc.Mapping`
- `values(self)` ← `collections.abc.Mapping`
- `__contains__(self, key)` ← `collections.abc.Mapping`
- `__eq__(self, other)` ← `collections.abc.Mapping`
- `__ne__(self, other)` ← `collections.abc.Mapping`


<a id="hotpot-cheminfo-views-sequence"></a>
## `hotpot.cheminfo.views.sequence`

`hotpot/cheminfo/views/sequence.py`；显式 __all__: 无


<a id="hotpot-cheminfo-views-sequence-atomseq"></a>
### `AtomSeq`

定义 owner `hotpot.cheminfo.views.sequence.AtomSeq`；基类 无。

- `__setattr__(self, name, value)`
- `__reduce_ex__(self, protocol)`
- `__init__(self, *atoms: Atom)`
- `__repr__(self)`
- `__getattr__(self, item)`
- `__len__(self)`
- `__contains__(self, item: Union[Atom, 'Bond'])`
- `__eq__(self, other)`
- `to_mol(self)`
- `atoms` [只读]: `tuple[Atom, ...]`
- `atoms_indices` [只读]
- `bonds` [只读]: `list['Bond']`
- `mol` [只读]: `Molecule`
- `has_metal` [只读]
- `atomN` [只读]: `Atom`
- `bondN` [只读]: `Bond`
- `aNidx` [只读]: `int`


<a id="hotpot-cheminfo-views-topology"></a>
## `hotpot.cheminfo.views.topology`

`hotpot/cheminfo/views/topology.py`；显式 __all__: 无


<a id="hotpot-cheminfo-views-topology-jointring"></a>
### `JointRing`

定义 owner `hotpot.cheminfo.views.topology.JointRing`；基类 无。

- `__setattr__(self, name, value)`
- `__reduce_ex__(self, protocol)`
- `__init__(self, *ring)`
- `__contains__(self, item: Union[Atom, Bond, 'Ring'])`
- `index(self, atom: Atom) -> Optional[int]`
- `index_bond(self, bond: Bond)`
- `atom_neigh_atom(self, atom: Atom)`
- `atom_neigh_bond(self, atom: Atom)`
- `bond_neigh_bond(self, bond: Bond)`
- `check_kekulize(self)`
- `rings` [只读]
- `atoms` [只读]
- `bonds` [只读]
- `mol` [只读]
- `empty` [只读]: `bool`

<a id="hotpot-cheminfo-views-topology-ring"></a>
### `Ring`

定义 owner `hotpot.cheminfo.views.topology.Ring`；基类 `AtomSeq`。

- `__init__(self, *atoms: Atom)`
- `to_mol(self)`
- `to_pair_edge(self)`
- `joint_with(self, other: 'Ring') -> bool`
- `joint_ring(self, aromatic=True) -> Optional['JointRing']`
- `next_atom(self, atom: Atom, reverse: bool=False) -> Atom`
- `relation_to_bond(self, bond: Bond) -> 'geometry_adapter.BondRingFinding[Ring, Bond]'`
- `closest_edge_to_bond(self, bond: Bond) -> Optional['geometry_adapter.RingEdgeDistance[Bond]']`
- `is_aromatic` [只读]: `bool`
- `is_disorder` [只读]
- `has_metal` [只读]
- `has_3d` [只读]
- `geometry_cycle` [只读]: `geometry.Cycle`
- `__setattr__(self, name, value)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__reduce_ex__(self, protocol)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__repr__(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__getattr__(self, item)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__len__(self)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__contains__(self, item: Union[Atom, 'Bond'])` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `__eq__(self, other)` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms` [只读]: `tuple[Atom, ...]` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atoms_indices` [只读] ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bonds` [只读]: `list['Bond']` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `mol` [只读]: `Molecule` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `atomN` [只读]: `Atom` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `bondN` [只读]: `Bond` ← `hotpot.cheminfo.views.sequence.AtomSeq`
- `aNidx` [只读]: `int` ← `hotpot.cheminfo.views.sequence.AtomSeq`

模块值/类型别名（声明存在不构成事实写许可）：

- `geometry` = `_geometry_namespaces()`
- `geometry_adapter` = `_geometry_namespaces()`
