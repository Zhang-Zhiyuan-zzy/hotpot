"""
python v3.9.0
@Project: hotpot
@File   : search
@Auther : Zhiyuan Zhang
@Data   : 2024/12/13
@Time   : 16:27
"""
from abc import abstractproperty
from typing import Union, Sequence, Literal
import networkx as nx
from hotpot.cheminfo.core import Molecule, Atom, Bond
from typing import List, Tuple, Optional  # 修改：导入 Optional 用于返回类型
from networkx.algorithms import isomorphism

def raise_not_implemented(self): raise NotImplemented(f"{self.__class__.__name__} not implemented")

class Query:
    _match_class = abstractproperty(raise_not_implemented)

    def __init__(self, **kwargs: set):
        self.kwargs = kwargs

    def __eq__(self, other: dict):
        if isinstance(other, self._match_class):
            other = {attr: getattr(other, attr) for attr in self.kwargs}

        return all(other[attr] in self.kwargs[attr] for attr in self.kwargs)


class QueryAtom(Query):
    _match_class = Atom

    def __init__(self, mol: "Substructure" = None, **attrs):
        self.mol = mol
        super().__init__(**attrs)

    @property
    def idx(self) -> int:
        return self._idx

    @classmethod
    def from_atom(
            cls, atom: Atom,
            include_attrs: Sequence[str] = None,
            exclude_attrs: Sequence[Literal['atomic_number', 'is_aromatic']] = None
    ):
        kwargs = {
            "atomic_number": atom.atomic_number,
            "is_aromatic": atom.is_aromatic
        }

        if include_attrs:
            for attr in include_attrs:
                kwargs[attr] = set(getattr(atom, attr))

        if exclude_attrs:
            for attr in exclude_attrs:
                kwargs.pop(attr)
        return cls(**kwargs)

    def __eq__(self, other):
        if not isinstance(other, QueryAtom):
            return False
        # 只比较属性，避免对 QueryAtom 对象进行索引
        return self.kwargs == other.kwargs

class QueryBond(Query):

    _match_class = Bond

    def __init__(self, atom1: QueryAtom, atom2: QueryAtom, bond_order: float, **attrs):
        self.atom1 = atom1
        self.atom2 = atom2
        self.bond_order = bond_order  # 存储bond_order
        super().__init__(**attrs)

    @property
    def a1idx(self) -> int:
        return self.atom1.idx

    @property
    def a2idx(self) -> int:
        return self.atom2.idx


class Substructure:
    """"""
    def __init__(self):
        self.query_atoms = []
        self.query_bonds = []
        self.query_graph = None  # 确保这里初始化图对象
        self.atom_idx_counter = 0  # 用于生成唯一的原子索引

    def add_atom(self, atom: Union[Atom, QueryAtom]):
        if isinstance(atom, Atom):
            atom = QueryAtom.from_atom(atom)

        # 确保每个原子有唯一的索引
        atom._idx = self.atom_idx_counter
        self.atom_idx_counter += 1

        self.query_atoms.append(atom)
        atom.mol = self

        return atom

    def add_bond(self, atom1: Union[int, QueryAtom], atom2: Union[int, QueryAtom], **bond_attrs):
        if isinstance(atom1, int):
            atom1 = self.query_atoms[atom1]
        if isinstance(atom2, int):
            atom2 = self.query_atoms[atom2]

        bond = QueryBond(atom1, atom2, **bond_attrs)
        self.query_bonds.append(bond)

        return bond

    def construct_graph(self):

        self.query_graph = nx.Graph()
        self.query_graph.add_nodes_from([(a.idx, {'qa': a}) for a in self.query_atoms])
        self.query_graph.add_edges_from([(b.a1idx, b.a2idx, {'qb': b}) for b in self.query_bonds])

class Searcher:
    def __init__(self, substructure: "Substructure"):
        self.substructure = substructure
        self.substructure.construct_graph()

    def search(self, mol: Molecule) -> Optional[List[dict]]:
        """
        搜索目标分子中与子结构匹配的部分，并返回详细的匹配结果。

        :param mol: 目标分子对象
        :return: 包含匹配详细信息的列表，如果没有匹配，返回 None
        """
        print(f"Starting search with molecule: {mol}")

        # 构建目标分子的图表示
        mol_graph = self._build_molecule_graph(mol)

        # 查找匹配
        matchings = self._find_subgraph_matches(mol_graph)

        # 如果没有匹配，返回 None
        if not matchings:
            print("No matches found.")
            return None

        # 构造详细的匹配结果
        detailed_matches = []

        for match in matchings:
            # 提取当前匹配的详细信息
            match_info = {
                "nodes": [],  # 存储节点的详细信息
                "edges": []  # 存储边的详细信息
            }

            # 处理节点信息
            for sub_node, mol_node in match:
                #sub_atom = self.substructure.query_graph.nodes[sub_node]["qa"]
                mol_atom = mol_graph.nodes[mol_node]

                match_info["nodes"].append({
                    "substructure_node": sub_node,
                    "molecule_node": mol_node,
                    "atomic_number": mol_atom.get("atomic_number"),
                    "is_aromatic": mol_atom.get("is_aromatic")
                })

            # 处理边信息
            # 修复处理边信息部分
            for sub_edge in self.substructure.query_graph.edges():
                sub_node1, sub_node2 = sub_edge
                mol_node1 = match[sub_node1]
                mol_node2 = match[sub_node2]

                # 提取原子编号
                mol_node1_idx = mol_node1[0]
                mol_node2_idx = mol_node2[0]

                sub_bond = self.substructure.query_graph.edges[sub_node1, sub_node2]["qb"]

                if mol_graph.has_edge(mol_node1_idx, mol_node2_idx):
                    mol_bond = mol_graph.edges[mol_node1_idx, mol_node2_idx]
                elif mol_graph.has_edge(mol_node2_idx, mol_node1_idx):  # 检查反向边
                    mol_bond = mol_graph.edges[mol_node2_idx, mol_node1_idx]
                else:
                    mol_bond = None
                    print(f"No edge found between {mol_node1_idx} and {mol_node2_idx}")

                # 记录边的信息
                match_info["edges"].append({
                    "substructure_edge": (sub_node1, sub_node2),
                    "molecule_edge": (mol_node1_idx, mol_node2_idx),
                    "sub_bond_order": sub_bond.bond_order,
                    "molecule_bond_order": mol_bond.get("bond_order") if mol_bond else None
                })

            # 添加到匹配结果中
            detailed_matches.append(match_info)

            print(detailed_matches)

        return detailed_matches

    def _build_molecule_graph(self, mol: Molecule) -> nx.Graph:
        """
        构建目标分子的图表示。
        :param mol: 目标分子对象
        :return: 分子的图表示
        """
        mol_graph = nx.Graph()

        # 添加原子节点到图中
        for idx, atom in enumerate(mol.atoms):
            mol_graph.add_node(idx, atomic_number=atom.atomic_number, is_aromatic=atom.is_aromatic)

        # 添加键（边）到图中
        for bond in mol.bonds:
            mol_graph.add_edge(bond.atom1.idx, bond.atom2.idx, bond_order=bond.bond_order)

        return mol_graph

    def _find_subgraph_matches(self, mol_graph: nx.Graph) -> List[Tuple[int, int]]:
        """
        查找子结构图和目标分子图的匹配，增加详细的调试信息。
        """
        matches = []

        GM = isomorphism.GraphMatcher(
            mol_graph,
            self.substructure.query_graph,
            node_match=self._node_match,
            edge_match=self._edge_match
        )

        for subgraph_match in GM.subgraph_isomorphisms_iter():
            print(f"匹配成功的子图：{subgraph_match}")  # 输出匹配信息
            matches.append(list(subgraph_match.items()))

        return matches

    def _node_match(self, mol_node_data: dict, sub_node_data: dict) -> bool:
        sub_atom = sub_node_data.get('qa')  # 子结构中的 QueryAtom
        mol_atom = mol_node_data

        # 防止 sub_atom 或 mol_atom 为空
        if not sub_atom or not mol_atom:
            return False

        # 获取和比较 aromatic 属性
        is_aromatic_sub = sub_atom.kwargs.get('is_aromatic', None)
        is_aromatic_mol = mol_atom.get('is_aromatic', None)
        if is_aromatic_sub is not None and is_aromatic_mol is not None and is_aromatic_sub != is_aromatic_mol:
            return False

        # 判断原子类型是否匹配
        allowed_atomic_numbers = {
            'C': {6},  # C 原子
            'N': {7},  # N 原子
            'O': {8},  # O 原子
            'P': {15},  # P 原子
        }

        # 获取原子类型并进行匹配
        atomic_number_sub = sub_atom.kwargs.get('atomic_number', set())
        atomic_number_mol = mol_atom.get('atomic_number', None)

        # 确保 atomic_number_sub 是集合
        #if isinstance(atomic_number_sub, set):

        if atomic_number_sub == atomic_number_mol:
            return True

        # elif atomic_number_mol and atomic_number_sub in {num for nums in allowed_atomic_numbers.values() for num in nums}:
        # return True

        return False

    def _edge_match(self, mol_edge_data: dict, sub_edge_data: dict) -> bool:

        sub_bond = sub_edge_data.get('qb')  # 子结构中的 QueryBond
        mol_bond_order = mol_edge_data.get('bond_order', None)

        if not sub_bond or mol_bond_order is None:
            return False

        # 确保 sub_bond.bond_order 和 mol_bond_order 都是相同类型（转换为 float）
        sub_bond_order = float(sub_bond.bond_order)

        # 定义模糊匹配规则
        allowed_bond_orders = {
            'single': {1.0},  # 单键
            'aromatic': {1.5},
            'double': {2.0},  # 双键
        }

        # 检查模糊匹配规则
        if sub_bond_order in {k for ks in allowed_bond_orders.values() for k in ks}:
            if mol_bond_order in {k for ks in allowed_bond_orders.values() for k in ks}:
                return True

        # 默认要求键类型完全匹配
        if sub_bond_order == mol_bond_order:
            return True

        return False