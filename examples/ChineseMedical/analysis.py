"""
@File Name:        analysis
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/10 18:39
@Project:          Hotpot
"""
import io
import os.path as osp
from functools import reduce
from operator import or_
from typing import Literal, Optional, List, Tuple, Dict
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, rdFMCS

import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


from hotpot.cheminfo import draw, rdworks

mols = [
    "O=C1C(O)=C(C)[C@@](C[C@](O2)([H])[C@]3(CO4)[C@]5([H])[C@@H](O)[C@H](O)[C@]4(C(OC)=O)[C@]3([H])[C@@H](OC(/C([H])=C(C)/C(C)C)=O)C2=O)([H])[C@]5(C)C1",
    "O=C6C(O)=C(C)[C@@](C[C@](O7)([H])[C@]8(CO9)[C@]%10([H])[C@@H](O)[C@H](O)[C@]9(C(OC)=O)[C@]8([H])[C@@H](OC(/C([H])=C(C)/C(C)(OC(C)=O)C)=O)C7=O)([H])[C@]%10(C)C6",
    "O=C1C(O)=C(C)[C@@](C[C@](O2)([H])[C@]3(CO4)[C@]5([H])[C@@H](O)[C@H](O)[C@]4(C(OC)=O)[C@]3([H])[C@@H](OC(CC(C)C)=O)C2=O)([H])[C@]5(C)C1",
    "O=C1C(O)=C(C)[C@@](C[C@](O2)([H])[C@]3(CO4)[C@]5([H])[C@@H](O)[C@H](O)[C@]4(C(OC)=O)[C@]3([H])[C@@H](OC(C)=O)C2=O)([H])[C@]5(C)C1",
    "O=C%21C=C(C)[C@@](C[C@](O%22)([H])[C@]%23(CO%24)[C@]%25([H])[C@@H](O)[C@H](O)[C@]%24(C)[C@]%23(O)[C@@H](O)C%22=O)([H])[C@]%25(C)C%21O",
    "O=C%26C(O)=C(C)[C@@](C[C@](O%27)([H])[C@]%28(CO%29)[C@]%30([H])[C@@H](O)[C@H](O)[C@]%29(C(OC)=O)[C@]%28([H])[C@@H](OC(/C=C(C)/C)=O)C%27=O)([H])[C@]%30(C)C%26",
    "O=C%31C=C(C)[C@@](C[C@](O%32)([H])[C@]%33(CO%34)[C@]%35([H])[C@@H](O)[C@H](O)[C@]%34(CO)[C@]%33(O)[C@@H](O)C%32=O)([H])[C@]%35(C)C%31O",
    "OC%36=C[C@@]%37(C)C(C[C@](O%38)([H])[C@]%39(CO%40)[C@]%37([H])[C@@H](O)[C@H](O)[C@]%40(C(OC)=O)[C@]%39([H])[C@@H](OC(/C=C(C)/C)=O)C%38=O)=C(C)C%36=O",
    "OC%41C=C(C)[C@@](C[C@](O%42)([H])[C@]%43(CO%44)[C@]%45([H])[C@@H](O)[C@H](O)[C@]%44(C)[C@]%43(O)[C@@H](O)C%42=O)([H])[C@]%45(C)C%41O",
    "C=C%46C(O[C@@H]%47[C@@H](O)[C@H](O)[C@@H](O)[C@H](CO)O%47)=C[C@]%48(C)[C@@](C[C@@](O%49)([H])[C@@]%50(CO%51)[C@@]%48([H])[C@H](O)[C@@H](O)[C@]%51(C(OC)=O)[C@@]%50([H])[C@H](OC(CC(C)C)=O)C%49=O)([H])[C@H]%46C",
    "O=C%52C(O[C@@H]%53[C@@H](O)[C@H](O)[C@@H](O)[C@H](CO)O%53)=C[C@]%54(C)[C@@](C[C@@](O%55)([H])[C@@]%56(CO%57)[C@@]%54([H])[C@H](O)[C@@H](O)[C@]%57(C(OC)=O)[C@@]%56([H])[C@H](OC(C)=O)C%55=O)([H])[C@H]%52C",
    "O=C%58C(O[C@@H]%59[C@@H](O)[C@H](O)[C@@H](O)[C@H](CO)O%59)=C[C@]%60(C)[C@@](C[C@@](O%61)([H])[C@@]%62(CO%63)[C@@]%60([H])[C@H](O)[C@@H](O)[C@]%63(C(OC)=O)[C@@]%62([H])[C@H](OC(/C([H])=C(C)/C(C)(OC(C)=O)C)=O)C%61=O)([H])[C@H]%58C",
    "O=C%64C(O[C@@H]%65[C@@H](O)[C@H](O)[C@@H](O)[C@H](CO)O%65)=C[C@]%66(C)[C@@](C[C@@](O%67)([H])[C@@]%68(CO%69)[C@@]%66([H])[C@H](O)[C@@H](O)[C@]%69(C(OC)=O)[C@@]%68([H])[C@H](OC(/C=C(C)/C(C)(O)C)=O)C%67=O)([H])[C@H]%64C",
    "O=C%70C(O[C@@H]%71[C@@H](O)[C@H](O)[C@@H](O)[C@H](CO)O%71)=C[C@]%72(C)[C@@](C[C@@](O%73)([H])[C@@]%74(CO%75)[C@@]%72([H])[C@H](O)[C@@H](O)[C@]%75(C(OC)=O)[C@@]%74([H])[C@H](OC(/C([H])=C(C)/C(C)C)=O)C%73=O)([H])[C@H]%70C",
]

ec50 = [
    0.08,
    0.32,
    0.52,
    0.24,
    0.97,
    0.08,
    0.85,
    1.86,
    float('inf'),
    float('inf'),
    float('inf'),
    float('inf'),
    float('inf'),
    float('inf'),
]

classes = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


_generators = {
    'Morgen': AllChem.GetMorganGenerator,
    'RdKit': AllChem.GetRDKitFPGenerator,
    'Torison': AllChem.GetTopologicalTorsionGenerator,
    'AtomPair': AllChem.GetAtomPairGenerator,

}
def load_fp(
        generator: Literal['Morgen', 'RdKit', 'AtomPair', 'Torison'] = 'Morgen',
        gen_kw: dict = None,
        sparse: bool = True,
        counts: bool = True,
):
    if gen_kw is None:
        gen_kw = {}
    if generator not in _generators:
        raise ValueError(f'invalid generator {generator}, please choose from {_generators.keys()}')
    else:
        fpgen = _generators[generator](**gen_kw)

    if counts:
        if sparse:
            gen_fn = fpgen.GetSparseCountFingerprint
        else:
            gen_fn = fpgen.GetCountFingerprint
    else:
        if sparse:
            gen_fn = fpgen.GetSparseFingerprint
        else:
            gen_fn = fpgen.GetFingerprint

    list_mols = [Chem.MolFromSmiles(smi) for smi in mols]

    if generator not in  ['Morgen', 'RdKit']:
        list_fp = [gen_fn(m) for m in list_mols]
        list_bit = []
    else:
        list_fp = []
        list_bit = []
        if generator == 'Morgen':
            collector = 'CollectBitInfoMap'
            getter = 'GetBitInfoMap'
        elif generator == 'RdKit':
            collector = 'CollectBitPaths'
            getter = 'GetBitPaths'
        else:
            raise ValueError(f'invalid generator {generator}')

        for mol in list_mols:
            ao = AllChem.AdditionalOutput()
            getattr(ao, collector)()
            fp = gen_fn(mol, additionalOutput=ao)
            bit = getattr(ao, getter)()

            list_fp.append(fp)
            list_bit.append(bit)

    return list_mols, list_fp, list_bit

def merge_fps(list_fp: list):
    return reduce(or_, list_fp)

def fps_to_array(fps: list, to_bit: bool = False):
    mfps = list(merge_fps(fps).GetNonzeroElements())
    arr = np.zeros((len(fps), len(mfps)))

    for i, fp in enumerate(fps):
        for f, counts in fp.GetNonzeroElements().items():
            arr[i, mfps.index(f)] = counts

    if to_bit:
        arr = arr.astype(bool).astype(int)

    return arr, mfps


def mannwhitneyu_test(arr, cls, fp_hash: list[int] = None):
    cls = np.asarray(cls)
    arr = np.asarray(arr)

    p_values = []
    U_values = []

    for i in range(arr.shape[1]):
        group0 = arr[cls == 0, i]
        group1 = arr[cls == 1, i]

        # Perform two-sided Mann-Whitney U test
        stat, p = mannwhitneyu(group0, group1, alternative='two-sided')

        U_values.append(stat)
        p_values.append(p)

    results = pd.DataFrame({
        'fingerprint': range(arr.shape[1]),
        'U_stat': U_values,
        'p_value': p_values
    })

    if isinstance(fp_hash, list):
        if len(fp_hash) != arr.shape[1]:
            raise ValueError(f'fp_hash should equal to arr.shape[1], but {len(fp_hash)} != {arr.shape[1]}')

        results['fp_hash'] = np.asarray(fp_hash)

    return results


def show_results(arr, cls, results):
    cls = np.asarray(cls)
    arr = np.asarray(arr)

    # Compute median difference as a simple effect size
    median_diff = []
    for i in range(arr.shape[1]):
        diff = np.median(arr[cls == 1, i]) - np.median(arr[cls == 0, i])
        median_diff.append(diff)
    results['median_diff'] = median_diff

    # Volcano plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='median_diff', y=-np.log10(results['p_value']), data=results)
    plt.xlabel('Median difference (Active - Inactive)')
    plt.ylabel('-log10(p-value)')
    plt.title('Mann–Whitney U Test — Fingerprint Importance')
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    plt.legend()
    plt.show()


def extract_top_fp(results: pd.DataFrame, head: int = 41):
    top_feats = results.sort_values('p_value').head(head)
    print(top_feats)
    return top_feats


############# BUILD Machine Learning Models ##########################
def validate_rf(arr, cls, cv=6):
    classifier = RandomForestClassifier(n_estimators=100)
    # classifier = GradientBoostingClassifier(n_estimators=100)
    arr = np.asarray(arr)
    cls = np.asarray(cls)
    scores = cross_val_score(classifier, arr, cls, cv=cv)
    print(f"Scores(CV={cv}): {scores}")
    print(f"Scores Mean: {scores.mean()}")
    print(f"Scores Std: {scores.std()}")


def get_feature_importance(arr, cls):
    arr = np.asarray(arr)
    cls = np.asarray(cls)

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(arr, cls)
    return classifier


def explain_fingerprint_shap(model, fp_array, plot: bool = True):
    """
    用 SHAP 解释随机森林分类模型对分子指纹的贡献。

    参数
    ----
    model : sklearn.ensemble.RandomForestClassifier
        已训练好的随机森林模型。
    list_mol : list of rdkit.Chem.Mol
        需要解释的 RDKit 分子对象列表。
    n_bits : int
        Morgan 指纹的长度 (默认 2048)。
    radius : int
        Morgan 指纹的半径 (默认 2)。
    plot : bool
        是否绘制 SHAP 总结图。

    返回
    ----
    shap_values : numpy.ndarray
        每个分子对应的每一位指纹的 SHAP 值 (形状: [n_samples, n_bits])
    fingerprints : numpy.ndarray
        输入的二进制指纹矩阵
    """
    # 2. 初始化 SHAP 解释器（用于树模型）
    explainer = shap.TreeExplainer(model)

    # 3. 计算 SHAP 值
    return np.asarray(explainer.shap_values(fp_array))


def attribute_sv_to_atoms(
    mol,
    shap_values: np.ndarray,  # [E, C]
    list_fp: list[int],       # 长度 E，对应 shap 行顺序
    fp_map: dict[int, list[tuple[int, int]]],  # 哈希 → [(中心原子, 半径), ...]
    mode: str = "sum",        # "sum" or "mean"
    scale_to_unit: bool = True  # 是否缩放到 [-1, 1]
):
    """
    将指纹级 SHAP 值分摊到分子每个原子上，可选择累加模式或平均模式。

    参数
    ----
    mol : rdkit.Chem.Mol
        目标分子
    shap_values : array [E, C]
        每个指纹（bit）的 SHAP 值矩阵
    list_fp : list[int]
        指纹哈希值列表，与 shap_values 对齐
    fp_map : dict
        哈希 → [(中心原子index, 半径), ...]
    mode : str
        'sum' 表示对包含原子的所有指纹 SHAP 求和；
        'mean' 表示求均值（当一个原子命中多个指纹时平滑）

    返回
    ----
    atom_sv : np.ndarray [N_atoms, C]
        每个原子的 SHAP 贡献（按类别分列）
    """
    if mode not in {"sum", "mean"}:
        raise ValueError(f"Invalid mode '{mode}', must be 'sum' or 'mean'.")

    n_atoms = mol.GetNumAtoms()
    n_classes = shap_values.shape[1]
    atom_sv = np.zeros((n_atoms, n_classes), dtype=float)

    # 每个原子对应的指纹集合
    atom_to_fps = {i: [] for i in range(n_atoms)}

    for fp_hash, centers in fp_map.items():
        for center_idx, radius in centers:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx)
            if not env:
                continue
            sub_atoms = set()
            for bidx in env:
                bond = mol.GetBondWithIdx(bidx)
                sub_atoms.add(bond.GetBeginAtomIdx())
                sub_atoms.add(bond.GetEndAtomIdx())
            for aidx in sub_atoms:
                atom_to_fps[aidx].append(fp_hash)

    fp_index_map = {fp: i for i, fp in enumerate(list_fp)}  # 哈希→行索引

    # 遍历每个原子累积 shap
    for atom_idx, fps in atom_to_fps.items():
        if not fps:
            continue
        contribs = []
        for fp_hash in fps:
            if fp_hash not in fp_index_map:
                continue
            fp_row = fp_index_map[fp_hash]
            contribs.append(shap_values[fp_row])
        if not contribs:
            continue
        contribs = np.stack(contribs)
        if mode == "sum":
            atom_sv[atom_idx] = contribs.sum(axis=0)
        else:  # mean 模式
            atom_sv[atom_idx] = contribs.mean(axis=0)

    if scale_to_unit:
        abs_max = np.max(np.abs(atom_sv))
        if abs_max > 0:
            atom_sv = atom_sv / abs_max  # 保持符号, 缩放范围 [-1,1]

    return atom_sv


def compute_atom_bond_colors(
        mol: Chem.Mol,
        atom_values: np.ndarray,
        cmap_name: str = "coolwarm",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: float = 0.75
) -> Tuple[
        Dict[int, Tuple[float, float, float]],
        Dict[int, Tuple[float, float, float]],
        mcolors.Normalize,
        cm.ScalarMappable
]:
    """
    计算原子和键的颜色映射。

    Parameters
    ----------
    mol : Chem.Mol
        RDKit分子对象
    atom_values : np.ndarray
        每个原子的SHAP值，shape = [n_atoms]
    cmap_name : str
        Matplotlib colormap名称
    vmin, vmax : float, optional
        颜色映射的最小/最大值。如果为None，自动计算
    threshold : float
        低于此阈值的值不着色

    Returns
    -------
    atom_colors : dict
        原子索引 -> RGB颜色元组
    bond_colors : dict
        键索引 -> RGB颜色元组
    norm : mcolors.Normalize
        归一化对象
    sm : cm.ScalarMappable
        用于生成colorbar的对象
    """
    n_atoms = mol.GetNumAtoms()
    atom_values = np.asarray(atom_values).flatten()
    assert len(atom_values) == n_atoms, f"atom_values length {len(atom_values)} != n_atoms {n_atoms}"

    # 设置颜色范围
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(atom_values))
        if abs_max == 0 or np.isnan(abs_max):
            abs_max = 1.0
        vmin = -abs_max
        vmax = abs_max

    # 创建颜色映射
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    # 计算原子颜色
    atom_colors = {}
    for i, val in enumerate(atom_values):
        if not np.isnan(val) and abs(val) > threshold:
            rgba = cmap(norm(val))
            atom_colors[i] = tuple(rgba[:3])  # RGB only

    # 计算键颜色（基于两端原子的平均值）
    bond_colors = {}
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()

        # 计算键的值为两端原子的平均
        bond_val = (atom_values[begin_atom] + atom_values[end_atom]) / 2

        if not np.isnan(bond_val) and abs(bond_val) > threshold:
            rgba = cmap(norm(bond_val))
            bond_colors[bond_idx] = tuple(rgba[:3])

    return atom_colors, bond_colors, norm, sm


if __name__ == "__main__":
    ms = [Chem.MolFromSmiles(smi) for smi in mols]
    list_mols, fps, bits = rdworks.load_fp(ms)
    arr, mfps = rdworks.fps_to_array(fps, to_bit=True)

    model = get_feature_importance(arr, classes)
    sv = explain_fingerprint_shap(model, arr)[1]

    save_path = '/mnt/d/zhang/OneDrive/Liu/BC_medical/medicals'
    fmt = ('svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp')
    # for i in range(arr.shape[0]):
    #     atom_sv = rdworks.assign_fp_values_atoms(ms[i], sv[i], mfps, bits[i], mode='mean')
    #     # colors, bonds, norm, sm = draw.compute_atom_bond_colors(ms[i], atom_sv)
    #     draw.draw_single_mol(
    #         ms[i],
    #         osp.join(save_path, f'm{i}.{fmt[i]}'),
    #         colorful_atom=True,
    #         atom_hl_values=atom_sv,
    #         colorbar=True
    #     )
    #     if i+1 == len(fmt):
    #         break

    atom_colors = []
    bond_colors = []
    list_atom_values = []
    for i in range(arr.shape[0]):
        atom_sv = rdworks.assign_fp_values_atoms(ms[i], sv[i], mfps, bits[i], mode='mean')
        colors, bonds, norm, sm = draw.compute_atom_bond_colors(ms[i], atom_sv)
        atom_colors.append(colors)
        bond_colors.append(bonds)
        list_atom_values.append(atom_sv)

    draw.draw_grid(ms, osp.join(save_path, f'm.svg'), list_atom_values=list_atom_values, colorbar=True)
