# 其他辅助插件：完整静态声明

库存基准提交 `2c2ef889abb6adcceba043b606da2e718dbea109`；日期 `2026-11-05`。逐模块 SHA-256 保留如下。

仅按现有 AST 库存展开非私有本地定义、显式导出目标、声明字段、property、常用 magic 方法及可静态解析的本地继承。参数拼写、默认值与注解采用库存中的源码 AST 表达式，未经运行时求值。

这份参考不声称辅助模块已在 Plan20 中重新科学验收。私有模块内的非私有名称仍可能是内部实现；直接模块可访问面与显式 __all__ 分开列出，不扩大批准的公共合同。

property 可写指源码声明 setter；只读 getter 不保证返回对象不可变。类声明值、self 赋值字段和 frozen dataclass 权限分别标示，均不自动授予权威化学事实写权限。

未穷尽第三方继承成员、运行时安装属性、元类生成行为或可选依赖实际可用性。静态解析的基类名称不保证是可导入的 owner，特别是内置类及泛型表达式；原始基类拼写另列。

[辅助入口与类索引](auxiliary_index.md) · [主交付树](../delivery_api_tree.md) · [完整机器库存](README.md#raw-inventory) · [主内核动态权限](dynamic_permissions.md)

## 模块导航

- [`hotpot.plugins.DataGen`](#hotpot-plugins-datagen)
- [`hotpot.plugins.DataGen.curve`](#hotpot-plugins-datagen-curve)
- [`hotpot.plugins.DataGen.samper_synthesizer`](#hotpot-plugins-datagen-samper-synthesizer)
- [`hotpot.plugins.PyG`](#hotpot-plugins-pyg)
- [`hotpot.plugins.PyG.data`](#hotpot-plugins-pyg-data)
- [`hotpot.plugins.PyG.data._base`](#hotpot-plugins-pyg-data--base)
- [`hotpot.plugins.PyG.data.ccdc`](#hotpot-plugins-pyg-data-ccdc)
- [`hotpot.plugins.PyG.data.consts`](#hotpot-plugins-pyg-data-consts)
- [`hotpot.plugins.PyG.data.tmqm`](#hotpot-plugins-pyg-data-tmqm)
- [`hotpot.plugins.PyG.data.utils`](#hotpot-plugins-pyg-data-utils)
- [`hotpot.plugins`](#hotpot-plugins)
- [`hotpot.plugins._harness`](#hotpot-plugins--harness)
- [`hotpot.plugins._harness.contracts`](#hotpot-plugins--harness-contracts)
- [`hotpot.plugins._harness.executable`](#hotpot-plugins--harness-executable)
- [`hotpot.plugins._harness.process`](#hotpot-plugins--harness-process)
- [`hotpot.plugins._harness.provenance`](#hotpot-plugins--harness-provenance)
- [`hotpot.plugins._harness.workspace`](#hotpot-plugins--harness-workspace)
- [`hotpot.plugins.ccdc_api`](#hotpot-plugins-ccdc-api)
- [`hotpot.plugins.ccdc_api.convert`](#hotpot-plugins-ccdc-api-convert)
- [`hotpot.plugins.ccdc_api.miner`](#hotpot-plugins-ccdc-api-miner)
- [`hotpot.plugins.ccdc_api.solvents`](#hotpot-plugins-ccdc-api-solvents)
- [`hotpot.plugins.ccdc_api.statistics`](#hotpot-plugins-ccdc-api-statistics)
- [`hotpot.plugins.dl`](#hotpot-plugins-dl)
- [`hotpot.plugins.dl.function`](#hotpot-plugins-dl-function)
- [`hotpot.plugins.dl.function.base`](#hotpot-plugins-dl-function-base)
- [`hotpot.plugins.dl.function.graph`](#hotpot-plugins-dl-function-graph)
- [`hotpot.plugins.dl.generate`](#hotpot-plugins-dl-generate)
- [`hotpot.plugins.dl.generate.vae`](#hotpot-plugins-dl-generate-vae)
- [`hotpot.plugins.dl.graph`](#hotpot-plugins-dl-graph)
- [`hotpot.plugins.dl.graph.gen_decoder`](#hotpot-plugins-dl-graph-gen-decoder)
- [`hotpot.plugins.dl.pytorch_func`](#hotpot-plugins-dl-pytorch-func)
- [`hotpot.plugins.ml`](#hotpot-plugins-ml)
- [`hotpot.plugins.ml.features`](#hotpot-plugins-ml-features)
- [`hotpot.plugins.ml.features._features`](#hotpot-plugins-ml-features--features)
- [`hotpot.plugins.ml.features.elements`](#hotpot-plugins-ml-features-elements)
- [`hotpot.plugins.ml.features.elements.elements`](#hotpot-plugins-ml-features-elements-elements)
- [`hotpot.plugins.ml.features.sol_media`](#hotpot-plugins-ml-features-sol-media)
- [`hotpot.plugins.ml.features.solvents`](#hotpot-plugins-ml-features-solvents)
- [`hotpot.plugins.ml.utils`](#hotpot-plugins-ml-utils)
- [`hotpot.plugins.ml.wf`](#hotpot-plugins-ml-wf)
- [`hotpot.plugins.opti`](#hotpot-plugins-opti)
- [`hotpot.plugins.opti.beyes`](#hotpot-plugins-opti-beyes)
- [`hotpot.plugins.opti.beyes.opti`](#hotpot-plugins-opti-beyes-opti)
- [`hotpot.plugins.opti.ev`](#hotpot-plugins-opti-ev)
- [`hotpot.plugins.opti.ev.core`](#hotpot-plugins-opti-ev-core)
- [`hotpot.plugins.opti.ev.func`](#hotpot-plugins-opti-ev-func)
- [`hotpot.plugins.opti.exp_design`](#hotpot-plugins-opti-exp-design)
- [`hotpot.plugins.opti.exp_design.core`](#hotpot-plugins-opti-exp-design-core)
- [`hotpot.plugins.opti.params_space`](#hotpot-plugins-opti-params-space)
- [`hotpot.plugins.plots`](#hotpot-plugins-plots)
- [`hotpot.plugins.plots.base`](#hotpot-plugins-plots-base)
- [`hotpot.plugins.plots.colors`](#hotpot-plugins-plots-colors)
- [`hotpot.plugins.plots.defaults`](#hotpot-plugins-plots-defaults)
- [`hotpot.plugins.plots.plots`](#hotpot-plugins-plots-plots)
- [`hotpot.plugins.plots.plots.base`](#hotpot-plugins-plots-plots-base)
- [`hotpot.plugins.plots.plots.beyes`](#hotpot-plugins-plots-plots-beyes)
- [`hotpot.plugins.plots.plots.general`](#hotpot-plugins-plots-plots-general)
- [`hotpot.plugins.plots.plots.ml`](#hotpot-plugins-plots-plots-ml)
- [`hotpot.plugins.plots.plots.perodotable`](#hotpot-plugins-plots-plots-perodotable)
- [`hotpot.plugins.plots.plotter`](#hotpot-plugins-plots-plotter)
- [`hotpot.plugins.plots.utils`](#hotpot-plugins-plots-utils)
- [`hotpot.plugins.pubchem`](#hotpot-plugins-pubchem)
- [`hotpot.plugins.pubchem.get_data`](#hotpot-plugins-pubchem-get-data)
- [`hotpot.plugins.pubchem.utils`](#hotpot-plugins-pubchem-utils)
- [`hotpot.plugins.qm`](#hotpot-plugins-qm)
- [`hotpot.plugins.qm.gaussian`](#hotpot-plugins-qm-gaussian)
- [`hotpot.plugins.qm.gaussian._works`](#hotpot-plugins-qm-gaussian--works)
- [`hotpot.plugins.qm.gaussian.gauss`](#hotpot-plugins-qm-gaussian-gauss)
- [`hotpot.plugins.zeoxx`](#hotpot-plugins-zeoxx)
- [`hotpot.plugins.zeoxx.core`](#hotpot-plugins-zeoxx-core)

<a id="hotpot-plugins-datagen"></a>
## `hotpot.plugins.DataGen`

源码 `hotpot/plugins/DataGen/__init__.py`；SHA-256 `306b2f9d53700087e9f6ffa841188cccdf9bd9d861d05a8abcff55daf6c6aa8a`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-datagen-curve"></a>
## `hotpot.plugins.DataGen.curve`

源码 `hotpot/plugins/DataGen/curve.py`；SHA-256 `a21ba985965d2a9172f54b65aa3a4b5b6e639bc60fba1e13178f2f2bb2db687d`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `curve_interpolate` → `hotpot.plugins.DataGen.curve.curve_interpolate`；`curve_interpolate(X, Y, method=PchipInterpolator, num: int=10000)`
- `nice_extended_max` → `hotpot.plugins.DataGen.curve.nice_extended_max`；`nice_extended_max(x: float) -> float`
- `plot_piecewise_colored_curve` → `hotpot.plugins.DataGen.curve.plot_piecewise_colored_curve`；`plot_piecewise_colored_curve(X, Y, segment_bounds=None, color_cycle=None, show_points=False, xi=None, yi=None, point_kwargs=None, curve_kwargs=None, xlabel='x', ylabel='y', title=None, figsize=(10, 7), axis_label_fontsize=14, axis_label_weight='regular', xtick_fontsize=12, ytick_fontsize=16, save_path: str=None)`
- `origin_colors` → `hotpot.plugins.DataGen.curve.origin_colors`
- `name` → `hotpot.plugins.DataGen.curve.name`
- `xi` → `hotpot.plugins.DataGen.curve.xi`
- `yi` → `hotpot.plugins.DataGen.curve.yi`
- `X` → `hotpot.plugins.DataGen.curve.X`
- `Y` → `hotpot.plugins.DataGen.curve.Y`
- `seg_x` → `hotpot.plugins.DataGen.curve.seg_x`
- `idx_bounds` → `hotpot.plugins.DataGen.curve.idx_bounds`

<a id="hotpot-plugins-datagen-curve-curve-interpolate"></a>
### `curve_interpolate`

定义 owner `hotpot.plugins.DataGen.curve.curve_interpolate`；源码 `hotpot/plugins/DataGen/curve.py:21`。

```python
curve_interpolate(X, Y, method=PchipInterpolator, num: int=10000)
```

<a id="hotpot-plugins-datagen-curve-nice-extended-max"></a>
### `nice_extended_max`

定义 owner `hotpot.plugins.DataGen.curve.nice_extended_max`；源码 `hotpot/plugins/DataGen/curve.py:37`。

```python
nice_extended_max(x: float) -> float
```

<a id="hotpot-plugins-datagen-curve-plot-piecewise-colored-curve"></a>
### `plot_piecewise_colored_curve`

定义 owner `hotpot.plugins.DataGen.curve.plot_piecewise_colored_curve`；源码 `hotpot/plugins/DataGen/curve.py:66`。

```python
plot_piecewise_colored_curve(X, Y, segment_bounds=None, color_cycle=None, show_points=False, xi=None, yi=None, point_kwargs=None, curve_kwargs=None, xlabel='x', ylabel='y', title=None, figsize=(10, 7), axis_label_fontsize=14, axis_label_weight='regular', xtick_fontsize=12, ytick_fontsize=16, save_path: str=None)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `origin_colors` = `['#0000FF', '#228B22', '#B22222', '#FF00FF']`；`hotpot/plugins/DataGen/curve.py:133`
- `name` = `'锆基RN-5@AWUPOZ'`；`hotpot/plugins/DataGen/curve.py:162`
- `xi` = `[0, 10, 15, 20, 30, 50, 80, 120, 125, 130, 135, 140, 250, 375, 500, 1000, 1500, 2000]`；`hotpot/plugins/DataGen/curve.py:163`
- `yi` = `[100, 87.4, 87.2, 87.1, 87.0, 86.95, 86.9, 86.9, 86.8, 87.2, 87.7, 88.4, 90.1, 91.2, 92.1, 93.1, 93.15, 93.17]`；`hotpot/plugins/DataGen/curve.py:164`
- `X` = `curve_interpolate(xi, yi)`；`hotpot/plugins/DataGen/curve.py:176`
- `Y` = `curve_interpolate(xi, yi)`；`hotpot/plugins/DataGen/curve.py:176`
- `seg_x` = `[0, 12.5, 115, 130, 2000]`；`hotpot/plugins/DataGen/curve.py:179`
- `idx_bounds` = `[np.searchsorted(X, xx, side='left') for xx in seg_x]`；`hotpot/plugins/DataGen/curve.py:180`


<a id="hotpot-plugins-datagen-samper-synthesizer"></a>
## `hotpot.plugins.DataGen.samper_synthesizer`

源码 `hotpot/plugins/DataGen/samper_synthesizer.py`；SHA-256 `8640b3c8d6f0d0a9de07756840d447f79bd44d249a6313a0cbad75ef031746f1`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `generate_predictions_with_r2` → `hotpot.plugins.DataGen.samper_synthesizer.generate_predictions_with_r2`；`generate_predictions_with_r2(r2, sample_num: int=1000, low: float=0, high: float=1, mu: float=0, sigma: float=1, df: int=1.0, y_true=None, rand_generator='uniform', random_state=None)`
- `y_pred` → `hotpot.plugins.DataGen.samper_synthesizer.y_pred`
- `y_true` → `hotpot.plugins.DataGen.samper_synthesizer.y_true`
- `xy` → `hotpot.plugins.DataGen.samper_synthesizer.xy`
- `plotter` → `hotpot.plugins.DataGen.samper_synthesizer.plotter`
- `fig` → `hotpot.plugins.DataGen.samper_synthesizer.fig`
- `axs` → `hotpot.plugins.DataGen.samper_synthesizer.axs`

<a id="hotpot-plugins-datagen-samper-synthesizer-generate-predictions-with-r2"></a>
### `generate_predictions_with_r2`

定义 owner `hotpot.plugins.DataGen.samper_synthesizer.generate_predictions_with_r2`；源码 `hotpot/plugins/DataGen/samper_synthesizer.py:33`。

```python
generate_predictions_with_r2(r2, sample_num: int=1000, low: float=0, high: float=1, mu: float=0, sigma: float=1, df: int=1.0, y_true=None, rand_generator='uniform', random_state=None)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `y_pred` = `generate_predictions_with_r2(0.78, mu=20, sigma=9, rand_generator='t-distributed', sample_num=150, df=20)`；`hotpot/plugins/DataGen/samper_synthesizer.py:64`
- `y_true` = `generate_predictions_with_r2(0.78, mu=20, sigma=9, rand_generator='t-distributed', sample_num=150, df=20)`；`hotpot/plugins/DataGen/samper_synthesizer.py:64`
- `xy` = `np.array([y_true, y_pred])`；`hotpot/plugins/DataGen/samper_synthesizer.py:65`
- `plotter` = `SciPlotter(R2Regression(xy, c1='darkgreen', marker1='x', s1=150, target_name='True logK', prediction_name='Predicted logK'))`；`hotpot/plugins/DataGen/samper_synthesizer.py:67`
- `fig` = `plotter()`；`hotpot/plugins/DataGen/samper_synthesizer.py:70`
- `axs` = `plotter()`；`hotpot/plugins/DataGen/samper_synthesizer.py:70`


<a id="hotpot-plugins-pyg"></a>
## `hotpot.plugins.PyG`

源码 `hotpot/plugins/PyG/__init__.py`；SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-pyg-data"></a>
## `hotpot.plugins.PyG.data`

源码 `hotpot/plugins/PyG/data/__init__.py`；SHA-256 `7f2055ac43687ba57d820dbe8ee10560ec9f5e90b0d0f4db25bd56fedcedf92b`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-pyg-data--base"></a>
## `hotpot.plugins.PyG.data._base`

源码 `hotpot/plugins/PyG/data/_base.py`；SHA-256 `87e74db2575e47618b8d7aa7a6772e7ef9b13662055ec3d1edb06994ae05a2b4`。

显式 __all__：`('BaseDataset',)`。

显式导出绑定：

- `BaseDataset` → `hotpot.plugins.PyG.data._base.BaseDataset`；构造声明 `__init__(self, data_dir: str, test_num=None, nproc=None, timeout=None, prefilter: Callable=None, transform: Callable=None)`

<a id="hotpot-plugins-pyg-data--base-basedataset"></a>
### `BaseDataset`

定义 owner `hotpot.plugins.PyG.data._base.BaseDataset`；源码 `hotpot/plugins/PyG/data/_base.py:16`。

基类原始声明：`ABC`；装饰器：无。

显式导出位置：`hotpot.plugins.PyG.data._base.BaseDataset`。

本类属性与方法：

- `__init__(self, data_dir: str, test_num=None, nproc=None, timeout=None, prefilter: Callable=None, transform: Callable=None)`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:19`
- `__getitem__(self, idx: int) -> Data`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:50`
- `__len__(self) -> int`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:52`
- `__iter__(self) -> Iterable[Data]`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:54`
- `get(self, idx: int) -> Data`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:56`
- `to_data(self, mol, *args, **kwargs)`；装饰器 `abstractmethod`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:69`
- `process(self)`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:76`
- `mp_process(self, mol, processes)`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:79`
- `attr_names(self)` [读；只读]；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:101`

本类字段与枚举值：

- `data_dir` = `data_dir` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:28`
- `prefilter` = `prefilter` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:29`
- `transform` = `transform` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:30`
- `test_num` = `test_num` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:31`
- `timeout` = `timeout` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:32`
- `datalist` = `[p for p in glob(osp.join(self.data_dir, '*.pt'))]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:42`
- `len` = `len(self.datalist)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:47`
- `nproc` = `os.cpu_count() // 2` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:35`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`abc.ABC`。


<a id="hotpot-plugins-pyg-data-ccdc"></a>
## `hotpot.plugins.PyG.data.ccdc`

源码 `hotpot/plugins/PyG/data/ccdc.py`；SHA-256 `982589ee1e5c7f398296db5ded1b0d6a6b0f065d5ea817f4a8168cc36c5c9083`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `ComplexDataset` → `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；构造声明 `__init__(self, data_dir: str, raw_mol2_dir: str=None, test_num=None, nproc=None, timeout=None, prefilter: Callable=None, transform: Callable=None)`

<a id="hotpot-plugins-pyg-data-ccdc-complexdataset"></a>
### `ComplexDataset`

定义 owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；源码 `hotpot/plugins/PyG/data/ccdc.py:16`。

基类原始声明：`BaseDataset`；装饰器：无。

本类属性与方法：

- `__init__(self, data_dir: str, raw_mol2_dir: str=None, test_num=None, nproc=None, timeout=None, prefilter: Callable=None, transform: Callable=None)`；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:28`
- `to_data(mol: Molecule, *args, **kwargs) -> Data` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:46`
- `process(self)`；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:81`
- `__iter__(self)`；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:121`
- `__len__(self)`；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:125`
- `__getitem__(self, idx)`；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:128`
- `get(self, idx)`；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:131`

本类字段与枚举值：

- `raw_mol2_dir` = `raw_mol2_dir` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data.ccdc.ComplexDataset`；`hotpot/plugins/PyG/data/ccdc.py:38`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.PyG.data.ccdc.BaseDataset`。


<a id="hotpot-plugins-pyg-data-consts"></a>
## `hotpot.plugins.PyG.data.consts`

源码 `hotpot/plugins/PyG/data/consts.py`；SHA-256 `499854e3d74612fdbd3018a9e3363c4f3fb37bcb002dd8af69d67762fbaaeebc`。

显式 __all__：`('edge_attr_names', 'rings_attr_names', 'pair_attr_names', 'num_atom_pair_attr')`。

显式导出绑定：

- `edge_attr_names` → `hotpot.plugins.PyG.data.consts.edge_attr_names`
- `rings_attr_names` → `hotpot.plugins.PyG.data.consts.rings_attr_names`
- `pair_attr_names` → `hotpot.plugins.PyG.data.consts.pair_attr_names`
- `num_atom_pair_attr` → `hotpot.plugins.PyG.data.consts.num_atom_pair_attr`

模块值/类型别名（声明存在不构成事实写许可）：

- `edge_attr_names` = `('bond_order', 'is_aromatic', 'is_metal_ligand_bond')`；`hotpot/plugins/PyG/data/consts.py:25`
- `rings_attr_names` = `('is_aromatic', 'has_metal')`；`hotpot/plugins/PyG/data/consts.py:26`
- `pair_attr_names` = `tuple(AtomPair.attr_names)`；`hotpot/plugins/PyG/data/consts.py:27`
- `num_atom_pair_attr` = `len(AtomPair.attr_names)`；`hotpot/plugins/PyG/data/consts.py:28`


<a id="hotpot-plugins-pyg-data-tmqm"></a>
## `hotpot.plugins.PyG.data.tmqm`

源码 `hotpot/plugins/PyG/data/tmqm.py`；SHA-256 `f6f4108d296cd79b1f1fad68e8ff9b7b9f947d25edb270ecb8553f63be02a0ad`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `tmQmDataset` → `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；构造声明 `__init__(self, data_dir: str, test_num=None, nproc=None, timeout=None, prefilter: Callable=None, transform: Callable=None)`

<a id="hotpot-plugins-pyg-data-tmqm-tmqmdataset"></a>
### `tmQmDataset`

定义 owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；源码 `hotpot/plugins/PyG/data/tmqm.py:18`。

基类原始声明：`BaseDataset`；装饰器：无。

本类属性与方法：

- `__init__(self, data_dir: str, test_num=None, nproc=None, timeout=None, prefilter: Callable=None, transform: Callable=None)`；owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；`hotpot/plugins/PyG/data/tmqm.py:30`
- `to_data(mol: Molecule, *args, **kwargs) -> Data` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；`hotpot/plugins/PyG/data/tmqm.py:42`
- `process(self)`；owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；`hotpot/plugins/PyG/data/tmqm.py:83`
- `__iter__(self)`；owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；`hotpot/plugins/PyG/data/tmqm.py:134`
- `__len__(self)`；owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；`hotpot/plugins/PyG/data/tmqm.py:138`
- `__getitem__(self, idx)`；owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；`hotpot/plugins/PyG/data/tmqm.py:141`
- `get(self, idx)`；owner `hotpot.plugins.PyG.data.tmqm.tmQmDataset`；`hotpot/plugins/PyG/data/tmqm.py:144`

继承成员（逐项保留声明 owner）：

- `mp_process(self, mol, processes)`；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:79`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `attr_names(self)` [读；只读]；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:101`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `data_dir` = `data_dir` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:28`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `prefilter` = `prefilter` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:29`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `transform` = `transform` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:30`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `test_num` = `test_num` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:31`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `timeout` = `timeout` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:32`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `datalist` = `[p for p in glob(osp.join(self.data_dir, '*.pt'))]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:42`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `len` = `len(self.datalist)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:47`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`
- `nproc` = `os.cpu_count() // 2` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.PyG.data._base.BaseDataset`；`hotpot/plugins/PyG/data/_base.py:35`；继承经 `hotpot.plugins.PyG.data._base.BaseDataset`


<a id="hotpot-plugins-pyg-data-utils"></a>
## `hotpot.plugins.PyG.data.utils`

源码 `hotpot/plugins/PyG/data/utils.py`；SHA-256 `dbd4fa3a63e1f269b2034eee40b02c2af68fafe6beaa47e30ca8ef86f27511e4`。

显式 __all__：`('direct_edge_to_indirect', 'extract_atom_attrs', 'extract_bond_attrs', 'extract_atom_pairs', 'extract_ring_attrs', 'merge_individual_data_to_block', 'make_empty_graph', 'mol_to_pyg_data', 'graph_extraction', 'extract_potentials_cbonds')`。

显式导出绑定：

- `direct_edge_to_indirect` → `hotpot.plugins.PyG.data.utils.direct_edge_to_indirect`；`direct_edge_to_indirect(attr_or_index: torch.Tensor, is_index=True) -> torch.Tensor`
- `extract_atom_attrs` → `hotpot.plugins.PyG.data.utils.extract_atom_attrs`；`extract_atom_attrs(mol: Molecule)`
- `extract_bond_attrs` → `hotpot.plugins.PyG.data.utils.extract_bond_attrs`；`extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str])`
- `extract_atom_pairs` → `hotpot.plugins.PyG.data.utils.extract_atom_pairs`；`extract_atom_pairs(mol: Molecule)`
- `extract_ring_attrs` → `hotpot.plugins.PyG.data.utils.extract_ring_attrs`；`extract_ring_attrs(mol: Molecule, ring_attr_names: Iterable[str])`
- `merge_individual_data_to_block` → `hotpot.plugins.PyG.data.utils.merge_individual_data_to_block`；`merge_individual_data_to_block(indiv_data_dir, merged_data_dir, bundle_size: int=200000)`
- `make_empty_graph` → `hotpot.plugins.PyG.data.utils.make_empty_graph`；`make_empty_graph(prefix: str='')`
- `mol_to_pyg_data` → `hotpot.plugins.PyG.data.utils.mol_to_pyg_data`；`mol_to_pyg_data(mol: Molecule=None, prefix: str='', with_batch: bool=True) -> Data`
- `graph_extraction` → `hotpot.plugins.PyG.data.utils.graph_extraction`；`graph_extraction(mol: Molecule=None, prefix: str='', with_batch: bool=False) -> dict`
- `extract_potentials_cbonds` → `hotpot.plugins.PyG.data.utils.extract_potentials_cbonds`；`extract_potentials_cbonds(mol: Molecule, catoms_symbols: Optional[Iterable[str]]=None)`

<a id="hotpot-plugins-pyg-data-utils-make-empty-graph"></a>
### `make_empty_graph`

定义 owner `hotpot.plugins.PyG.data.utils.make_empty_graph`；源码 `hotpot/plugins/PyG/data/utils.py:32`。

```python
make_empty_graph(prefix: str='')
```

<a id="hotpot-plugins-pyg-data-utils-mol-to-pyg-data"></a>
### `mol_to_pyg_data`

定义 owner `hotpot.plugins.PyG.data.utils.mol_to_pyg_data`；源码 `hotpot/plugins/PyG/data/utils.py:50`。

```python
mol_to_pyg_data(mol: Molecule=None, prefix: str='', with_batch: bool=True) -> Data
```

<a id="hotpot-plugins-pyg-data-utils-graph-extraction"></a>
### `graph_extraction`

定义 owner `hotpot.plugins.PyG.data.utils.graph_extraction`；源码 `hotpot/plugins/PyG/data/utils.py:53`。

```python
graph_extraction(mol: Molecule=None, prefix: str='', with_batch: bool=False) -> dict
```

<a id="hotpot-plugins-pyg-data-utils-direct-edge-to-indirect"></a>
### `direct_edge_to_indirect`

定义 owner `hotpot.plugins.PyG.data.utils.direct_edge_to_indirect`；源码 `hotpot/plugins/PyG/data/utils.py:92`。

```python
direct_edge_to_indirect(attr_or_index: torch.Tensor, is_index=True) -> torch.Tensor
```

<a id="hotpot-plugins-pyg-data-utils-extract-atom-attrs"></a>
### `extract_atom_attrs`

定义 owner `hotpot.plugins.PyG.data.utils.extract_atom_attrs`；源码 `hotpot/plugins/PyG/data/utils.py:100`。

```python
extract_atom_attrs(mol: Molecule)
```

<a id="hotpot-plugins-pyg-data-utils-extract-bond-attrs"></a>
### `extract_bond_attrs`

定义 owner `hotpot.plugins.PyG.data.utils.extract_bond_attrs`；源码 `hotpot/plugins/PyG/data/utils.py:104`。

```python
extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str])
```

<a id="hotpot-plugins-pyg-data-utils-extract-atom-pairs"></a>
### `extract_atom_pairs`

定义 owner `hotpot.plugins.PyG.data.utils.extract_atom_pairs`；源码 `hotpot/plugins/PyG/data/utils.py:117`。

```python
extract_atom_pairs(mol: Molecule)
```

<a id="hotpot-plugins-pyg-data-utils-extract-potentials-cbonds"></a>
### `extract_potentials_cbonds`

定义 owner `hotpot.plugins.PyG.data.utils.extract_potentials_cbonds`；源码 `hotpot/plugins/PyG/data/utils.py:131`。

```python
extract_potentials_cbonds(mol: Molecule, catoms_symbols: Optional[Iterable[str]]=None)
```

<a id="hotpot-plugins-pyg-data-utils-extract-ring-attrs"></a>
### `extract_ring_attrs`

定义 owner `hotpot.plugins.PyG.data.utils.extract_ring_attrs`；源码 `hotpot/plugins/PyG/data/utils.py:158`。

```python
extract_ring_attrs(mol: Molecule, ring_attr_names: Iterable[str])
```

<a id="hotpot-plugins-pyg-data-utils-merge-individual-data-to-block"></a>
### `merge_individual_data_to_block`

定义 owner `hotpot.plugins.PyG.data.utils.merge_individual_data_to_block`；源码 `hotpot/plugins/PyG/data/utils.py:196`。

```python
merge_individual_data_to_block(indiv_data_dir, merged_data_dir, bundle_size: int=200000)
```


<a id="hotpot-plugins"></a>
## `hotpot.plugins`

源码 `hotpot/plugins/__init__.py`；SHA-256 `6e64ab18b4c52aa517b682a38944baca3b965e1186b11c77972aa7fd102f71c9`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins--harness"></a>
## `hotpot.plugins._harness`

源码 `hotpot/plugins/_harness/__init__.py`；SHA-256 `4355a2568e5eab13f05dde510ccd13d45dc996812b857ce8bb4aa72c167b16c6`。

显式 __all__：`('ArtifactProvenance', 'ExecutableNotFoundError', 'NativeProcessResult', 'ProcessProvenance', 'ProcessRequest', 'ProcessTimeoutError', 'build_process_provenance', 'isolated_workspace', 'resolve_executable', 'run_process', 'sha256_file')`。

显式导出绑定：

- `ArtifactProvenance` → `hotpot.plugins._harness.contracts.ArtifactProvenance`；构造声明 `ArtifactProvenance(path: Path, sha256: str)`
- `ExecutableNotFoundError` → `hotpot.plugins._harness.executable.ExecutableNotFoundError`
- `NativeProcessResult` → `hotpot.plugins._harness.contracts.NativeProcessResult`；构造声明 `NativeProcessResult(argv: Tuple[str, ...], cwd: Path, return_code: int, stdout: str, stderr: str, elapsed_seconds: float)`
- `ProcessProvenance` → `hotpot.plugins._harness.contracts.ProcessProvenance`；构造声明 `ProcessProvenance(executable: Path, executable_sha256: str, argv: Tuple[str, ...], cwd: Path, return_code: int, stdout: str, stderr: str, elapsed_seconds: float, artifacts: Tuple[ArtifactProvenance, ...])`
- `ProcessRequest` → `hotpot.plugins._harness.contracts.ProcessRequest`；构造声明 `ProcessRequest(argv: Tuple[str, ...], cwd: Path, env: Mapping[str, str], timeout_seconds: Optional[float] = None)`
- `ProcessTimeoutError` → `hotpot.plugins._harness.process.ProcessTimeoutError`；构造声明 `__init__(self, request: ProcessRequest) -> None`
- `build_process_provenance` → `hotpot.plugins._harness.provenance.build_process_provenance`；`build_process_provenance(*, executable: Path, result: NativeProcessResult, artifacts: Iterable[Path]) -> ProcessProvenance`
- `isolated_workspace` → `hotpot.plugins._harness.workspace.isolated_workspace`；`isolated_workspace(parent: Optional[Path]=None, *, prefix: str='hotpot-', retain: bool=False) -> Iterator[Path]`
- `resolve_executable` → `hotpot.plugins._harness.executable.resolve_executable`；`resolve_executable(name: str, *, explicit: Optional[_ExecutablePath]=None, env_var: Optional[str]=None, env: Optional[Mapping[str, str]]=None) -> Path`
- `run_process` → `hotpot.plugins._harness.process.run_process`；`run_process(request: ProcessRequest) -> NativeProcessResult`
- `sha256_file` → `hotpot.plugins._harness.provenance.sha256_file`；`sha256_file(path: Path) -> str`

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins--harness-contracts"></a>
## `hotpot.plugins._harness.contracts`

源码 `hotpot/plugins/_harness/contracts.py`；SHA-256 `84e46191cefd05a859b2505d1364efd8f40269775360db6eed5b28c721e666e6`。

显式 __all__：`('ArtifactProvenance', 'NativeProcessResult', 'ProcessProvenance', 'ProcessRequest')`。

显式导出绑定：

- `ArtifactProvenance` → `hotpot.plugins._harness.contracts.ArtifactProvenance`；构造声明 `ArtifactProvenance(path: Path, sha256: str)`
- `NativeProcessResult` → `hotpot.plugins._harness.contracts.NativeProcessResult`；构造声明 `NativeProcessResult(argv: Tuple[str, ...], cwd: Path, return_code: int, stdout: str, stderr: str, elapsed_seconds: float)`
- `ProcessProvenance` → `hotpot.plugins._harness.contracts.ProcessProvenance`；构造声明 `ProcessProvenance(executable: Path, executable_sha256: str, argv: Tuple[str, ...], cwd: Path, return_code: int, stdout: str, stderr: str, elapsed_seconds: float, artifacts: Tuple[ArtifactProvenance, ...])`
- `ProcessRequest` → `hotpot.plugins._harness.contracts.ProcessRequest`；构造声明 `ProcessRequest(argv: Tuple[str, ...], cwd: Path, env: Mapping[str, str], timeout_seconds: Optional[float] = None)`

<a id="hotpot-plugins--harness-contracts-processrequest"></a>
### `ProcessRequest`

定义 owner `hotpot.plugins._harness.contracts.ProcessRequest`；源码 `hotpot/plugins/_harness/contracts.py:19`。

基类原始声明：无显式基类；装饰器：`dataclass(frozen=True)`。

显式导出位置：`hotpot.plugins._harness.ProcessRequest`, `hotpot.plugins._harness.contracts.ProcessRequest`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
ProcessRequest(argv: Tuple[str, ...], cwd: Path, env: Mapping[str, str], timeout_seconds: Optional[float] = None)
```

本类字段与枚举值：

- `argv: Tuple[str, ...]` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessRequest`；`hotpot/plugins/_harness/contracts.py:22`
- `cwd: Path` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessRequest`；`hotpot/plugins/_harness/contracts.py:23`
- `env: Mapping[str, str]` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessRequest`；`hotpot/plugins/_harness/contracts.py:24`
- `timeout_seconds: Optional[float]` = `None` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessRequest`；`hotpot/plugins/_harness/contracts.py:25`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins--harness-contracts-nativeprocessresult"></a>
### `NativeProcessResult`

定义 owner `hotpot.plugins._harness.contracts.NativeProcessResult`；源码 `hotpot/plugins/_harness/contracts.py:29`。

基类原始声明：无显式基类；装饰器：`dataclass(frozen=True)`。

显式导出位置：`hotpot.plugins._harness.NativeProcessResult`, `hotpot.plugins._harness.contracts.NativeProcessResult`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
NativeProcessResult(argv: Tuple[str, ...], cwd: Path, return_code: int, stdout: str, stderr: str, elapsed_seconds: float)
```

本类字段与枚举值：

- `argv: Tuple[str, ...]` [只读值字段]；owner `hotpot.plugins._harness.contracts.NativeProcessResult`；`hotpot/plugins/_harness/contracts.py:32`
- `cwd: Path` [只读值字段]；owner `hotpot.plugins._harness.contracts.NativeProcessResult`；`hotpot/plugins/_harness/contracts.py:33`
- `return_code: int` [只读值字段]；owner `hotpot.plugins._harness.contracts.NativeProcessResult`；`hotpot/plugins/_harness/contracts.py:34`
- `stdout: str` [只读值字段]；owner `hotpot.plugins._harness.contracts.NativeProcessResult`；`hotpot/plugins/_harness/contracts.py:35`
- `stderr: str` [只读值字段]；owner `hotpot.plugins._harness.contracts.NativeProcessResult`；`hotpot/plugins/_harness/contracts.py:36`
- `elapsed_seconds: float` [只读值字段]；owner `hotpot.plugins._harness.contracts.NativeProcessResult`；`hotpot/plugins/_harness/contracts.py:37`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins--harness-contracts-artifactprovenance"></a>
### `ArtifactProvenance`

定义 owner `hotpot.plugins._harness.contracts.ArtifactProvenance`；源码 `hotpot/plugins/_harness/contracts.py:41`。

基类原始声明：无显式基类；装饰器：`dataclass(frozen=True)`。

显式导出位置：`hotpot.plugins._harness.ArtifactProvenance`, `hotpot.plugins._harness.contracts.ArtifactProvenance`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
ArtifactProvenance(path: Path, sha256: str)
```

本类字段与枚举值：

- `path: Path` [只读值字段]；owner `hotpot.plugins._harness.contracts.ArtifactProvenance`；`hotpot/plugins/_harness/contracts.py:44`
- `sha256: str` [只读值字段]；owner `hotpot.plugins._harness.contracts.ArtifactProvenance`；`hotpot/plugins/_harness/contracts.py:45`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins--harness-contracts-processprovenance"></a>
### `ProcessProvenance`

定义 owner `hotpot.plugins._harness.contracts.ProcessProvenance`；源码 `hotpot/plugins/_harness/contracts.py:49`。

基类原始声明：无显式基类；装饰器：`dataclass(frozen=True)`。

显式导出位置：`hotpot.plugins._harness.ProcessProvenance`, `hotpot.plugins._harness.contracts.ProcessProvenance`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
ProcessProvenance(executable: Path, executable_sha256: str, argv: Tuple[str, ...], cwd: Path, return_code: int, stdout: str, stderr: str, elapsed_seconds: float, artifacts: Tuple[ArtifactProvenance, ...])
```

本类字段与枚举值：

- `executable: Path` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:52`
- `executable_sha256: str` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:53`
- `argv: Tuple[str, ...]` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:54`
- `cwd: Path` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:55`
- `return_code: int` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:56`
- `stdout: str` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:57`
- `stderr: str` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:58`
- `elapsed_seconds: float` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:59`
- `artifacts: Tuple[ArtifactProvenance, ...]` [只读值字段]；owner `hotpot.plugins._harness.contracts.ProcessProvenance`；`hotpot/plugins/_harness/contracts.py:60`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.


<a id="hotpot-plugins--harness-executable"></a>
## `hotpot.plugins._harness.executable`

源码 `hotpot/plugins/_harness/executable.py`；SHA-256 `8c5493d5ed634382af519ab3f7c0c3c5f424900a77ca599863d9bdcd1207f4be`。

显式 __all__：`('ExecutableNotFoundError', 'resolve_executable')`。

显式导出绑定：

- `ExecutableNotFoundError` → `hotpot.plugins._harness.executable.ExecutableNotFoundError`
- `resolve_executable` → `hotpot.plugins._harness.executable.resolve_executable`；`resolve_executable(name: str, *, explicit: Optional[_ExecutablePath]=None, env_var: Optional[str]=None, env: Optional[Mapping[str, str]]=None) -> Path`

<a id="hotpot-plugins--harness-executable-executablenotfounderror"></a>
### `ExecutableNotFoundError`

定义 owner `hotpot.plugins._harness.executable.ExecutableNotFoundError`；源码 `hotpot/plugins/_harness/executable.py:14`。

基类原始声明：`FileNotFoundError`；装饰器：无。

显式导出位置：`hotpot.plugins._harness.ExecutableNotFoundError`, `hotpot.plugins._harness.executable.ExecutableNotFoundError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins._harness.executable.FileNotFoundError`。

<a id="hotpot-plugins--harness-executable-resolve-executable"></a>
### `resolve_executable`

定义 owner `hotpot.plugins._harness.executable.resolve_executable`；源码 `hotpot/plugins/_harness/executable.py:49`。

```python
resolve_executable(name: str, *, explicit: Optional[_ExecutablePath]=None, env_var: Optional[str]=None, env: Optional[Mapping[str, str]]=None) -> Path
```


<a id="hotpot-plugins--harness-process"></a>
## `hotpot.plugins._harness.process`

源码 `hotpot/plugins/_harness/process.py`；SHA-256 `a31150029f487819d995fb41234ad8c7311acaa944ddb67df7da6230b78c1288`。

显式 __all__：`('ProcessTimeoutError', 'run_process')`。

显式导出绑定：

- `ProcessTimeoutError` → `hotpot.plugins._harness.process.ProcessTimeoutError`；构造声明 `__init__(self, request: ProcessRequest) -> None`
- `run_process` → `hotpot.plugins._harness.process.run_process`；`run_process(request: ProcessRequest) -> NativeProcessResult`

<a id="hotpot-plugins--harness-process-processtimeouterror"></a>
### `ProcessTimeoutError`

定义 owner `hotpot.plugins._harness.process.ProcessTimeoutError`；源码 `hotpot/plugins/_harness/process.py:14`。

基类原始声明：`TimeoutError`；装饰器：无。

显式导出位置：`hotpot.plugins._harness.ProcessTimeoutError`, `hotpot.plugins._harness.process.ProcessTimeoutError`。

本类属性与方法：

- `__init__(self, request: ProcessRequest) -> None`；owner `hotpot.plugins._harness.process.ProcessTimeoutError`；`hotpot/plugins/_harness/process.py:17`

本类字段与枚举值：

- `request` = `request` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins._harness.process.ProcessTimeoutError`；`hotpot/plugins/_harness/process.py:18`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins._harness.process.TimeoutError`。

<a id="hotpot-plugins--harness-process-run-process"></a>
### `run_process`

定义 owner `hotpot.plugins._harness.process.run_process`；源码 `hotpot/plugins/_harness/process.py:25`。

```python
run_process(request: ProcessRequest) -> NativeProcessResult
```


<a id="hotpot-plugins--harness-provenance"></a>
## `hotpot.plugins._harness.provenance`

源码 `hotpot/plugins/_harness/provenance.py`；SHA-256 `863406c63e2e4bbb7a208615f82358a46ec3694b90f8dc7794a6bd0844102525`。

显式 __all__：`('build_process_provenance', 'sha256_file')`。

显式导出绑定：

- `build_process_provenance` → `hotpot.plugins._harness.provenance.build_process_provenance`；`build_process_provenance(*, executable: Path, result: NativeProcessResult, artifacts: Iterable[Path]) -> ProcessProvenance`
- `sha256_file` → `hotpot.plugins._harness.provenance.sha256_file`；`sha256_file(path: Path) -> str`

<a id="hotpot-plugins--harness-provenance-sha256-file"></a>
### `sha256_file`

定义 owner `hotpot.plugins._harness.provenance.sha256_file`；源码 `hotpot/plugins/_harness/provenance.py:15`。

```python
sha256_file(path: Path) -> str
```

<a id="hotpot-plugins--harness-provenance-build-process-provenance"></a>
### `build_process_provenance`

定义 owner `hotpot.plugins._harness.provenance.build_process_provenance`；源码 `hotpot/plugins/_harness/provenance.py:25`。

```python
build_process_provenance(*, executable: Path, result: NativeProcessResult, artifacts: Iterable[Path]) -> ProcessProvenance
```


<a id="hotpot-plugins--harness-workspace"></a>
## `hotpot.plugins._harness.workspace`

源码 `hotpot/plugins/_harness/workspace.py`；SHA-256 `1eb32f43c2198db7aaaedb76bb625086ab46c273fab49120998b62913b25906f`。

显式 __all__：`('isolated_workspace',)`。

显式导出绑定：

- `isolated_workspace` → `hotpot.plugins._harness.workspace.isolated_workspace`；`isolated_workspace(parent: Optional[Path]=None, *, prefix: str='hotpot-', retain: bool=False) -> Iterator[Path]`

<a id="hotpot-plugins--harness-workspace-isolated-workspace"></a>
### `isolated_workspace`

定义 owner `hotpot.plugins._harness.workspace.isolated_workspace`；源码 `hotpot/plugins/_harness/workspace.py:17`。

```python
isolated_workspace(parent: Optional[Path]=None, *, prefix: str='hotpot-', retain: bool=False) -> Iterator[Path]
```

装饰器：`contextmanager`。


<a id="hotpot-plugins-ccdc-api"></a>
## `hotpot.plugins.ccdc_api`

源码 `hotpot/plugins/ccdc_api/__init__.py`；SHA-256 `160ef96a0ede7fe8bad0ba7cacaf7c4763b1bbef8653a0bce0977adfecc2b1e0`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-ccdc-api-convert"></a>
## `hotpot.plugins.ccdc_api.convert`

源码 `hotpot/plugins/ccdc_api/convert.py`；SHA-256 `ddbd1be4dce72b63f77edc5fbff107bc94db4accd63fb303ebed16db46e924f8`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `mol_from_ccdc_to_hp` → `hotpot.plugins.ccdc_api.convert.mol_from_ccdc_to_hp`；`mol_from_ccdc_to_hp(c_mol: CMol)`
- `ccdc_all_bond_type` → `hotpot.plugins.ccdc_api.convert.ccdc_all_bond_type`

<a id="hotpot-plugins-ccdc-api-convert-mol-from-ccdc-to-hp"></a>
### `mol_from_ccdc_to_hp`

定义 owner `hotpot.plugins.ccdc_api.convert.mol_from_ccdc_to_hp`；源码 `hotpot/plugins/ccdc_api/convert.py:29`。

```python
mol_from_ccdc_to_hp(c_mol: CMol)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `ccdc_all_bond_type` = `CBond.BondType.all_bond_types()`；`hotpot/plugins/ccdc_api/convert.py:21`


<a id="hotpot-plugins-ccdc-api-miner"></a>
## `hotpot.plugins.ccdc_api.miner`

源码 `hotpot/plugins/ccdc_api/miner.py`；SHA-256 `5386f0083048393e50589e62465925b87ba364ec6149ddb590aa79594bd8a458`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `EntryReader` → `hotpot.plugins.ccdc_api.miner.EntryReader`；构造声明 `__init__(self, start: int=0)`
- `extract_metal_complexes` → `hotpot.plugins.ccdc_api.miner.extract_metal_complexes`；`extract_metal_complexes(output_dir: str, fmt: str='mol2', nproc: int=None, timeout: float=100)`
- `extract_metal_complexes_` → `hotpot.plugins.ccdc_api.miner.extract_metal_complexes_`；`extract_metal_complexes_(output_dir: str, fmt: str='mol2', nproc: int=None, timeout: float=20)`
- `extract_entry_info` → `hotpot.plugins.ccdc_api.miner.extract_entry_info`；`extract_entry_info()`

<a id="hotpot-plugins-ccdc-api-miner-entryreader"></a>
### `EntryReader`

定义 owner `hotpot.plugins.ccdc_api.miner.EntryReader`；源码 `hotpot/plugins/ccdc_api/miner.py:249`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, start: int=0)`；owner `hotpot.plugins.ccdc_api.miner.EntryReader`；`hotpot/plugins/ccdc_api/miner.py:250`
- `__len__(self)`；owner `hotpot.plugins.ccdc_api.miner.EntryReader`；`hotpot/plugins/ccdc_api/miner.py:255`
- `__next__(self)`；owner `hotpot.plugins.ccdc_api.miner.EntryReader`；`hotpot/plugins/ccdc_api/miner.py:258`
- `__iter__(self)`；owner `hotpot.plugins.ccdc_api.miner.EntryReader`；`hotpot/plugins/ccdc_api/miner.py:267`
- `refresh(self)`；owner `hotpot.plugins.ccdc_api.miner.EntryReader`；`hotpot/plugins/ccdc_api/miner.py:272`

本类字段与枚举值：

- `start` = `start` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.miner.EntryReader`；`hotpot/plugins/ccdc_api/miner.py:252`
- `index` = `start` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.miner.EntryReader`；`hotpot/plugins/ccdc_api/miner.py:253`

<a id="hotpot-plugins-ccdc-api-miner-extract-metal-complexes"></a>
### `extract_metal_complexes`

定义 owner `hotpot.plugins.ccdc_api.miner.extract_metal_complexes`；源码 `hotpot/plugins/ccdc_api/miner.py:276`。

```python
extract_metal_complexes(output_dir: str, fmt: str='mol2', nproc: int=None, timeout: float=100)
```

<a id="hotpot-plugins-ccdc-api-miner-extract-metal-complexes-"></a>
### `extract_metal_complexes_`

定义 owner `hotpot.plugins.ccdc_api.miner.extract_metal_complexes_`；源码 `hotpot/plugins/ccdc_api/miner.py:357`。

```python
extract_metal_complexes_(output_dir: str, fmt: str='mol2', nproc: int=None, timeout: float=20)
```

<a id="hotpot-plugins-ccdc-api-miner-extract-entry-info"></a>
### `extract_entry_info`

定义 owner `hotpot.plugins.ccdc_api.miner.extract_entry_info`；源码 `hotpot/plugins/ccdc_api/miner.py:481`。

```python
extract_entry_info()
```


<a id="hotpot-plugins-ccdc-api-solvents"></a>
## `hotpot.plugins.ccdc_api.solvents`

源码 `hotpot/plugins/ccdc_api/solvents.py`；SHA-256 `598e83752bbbfa2ea56aabb526825d80efa9e15c84a26a5f109a9baa71751376`。

显式 __all__：`('solvents', 'solvents_smiles')`。

显式导出绑定：

- `solvents` → `hotpot.plugins.ccdc_api.solvents.solvents`
- `solvents_smiles` → `hotpot.plugins.ccdc_api.solvents.solvents_smiles`

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-ccdc-api-statistics"></a>
## `hotpot.plugins.ccdc_api.statistics`

源码 `hotpot/plugins/ccdc_api/statistics.py`；SHA-256 `390a4248fc87d03c64f247c215fea81e0ac28ea4e3bc53c2e6b45cc77fa8ae0e`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `get_stem` → `hotpot.plugins.ccdc_api.statistics.get_stem`；`get_stem(filename)`
- `get_metals` → `hotpot.plugins.ccdc_api.statistics.get_metals`；`get_metals(mol)`
- `ComplexStatistics` → `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；构造声明 `__init__(self, ccdc_complex_dir, *skips)`

<a id="hotpot-plugins-ccdc-api-statistics-get-stem"></a>
### `get_stem`

定义 owner `hotpot.plugins.ccdc_api.statistics.get_stem`；源码 `hotpot/plugins/ccdc_api/statistics.py:8`。

```python
get_stem(filename)
```

<a id="hotpot-plugins-ccdc-api-statistics-get-metals"></a>
### `get_metals`

定义 owner `hotpot.plugins.ccdc_api.statistics.get_metals`；源码 `hotpot/plugins/ccdc_api/statistics.py:11`。

```python
get_metals(mol)
```

<a id="hotpot-plugins-ccdc-api-statistics-complexstatistics"></a>
### `ComplexStatistics`

定义 owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；源码 `hotpot/plugins/ccdc_api/statistics.py:14`。

基类原始声明：`object`；装饰器：无。

本类属性与方法：

- `__init__(self, ccdc_complex_dir, *skips)`；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:15`
- `statistic(self)`；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:24`

本类字段与枚举值：

- `ccdc_complex_dir` = `ccdc_complex_dir` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:16`
- `skips` = `skips` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:17`
- `atom_counts` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:18`
- `metal_counts` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:19`
- `metal_types` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:20`
- `mol_weights` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:21`
- `solvents` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ccdc_api.statistics.ComplexStatistics`；`hotpot/plugins/ccdc_api/statistics.py:22`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ccdc_api.statistics.object`。


<a id="hotpot-plugins-dl"></a>
## `hotpot.plugins.dl`

源码 `hotpot/plugins/dl/__init__.py`；SHA-256 `b80f32e9af0f9eb746a4b98e037a84dc4ec5ce1b7431dff86634070c46d8822c`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-dl-function"></a>
## `hotpot.plugins.dl.function`

源码 `hotpot/plugins/dl/function/__init__.py`；SHA-256 `d61e681b74bf093c5b610b5c2637a19686598fc35b242d836dc0b300e04edbb8`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-dl-function-base"></a>
## `hotpot.plugins.dl.function.base`

源码 `hotpot/plugins/dl/function/base.py`；SHA-256 `b554353a9234e4fbe40e7b91d3069c25377088c66b770a9ac60a7c3426ad82a8`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `eigen_abs` → `hotpot.plugins.dl.function.base.eigen_abs`；`eigen_abs(x: torch.Tensor) -> torch.Tensor`
- `is_symmetric` → `hotpot.plugins.dl.function.base.is_symmetric`；`is_symmetric(x, tol: float=5e-07) -> bool`
- `matmul_power` → `hotpot.plugins.dl.function.base.matmul_power`；`matmul_power(x: torch.Tensor, exp: Union[int, float, torch.Tensor], abs_val: bool=False) -> torch.Tensor`
- `kl_div_with_multivariate_normal` → `hotpot.plugins.dl.function.base.kl_div_with_multivariate_normal`；`kl_div_with_multivariate_normal(z: torch.Tensor, eps: float=0.01) -> torch.Tensor`
- `get_positional_encoding` → `hotpot.plugins.dl.function.base.get_positional_encoding`；`get_positional_encoding(max_seq_len: int, embed_dim: int, batch_size: int=None, base: int=None, device: torch.device=None)`

<a id="hotpot-plugins-dl-function-base-eigen-abs"></a>
### `eigen_abs`

定义 owner `hotpot.plugins.dl.function.base.eigen_abs`；源码 `hotpot/plugins/dl/function/base.py:16`。

```python
eigen_abs(x: torch.Tensor) -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-base-is-symmetric"></a>
### `is_symmetric`

定义 owner `hotpot.plugins.dl.function.base.is_symmetric`；源码 `hotpot/plugins/dl/function/base.py:28`。

```python
is_symmetric(x, tol: float=5e-07) -> bool
```

<a id="hotpot-plugins-dl-function-base-matmul-power"></a>
### `matmul_power`

定义 owner `hotpot.plugins.dl.function.base.matmul_power`；源码 `hotpot/plugins/dl/function/base.py:35`。

```python
matmul_power(x: torch.Tensor, exp: Union[int, float, torch.Tensor], abs_val: bool=False) -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-base-kl-div-with-multivariate-normal"></a>
### `kl_div_with_multivariate_normal`

定义 owner `hotpot.plugins.dl.function.base.kl_div_with_multivariate_normal`；源码 `hotpot/plugins/dl/function/base.py:57`。

```python
kl_div_with_multivariate_normal(z: torch.Tensor, eps: float=0.01) -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-base-get-positional-encoding"></a>
### `get_positional_encoding`

定义 owner `hotpot.plugins.dl.function.base.get_positional_encoding`；源码 `hotpot/plugins/dl/function/base.py:67`。

```python
get_positional_encoding(max_seq_len: int, embed_dim: int, batch_size: int=None, base: int=None, device: torch.device=None)
```


<a id="hotpot-plugins-dl-function-graph"></a>
## `hotpot.plugins.dl.function.graph`

源码 `hotpot/plugins/dl/function/graph.py`；SHA-256 `00db4522313fac17873f7e97a5e42ceadee24c99768091cf2599fc44d05e9c5c`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `spectral_similarity` → `hotpot.plugins.dl.function.graph.spectral_similarity`；`spectral_similarity(spec1: Union[torch.Tensor, Sequence[torch.Tensor]], spec2: Union[torch.Tensor, Sequence[torch.Tensor]], metric: Union[Literal['min', 'mean'], Callable]=None)`
- `smiles_to_spectrum` → `hotpot.plugins.dl.function.graph.smiles_to_spectrum`；`smiles_to_spectrum(smiles: str, length: int=4) -> torch.Tensor`
- `edge_index2adj` → `hotpot.plugins.dl.function.graph.edge_index2adj`；`edge_index2adj(note_num: int, edge_index: torch.Tensor) -> torch.Tensor`
- `graph_spectrum` → `hotpot.plugins.dl.function.graph.graph_spectrum`；`graph_spectrum(mat: torch.Tensor, descending: bool=False)`
- `adj2laplacian` → `hotpot.plugins.dl.function.graph.adj2laplacian`；`adj2laplacian(adj: torch.Tensor, norm=True) -> torch.Tensor`
- `graph_data2spectrum` → `hotpot.plugins.dl.function.graph.graph_data2spectrum`；`graph_data2spectrum(data: Data, confs_length: int=4, atomic_numbers_col=0) -> torch.Tensor`
- `smiles2spectrum_data` → `hotpot.plugins.dl.function.graph.smiles2spectrum_data`；`smiles2spectrum_data(smiles: str, length: int=4, atomic_numbers_col: int=0) -> Data`
- `graph_spectrum_similarity` → `hotpot.plugins.dl.function.graph.graph_spectrum_similarity`；`graph_spectrum_similarity(spec1: torch.Tensor, spec2: torch.Tensor, metric: Union[None, Literal['mean', 'mix'], Callable]='min') -> torch.Tensor`
- `batch_spectrum` → `hotpot.plugins.dl.function.graph.batch_spectrum`；`batch_spectrum(batch_matrix: torch.Tensor, batch_atom_num: torch.Tensor)`
- `tensor_sort` → `hotpot.plugins.dl.function.graph.tensor_sort`；`tensor_sort(inputs: torch.Tensor, key: Callable=None, dim: int=None, descending: bool=True)`

<a id="hotpot-plugins-dl-function-graph-spectral-similarity"></a>
### `spectral_similarity`

定义 owner `hotpot.plugins.dl.function.graph.spectral_similarity`；源码 `hotpot/plugins/dl/function/graph.py:55`。

```python
spectral_similarity(spec1: Union[torch.Tensor, Sequence[torch.Tensor]], spec2: Union[torch.Tensor, Sequence[torch.Tensor]], metric: Union[Literal['min', 'mean'], Callable]=None)
```

<a id="hotpot-plugins-dl-function-graph-smiles-to-spectrum"></a>
### `smiles_to_spectrum`

定义 owner `hotpot.plugins.dl.function.graph.smiles_to_spectrum`；源码 `hotpot/plugins/dl/function/graph.py:88`。

```python
smiles_to_spectrum(smiles: str, length: int=4) -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-graph-edge-index2adj"></a>
### `edge_index2adj`

定义 owner `hotpot.plugins.dl.function.graph.edge_index2adj`；源码 `hotpot/plugins/dl/function/graph.py:93`。

```python
edge_index2adj(note_num: int, edge_index: torch.Tensor) -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-graph-graph-spectrum"></a>
### `graph_spectrum`

定义 owner `hotpot.plugins.dl.function.graph.graph_spectrum`；源码 `hotpot/plugins/dl/function/graph.py:106`。

```python
graph_spectrum(mat: torch.Tensor, descending: bool=False)
```

<a id="hotpot-plugins-dl-function-graph-adj2laplacian"></a>
### `adj2laplacian`

定义 owner `hotpot.plugins.dl.function.graph.adj2laplacian`；源码 `hotpot/plugins/dl/function/graph.py:112`。

```python
adj2laplacian(adj: torch.Tensor, norm=True) -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-graph-graph-data2spectrum"></a>
### `graph_data2spectrum`

定义 owner `hotpot.plugins.dl.function.graph.graph_data2spectrum`；源码 `hotpot/plugins/dl/function/graph.py:123`。

```python
graph_data2spectrum(data: Data, confs_length: int=4, atomic_numbers_col=0) -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-graph-smiles2spectrum-data"></a>
### `smiles2spectrum_data`

定义 owner `hotpot.plugins.dl.function.graph.smiles2spectrum_data`；源码 `hotpot/plugins/dl/function/graph.py:135`。

```python
smiles2spectrum_data(smiles: str, length: int=4, atomic_numbers_col: int=0) -> Data
```

<a id="hotpot-plugins-dl-function-graph-graph-spectrum-similarity"></a>
### `graph_spectrum_similarity`

定义 owner `hotpot.plugins.dl.function.graph.graph_spectrum_similarity`；源码 `hotpot/plugins/dl/function/graph.py:147`。

```python
graph_spectrum_similarity(spec1: torch.Tensor, spec2: torch.Tensor, metric: Union[None, Literal['mean', 'mix'], Callable]='min') -> torch.Tensor
```

<a id="hotpot-plugins-dl-function-graph-batch-spectrum"></a>
### `batch_spectrum`

定义 owner `hotpot.plugins.dl.function.graph.batch_spectrum`；源码 `hotpot/plugins/dl/function/graph.py:179`。

```python
batch_spectrum(batch_matrix: torch.Tensor, batch_atom_num: torch.Tensor)
```

<a id="hotpot-plugins-dl-function-graph-tensor-sort"></a>
### `tensor_sort`

定义 owner `hotpot.plugins.dl.function.graph.tensor_sort`；源码 `hotpot/plugins/dl/function/graph.py:193`。

```python
tensor_sort(inputs: torch.Tensor, key: Callable=None, dim: int=None, descending: bool=True)
```


<a id="hotpot-plugins-dl-generate"></a>
## `hotpot.plugins.dl.generate`

源码 `hotpot/plugins/dl/generate/__init__.py`；SHA-256 `e20bc87dc7cc61e3a2b415b93adb5ecb1fb5582a107eb22b25ac32b58195945e`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-dl-generate-vae"></a>
## `hotpot.plugins.dl.generate.vae`

源码 `hotpot/plugins/dl/generate/vae.py`；SHA-256 `df6a91019ab6da25a84e3e761b6a1c517b057b04f34814c16aa086073832b239`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `VAE` → `hotpot.plugins.dl.generate.vae.VAE`；构造声明 `__init__(self, encoder, decoder, latent_dim=1024)`
- `GraphEncoder` → `hotpot.plugins.dl.generate.vae.GraphEncoder`；构造声明 `__init__(self, in_channels: int, hidden_channels: int=16, gnn_layers: int=8, gnn_pool_ratio: Union[int, float, Sequence]=None, latent_size: int=512, num_layers: int=10, n_head: int=4, square_sigma: bool=True, batch_first: bool=False)`
- `NodeNumDecoder` → `hotpot.plugins.dl.generate.vae.NodeNumDecoder`；构造声明 `__init__(self, latent_dim: int)`
- `MatrixDecoder` → `hotpot.plugins.dl.generate.vae.MatrixDecoder`；构造声明 `__init__(self, latent_dim: int, n_heads: int=4, num_layers: int=3)`
- `GraphDecoder` → `hotpot.plugins.dl.generate.vae.GraphDecoder`；构造声明 `__init__(self, latent_size: int, kernel_size: int=16, mean_node_num: int=32, mat_act: Callable=None, batch_first: bool=False)`
- `KernelLinear` → `hotpot.plugins.dl.generate.vae.KernelLinear`；构造声明 `__init__(self, in_features: int, out_features: int, kernel_num: int=1, bias=True) -> None`
- `MultiKernelMLP` → `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；构造声明 `__init__(self, in_feature: int=None, out_feature: int=None, hidden_feature: int=None, filter_number: int=None, layer_number: int=None)`
- `SpectrumLoss` → `hotpot.plugins.dl.generate.vae.SpectrumLoss`；构造声明 `__init__(self)`
- `spectrum_loss` → `hotpot.plugins.dl.generate.vae.spectrum_loss`；`spectrum_loss(pred_spectrum: torch.Tensor, target_spectrum: torch.Tensor, metric: Union[Literal['min', 'mean'], Callable]='min', to_distance: bool=False, eps: float=1e-06) -> torch.Tensor`
- `graph_vae_loss` → `hotpot.plugins.dl.generate.vae.graph_vae_loss`；`graph_vae_loss(true_atom_num: torch.Tensor, pred_atom_num: torch.Tensor, true_spectrum: torch.Tensor, pred_spectrum: torch.Tensor, latent_vector: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor)`

<a id="hotpot-plugins-dl-generate-vae-vae"></a>
### `VAE`

定义 owner `hotpot.plugins.dl.generate.vae.VAE`；源码 `hotpot/plugins/dl/generate/vae.py:21`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, encoder, decoder, latent_dim=1024)`；owner `hotpot.plugins.dl.generate.vae.VAE`；`hotpot/plugins/dl/generate/vae.py:22`
- `forward(self, *args, **kwargs)`；owner `hotpot.plugins.dl.generate.vae.VAE`；`hotpot/plugins/dl/generate/vae.py:28`
- `reparameterize(self, mu, logvar)`；owner `hotpot.plugins.dl.generate.vae.VAE`；`hotpot/plugins/dl/generate/vae.py:37`
- `kld_loss(mu, logvar)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.dl.generate.vae.VAE`；`hotpot/plugins/dl/generate/vae.py:53`

本类字段与枚举值：

- `encoder` = `encoder` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.VAE`；`hotpot/plugins/dl/generate/vae.py:24`
- `decoder` = `decoder` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.VAE`；`hotpot/plugins/dl/generate/vae.py:25`
- `latent_norm` = `nn.BatchNorm1d(latent_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.VAE`；`hotpot/plugins/dl/generate/vae.py:26`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-graphencoder"></a>
### `GraphEncoder`

定义 owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；源码 `hotpot/plugins/dl/generate/vae.py:57`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, in_channels: int, hidden_channels: int=16, gnn_layers: int=8, gnn_pool_ratio: Union[int, float, Sequence]=None, latent_size: int=512, num_layers: int=10, n_head: int=4, square_sigma: bool=True, batch_first: bool=False)`；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:59`
- `pool_graph(self, x, delta_ptr)`；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:135`
- `forward(self, x, edge_index, batch, ptr)`；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:149`

本类字段与枚举值：

- `in_channels` = `in_channels` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:73`
- `hidden_channels` = `hidden_channels` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:74`
- `latent_size` = `latent_size` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:75`
- `num_layers` = `num_layers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:76`
- `n_head` = `n_head` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:77`
- `square_sigma` = `square_sigma` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:78`
- `batch_first` = `batch_first` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:79`
- `gnn_layers` = `gnn_layers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:80`
- `gnn_dim_add` = `(latent_size - hidden_channels) // gnn_layers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:81`
- `encoder0` = `pyg_nn.MLP([in_channels, hidden_channels])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:84`
- `gnn_channels` = `[hidden_channels] + [hidden_channels + i * self.gnn_dim_add for i in range(1, num_layers)] + [latent_size]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:87`
- `gin` = `nn.ModuleList([pyg_nn.GINConv(nn.Linear(in_dim, out_dim), eps=10 - 7) for in_dim, out_dim in zip(self.gnn_channels[:-1], self.gnn_channels[1:])])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:92`
- `pool_ratio` = `1 - 1 / num_layers if not gnn_pool_ratio else gnn_pool_ratio if not isinstance(gnn_pool_ratio, Sequence) else list(gnn_pool_ratio)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:98`
- `SAGPools` = `nn.ModuleList([pyg_nn.SAGPooling(dim, ratio=self.pool_ratio) for dim in self.gnn_channels[1:]])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:107`
- `gnn_norm` = `nn.ModuleList([nn.BatchNorm1d(dim, eps=0.0001) for dim in self.gnn_channels[1:]])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:110`
- `global_pool` = `pyg_nn.SAGPooling(latent_size, ratio=1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:114`
- `batch_norm` = `nn.BatchNorm1d(latent_size, eps=0.0001)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:115`
- `num_query` = `latent_size // hidden_channels + 1` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:125`
- `dim_after_transformer` = `hidden_channels * self.num_query` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:126`
- `to_mu` = `pyg_nn.MLP(2 * [latent_size])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:132`
- `to_logvar` = `pyg_nn.MLP(2 * [latent_size])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphEncoder`；`hotpot/plugins/dl/generate/vae.py:133`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-nodenumdecoder"></a>
### `NodeNumDecoder`

定义 owner `hotpot.plugins.dl.generate.vae.NodeNumDecoder`；源码 `hotpot/plugins/dl/generate/vae.py:224`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, latent_dim: int)`；owner `hotpot.plugins.dl.generate.vae.NodeNumDecoder`；`hotpot/plugins/dl/generate/vae.py:226`
- `forward(self, z: torch.Tensor)`；owner `hotpot.plugins.dl.generate.vae.NodeNumDecoder`；`hotpot/plugins/dl/generate/vae.py:230`

本类字段与枚举值：

- `node_decoder` = `pyg_nn.MLP([latent_dim, latent_dim // 2, latent_dim // 8, latent_dim // 16, 1])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.NodeNumDecoder`；`hotpot/plugins/dl/generate/vae.py:228`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-matrixdecoder"></a>
### `MatrixDecoder`

定义 owner `hotpot.plugins.dl.generate.vae.MatrixDecoder`；源码 `hotpot/plugins/dl/generate/vae.py:234`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, latent_dim: int, n_heads: int=4, num_layers: int=3)`；owner `hotpot.plugins.dl.generate.vae.MatrixDecoder`；`hotpot/plugins/dl/generate/vae.py:236`

本类字段与枚举值：

- `decoder0` = `pyg_nn.MLP([latent_dim, latent_dim])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MatrixDecoder`；`hotpot/plugins/dl/generate/vae.py:239`
- `encoder_start` = `pyg_nn.MLP([latent_dim, latent_dim])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MatrixDecoder`；`hotpot/plugins/dl/generate/vae.py:239`
- `transformer` = `nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=latent_dim, nhead=n_heads), num_layers=num_layers)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MatrixDecoder`；`hotpot/plugins/dl/generate/vae.py:240`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-graphdecoder"></a>
### `GraphDecoder`

定义 owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；源码 `hotpot/plugins/dl/generate/vae.py:246`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, latent_size: int, kernel_size: int=16, mean_node_num: int=32, mat_act: Callable=None, batch_first: bool=False)`；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:248`
- `forward(self, z: torch.Tensor)`；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:281`

本类字段与枚举值：

- `z_size` = `latent_size` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:258`
- `batch_first` = `batch_first` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:259`
- `mean_node_num` = `mean_node_num` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:260`
- `mlp` = `pyg_nn.MLP([latent_size, latent_size, latent_size, latent_size], act='LeakyReLU')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:264`
- `pred_node_num` = `nn.Linear(latent_size, 1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:265`
- `latent_proliferation` = `KernelLinear(latent_size, latent_size, kernel_size)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:267`
- `encoder_start` = `pyg_nn.MLP([latent_size, latent_size])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:269`
- `transformer` = `nn.Transformer(d_model=latent_size, nhead=2, num_encoder_layers=4, num_decoder_layers=2)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:271`
- `one_to_six` = `KernelLinear(latent_size, latent_size, 6)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:278`
- `to_seq_vector` = `pyg_nn.MLP([latent_size] * 3, norm='layer_norm')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.GraphDecoder`；`hotpot/plugins/dl/generate/vae.py:279`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-kernellinear"></a>
### `KernelLinear`

定义 owner `hotpot.plugins.dl.generate.vae.KernelLinear`；源码 `hotpot/plugins/dl/generate/vae.py:314`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, in_features: int, out_features: int, kernel_num: int=1, bias=True) -> None`；owner `hotpot.plugins.dl.generate.vae.KernelLinear`；`hotpot/plugins/dl/generate/vae.py:316`
- `forward(self, x: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.dl.generate.vae.KernelLinear`；`hotpot/plugins/dl/generate/vae.py:325`

本类字段与枚举值：

- `kernel_num` = `kernel_num` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.KernelLinear`；`hotpot/plugins/dl/generate/vae.py:318`
- `weight` = `nn.Parameter(torch.randn(kernel_num, out_features, in_features))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.KernelLinear`；`hotpot/plugins/dl/generate/vae.py:319`
- `bias` = `nn.Parameter(torch.randn(kernel_num, 1, out_features))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.KernelLinear`；`hotpot/plugins/dl/generate/vae.py:321`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-multikernelmlp"></a>
### `MultiKernelMLP`

定义 owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；源码 `hotpot/plugins/dl/generate/vae.py:333`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, in_feature: int=None, out_feature: int=None, hidden_feature: int=None, filter_number: int=None, layer_number: int=None)`；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:335`
- `forward(self, x: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:352`

本类字段与枚举值：

- `in_feature` = `in_feature` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:344`
- `out_feature` = `out_feature` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:345`
- `hidden_feature` = `hidden_feature` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:346`
- `filter_number` = `filter_number` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:347`
- `layer_number` = `layer_number` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:348`
- `kernel_linear` = `KernelLinear(in_feature, hidden_feature, filter_number)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.dl.generate.vae.MultiKernelMLP`；`hotpot/plugins/dl/generate/vae.py:350`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-spectrumloss"></a>
### `SpectrumLoss`

定义 owner `hotpot.plugins.dl.generate.vae.SpectrumLoss`；源码 `hotpot/plugins/dl/generate/vae.py:356`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self)`；owner `hotpot.plugins.dl.generate.vae.SpectrumLoss`；`hotpot/plugins/dl/generate/vae.py:357`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.generate.vae.nn.Module`。

<a id="hotpot-plugins-dl-generate-vae-spectrum-loss"></a>
### `spectrum_loss`

定义 owner `hotpot.plugins.dl.generate.vae.spectrum_loss`；源码 `hotpot/plugins/dl/generate/vae.py:361`。

```python
spectrum_loss(pred_spectrum: torch.Tensor, target_spectrum: torch.Tensor, metric: Union[Literal['min', 'mean'], Callable]='min', to_distance: bool=False, eps: float=1e-06) -> torch.Tensor
```

<a id="hotpot-plugins-dl-generate-vae-graph-vae-loss"></a>
### `graph_vae_loss`

定义 owner `hotpot.plugins.dl.generate.vae.graph_vae_loss`；源码 `hotpot/plugins/dl/generate/vae.py:401`。

```python
graph_vae_loss(true_atom_num: torch.Tensor, pred_atom_num: torch.Tensor, true_spectrum: torch.Tensor, pred_spectrum: torch.Tensor, latent_vector: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor)
```


<a id="hotpot-plugins-dl-graph"></a>
## `hotpot.plugins.dl.graph`

源码 `hotpot/plugins/dl/graph/__init__.py`；SHA-256 `b80f32e9af0f9eb746a4b98e037a84dc4ec5ce1b7431dff86634070c46d8822c`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-dl-graph-gen-decoder"></a>
## `hotpot.plugins.dl.graph.gen_decoder`

源码 `hotpot/plugins/dl/graph/gen_decoder.py`；SHA-256 `892a1f2932ff5309dc9b39d2967cb15633fcdd30a4a6a3ae2fc76d31770cf118`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `NodeEdgeDecoder` → `hotpot.plugins.dl.graph.gen_decoder.NodeEdgeDecoder`；构造声明 `__init__(self)`

<a id="hotpot-plugins-dl-graph-gen-decoder-nodeedgedecoder"></a>
### `NodeEdgeDecoder`

定义 owner `hotpot.plugins.dl.graph.gen_decoder.NodeEdgeDecoder`；源码 `hotpot/plugins/dl/graph/gen_decoder.py:13`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self)`；owner `hotpot.plugins.dl.graph.gen_decoder.NodeEdgeDecoder`；`hotpot/plugins/dl/graph/gen_decoder.py:15`
- `forward(self, z: torch.Tensor)`；owner `hotpot.plugins.dl.graph.gen_decoder.NodeEdgeDecoder`；`hotpot/plugins/dl/graph/gen_decoder.py:18`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.dl.graph.gen_decoder.nn.Module`。


<a id="hotpot-plugins-dl-pytorch-func"></a>
## `hotpot.plugins.dl.pytorch_func`

源码 `hotpot/plugins/dl/pytorch_func.py`；SHA-256 `1a394a3cf6da933bd0b21c8fe89bae1da3d7a73671edf469c15f894c595b1f08`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `TensorContainer` → `hotpot.plugins.dl.pytorch_func.TensorContainer`；构造声明 `__init__(self, *args, **kwargs: torch.Tensor)`
- `train` → `hotpot.plugins.dl.pytorch_func.train`；`train(model, dataloader, loss_fn, opti_cls, learning_rate: float, epoch_num: int, device=torch.device('cpu'), data_organizer: Callable=None, loss_organizer: Callable=None, valid_func: Callable=None, valid_per_batch: int=100)`
- `validate` → `hotpot.plugins.dl.pytorch_func.validate`；`validate(item_num: int, model, batch, valid_func: Callable, device: torch, data_organizer: Callable=None)`
- `show_gpu_memory_utils` → `hotpot.plugins.dl.pytorch_func.show_gpu_memory_utils`；`show_gpu_memory_utils(device=torch.device('cuda'))`
- `get_model_size` → `hotpot.plugins.dl.pytorch_func.get_model_size`；`get_model_size(module)`
- `loss_organizer` → `hotpot.plugins.dl.pytorch_func.loss_organizer`；`loss_organizer(target, out)`
- `to_valid` → `hotpot.plugins.dl.pytorch_func.to_valid`；`to_valid(target, out)`

<a id="hotpot-plugins-dl-pytorch-func-tensorcontainer"></a>
### `TensorContainer`

定义 owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；源码 `hotpot/plugins/dl/pytorch_func.py:44`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, *args, **kwargs: torch.Tensor)`；owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；`hotpot/plugins/dl/pytorch_func.py:46`
- `__dir__(self) -> Iterable[str]`；owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；`hotpot/plugins/dl/pytorch_func.py:57`
- `__getattr__(self, item)`；owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；`hotpot/plugins/dl/pytorch_func.py:60`
- `__setattr__(self, key, value)`；owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；`hotpot/plugins/dl/pytorch_func.py:68`
- `to(self, where)`；owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；`hotpot/plugins/dl/pytorch_func.py:76`
- `args(self)` [读；只读]；owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；`hotpot/plugins/dl/pytorch_func.py:84`
- `kwargs(self)` [读；只读]；owner `hotpot.plugins.dl.pytorch_func.TensorContainer`；`hotpot/plugins/dl/pytorch_func.py:88`

<a id="hotpot-plugins-dl-pytorch-func-train"></a>
### `train`

定义 owner `hotpot.plugins.dl.pytorch_func.train`；源码 `hotpot/plugins/dl/pytorch_func.py:92`。

```python
train(model, dataloader, loss_fn, opti_cls, learning_rate: float, epoch_num: int, device=torch.device('cpu'), data_organizer: Callable=None, loss_organizer: Callable=None, valid_func: Callable=None, valid_per_batch: int=100)
```

<a id="hotpot-plugins-dl-pytorch-func-validate"></a>
### `validate`

定义 owner `hotpot.plugins.dl.pytorch_func.validate`；源码 `hotpot/plugins/dl/pytorch_func.py:146`。

```python
validate(item_num: int, model, batch, valid_func: Callable, device: torch, data_organizer: Callable=None)
```

<a id="hotpot-plugins-dl-pytorch-func-show-gpu-memory-utils"></a>
### `show_gpu_memory_utils`

定义 owner `hotpot.plugins.dl.pytorch_func.show_gpu_memory_utils`；源码 `hotpot/plugins/dl/pytorch_func.py:172`。

```python
show_gpu_memory_utils(device=torch.device('cuda'))
```

<a id="hotpot-plugins-dl-pytorch-func-get-model-size"></a>
### `get_model_size`

定义 owner `hotpot.plugins.dl.pytorch_func.get_model_size`；源码 `hotpot/plugins/dl/pytorch_func.py:182`。

```python
get_model_size(module)
```

<a id="hotpot-plugins-dl-pytorch-func-loss-organizer"></a>
### `loss_organizer`

定义 owner `hotpot.plugins.dl.pytorch_func.loss_organizer`；源码 `hotpot/plugins/dl/pytorch_func.py:191`。

```python
loss_organizer(target, out)
```

<a id="hotpot-plugins-dl-pytorch-func-to-valid"></a>
### `to_valid`

定义 owner `hotpot.plugins.dl.pytorch_func.to_valid`；源码 `hotpot/plugins/dl/pytorch_func.py:209`。

```python
to_valid(target, out)
```


<a id="hotpot-plugins-ml"></a>
## `hotpot.plugins.ml`

源码 `hotpot/plugins/ml/__init__.py`；SHA-256 `4195ab104de60425e6ac8c1549e4d4be0581b212ba5a22f0ef5e22f923b0d55a`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-ml-features"></a>
## `hotpot.plugins.ml.features`

源码 `hotpot/plugins/ml/features/__init__.py`；SHA-256 `35fb9f6c22e3c623358b57bc17b3f406fb099a7d2ee6704edc1d19410a2c9b67`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-ml-features--features"></a>
## `hotpot.plugins.ml.features._features`

源码 `hotpot/plugins/ml/features/_features.py`；SHA-256 `4c936872deb003ecd131654cf17081165ef595c5f860987b258545111676cd89`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `get_rdkit_mol_descriptor` → `hotpot.plugins.ml.features._features.get_rdkit_mol_descriptor`；`get_rdkit_mol_descriptor(list_smi, mol_names=None) -> pd.DataFrame`

<a id="hotpot-plugins-ml-features--features-get-rdkit-mol-descriptor"></a>
### `get_rdkit_mol_descriptor`

定义 owner `hotpot.plugins.ml.features._features.get_rdkit_mol_descriptor`；源码 `hotpot/plugins/ml/features/_features.py:17`。

```python
get_rdkit_mol_descriptor(list_smi, mol_names=None) -> pd.DataFrame
```


<a id="hotpot-plugins-ml-features-elements"></a>
## `hotpot.plugins.ml.features.elements`

源码 `hotpot/plugins/ml/features/elements/__init__.py`；SHA-256 `664be5bf7ac69379ac1590893599554861dab77c18ba0073c2aebf1c06e7cf6c`。

显式 __all__：`('element_features',)`。

显式导出绑定：

- `element_features` → `hotpot.plugins.ml.features.elements.element_features`

<a id="hotpot-plugins-ml-features-elements-elementfeature"></a>
### `ElementFeature`

定义 owner `hotpot.plugins.ml.features.elements.ElementFeature`；源码 `hotpot/plugins/ml/features/elements/__init__.py:15`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, *, data=None)`；owner `hotpot.plugins.ml.features.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/__init__.py:16`
- `__repr__(self)`；owner `hotpot.plugins.ml.features.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/__init__.py:26`
- `__getitem__(self, item)`；owner `hotpot.plugins.ml.features.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/__init__.py:29`
- `dropna(self)`；owner `hotpot.plugins.ml.features.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/__init__.py:32`
- `dataframe(self)` [读；只读]；owner `hotpot.plugins.ml.features.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/__init__.py:36`

<a id="hotpot-plugins-ml-features-elements-determine-elements-block"></a>
### `determine_elements_block`

定义 owner `hotpot.plugins.ml.features.elements.determine_elements_block`；源码 `hotpot/plugins/ml/features/elements/__init__.py:40`。

```python
determine_elements_block(atom_numbers: pd.Series)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `element_features` = `ElementFeature()`；`hotpot/plugins/ml/features/elements/__init__.py:77`
- `data` = `pd.read_pickle(os.path.join(os.path.dirname(__file__), '_elements.pd.xz'))`；`hotpot/plugins/ml/features/elements/__init__.py:82`


<a id="hotpot-plugins-ml-features-elements-elements"></a>
## `hotpot.plugins.ml.features.elements.elements`

源码 `hotpot/plugins/ml/features/elements/elements.py`；SHA-256 `6ba5439a5f36b84ddfbc1df4c53cb451d845678fdfdee1daad6ad5b768fb7863`。

显式 __all__：`('element_features',)`。

显式导出绑定：

- `element_features` → `hotpot.plugins.ml.features.elements.elements.element_features`

<a id="hotpot-plugins-ml-features-elements-elements-elementfeature"></a>
### `ElementFeature`

定义 owner `hotpot.plugins.ml.features.elements.elements.ElementFeature`；源码 `hotpot/plugins/ml/features/elements/elements.py:15`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, *, data=None)`；owner `hotpot.plugins.ml.features.elements.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/elements.py:16`
- `__repr__(self)`；owner `hotpot.plugins.ml.features.elements.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/elements.py:26`
- `__getitem__(self, item)`；owner `hotpot.plugins.ml.features.elements.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/elements.py:29`
- `dropna(self)`；owner `hotpot.plugins.ml.features.elements.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/elements.py:32`
- `dataframe(self)` [读；只读]；owner `hotpot.plugins.ml.features.elements.elements.ElementFeature`；`hotpot/plugins/ml/features/elements/elements.py:36`

<a id="hotpot-plugins-ml-features-elements-elements-determine-elements-block"></a>
### `determine_elements_block`

定义 owner `hotpot.plugins.ml.features.elements.elements.determine_elements_block`；源码 `hotpot/plugins/ml/features/elements/elements.py:40`。

```python
determine_elements_block(atom_numbers: pd.Series)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `element_features` = `ElementFeature()`；`hotpot/plugins/ml/features/elements/elements.py:77`
- `data` = `pd.read_pickle(os.path.join(os.path.dirname(__file__), '_elements.pd.xz'))`；`hotpot/plugins/ml/features/elements/elements.py:82`


<a id="hotpot-plugins-ml-features-sol-media"></a>
## `hotpot.plugins.ml.features.sol_media`

源码 `hotpot/plugins/ml/features/sol_media/__init__.py`；SHA-256 `9996a184c73097fe613f7e001e86f1bc03e3a51219755d77ab15511916533413`。

显式 __all__：`('get_all_media_names', 'to_media_features')`。

显式导出绑定：

- `get_all_media_names` → `hotpot.plugins.ml.features.sol_media.get_all_media_names`；`get_all_media_names() -> list[str]`
- `to_media_features` → `hotpot.plugins.ml.features.sol_media.to_media_features`；`to_media_features(lst_media, fill_miss: Union[list[str], bool]=True) -> pd.DataFrame`

<a id="hotpot-plugins-ml-features-sol-media-get-all-media-names"></a>
### `get_all_media_names`

定义 owner `hotpot.plugins.ml.features.sol_media.get_all_media_names`；源码 `hotpot/plugins/ml/features/sol_media/__init__.py:18`。

```python
get_all_media_names() -> list[str]
```

<a id="hotpot-plugins-ml-features-sol-media-to-media-features"></a>
### `to_media_features`

定义 owner `hotpot.plugins.ml.features.sol_media.to_media_features`；源码 `hotpot/plugins/ml/features/sol_media/__init__.py:22`。

```python
to_media_features(lst_media, fill_miss: Union[list[str], bool]=True) -> pd.DataFrame
```


<a id="hotpot-plugins-ml-features-solvents"></a>
## `hotpot.plugins.ml.features.solvents`

源码 `hotpot/plugins/ml/features/solvents/__init__.py`；SHA-256 `3edebd04d14b0a03edcf746234f5882a352f92fa0ab2b4f1513fe34cdb3ceece`。

显式 __all__：`('get_all_solvents_names', 'to_solvent_features')`。

显式导出绑定：

- `get_all_solvents_names` → `hotpot.plugins.ml.features.solvents.get_all_solvents_names`；`get_all_solvents_names() -> list[str]`
- `to_solvent_features` → `hotpot.plugins.ml.features.solvents.to_solvent_features`；`to_solvent_features(lst_sols: list, fill_miss: Union[list, bool]=False) -> pd.DataFrame`

<a id="hotpot-plugins-ml-features-solvents-get-all-solvents-names"></a>
### `get_all_solvents_names`

定义 owner `hotpot.plugins.ml.features.solvents.get_all_solvents_names`；源码 `hotpot/plugins/ml/features/solvents/__init__.py:17`。

```python
get_all_solvents_names() -> list[str]
```

<a id="hotpot-plugins-ml-features-solvents-to-solvent-features"></a>
### `to_solvent_features`

定义 owner `hotpot.plugins.ml.features.solvents.to_solvent_features`；源码 `hotpot/plugins/ml/features/solvents/__init__.py:21`。

```python
to_solvent_features(lst_sols: list, fill_miss: Union[list, bool]=False) -> pd.DataFrame
```


<a id="hotpot-plugins-ml-utils"></a>
## `hotpot.plugins.ml.utils`

源码 `hotpot/plugins/ml/utils.py`；SHA-256 `db22b8e5f2338998923766973cdfb40e8566980c645dbb213b5295d52feaba00`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `scatter_discriminant_analysis` → `hotpot.plugins.ml.utils.scatter_discriminant_analysis`；`scatter_discriminant_analysis(X: np.ndarray, y: np.ndarray, split_points: Union[np.ndarray, Sequence]=None)`
- `plot_scatter_discriminant_lines` → `hotpot.plugins.ml.utils.plot_scatter_discriminant_lines`；`plot_scatter_discriminant_lines(X, y, lines, dim_method=TSNE(n_components=2))`

<a id="hotpot-plugins-ml-utils-scatter-discriminant-analysis"></a>
### `scatter_discriminant_analysis`

定义 owner `hotpot.plugins.ml.utils.scatter_discriminant_analysis`；源码 `hotpot/plugins/ml/utils.py:18`。

```python
scatter_discriminant_analysis(X: np.ndarray, y: np.ndarray, split_points: Union[np.ndarray, Sequence]=None)
```

<a id="hotpot-plugins-ml-utils-plot-scatter-discriminant-lines"></a>
### `plot_scatter_discriminant_lines`

定义 owner `hotpot.plugins.ml.utils.plot_scatter_discriminant_lines`；源码 `hotpot/plugins/ml/utils.py:54`。

```python
plot_scatter_discriminant_lines(X, y, lines, dim_method=TSNE(n_components=2))
```


<a id="hotpot-plugins-ml-wf"></a>
## `hotpot.plugins.ml.wf`

源码 `hotpot/plugins/ml/wf.py`；SHA-256 `e3b0ebf94418b2ef4ba60c746013db7ba9bf1e2432c98bf1d7b68161a951cf33`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `expectation_improvement` → `hotpot.plugins.ml.wf.expectation_improvement`；`expectation_improvement(y, mu, sigma)`
- `get_rdkit_mol_descriptor` → `hotpot.plugins.ml.wf.get_rdkit_mol_descriptor`；`get_rdkit_mol_descriptor(list_smi) -> pd.DataFrame`
- `LinearAddLightGBM` → `hotpot.plugins.ml.wf.LinearAddLightGBM`；构造声明 `__init__(self)`
- `CrossValidation` → `hotpot.plugins.ml.wf.CrossValidation`；构造声明 `__init__(self, X, y, cv=None)`
- `MachineLearning` → `hotpot.plugins.ml.wf.MachineLearning`；构造声明 `__init__(self, work_dir: Union[Path, str], data: pd.DataFrame, features: Union[Sequence, np.ndarray], target, estimator: BaseEstimator=GradientBoostingRegressor(), **kwargs)`
- `MachineLearning_` → `hotpot.plugins.ml.wf.MachineLearning_`；构造声明 `__init__(self, work_dir: Union[Path, str], data: pd.DataFrame, features: Sequence[str], target, estimator: BaseEstimator=GradientBoostingRegressor(), **kwargs)`
- `linear_leave_one_out_analysis` → `hotpot.plugins.ml.wf.linear_leave_one_out_analysis`；`linear_leave_one_out_analysis(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray)`
- `import_msg` → `hotpot.plugins.ml.wf.import_msg`
- `cross_valid` → `hotpot.plugins.ml.wf.cross_valid`
- `dataset` → `hotpot.plugins.ml.wf.dataset`
- `X` → `hotpot.plugins.ml.wf.X`
- `y` → `hotpot.plugins.ml.wf.y`

<a id="hotpot-plugins-ml-wf-expectation-improvement"></a>
### `expectation_improvement`

定义 owner `hotpot.plugins.ml.wf.expectation_improvement`；源码 `hotpot/plugins/ml/wf.py:55`。

```python
expectation_improvement(y, mu, sigma)
```

<a id="hotpot-plugins-ml-wf-get-rdkit-mol-descriptor"></a>
### `get_rdkit_mol_descriptor`

定义 owner `hotpot.plugins.ml.wf.get_rdkit_mol_descriptor`；源码 `hotpot/plugins/ml/wf.py:66`。

```python
get_rdkit_mol_descriptor(list_smi) -> pd.DataFrame
```

<a id="hotpot-plugins-ml-wf-linearaddlightgbm"></a>
### `LinearAddLightGBM`

定义 owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；源码 `hotpot/plugins/ml/wf.py:127`。

基类原始声明：`BaseEstimator`；装饰器：无。

本类属性与方法：

- `__init__(self)`；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:128`
- `__call__(self, *args, **kwargs)`；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:138`
- `fit(self, X, y=None, linear_feature_index: Union[Sequence[int], np.ndarray]=None, lightgbm_feature_index: Union[Sequence[int], np.ndarray]=None)`；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:141`
- `predict(self, X)`；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:162`
- `score(self, X, y=None)`；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:165`
- `linear_coeff_(self)` [读；只读]；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:170`

本类字段与枚举值：

- `linear` = `sklearn.linear_model.LinearRegression()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:130`
- `lightgbm` = `GradientBoostingRegressor()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:132`
- `linear_feature_index` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:133`
- `lightgbm_feature_index` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:134`
- `delta_y` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.LinearAddLightGBM`；`hotpot/plugins/ml/wf.py:136`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`sklearn.base.BaseEstimator`。

<a id="hotpot-plugins-ml-wf-crossvalidation"></a>
### `CrossValidation`

定义 owner `hotpot.plugins.ml.wf.CrossValidation`；源码 `hotpot/plugins/ml/wf.py:174`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, X, y, cv=None)`；owner `hotpot.plugins.ml.wf.CrossValidation`；`hotpot/plugins/ml/wf.py:175`
- `__call__(self, *args, **kwargs)`；owner `hotpot.plugins.ml.wf.CrossValidation`；`hotpot/plugins/ml/wf.py:181`

本类字段与枚举值：

- `cv` = `cv` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.CrossValidation`；`hotpot/plugins/ml/wf.py:177`
- `train_indices` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.CrossValidation`；`hotpot/plugins/ml/wf.py:178`
- `valid_indices` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.CrossValidation`；`hotpot/plugins/ml/wf.py:179`

<a id="hotpot-plugins-ml-wf-machinelearning"></a>
### `MachineLearning`

定义 owner `hotpot.plugins.ml.wf.MachineLearning`；源码 `hotpot/plugins/ml/wf.py:185`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, work_dir: Union[Path, str], data: pd.DataFrame, features: Union[Sequence, np.ndarray], target, estimator: BaseEstimator=GradientBoostingRegressor(), **kwargs)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:189`
- `work(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:299`
- `preprocess(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:323`
- `train_test_split(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:358`
- `feature_engineering(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:415`
- `quickly_feature_selection(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:438`
- `calc_pearson_coef(X)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:456`
- `calc_pearson_matrix(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:460`
- `make_pearson_matrix_plot(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:465`
- `feat_hiera_clustering(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:471`
- `make_hierarchical_tree(clustering, threshold)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:489`
- `pca_dimension_reduction(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:524`
- `recursive_feature_addition(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:590`
- `recursive_feature_elimination(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:638`
- `recursive_feature_selection(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:657`
- `permutate_importance(estimator, *args, **kwargs)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:687`
- `gini_importance(estimator, X, y)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:692`
- `cross_validation(self, feature_type: Literal['essential', 'reduced', 'scaled']='essential', sample_type: Literal['train', 'test', 'all']='train')`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:748`
- `train_model(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:769`
- `determine_importance(self, feature_type: Literal['essential', 'reduced', 'scaled']='essential', sample_type: Literal['train', 'test', 'all']='all')`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:793`
- `generate_test_X(self, template_X: np.ndarray=None, independent: bool=False, norm_uniform: Literal['norm', 'uniform']='uniform', min_offset: Union[float, np.ndarray]=0.0, max_offset: Union[float, np.ndarray]=0.0, sample_num: int=1000, seed_: int=None)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:824`
- `calc_shap_values(self, estimator, X, y, feature_names: Union[Sequence, np.ndarray]=None, sample_size: int=1000, X_test: np.ndarray=None, test_size: int=1000, explainer_cls: shap.Explainer=shap.TreeExplainer, shap_values_save_path: Union[str, os.PathLike]=None, **kwargs)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:893`
- `make_shap_bar_beeswarm(shap_values: shap.Explanation, max_display: int=15, savefig_path: Union[str, os.PathLike]=None, **kwargs)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:953`
- `shap_analysis(self)`；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:970`
- `scaled_X(self) -> np.ndarray` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:336`
- `scaled_y(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:340`
- `sXtr(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:372`
- `sYtr(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:376`
- `sXte(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:380`
- `sYte(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:384`
- `ntri_features(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:388`
- `ntri_X(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:392`
- `ntri_sX(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:396`
- `ntri_Xtr(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:400`
- `ntri_Xte(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:404`
- `ntri_sXtr(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:408`
- `ntri_sXte(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:412`
- `clustered_feat_names(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:517`
- `reduced_features(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:543`
- `reduced_sX(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:550`
- `reduced_sXtr(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:567`
- `reduced_sXte(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:572`
- `essential_features(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:805`
- `essential_sX(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:811`
- `essential_sXtr(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:817`
- `essential_sXte(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:821`

本类字段与枚举值：

- `work_dir` = `Path(work_dir)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:222`
- `picture_dir` = `self.work_dir.joinpath('pictures')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:223`
- `sheet_dir` = `self.work_dir.joinpath('sheet')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:224`
- `data` = `data` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:226`
- `features` = `list(features) if isinstance(features, Sequence) else features.tolist()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:227`
- `target` = `target` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:228`
- `estimator` = `estimator` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:229`
- `sample_num` = `len(self.data)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:231`
- `X` = `data.loc[:, features]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:232`
- `y` = `data[target]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:233`
- `other_info` = `data.drop(target, axis=1).drop(features, axis=1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:234`
- `xscaler` = `kwargs.get('xscaler', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:236`
- `yscaler` = `kwargs.get('yscaler', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:237`
- `scaled_data` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:238`
- `test_indices` = `kwargs.get('test_indices', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:241`
- `train_indices` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:242`
- `test_size` = `kwargs.get('test_size', 0.2)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:243`
- `data_shuffle` = `kwargs.get('data_shuffle', False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:244`
- `data_stratify` = `kwargs.get('data_stratify', False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:245`
- `stratify_log` = `kwargs.get('stratify_log', False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:246`
- `feat_imp_measurer` = `kwargs.get('feat_imp_measurer', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:249`
- `feat_imp_threshold` = `kwargs.get('feat_imp_threshold', 0.001)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:250`
- `feat_cluster_threshold` = `kwargs.get('feat_cluster_threshold', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:251`
- `recur_addi_threshold` = `kwargs.get('recur_addi_threshold', 0.0)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:252`
- `recur_addi_prev_threshold` = `kwargs.get('recur_addi_prev_threshold')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:258`
- `trivial_feat_indices` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:261`
- `ntri_feat_indices` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:262`
- `pearson_mat` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:263`
- `ntri_feat_cluster` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:264`
- `clustered_feat_indices` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:265`
- `clustered_map` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:266`
- `clt_X` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:267`
- `essential_feat_indices` = `self._get_essential_features(kwargs.get('essential_features'))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:269`
- `cv_best_metric` = `(None,)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:270`
- `how_to_essential` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:271`
- `feature_importance` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:273`
- `optimal_hyperparams` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:274`
- `valid_score` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:275`
- `valid_pred` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:276`
- `valid_true` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:277`
- `explainer` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:280`
- `shap_value` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning`；`hotpot/plugins/ml/wf.py:281`

<a id="hotpot-plugins-ml-wf-machinelearning-"></a>
### `MachineLearning_`

定义 owner `hotpot.plugins.ml.wf.MachineLearning_`；源码 `hotpot/plugins/ml/wf.py:989`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, work_dir: Union[Path, str], data: pd.DataFrame, features: Sequence[str], target, estimator: BaseEstimator=GradientBoostingRegressor(), **kwargs)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:993`
- `feature_names_to_indices_(feature_list: list, get_names: list)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1142`
- `work(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1154`
- `preprocess_(X, y, xscaler=None, yscaler=None) -> (np.ndarray, np.ndarray)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1184`
- `preprocess(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1197`
- `invert_scaling_X(self, X, features: list[str]=None, recovery_cluster_to_original: bool=False)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1206`
- `get_clustering_feat_and_X_by_clustering_name(self, clustering_name: str, which_stage: str='non_trivial')`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1321`
- `invert_pca_x_(inverted_x, X_clt, x_pca, show_cv_results=True)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1340`
- `invert_clustering_x(self, X_clt, clustering_names: Union[str, list[str]])`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1350`
- `train_test_split(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1368`
- `feature_engineering(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1382`
- `quickly_feature_selection_(X, y, feat_imp_measurer, feat_imp_threshold=0.0001)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1417`
- `quickly_feature_selection(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1434`
- `calc_pearson_matrix_(X)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1453`
- `calc_pearson_matrix(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1456`
- `make_pearson_matrix_plot(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1461`
- `make_hierarchical_tree(clustering, threshold)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1468`
- `feat_hiera_clustering_(X, features, y=None, cluster_threshold=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1496`
- `feat_hiera_clustering(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1516`
- `pca_dimension_reduction_(X: np.ndarray, features: list[str], clustering_map: dict=None, clustering_threshold: float=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1536`
- `pca_dimension_reduction(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1586`
- `recursive_feature_addition_(X_train: np.ndarray, y_train: np.ndarray, feature_measurer, prev_addi_threshold: float=0.2, recur_addi_threshold: float=0.0)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1623`
- `recursive_feature_addition(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1669`
- `recursive_feature_elimination_(X_train: np.ndarray, y_train: np.ndarray, feat_measurer)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1680`
- `recursive_feature_elimination(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1692`
- `recursive_feature_selection_(X, features, feat_indices_eli, score_eli, feat_indices_add, score_add)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1706`
- `recursive_feature_selection(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1736`
- `has_fitted(estimator: BaseEstimator)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1758`
- `permutate_importance(estimator, X, y, features, *args, **kwargs)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1766`
- `gini_importance(estimator, X, y, features: list[str])` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1770`
- `determine_importance(self, feature_type: Literal['essential', 'reduced', 'scaled']='essential', sample_type: Literal['train', 'test', 'all']='all')`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1786`
- `export_feature_importance(self, to_determine: bool=False)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1797`
- `cross_validation(self, feature_type: Literal['essential', 'reduced', 'scaled']='essential', sample_type: Literal['train', 'test', 'all']='train')`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1871`
- `train_model(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1893`
- `partial_dependence_(estimator: ModelLike, features: Sequence[str], X: np.ndarray, feature_indices: Sequence[int], target_name: str='Target', figsave_path: Union[str, os.PathLike]=None, **kwargs)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1967`
- `partial_dependence(self, estimator: ModelLike=None, feature_types: Literal['original', 'non_trivial', 'reduced', 'essential']=None, recover_to_original_scale: bool=False, analyze_in_original_data: bool=False, **kwargs)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2012`
- `train_surrogate_tree_(teacher_model: ModelLike, X: np.ndarray, feature_names, surrogate_model: ModelLike=DecisionTreeRegressor(), validate_surrogate: bool=True, generate_train_data: bool=False, generate_valid_data: bool=True, generator_kwargs: dict=None, tree_plot_kwargs: dict=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2084`
- `train_surrogate_tree(self, teacher_model: ModelLike=None, X: np.ndarray=None, feature_names: Sequence[str]=None, surrogate_model: ModelLike=DecisionTreeRegressor(), validate_surrogate: bool=True, generate_train_data: bool=False, generate_valid_data: bool=True, generator_kwargs: dict=None, tree_plot_kwargs: dict=None)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2149`
- `generate_test_X_(template_X, independent: bool=False, norm_uniform: Literal['norm', 'uniform']='uniform', min_offset: Union[float, np.ndarray]=0.0, max_offset: Union[float, np.ndarray]=0.0, X_scale: Union[float, np.ndarray]=None, sample_num: int=1000, seed_: int=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2193`
- `generate_test_X(self, template_X: np.ndarray=None, independent: bool=False, norm_uniform: Literal['norm', 'uniform']='uniform', min_offset: Union[float, np.ndarray]=0.0, max_offset: Union[float, np.ndarray]=0.0, sample_num: int=1000, seed_: int=None)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2265`
- `shap_analysis_(estimator, X, y, feature_names: Union[Sequence, np.ndarray]=None, gen_X_train: bool=False, sample_size: int=1000, X_test: np.ndarray=None, test_size: int=1000, explainer_cls: type=shap.TreeExplainer, shap_values_save_path: Union[str, os.PathLike]=None, **kwargs)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2305`
- `make_shap_bar_beeswarm(shap_values: shap.Explanation, max_display: int=15, savefig_path: Union[str, os.PathLike]=None, **kwargs)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2375`
- `shap_analysis(self)`；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:2392`
- `X(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1091`
- `y(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1095`
- `X_train(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1099`
- `y_train(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1103`
- `X_test(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1107`
- `y_test(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1111`
- `features(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1115`
- `feature_type(self)` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1119`
- `pearson_mat(self) -> np.ndarray` [读；只读]；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1449`

本类字段与枚举值：

- `work_dir` = `Path(work_dir)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1031`
- `picture_dir` = `self.work_dir.joinpath('pictures')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1032`
- `sheet_dir` = `self.work_dir.joinpath('sheet')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1033`
- `data` = `data` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1035`
- `target` = `target` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1036`
- `estimator` = `estimator` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1037`
- `sample_num` = `len(self.data)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1039`
- `other_info` = `data.drop(target, axis=1).drop(features, axis=1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1040`
- `xscaler` = `kwargs.get('xscaler', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1042`
- `yscaler` = `kwargs.get('yscaler', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1043`
- `test_indices` = `kwargs.get('test_indices', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1046`
- `train_indices` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1047`
- `test_size` = `kwargs.get('test_size', 0.2)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1048`
- `data_shuffle` = `kwargs.get('data_shuffle', True)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1049`
- `data_stratify` = `kwargs.get('data_stratify', False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1050`
- `stratify_log` = `kwargs.get('stratify_log', False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1051`
- `feat_imp_measurer` = `kwargs.get('feat_imp_measurer', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1054`
- `feat_imp_cutoff` = `kwargs.get('feat_imp_cutoff', 0.001)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1055`
- `feat_cluster_threshold` = `kwargs.get('feat_cluster_threshold', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1056`
- `recur_addi_threshold` = `kwargs.get('recur_addi_threshold', 0.0)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1057`
- `recur_addi_prev_threshold` = `kwargs.get('recur_addi_prev_threshold')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1063`
- `pearson_mat_dict` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1066`
- `clustered_maps` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1067`
- `clustering` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1068`
- `pca_features` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1069`
- `pca_X` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1070`
- `cv_best_metric` = `(None,)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1071`
- `how_to_essential` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1072`
- `feature_importance` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1074`
- `valid_score` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1075`
- `valid_pred` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1076`
- `valid_true` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1077`
- `explainer` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1080`
- `shap_value` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1081`
- `kwargs` = `kwargs` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ml.wf.MachineLearning_`；`hotpot/plugins/ml/wf.py:1088`

<a id="hotpot-plugins-ml-wf-linear-leave-one-out-analysis"></a>
### `linear_leave_one_out_analysis`

定义 owner `hotpot.plugins.ml.wf.linear_leave_one_out_analysis`；源码 `hotpot/plugins/ml/wf.py:2411`。

```python
linear_leave_one_out_analysis(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `import_msg` = `'xgboost or lightgbm not have not been installed, pip install xgboost or lightgbm!'`；`hotpot/plugins/ml/wf.py:52`
- `cross_valid` = `_cross_val`；`hotpot/plugins/ml/wf.py:2427`
- `dataset` = `fetch_california_housing()`；`hotpot/plugins/ml/wf.py:2430`
- `X` = `(dataset.data, dataset.target)`；`hotpot/plugins/ml/wf.py:2431`
- `y` = `(dataset.data, dataset.target)`；`hotpot/plugins/ml/wf.py:2431`


<a id="hotpot-plugins-opti"></a>
## `hotpot.plugins.opti`

源码 `hotpot/plugins/opti/__init__.py`；SHA-256 `5e905d1de8cde77b90fcc25bc6a3ccea48ba46d2d0ae7a18b23e5c30bb403214`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-opti-beyes"></a>
## `hotpot.plugins.opti.beyes`

源码 `hotpot/plugins/opti/beyes/__init__.py`；SHA-256 `8017f6f6b1e9c3a0b9df0976cea1a514e013440090e570828cbf60398b4ca079`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-opti-beyes-opti"></a>
## `hotpot.plugins.opti.beyes.opti`

源码 `hotpot/plugins/opti/beyes/opti.py`；SHA-256 `be01381a4f19a9eef82e03ed5633ef8094497cad35f24ebc31c0b19a27be1a36`。

显式 __all__：`('next_params', 'draw_comics_map')`。

显式导出绑定：

- `next_params` → `hotpot.plugins.opti.beyes.opti.next_params`；`next_params(X: ArrayLike, y: ArrayLike, param_range: np.ndarray, param_names: List[str], next_param_path: Union[str, os.PathLike], mesh_counts: int=DEFAULT_MESH_COUNTS, figpath: Optional[Union[str, os.PathLike]]=None, log_indices: Optional[Union[int, List[int]]]=None, to_coutourf: bool=True, cmap: str='Greys') -> None`
- `draw_comics_map` → `hotpot.plugins.opti.beyes.opti.draw_comics_map`；`draw_comics_map(X: ArrayLike, y: ArrayLike, init_index: int, batch_size: int, param_range: Optional[np.ndarray]=None, param_names: Optional[Sequence[str]]=None, mesh_counts: int=DEFAULT_MESH_COUNTS, log_indices: Optional[Union[int, List[int]]]=None, figpath_dir: Optional[Union[str, os.PathLike]]=None, emb_method: Union[TSNE, MDS]=TSNE(), to_coutourf: bool=True, cmap: str='Greys') -> None`

<a id="hotpot-plugins-opti-beyes-opti-generate-power-ladder"></a>
### `generate_power_ladder`

定义 owner `hotpot.plugins.opti.beyes.opti.generate_power_ladder`；源码 `hotpot/plugins/opti/beyes/opti.py:41`。

```python
generate_power_ladder(index: int, batch_size: int=DEFAULT_BATCH_SIZE, max_epsilon: float=DEFAULT_ACQ_EPS_MAX, power: float=DEFAULT_ACQ_POWER) -> float
```

<a id="hotpot-plugins-opti-beyes-opti-acquisitionfunction"></a>
### `AcquisitionFunction`

定义 owner `hotpot.plugins.opti.beyes.opti.AcquisitionFunction`；源码 `hotpot/plugins/opti/beyes/opti.py:53`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `expected_improvement(mean: torch.Tensor, sigma: torch.Tensor, best_observed: torch.Tensor, epsilon: float=0.02) -> torch.Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.opti.beyes.opti.AcquisitionFunction`；`hotpot/plugins/opti/beyes/opti.py:55`

<a id="hotpot-plugins-opti-beyes-opti-gaussianprocess"></a>
### `GaussianProcess`

定义 owner `hotpot.plugins.opti.beyes.opti.GaussianProcess`；源码 `hotpot/plugins/opti/beyes/opti.py:74`。

基类原始声明：`gpytorch.models.ExactGP`；装饰器：无。

本类属性与方法：

- `__init__(self, train_x: torch.Tensor, train_y: torch.Tensor, mean_module: Optional[gpytorch.means.Mean]=None, covar_module: Optional[gpytorch.kernels.Kernel]=None, likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood]=None) -> None`；owner `hotpot.plugins.opti.beyes.opti.GaussianProcess`；`hotpot/plugins/opti/beyes/opti.py:75`
- `forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal`；owner `hotpot.plugins.opti.beyes.opti.GaussianProcess`；`hotpot/plugins/opti/beyes/opti.py:96`

本类字段与枚举值：

- `mean_module` = `mean_module` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.GaussianProcess`；`hotpot/plugins/opti/beyes/opti.py:93`
- `covar_module` = `covar_module` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.GaussianProcess`；`hotpot/plugins/opti/beyes/opti.py:94`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.opti.beyes.opti.gpytorch.models.ExactGP`。

<a id="hotpot-plugins-opti-beyes-opti-bayesianoptimizer"></a>
### `BayesianOptimizer`

定义 owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；源码 `hotpot/plugins/opti/beyes/opti.py:102`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, surrogate: gpytorch.models.ExactGP, acquisition_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]=AcquisitionFunction.expected_improvement, batch_size: int=1) -> None`；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:105`
- `__call__(self, design_points: ArrayLike, n_iter: int=DEFAULT_BO_ITER, lr: float=DEFAULT_GP_LR) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:117`
- `train_gp(self, n_iter: int=DEFAULT_BO_ITER, lr: float=DEFAULT_GP_LR, report_gap: Optional[int]=None) -> None`；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:178`
- `predict(self, inputs: torch.Tensor, which: Literal['first', 'last']='last') -> Tuple[torch.Tensor, torch.Tensor]`；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:210`
- `generate_2d_embedding_design_space(self, design_points: ArrayLike, optimal_indices: Optional[ArrayLike]=None, n_iter: int=DEFAULT_BO_ITER, lr: float=DEFAULT_GP_LR, original_x: Optional[np.ndarray]=None, original_y: Optional[np.ndarray]=None, embedding_method: Union[TSNE, MDS, None]=TSNE(), figpath: Optional[Union[str, os.PathLike]]=None, embedded_x: Optional[np.ndarray]=None, show_fig: bool=False, y_scaler: Optional[StandardScaler]=None, to_contourf: bool=True, cmap: str='Greys') -> Tuple[np.ndarray, np.ndarray, np.ndarray]`；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:227`

本类字段与枚举值：

- `surrogate` = `surrogate` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:111`
- `acquisition_fn` = `acquisition_fn` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:112`
- `batch_size` = `batch_size` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:113`
- `is_trained` = `False` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:114`
- `surrogate_snapshots: List[gpytorch.models.ExactGP]` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.BayesianOptimizer`；`hotpot/plugins/opti/beyes/opti.py:115`

<a id="hotpot-plugins-opti-beyes-opti-parampreprocessor"></a>
### `ParamPreprocessor`

定义 owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；源码 `hotpot/plugins/opti/beyes/opti.py:300`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, scaler: Callable[..., MinMaxScaler]=MinMaxScaler(), y_scaler: Optional[StandardScaler]=StandardScaler(), param_range: Optional[Union[torch.Tensor, np.ndarray]]=None, param_names: Optional[List[str]]=None, param_mesh_counts: int=DEFAULT_MESH_COUNTS, logX_indices: Optional[Union[int, List[int]]]=None) -> None`；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:310`
- `get_design_space(self) -> torch.Tensor`；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:339`
- `fit(self, X: torch.Tensor, y: Optional[torch.Tensor]=None) -> None`；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:358`
- `log10_transform(self, X: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:363`
- `scale_features(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]`；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:375`
- `inverse_log10(self, X_log: ArrayLike) -> ArrayLike`；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:386`
- `inverse_scale(self, values: ArrayLike, mean: ArrayLike, sigma: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:397`

本类字段与枚举值：

- `scaler` = `scaler` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:319`
- `y_scaler` = `y_scaler` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:320`
- `param_range` = `param_range` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:321`
- `param_names` = `param_names` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:322`
- `param_mesh_counts` = `param_mesh_counts` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:323`
- `logX_indices: Optional[List[int]]` = `[logX_indices]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.beyes.opti.ParamPreprocessor`；`hotpot/plugins/opti/beyes/opti.py:335`

<a id="hotpot-plugins-opti-beyes-opti-preprocess-inputs"></a>
### `preprocess_inputs`

定义 owner `hotpot.plugins.opti.beyes.opti.preprocess_inputs`；源码 `hotpot/plugins/opti/beyes/opti.py:423`。

```python
preprocess_inputs(X: ArrayLike, y: ArrayLike, param_range: np.ndarray, param_names: List[str], log_indices: Optional[Union[int, List[int]]], mesh_counts: int) -> Tuple[ParamPreprocessor, np.ndarray, np.ndarray, torch.Tensor]
```

<a id="hotpot-plugins-opti-beyes-opti-run-bayesian-optimization"></a>
### `run_bayesian_optimization`

定义 owner `hotpot.plugins.opti.beyes.opti.run_bayesian_optimization`；源码 `hotpot/plugins/opti/beyes/opti.py:450`。

```python
run_bayesian_optimization(X: ArrayLike, y: ArrayLike, design_points: ArrayLike, batch_size: int=DEFAULT_BATCH_SIZE) -> Tuple[BayesianOptimizer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

<a id="hotpot-plugins-opti-beyes-opti-next-params"></a>
### `next_params`

定义 owner `hotpot.plugins.opti.beyes.opti.next_params`；源码 `hotpot/plugins/opti/beyes/opti.py:477`。

```python
next_params(X: ArrayLike, y: ArrayLike, param_range: np.ndarray, param_names: List[str], next_param_path: Union[str, os.PathLike], mesh_counts: int=DEFAULT_MESH_COUNTS, figpath: Optional[Union[str, os.PathLike]]=None, log_indices: Optional[Union[int, List[int]]]=None, to_coutourf: bool=True, cmap: str='Greys') -> None
```

<a id="hotpot-plugins-opti-beyes-opti-draw-comics-map"></a>
### `draw_comics_map`

定义 owner `hotpot.plugins.opti.beyes.opti.draw_comics_map`；源码 `hotpot/plugins/opti/beyes/opti.py:571`。

```python
draw_comics_map(X: ArrayLike, y: ArrayLike, init_index: int, batch_size: int, param_range: Optional[np.ndarray]=None, param_names: Optional[Sequence[str]]=None, mesh_counts: int=DEFAULT_MESH_COUNTS, log_indices: Optional[Union[int, List[int]]]=None, figpath_dir: Optional[Union[str, os.PathLike]]=None, emb_method: Union[TSNE, MDS]=TSNE(), to_coutourf: bool=True, cmap: str='Greys') -> None
```

模块值/类型别名（声明存在不构成事实写许可）：

- `ArrayLike` = `Union[np.ndarray, torch.Tensor]`；`hotpot/plugins/opti/beyes/opti.py:29`
- `DEFAULT_BATCH_SIZE` = `5`；`hotpot/plugins/opti/beyes/opti.py:31`
- `DEFAULT_ACQ_EPS_MAX` = `3.0`；`hotpot/plugins/opti/beyes/opti.py:32`
- `DEFAULT_ACQ_POWER` = `2.5`；`hotpot/plugins/opti/beyes/opti.py:33`
- `DEFAULT_EI_EPS` = `0.01`；`hotpot/plugins/opti/beyes/opti.py:34`
- `MIN_SIGMA` = `1e-09`；`hotpot/plugins/opti/beyes/opti.py:35`
- `DEFAULT_MESH_COUNTS` = `20`；`hotpot/plugins/opti/beyes/opti.py:36`
- `DEFAULT_BO_ITER` = `150`；`hotpot/plugins/opti/beyes/opti.py:37`
- `DEFAULT_GP_LR` = `0.1`；`hotpot/plugins/opti/beyes/opti.py:38`


<a id="hotpot-plugins-opti-ev"></a>
## `hotpot.plugins.opti.ev`

源码 `hotpot/plugins/opti/ev/__init__.py`；SHA-256 `df9d1b3a1992c7d748d04e90d83c736104a74fb0447a9b7a6a743c4ebe1c9f59`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-opti-ev-core"></a>
## `hotpot.plugins.opti.ev.core`

源码 `hotpot/plugins/opti/ev/core.py`；SHA-256 `0a5431c92d42f8ace51e146e33b0ca56b6f7dce36c4232c5187d713650a5c528`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `get_params` → `hotpot.plugins.opti.ev.core.get_params`；`get_params(param_name: str, *objs)`
- `UniversalParams` → `hotpot.plugins.opti.ev.core.UniversalParams`
- `GeneLibrary` → `hotpot.plugins.opti.ev.core.GeneLibrary`
- `Individual` → `hotpot.plugins.opti.ev.core.Individual`；构造声明 `__init__(self, species: 'Species', gene_series: 'GeneSeries', fitness_function: Callable=None, mutation_rate: float=0.5)`
- `Gene` → `hotpot.plugins.opti.ev.core.Gene`；构造声明 `__init__(self, gene_space: 'GeneSpace', mutation_rate: float=None, value=None)`
- `GeneSeries` → `hotpot.plugins.opti.ev.core.GeneSeries`；构造声明 `__init__(self, *gene)`
- `GeneSpace` → `hotpot.plugins.opti.ev.core.GeneSpace`；构造声明 `__init__(self, name: str, species: str=None)`
- `DiscreteGenSpace` → `hotpot.plugins.opti.ev.core.DiscreteGenSpace`；构造声明 `__init__(self, name: str, *gen_vals: Sequence)`
- `ContinuousGenSpace` → `hotpot.plugins.opti.ev.core.ContinuousGenSpace`；构造声明 `__init__(self, name: str, min_val: int, max_val: int)`
- `formalize_params_space` → `hotpot.plugins.opti.ev.core.formalize_params_space`；`formalize_params_space(name, space: Sequence)`
- `Species` → `hotpot.plugins.opti.ev.core.Species`；构造声明 `__init__(self, name: str, *gene_spaces: 'GeneSpace', mutation_rate: float=None, lifetime: Union[int, Callable]=None, fitness_function: Callable=None)`
- `Earth` → `hotpot.plugins.opti.ev.core.Earth`；构造声明 `__init__(self, survival_func: Callable=None, living_ratio: float=0.6, fitness_threshold: float=0.8, max_living_space: int=25, fitness_function: Callable=None, *, keep_ratio: float=None, mutation_rate: float=None, **kwargs)`
- `EvoluteOptimizer` → `hotpot.plugins.opti.ev.core.EvoluteOptimizer`；构造声明 `__init__(self, gene_series, pop_size)`
- `ml_optimizer` → `hotpot.plugins.opti.ev.core.ml_optimizer`；`ml_optimizer(models: dict[str, type(ModelLike)], hypers: dict[str, Sequence], X: np.ndarray, y: np.ndarray, X_test: np.ndarray=None, y_test: np.ndarray=None, features: list[str]=None, fitness_function: Callable=None, initial_size: Union[int, list[int]]=3, *, feature_engineering: bool=False, fe_kwargs: dict[str, Any]=None, which_return: Union[Literal['best', 'top'], int]='best', **kwargs)`
- `MachineLearningOptimizer` → `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；构造声明 `__init__(self, models: dict[str, type(ModelLike)], hypers: dict[str, dict], X: np.ndarray, y: np.ndarray, X_test: np.ndarray=None, y_test: np.ndarray=None, features: list[str]=None, *, fitness_function: Callable=None, initial_size: Union[int, list[int]]=3, feature_engineering: bool=False, fe_kwargs: dict[str, Any]=None, **kwargs)`
- `MLOptimizer` → `hotpot.plugins.opti.ev.core.MLOptimizer`；构造声明 `__init__(self, model: ModelLike, X_train: np.ndarray, y_train: np.ndarray, params_space: dict[str, Union[list, tuple]], cv: Union[int, Callable]=5, shuffle: bool=True, *, mutation_rate: float=0.2, init_params: Union[int, list[dict]]=3, living_space: int=10, survival_func: Callable=None, survival_rate: float=0.4, survival_threshold: float=0.8)`
- `gene_library` → `hotpot.plugins.opti.ev.core.gene_library`

<a id="hotpot-plugins-opti-ev-core-get-params"></a>
### `get_params`

定义 owner `hotpot.plugins.opti.ev.core.get_params`；源码 `hotpot/plugins/opti/ev/core.py:33`。

```python
get_params(param_name: str, *objs)
```

<a id="hotpot-plugins-opti-ev-core-universalparams"></a>
### `UniversalParams`

定义 owner `hotpot.plugins.opti.ev.core.UniversalParams`；源码 `hotpot/plugins/opti/ev/core.py:41`。

基类原始声明：无显式基类；装饰器：无。

本类字段与枚举值：

- `mutation_rate: float` = `0.1` [字段声明；写许可依具体合同]；owner `hotpot.plugins.opti.ev.core.UniversalParams`；`hotpot/plugins/opti/ev/core.py:43`
- `num_children: int` = `2` [字段声明；写许可依具体合同]；owner `hotpot.plugins.opti.ev.core.UniversalParams`；`hotpot/plugins/opti/ev/core.py:44`
- `fitness_function: Callable` = `None` [字段声明；写许可依具体合同]；owner `hotpot.plugins.opti.ev.core.UniversalParams`；`hotpot/plugins/opti/ev/core.py:45`

<a id="hotpot-plugins-opti-ev-core-genelibrary"></a>
### `GeneLibrary`

定义 owner `hotpot.plugins.opti.ev.core.GeneLibrary`；源码 `hotpot/plugins/opti/ev/core.py:48`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__new__(cls, *args, **kwargs)`；owner `hotpot.plugins.opti.ev.core.GeneLibrary`；`hotpot/plugins/opti/ev/core.py:54`
- `append_gene_space(self, gene_space)`；owner `hotpot.plugins.opti.ev.core.GeneLibrary`；`hotpot/plugins/opti/ev/core.py:70`
- `clear(self)`；owner `hotpot.plugins.opti.ev.core.GeneLibrary`；`hotpot/plugins/opti/ev/core.py:73`
- `library(self)` [读；只读]；owner `hotpot.plugins.opti.ev.core.GeneLibrary`；`hotpot/plugins/opti/ev/core.py:63`
- `library_dict(self)` [读；只读]；owner `hotpot.plugins.opti.ev.core.GeneLibrary`；`hotpot/plugins/opti/ev/core.py:67`

<a id="hotpot-plugins-opti-ev-core-individual"></a>
### `Individual`

定义 owner `hotpot.plugins.opti.ev.core.Individual`；源码 `hotpot/plugins/opti/ev/core.py:81`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, species: 'Species', gene_series: 'GeneSeries', fitness_function: Callable=None, mutation_rate: float=0.5)`；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:162`
- `__repr__(self)`；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:186`
- `gene_mutate(self)`；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:209`
- `mate(self, other: 'Individual', gene_mutate: bool=True)`；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:216`
- `get_fitness_function(self) -> Callable`；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:246`
- `calc_fitness(self, timeout=None)`；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:249`
- `die(self)`；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:262`
- `earth(self)` [读；只读]；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:190`
- `expected_lifetime(self)` [读；只读]；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:194`
- `gene_dict(self) -> dict` [读；只读]；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:199`
- `species_name(self) -> str` [读；只读]；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:203`

本类字段与枚举值：

- `species` = `species` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:169`
- `gene_series` = `gene_series` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:170`
- `fitness_` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:171`
- `fitness_function` = `fitness_function` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:172`
- `fitness_process: Union[None, Individual.FitnessProcess]` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:173`
- `age` = `0` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:174`
- `results` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:184`
- `mutation_rate` = `mutation_rate` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Individual`；`hotpot/plugins/opti/ev/core.py:178`

<a id="hotpot-plugins-opti-ev-core-gene"></a>
### `Gene`

定义 owner `hotpot.plugins.opti.ev.core.Gene`；源码 `hotpot/plugins/opti/ev/core.py:267`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, gene_space: 'GeneSpace', mutation_rate: float=None, value=None)`；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:271`
- `__gt__(self, other)`；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:287`
- `__repr__(self)`；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:290`
- `mutate(self)`；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:301`
- `gene_mutation_rate(self) -> float`；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:305`
- `name(self) -> str` [读；只读]；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:294`
- `species(self) -> str` [读；只读]；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:298`

本类字段与枚举值：

- `gene_space` = `gene_space` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:277`
- `value` = `value` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:278`
- `mutation_rate` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Gene`；`hotpot/plugins/opti/ev/core.py:280`

<a id="hotpot-plugins-opti-ev-core-geneseries"></a>
### `GeneSeries`

定义 owner `hotpot.plugins.opti.ev.core.GeneSeries`；源码 `hotpot/plugins/opti/ev/core.py:309`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, *gene)`；owner `hotpot.plugins.opti.ev.core.GeneSeries`；`hotpot/plugins/opti/ev/core.py:311`
- `__repr__(self)`；owner `hotpot.plugins.opti.ev.core.GeneSeries`；`hotpot/plugins/opti/ev/core.py:314`
- `__iter__(self)`；owner `hotpot.plugins.opti.ev.core.GeneSeries`；`hotpot/plugins/opti/ev/core.py:317`
- `__eq__(self, other)`；owner `hotpot.plugins.opti.ev.core.GeneSeries`；`hotpot/plugins/opti/ev/core.py:320`
- `__len__(self)`；owner `hotpot.plugins.opti.ev.core.GeneSeries`；`hotpot/plugins/opti/ev/core.py:323`
- `series(self) -> list['Gene']` [读；只读]；owner `hotpot.plugins.opti.ev.core.GeneSeries`；`hotpot/plugins/opti/ev/core.py:327`
- `series_dict(self) -> dict` [读；只读]；owner `hotpot.plugins.opti.ev.core.GeneSeries`；`hotpot/plugins/opti/ev/core.py:331`

<a id="hotpot-plugins-opti-ev-core-genespace"></a>
### `GeneSpace`

定义 owner `hotpot.plugins.opti.ev.core.GeneSpace`；源码 `hotpot/plugins/opti/ev/core.py:335`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__new__(cls, *args, **kwargs)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:337`
- `__init__(self, name: str, species: str=None)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:343`
- `__contains__(self, item)`；装饰器 `abstractmethod`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:348`
- `__repr__(self)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:351`
- `generate_gene(self)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:354`
- `get_random_gene(self)`；装饰器 `abstractmethod`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:361`

本类字段与枚举值：

- `name` = `name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:344`
- `species` = `species` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:345`

<a id="hotpot-plugins-opti-ev-core-discretegenspace"></a>
### `DiscreteGenSpace`

定义 owner `hotpot.plugins.opti.ev.core.DiscreteGenSpace`；源码 `hotpot/plugins/opti/ev/core.py:365`。

基类原始声明：`GeneSpace`；装饰器：无。

本类属性与方法：

- `__init__(self, name: str, *gen_vals: Sequence)`；owner `hotpot.plugins.opti.ev.core.DiscreteGenSpace`；`hotpot/plugins/opti/ev/core.py:367`
- `__contains__(self, item)`；owner `hotpot.plugins.opti.ev.core.DiscreteGenSpace`；`hotpot/plugins/opti/ev/core.py:371`
- `get_random_gene(self)`；owner `hotpot.plugins.opti.ev.core.DiscreteGenSpace`；`hotpot/plugins/opti/ev/core.py:374`

本类字段与枚举值：

- `gen_vals` = `list(gen_vals)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.DiscreteGenSpace`；`hotpot/plugins/opti/ev/core.py:369`

继承成员（逐项保留声明 owner）：

- `__new__(cls, *args, **kwargs)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:337`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `__repr__(self)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:351`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `generate_gene(self)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:354`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `name` = `name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:344`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `species` = `species` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:345`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`

<a id="hotpot-plugins-opti-ev-core-continuousgenspace"></a>
### `ContinuousGenSpace`

定义 owner `hotpot.plugins.opti.ev.core.ContinuousGenSpace`；源码 `hotpot/plugins/opti/ev/core.py:378`。

基类原始声明：`GeneSpace`；装饰器：无。

本类属性与方法：

- `__init__(self, name: str, min_val: int, max_val: int)`；owner `hotpot.plugins.opti.ev.core.ContinuousGenSpace`；`hotpot/plugins/opti/ev/core.py:380`
- `__contains__(self, item)`；owner `hotpot.plugins.opti.ev.core.ContinuousGenSpace`；`hotpot/plugins/opti/ev/core.py:385`
- `get_random_gene(self)`；owner `hotpot.plugins.opti.ev.core.ContinuousGenSpace`；`hotpot/plugins/opti/ev/core.py:388`

本类字段与枚举值：

- `min_val` = `min_val` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.ContinuousGenSpace`；`hotpot/plugins/opti/ev/core.py:382`
- `max_val` = `max_val` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.ContinuousGenSpace`；`hotpot/plugins/opti/ev/core.py:383`

继承成员（逐项保留声明 owner）：

- `__new__(cls, *args, **kwargs)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:337`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `__repr__(self)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:351`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `generate_gene(self)`；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:354`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `name` = `name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:344`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`
- `species` = `species` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.GeneSpace`；`hotpot/plugins/opti/ev/core.py:345`；继承经 `hotpot.plugins.opti.ev.core.GeneSpace`

<a id="hotpot-plugins-opti-ev-core-formalize-params-space"></a>
### `formalize_params_space`

定义 owner `hotpot.plugins.opti.ev.core.formalize_params_space`；源码 `hotpot/plugins/opti/ev/core.py:392`。

```python
formalize_params_space(name, space: Sequence)
```

<a id="hotpot-plugins-opti-ev-core-species"></a>
### `Species`

定义 owner `hotpot.plugins.opti.ev.core.Species`；源码 `hotpot/plugins/opti/ev/core.py:449`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, name: str, *gene_spaces: 'GeneSpace', mutation_rate: float=None, lifetime: Union[int, Callable]=None, fitness_function: Callable=None)`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:452`
- `__repr__(self)`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:479`
- `create_individual(self, **kwargs) -> 'Individual'`；装饰器 `_AllSpeciesMeta.register`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:491`
- `get_gene_space(self, name: str) -> 'GeneSpace'`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:512`
- `create_individuals(self, nums: int=2)`；装饰器 `_AllSpeciesMeta.register`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:520`
- `get_gene_series(self) -> 'GeneSeries'`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:523`
- `is_fitted_gene(self, gene: 'Gene') -> bool`；装饰器 `overload`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:527`
- `is_fitted_gene(self, name: str, value) -> bool`；装饰器 `overload`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:531`
- `is_fitted_gene(self, *args, **kwargs) -> bool`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:534`
- `add_individual(self, individual: 'Individual')`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:546`
- `reproduce(self)`；装饰器 `_AllSpeciesMeta.register`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:563`
- `clear_population(self)`；装饰器 `_AllSpeciesMeta.register`；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:577`
- `expected_lifetime(self) -> int` [读；只读]；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:483`

本类字段与枚举值：

- `name` = `name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:459`
- `gene_spaces` = `gene_spaces` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:460`
- `fitness_function` = `fitness_function` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:462`
- `individuals` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:464`
- `mutation_rate` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:466`
- `earth` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Species`；`hotpot/plugins/opti/ev/core.py:477`

<a id="hotpot-plugins-opti-ev-core-earth"></a>
### `Earth`

定义 owner `hotpot.plugins.opti.ev.core.Earth`；源码 `hotpot/plugins/opti/ev/core.py:609`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, survival_func: Callable=None, living_ratio: float=0.6, fitness_threshold: float=0.8, max_living_space: int=25, fitness_function: Callable=None, *, keep_ratio: float=None, mutation_rate: float=None, **kwargs)`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:611`
- `is_extinct(self, species: Species) -> bool`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:656`
- `create_species(self, species: Species) -> Species`；装饰器 `overload`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:660`
- `create_species(self, name: str, *gene_species: GeneSpace, mutation_rate: float=None, lifetime: Union[int, Callable]=None, fitness_function: Callable=None, **kwargs) -> Species`；装饰器 `overload`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:664`
- `create_species(self, name: str, params_dict: Dict[str, Sequence])`；装饰器 `overload`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:676`
- `create_species(self, *args, **kwargs) -> Species`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:683`
- `clear_population(self)`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:709`
- `create_individuals(self, num_individuals: Union[Sequence[int], int])`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:720`
- `calc_fitness(self, nproc: int=None, timeout: int=None, not_calc_existed_fitness: bool=False, false_to_mutate: bool=False)`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:743`
- `sorted_individuals(self, decending: bool=False) -> list[Individual]`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:808`
- `progress_time(self, nproc=None, timeout=None, not_calc_existing_fitness: bool=True, false_to_mutate: bool=False)`；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:822`
- `living_species(self) -> set[Species]` [读；只读]；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:653`
- `species(self) -> list[Species]` [读；只读]；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:713`
- `num_species(self) -> int` [读；只读]；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:717`
- `individuals(self) -> list[Individual]` [读；只读]；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:805`

本类字段与枚举值：

- `survival_func` = `survival_func if isinstance(survival_func, Callable) else LFunc.top_living` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:625`
- `living_rate` = `living_ratio if 0 < living_ratio < 1 else 0 if living_ratio <= 0 else 1` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:626`
- `fitness_threshold` = `fitness_threshold` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:643`
- `fitness_function` = `fitness_function` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:644`
- `max_living_space` = `max_living_space` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:645`
- `mutation_rate` = `mutation_rate` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:646`
- `time` = `0` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:647`
- `env_func` = `kwargs.get('env_func', None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:650`
- `keep_ratio` = `0.0` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.Earth`；`hotpot/plugins/opti/ev/core.py:635`

<a id="hotpot-plugins-opti-ev-core-evoluteoptimizer"></a>
### `EvoluteOptimizer`

定义 owner `hotpot.plugins.opti.ev.core.EvoluteOptimizer`；源码 `hotpot/plugins/opti/ev/core.py:873`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, gene_series, pop_size)`；owner `hotpot.plugins.opti.ev.core.EvoluteOptimizer`；`hotpot/plugins/opti/ev/core.py:874`

<a id="hotpot-plugins-opti-ev-core-ml-optimizer"></a>
### `ml_optimizer`

定义 owner `hotpot.plugins.opti.ev.core.ml_optimizer`；源码 `hotpot/plugins/opti/ev/core.py:896`。

```python
ml_optimizer(models: dict[str, type(ModelLike)], hypers: dict[str, Sequence], X: np.ndarray, y: np.ndarray, X_test: np.ndarray=None, y_test: np.ndarray=None, features: list[str]=None, fitness_function: Callable=None, initial_size: Union[int, list[int]]=3, *, feature_engineering: bool=False, fe_kwargs: dict[str, Any]=None, which_return: Union[Literal['best', 'top'], int]='best', **kwargs)
```

<a id="hotpot-plugins-opti-ev-core-machinelearningoptimizer"></a>
### `MachineLearningOptimizer`

定义 owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；源码 `hotpot/plugins/opti/ev/core.py:940`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, models: dict[str, type(ModelLike)], hypers: dict[str, dict], X: np.ndarray, y: np.ndarray, X_test: np.ndarray=None, y_test: np.ndarray=None, features: list[str]=None, *, fitness_function: Callable=None, initial_size: Union[int, list[int]]=3, feature_engineering: bool=False, fe_kwargs: dict[str, Any]=None, **kwargs)`；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:941`
- `train_test_fitness(self, individual)`；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:1045`
- `cross_validation_fitness(self, individual)`；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:1063`
- `optimize(self, n_iters: int=1, **kwargs)`；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:1077`
- `sorted_results(self) -> list[tuple[type(ModelLike), dict]]` [读；只读]；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:1090`
- `optimal_result(self) -> tuple[type(ModelLike), dict]` [读；只读]；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:1097`
- `optimal_model(self) -> ModelLike` [读；只读]；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:1101`

本类字段与枚举值：

- `models` = `models` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:972`
- `hypers` = `hypers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:973`
- `X` = `X` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:974`
- `y` = `y` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:975`
- `X_test` = `X_test` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:976`
- `y_test` = `y_test` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:977`
- `features` = `self._get_feature_names(features)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:978`
- `feature_engineering` = `feature_engineering` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:979`
- `fe_kwargs` = `fe_kwargs if fe_kwargs else {}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:980`
- `initial_size` = `initial_size` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:989`
- `earth` = `Earth(fitness_function=self.fitness_function, **kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:990`
- `fitness_function` = `fitness_function` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MachineLearningOptimizer`；`hotpot/plugins/opti/ev/core.py:983`

<a id="hotpot-plugins-opti-ev-core-mloptimizer"></a>
### `MLOptimizer`

定义 owner `hotpot.plugins.opti.ev.core.MLOptimizer`；源码 `hotpot/plugins/opti/ev/core.py:1106`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, model: ModelLike, X_train: np.ndarray, y_train: np.ndarray, params_space: dict[str, Union[list, tuple]], cv: Union[int, Callable]=5, shuffle: bool=True, *, mutation_rate: float=0.2, init_params: Union[int, list[dict]]=3, living_space: int=10, survival_func: Callable=None, survival_rate: float=0.4, survival_threshold: float=0.8)`；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1107`
- `optimize(self, iter_num: int=5, nproc: int=-1)`；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1140`
- `surviving_individuals(self)` [读；只读]；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1199`
- `best_individual(self) -> Individual` [读；只读]；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1203`
- `best_params(self) -> dict` [读；只读]；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1207`
- `best_fitness(self) -> float` [读；只读]；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1211`
- `best_estimator(self) -> BaseEstimator` [读；只读]；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1215`

本类字段与枚举值：

- `model` = `model` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1123`
- `X_train` = `X_train` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1124`
- `y_train` = `y_train` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1125`
- `params_space` = `params_space` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1126`
- `cv` = `cv` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1127`
- `shuffle` = `shuffle` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1128`
- `init_params` = `init_params` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1130`
- `living_space` = `living_space` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1131`
- `mutation_rate` = `mutation_rate` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1132`
- `survival_func` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1134`
- `survival_rate` = `survival_rate` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1135`
- `survival_threshold` = `survival_threshold` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.core.MLOptimizer`；`hotpot/plugins/opti/ev/core.py:1136`

模块值/类型别名（声明存在不构成事实写许可）：

- `gene_library` = `GeneLibrary()`；`hotpot/plugins/opti/ev/core.py:78`


<a id="hotpot-plugins-opti-ev-func"></a>
## `hotpot.plugins.opti.ev.func`

源码 `hotpot/plugins/opti/ev/func.py`；SHA-256 `aafd0a8f03c79ee4e31f169c59890d0661f4e8de228b10f399b8139f638ec8ae`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `LivingFunc` → `hotpot.plugins.opti.ev.func.LivingFunc`
- `MachineLearningR2Fitness` → `hotpot.plugins.opti.ev.func.MachineLearningR2Fitness`；构造声明 `__init__(self, X, y, algorithm: type)`

<a id="hotpot-plugins-opti-ev-func-livingfunc"></a>
### `LivingFunc`

定义 owner `hotpot.plugins.opti.ev.func.LivingFunc`；源码 `hotpot/plugins/opti/ev/func.py:14`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `top_living(earth)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.opti.ev.func.LivingFunc`；`hotpot/plugins/opti/ev/func.py:17`
- `healthy_living(earth)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.opti.ev.func.LivingFunc`；`hotpot/plugins/opti/ev/func.py:32`

<a id="hotpot-plugins-opti-ev-func-machinelearningr2fitness"></a>
### `MachineLearningR2Fitness`

定义 owner `hotpot.plugins.opti.ev.func.MachineLearningR2Fitness`；源码 `hotpot/plugins/opti/ev/func.py:41`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, X, y, algorithm: type)`；owner `hotpot.plugins.opti.ev.func.MachineLearningR2Fitness`；`hotpot/plugins/opti/ev/func.py:42`
- `__call__(self, individual)`；owner `hotpot.plugins.opti.ev.func.MachineLearningR2Fitness`；`hotpot/plugins/opti/ev/func.py:46`

本类字段与枚举值：

- `algorithm` = `algorithm` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.ev.func.MachineLearningR2Fitness`；`hotpot/plugins/opti/ev/func.py:44`


<a id="hotpot-plugins-opti-exp-design"></a>
## `hotpot.plugins.opti.exp_design`

源码 `hotpot/plugins/opti/exp_design/__init__.py`；SHA-256 `3af3a5b98a9654030d195f4c8dcf456d1acbcb22416fac49f689d92be6fc05af`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-opti-exp-design-core"></a>
## `hotpot.plugins.opti.exp_design.core`

源码 `hotpot/plugins/opti/exp_design/core.py`；SHA-256 `ba55454af41fa2ddc0675edeb2ccee9963030e2fc69bd8b848a6fa3e7f2c34c6`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `orthogonal_design` → `hotpot.plugins.opti.exp_design.core.orthogonal_design`；`orthogonal_design(seq_of_params: Sequence[Sequence], num_exper: int)`
- `params_space` → `hotpot.plugins.opti.exp_design.core.params_space`
- `params` → `hotpot.plugins.opti.exp_design.core.params`

<a id="hotpot-plugins-opti-exp-design-core-orthogonal-design"></a>
### `orthogonal_design`

定义 owner `hotpot.plugins.opti.exp_design.core.orthogonal_design`；源码 `hotpot/plugins/opti/exp_design/core.py:16`。

```python
orthogonal_design(seq_of_params: Sequence[Sequence], num_exper: int)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `params_space` = `[[10 * i for i in range(1, 7)], [20 * i for i in range(1, 7)], [2 * i for i in range(1, 6)], [2 * i for i in range(1, 6)]]`；`hotpot/plugins/opti/exp_design/core.py:35`
- `params` = `orthogonal_design(params_space, 10)`；`hotpot/plugins/opti/exp_design/core.py:42`
- `params` = `pd.DataFrame(params, columns=['H4', 'Mg', 'DMF', 'CH4OH'])`；`hotpot/plugins/opti/exp_design/core.py:43`


<a id="hotpot-plugins-opti-params-space"></a>
## `hotpot.plugins.opti.params_space`

源码 `hotpot/plugins/opti/params_space.py`；SHA-256 `569b7acd24da7acb42bfe24babf7cac8773ff3d6d5d2f47ce923409432750b1a`。

显式 __all__：`('ParamSpace', 'optuna_optimize', 'ParamSets')`。

显式导出绑定：

- `ParamSpace` → `hotpot.plugins.opti.params_space.ParamSpace`；构造声明 `__init__(self)`
- `optuna_optimize` → `hotpot.plugins.opti.params_space.optuna_optimize`；`optuna_optimize(response: Callable[[dict], float], space: ParamSpace, n_trials: int=100, sampler: optuna.samplers.BaseSampler=optuna.samplers.GPSampler, return_study: bool=False)`
- `ParamSets` → `hotpot.plugins.opti.params_space.ParamSets`；构造声明 `__init__(self, params: Optional[dict[str, Any]]=None) -> None`

<a id="hotpot-plugins-opti-params-space-categoricalparams"></a>
### `CategoricalParams`

定义 owner `hotpot.plugins.opti.params_space.CategoricalParams`；源码 `hotpot/plugins/opti/params_space.py:16`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
CategoricalParams(name: str, params: list[Any])
```

本类字段与枚举值：

- `name: str` [可写值字段]；owner `hotpot.plugins.opti.params_space.CategoricalParams`；`hotpot/plugins/opti/params_space.py:17`
- `params: list[Any]` [可写值字段]；owner `hotpot.plugins.opti.params_space.CategoricalParams`；`hotpot/plugins/opti/params_space.py:18`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-opti-params-space-discreteuniformparams"></a>
### `DiscreteUniformParams`

定义 owner `hotpot.plugins.opti.params_space.DiscreteUniformParams`；源码 `hotpot/plugins/opti/params_space.py:21`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
DiscreteUniformParams(name: str, low: float, high: float, q: float)
```

本类字段与枚举值：

- `name: str` [可写值字段]；owner `hotpot.plugins.opti.params_space.DiscreteUniformParams`；`hotpot/plugins/opti/params_space.py:22`
- `low: float` [可写值字段]；owner `hotpot.plugins.opti.params_space.DiscreteUniformParams`；`hotpot/plugins/opti/params_space.py:23`
- `high: float` [可写值字段]；owner `hotpot.plugins.opti.params_space.DiscreteUniformParams`；`hotpot/plugins/opti/params_space.py:24`
- `q: float` [可写值字段]；owner `hotpot.plugins.opti.params_space.DiscreteUniformParams`；`hotpot/plugins/opti/params_space.py:25`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-opti-params-space-floatparams"></a>
### `FloatParams`

定义 owner `hotpot.plugins.opti.params_space.FloatParams`；源码 `hotpot/plugins/opti/params_space.py:28`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
FloatParams(name: str, low: float, high: float, log: bool = False)
```

本类字段与枚举值：

- `name: str` [可写值字段]；owner `hotpot.plugins.opti.params_space.FloatParams`；`hotpot/plugins/opti/params_space.py:29`
- `low: float` [可写值字段]；owner `hotpot.plugins.opti.params_space.FloatParams`；`hotpot/plugins/opti/params_space.py:30`
- `high: float` [可写值字段]；owner `hotpot.plugins.opti.params_space.FloatParams`；`hotpot/plugins/opti/params_space.py:31`
- `log: bool` = `False` [可写值字段]；owner `hotpot.plugins.opti.params_space.FloatParams`；`hotpot/plugins/opti/params_space.py:32`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-opti-params-space-intparams"></a>
### `IntParams`

定义 owner `hotpot.plugins.opti.params_space.IntParams`；源码 `hotpot/plugins/opti/params_space.py:35`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
IntParams(name: str, low: int, high: int, step: int = 1, log: bool = False)
```

本类字段与枚举值：

- `name: str` [可写值字段]；owner `hotpot.plugins.opti.params_space.IntParams`；`hotpot/plugins/opti/params_space.py:36`
- `low: int` [可写值字段]；owner `hotpot.plugins.opti.params_space.IntParams`；`hotpot/plugins/opti/params_space.py:37`
- `high: int` [可写值字段]；owner `hotpot.plugins.opti.params_space.IntParams`；`hotpot/plugins/opti/params_space.py:38`
- `step: int` = `1` [可写值字段]；owner `hotpot.plugins.opti.params_space.IntParams`；`hotpot/plugins/opti/params_space.py:39`
- `log: bool` = `False` [可写值字段]；owner `hotpot.plugins.opti.params_space.IntParams`；`hotpot/plugins/opti/params_space.py:40`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-opti-params-space-paramspace"></a>
### `ParamSpace`

定义 owner `hotpot.plugins.opti.params_space.ParamSpace`；源码 `hotpot/plugins/opti/params_space.py:43`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.plugins.opti.params_space.ParamSpace`。

本类属性与方法：

- `__init__(self)`；owner `hotpot.plugins.opti.params_space.ParamSpace`；`hotpot/plugins/opti/params_space.py:44`
- `add_categorical_params(self, name, params)`；owner `hotpot.plugins.opti.params_space.ParamSpace`；`hotpot/plugins/opti/params_space.py:47`
- `add_discrete_uniform_params(self, name, low, high, q: float=1.0)`；owner `hotpot.plugins.opti.params_space.ParamSpace`；`hotpot/plugins/opti/params_space.py:50`
- `add_float_params(self, name, low, high, log=False)`；owner `hotpot.plugins.opti.params_space.ParamSpace`；`hotpot/plugins/opti/params_space.py:53`
- `add_int_params(self, name, low, high, step=1, log=False)`；owner `hotpot.plugins.opti.params_space.ParamSpace`；`hotpot/plugins/opti/params_space.py:56`
- `copy_to_optuna_trial(self, trial: optuna.Trial)`；owner `hotpot.plugins.opti.params_space.ParamSpace`；`hotpot/plugins/opti/params_space.py:59`

本类字段与枚举值：

- `space` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.opti.params_space.ParamSpace`；`hotpot/plugins/opti/params_space.py:45`

<a id="hotpot-plugins-opti-params-space-optuna-optimize"></a>
### `optuna_optimize`

定义 owner `hotpot.plugins.opti.params_space.optuna_optimize`；源码 `hotpot/plugins/opti/params_space.py:78`。

```python
optuna_optimize(response: Callable[[dict], float], space: ParamSpace, n_trials: int=100, sampler: optuna.samplers.BaseSampler=optuna.samplers.GPSampler, return_study: bool=False)
```

<a id="hotpot-plugins-opti-params-space-paramsets"></a>
### `ParamSets`

定义 owner `hotpot.plugins.opti.params_space.ParamSets`；源码 `hotpot/plugins/opti/params_space.py:98`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.plugins.opti.params_space.ParamSets`。

本类属性与方法：

- `__init__(self, params: Optional[dict[str, Any]]=None) -> None`；owner `hotpot.plugins.opti.params_space.ParamSets`；`hotpot/plugins/opti/params_space.py:100`
- `__getattr__(self, key: str) -> Any`；owner `hotpot.plugins.opti.params_space.ParamSets`；`hotpot/plugins/opti/params_space.py:109`
- `__setattr__(self, key: str, value: Any) -> None`；owner `hotpot.plugins.opti.params_space.ParamSets`；`hotpot/plugins/opti/params_space.py:115`
- `export(self, path: str) -> None`；owner `hotpot.plugins.opti.params_space.ParamSets`；`hotpot/plugins/opti/params_space.py:132`
- `from_json(cls, path: str) -> 'ParamSets'` [classmethod]；装饰器 `classmethod`；owner `hotpot.plugins.opti.params_space.ParamSets`；`hotpot/plugins/opti/params_space.py:138`
- `to_dict(self) -> dict[str, Any]`；owner `hotpot.plugins.opti.params_space.ParamSets`；`hotpot/plugins/opti/params_space.py:147`


<a id="hotpot-plugins-plots"></a>
## `hotpot.plugins.plots`

源码 `hotpot/plugins/plots/__init__.py`；SHA-256 `1072998a891712080d26e755491cae6d420a68ec23379cad509c6d7e069f06ca`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `refresh_cache` → `hotpot.plugins.plots.refresh_cache`；`refresh_cache()`
- `init_fonts` → `hotpot.plugins.plots.init_fonts`；`init_fonts()`

<a id="hotpot-plugins-plots-refresh-cache"></a>
### `refresh_cache`

定义 owner `hotpot.plugins.plots.refresh_cache`；源码 `hotpot/plugins/plots/__init__.py:40`。

```python
refresh_cache()
```

<a id="hotpot-plugins-plots-init-fonts"></a>
### `init_fonts`

定义 owner `hotpot.plugins.plots.init_fonts`；源码 `hotpot/plugins/plots/__init__.py:48`。

```python
init_fonts()
```


<a id="hotpot-plugins-plots-base"></a>
## `hotpot.plugins.plots.base`

源码 `hotpot/plugins/plots/base.py`；SHA-256 `f2c810a1c2e907b4b96557e4fc5adfc1a938be67ee237999ef044254d28d4ede`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `ImmutableDict` → `hotpot.plugins.plots.base.ImmutableDict`
- `Plotter` → `hotpot.plugins.plots.base.Plotter`；构造声明 `__init__(self)`
- `Plot` → `hotpot.plugins.plots.base.Plot`
- `FigAxesModifier` → `hotpot.plugins.plots.base.FigAxesModifier`；构造声明 `__init__(self, plotter: Plotter, fig: plt.Figure, ax: plt.Axes, **kwargs)`

<a id="hotpot-plugins-plots-base-immutabledict"></a>
### `ImmutableDict`

定义 owner `hotpot.plugins.plots.base.ImmutableDict`；源码 `hotpot/plugins/plots/base.py:14`。

基类原始声明：`dict`；装饰器：无。

本类属性与方法：

- `__setitem__(self, key, value)`；owner `hotpot.plugins.plots.base.ImmutableDict`；`hotpot/plugins/plots/base.py:15`
- `__delitem__(self, key)`；owner `hotpot.plugins.plots.base.ImmutableDict`；`hotpot/plugins/plots/base.py:18`
- `update(self, __m, **kwargs)`；owner `hotpot.plugins.plots.base.ImmutableDict`；`hotpot/plugins/plots/base.py:21`
- `popitem(self)`；owner `hotpot.plugins.plots.base.ImmutableDict`；`hotpot/plugins/plots/base.py:24`
- `clear(self)`；owner `hotpot.plugins.plots.base.ImmutableDict`；`hotpot/plugins/plots/base.py:27`
- `pop(self, __key)`；owner `hotpot.plugins.plots.base.ImmutableDict`；`hotpot/plugins/plots/base.py:30`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.plots.base.dict`。

<a id="hotpot-plugins-plots-base-plotter"></a>
### `Plotter`

定义 owner `hotpot.plugins.plots.base.Plotter`；源码 `hotpot/plugins/plots/base.py:34`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self)`；owner `hotpot.plugins.plots.base.Plotter`；`hotpot/plugins/plots/base.py:36`
- `add_axes_modifier(self, which: Literal['fig', 'ax'], modifier: 'FigAxesModifier')`；owner `hotpot.plugins.plots.base.Plotter`；`hotpot/plugins/plots/base.py:40`
- `post_process_axes(self) -> None`；owner `hotpot.plugins.plots.base.Plotter`；`hotpot/plugins/plots/base.py:44`

<a id="hotpot-plugins-plots-base-plot"></a>
### `Plot`

定义 owner `hotpot.plugins.plots.base.Plot`；源码 `hotpot/plugins/plots/base.py:52`。

基类原始声明：`ABC`；装饰器：无。

本类属性与方法：

- `__call__(self, ax: plt.Axes, sciplot: Plotter=None)`；owner `hotpot.plugins.plots.base.Plot`；`hotpot/plugins/plots/base.py:54`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`abc.ABC`。

<a id="hotpot-plugins-plots-base-figaxesmodifier"></a>
### `FigAxesModifier`

定义 owner `hotpot.plugins.plots.base.FigAxesModifier`；源码 `hotpot/plugins/plots/base.py:63`。

基类原始声明：`ABC`；装饰器：无。

本类属性与方法：

- `__init__(self, plotter: Plotter, fig: plt.Figure, ax: plt.Axes, **kwargs)`；owner `hotpot.plugins.plots.base.FigAxesModifier`；`hotpot/plugins/plots/base.py:64`
- `modify(self)`；装饰器 `abstractmethod`；owner `hotpot.plugins.plots.base.FigAxesModifier`；`hotpot/plugins/plots/base.py:71`

本类字段与枚举值：

- `plotter` = `plotter` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.base.FigAxesModifier`；`hotpot/plugins/plots/base.py:65`
- `fig` = `fig` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.base.FigAxesModifier`；`hotpot/plugins/plots/base.py:66`
- `ax` = `ax` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.base.FigAxesModifier`；`hotpot/plugins/plots/base.py:67`
- `kwargs` = `kwargs` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.base.FigAxesModifier`；`hotpot/plugins/plots/base.py:68`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`abc.ABC`。


<a id="hotpot-plugins-plots-colors"></a>
## `hotpot.plugins.plots.colors`

源码 `hotpot/plugins/plots/colors.py`；SHA-256 `5155c6d956f36cbb45cf4e167697cea4ac675860ccd4644f9e17c97f67811e5e`。

显式 __all__：`('create_saturation_colormap', 'register_cmap_to_hotpot', 'load_defined_cmap')`。

显式导出绑定：

- `create_saturation_colormap` → `hotpot.plugins.plots.colors.create_saturation_colormap`；`create_saturation_colormap(name: str, color_rgb_names: Union[str, tuple[int, int, int]])`
- `register_cmap_to_hotpot` → `hotpot.plugins.plots.colors.register_cmap_to_hotpot`；`register_cmap_to_hotpot(cmap: mcolors.Colormap, name: Optional[str]=True)`
- `load_defined_cmap` → `hotpot.plugins.plots.colors.load_defined_cmap`；`load_defined_cmap(cmap_name: str, cmap_type: Literal['linear', 'list']='linear')`

<a id="hotpot-plugins-plots-colors-create-saturation-colormap"></a>
### `create_saturation_colormap`

定义 owner `hotpot.plugins.plots.colors.create_saturation_colormap`；源码 `hotpot/plugins/plots/colors.py:23`。

```python
create_saturation_colormap(name: str, color_rgb_names: Union[str, tuple[int, int, int]])
```

<a id="hotpot-plugins-plots-colors-register-cmap-to-hotpot"></a>
### `register_cmap_to_hotpot`

定义 owner `hotpot.plugins.plots.colors.register_cmap_to_hotpot`；源码 `hotpot/plugins/plots/colors.py:63`。

```python
register_cmap_to_hotpot(cmap: mcolors.Colormap, name: Optional[str]=True)
```

<a id="hotpot-plugins-plots-colors-get-defined-cmap-name"></a>
### `get_defined_cmap_name`

定义 owner `hotpot.plugins.plots.colors.get_defined_cmap_name`；源码 `hotpot/plugins/plots/colors.py:76`。

```python
get_defined_cmap_name()
```

<a id="hotpot-plugins-plots-colors-load-defined-cmap"></a>
### `load_defined_cmap`

定义 owner `hotpot.plugins.plots.colors.load_defined_cmap`；源码 `hotpot/plugins/plots/colors.py:79`。

```python
load_defined_cmap(cmap_name: str, cmap_type: Literal['linear', 'list']='linear')
```

<a id="hotpot-plugins-plots-colors-load-cmap"></a>
### `load_cmap`

定义 owner `hotpot.plugins.plots.colors.load_cmap`；源码 `hotpot/plugins/plots/colors.py:93`。

```python
load_cmap(cmap_name: str, cmap_type: Literal['linear', 'list']='linear')
```

模块值/类型别名（声明存在不构成事实写许可）：

- `c_m` = `create_saturation_colormap('BluesSat', 'blue')`；`hotpot/plugins/plots/colors.py:104`


<a id="hotpot-plugins-plots-defaults"></a>
## `hotpot.plugins.plots.defaults`

源码 `hotpot/plugins/plots/defaults.py`；SHA-256 `50e112312a1a4d5a1af81b994ebb95372cca02ce44ca2eb4911cf473f8d0ea0e`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Settings` → `hotpot.plugins.plots.defaults.Settings`

<a id="hotpot-plugins-plots-defaults-settings"></a>
### `Settings`

定义 owner `hotpot.plugins.plots.defaults.Settings`；源码 `hotpot/plugins/plots/defaults.py:9`。

基类原始声明：无显式基类；装饰器：无。

本类字段与枚举值：

- `font` = `'Arial'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:11`
- `figwidth` = `10.72` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:14`
- `figheight` = `8.205` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:15`
- `dpi` = `300` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:16`
- `axes_pos` = `[0.1483, 0.1657, 0.7163, 0.7185]` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:19`
- `splinewidth` = `3` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:20`
- `xy_label_fontsize` = `32` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:21`
- `ticklabels_fontsize` = `22` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:22`
- `xticklabels_fontsize` = `ticklabels_fontsize` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:23`
- `yticklabels_fontsize` = `ticklabels_fontsize` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:24`
- `ticklabels_rotation` = `None` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:25`
- `xticklabels_rotation` = `None` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:26`
- `yticklabels_rotation` = `None` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:27`
- `superscript_position` = `(-0.125, 1.075)` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:30`
- `superscript_font_dict` = `{'font': font, 'fontsize': 42, 'fontweight': 'bold'}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:31`
- `colorbar_pos` = `[0.9, 0.1657, 0.05, 0.7185]` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:34`
- `text_fontdict` = `{'fontsize': 20}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.plots.defaults.Settings`；`hotpot/plugins/plots/defaults.py:37`

本类名称别名：`xticklabels_fontsize` → `ticklabels_fontsize`, `yticklabels_fontsize` → `ticklabels_fontsize`。


<a id="hotpot-plugins-plots-plots"></a>
## `hotpot.plugins.plots.plots`

源码 `hotpot/plugins/plots/plots/__init__.py`；SHA-256 `b920f40125dd8fb5c95a06a5db2df70666d2fd84a60e7eb8cb8231a302d71c77`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-plots-plots-base"></a>
## `hotpot.plugins.plots.plots.base`

源码 `hotpot/plugins/plots/plots/base.py`；SHA-256 `eb38a48b6cea0cb354f259dd78cb473d1e9a074c8418f825b8b6ee9ef3619194`。

显式 __all__：`('Plot',)`。

显式导出绑定：

- `Plot` → `hotpot.plugins.plots.plots.base.Plot`；构造声明 `__init__(self, *args, **kwargs)`

<a id="hotpot-plugins-plots-plots-base-plot"></a>
### `Plot`

定义 owner `hotpot.plugins.plots.plots.base.Plot`；源码 `hotpot/plugins/plots/plots/base.py:16`。

基类原始声明：`ABC`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.base.Plot`。

本类属性与方法：

- `__init__(self, *args, **kwargs)`；装饰器 `abstractmethod`；owner `hotpot.plugins.plots.plots.base.Plot`；`hotpot/plugins/plots/plots/base.py:19`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；装饰器 `abstractmethod`；owner `hotpot.plugins.plots.plots.base.Plot`；`hotpot/plugins/plots/plots/base.py:23`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`abc.ABC`。


<a id="hotpot-plugins-plots-plots-beyes"></a>
## `hotpot.plugins.plots.plots.beyes`

源码 `hotpot/plugins/plots/plots/beyes.py`；SHA-256 `9842c8f4a75edd4d7874d9107899c26b8b570f85fdcbb0221c5f58ea207b7f2a`。

显式 __all__：`('BayesDesignSpaceMap',)`。

显式导出绑定：

- `BayesDesignSpaceMap` → `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；构造声明 `__init__(self, emb_x: Union[list[np.ndarray], np.ndarray], mus: Union[list[np.ndarray], np.ndarray], sigmas: Union[list[np.ndarray], np.ndarray], X_opti_idx: Union[list[np.ndarray], np.ndarray]=None, X_orig_idx: Union[list[np.ndarray], np.ndarray]=None, to_coutourf=True, mesh_num=100, mu_norm: tuple[float, float]=None, sigma_norm: tuple[float, float]=None, sigma_is_color: bool=False, sigma_label: str=None, cmap='Greys', **kwargs)`

<a id="hotpot-plugins-plots-plots-beyes-bayesdesignspacemap"></a>
### `BayesDesignSpaceMap`

定义 owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；源码 `hotpot/plugins/plots/plots/beyes.py:25`。

基类原始声明：`SciPlotter`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`。

本类属性与方法：

- `list_array_min_max(list_arr: list[np.ndarray]) -> tuple[float, float]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:78`
- `normalizelist_sigma(sigmas, mu_min, mu_max)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:84`
- `__init__(self, emb_x: Union[list[np.ndarray], np.ndarray], mus: Union[list[np.ndarray], np.ndarray], sigmas: Union[list[np.ndarray], np.ndarray], X_opti_idx: Union[list[np.ndarray], np.ndarray]=None, X_orig_idx: Union[list[np.ndarray], np.ndarray]=None, to_coutourf=True, mesh_num=100, mu_norm: tuple[float, float]=None, sigma_norm: tuple[float, float]=None, sigma_is_color: bool=False, sigma_label: str=None, cmap='Greys', **kwargs)`；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:110`

本类字段与枚举值：

- `plot_index` = `0` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:162`
- `plots_num` = `len(emb_x)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:163`
- `emb_x` = `emb_x` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:165`
- `mus` = `mus` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:166`
- `sigmas` = `sigmas` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:167`
- `X_opti_idx` = `X_opti_idx` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:168`
- `X_orig_idx` = `X_orig_idx` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:169`
- `to_coutourf` = `to_coutourf` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:170`
- `mesh_num` = `mesh_num` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:171`
- `cmap` = `colors.load_cmap(cmap)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:172`
- `sigma_ref_value` = `sigma_ref_value` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:181`
- `sigma_colorbar_label` = `sigma_label if isinstance(sigma_label, str) else sigma_colorbar_label` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:182`
- `sigma_colors` = `self.sigmas if sigma_is_color else sigma_colors` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:183`
- `mu_norm` = `Normalize(mu_min, mu_max)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:193`
- `sigma_norm` = `Normalize(*sigma_norm)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.beyes.BayesDesignSpaceMap`；`hotpot/plugins/plots/plots/beyes.py:194`

继承成员（逐项保留声明 owner）：

- `init_figure_size(self, nrows, ncols)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:96`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `set_arguments(self, **kwargs)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:112`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `print_settings(self)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:116`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `__call__(self, *args, **kwargs)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:120`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `set_spline(self, main_ax: plt.Axes, *twin_axs: plt.Axes)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:142`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `set_xylabel_font(self, main_ax: plt.Axes, *twin_axs: plt.Axes)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:152`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `set_ticks(self, main_ax: plt.Axes, *twin_axs: plt.Axes)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:180`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `set_axes_position(self, main_ax, *twin_axs)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:210`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `add_superscript(self, ax, i, font_dict: dict=None)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:219`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `add_colorbar(self, ax: plt.Axes, mappable=None, value_lim: Sequence=None, colorbar_label: str=None, colorbar_pos: Sequence[float]=None, orientation: Literal['horizontal', 'vertical']='vertical', **kwargs)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:223`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `rows_interval(self)` [读；只读]；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:85`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `cols_interval(self)` [读；只读]；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:91`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `settings` = `Settings()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:47`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `plotters` = `plotters` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:64`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `fig` = `fig` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:69`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `axs` = `np.array([[axs]])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:71`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `add_axes_modifier(self, which: Literal['fig', 'ax'], modifier: 'FigAxesModifier')`；owner `hotpot.plugins.plots.base.Plotter`；`hotpot/plugins/plots/base.py:40`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`
- `post_process_axes(self) -> None`；owner `hotpot.plugins.plots.base.Plotter`；`hotpot/plugins/plots/base.py:44`；继承经 `hotpot.plugins.plots.plotter.SciPlotter`

模块值/类型别名（声明存在不构成事实写许可）：

- `EPSILON` = `1e-08`；`hotpot/plugins/plots/plots/beyes.py:22`


<a id="hotpot-plugins-plots-plots-general"></a>
## `hotpot.plugins.plots.plots.general`

源码 `hotpot/plugins/plots/plots/general.py`；SHA-256 `c229f8db8239959df47b5c065333f569ac52fe91f99c011f72db7d1ffeb7dc90`。

显式 __all__：`('Hist',)`。

显式导出绑定：

- `Hist` → `hotpot.plugins.plots.plots.general.Hist`；构造声明 `__init__(self, x, x_name, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='#F5CD5E', label=None, stacked=False, edgecolor='black', linewidth=1.0, *, data=None, **kwargs)`

<a id="hotpot-plugins-plots-plots-general-hist"></a>
### `Hist`

定义 owner `hotpot.plugins.plots.plots.general.Hist`；源码 `hotpot/plugins/plots/plots/general.py:23`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.general.Hist`。

本类属性与方法：

- `__init__(self, x, x_name, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='#F5CD5E', label=None, stacked=False, edgecolor='black', linewidth=1.0, *, data=None, **kwargs)`；owner `hotpot.plugins.plots.plots.general.Hist`；`hotpot/plugins/plots/plots/general.py:24`
- `__call__(self, ax: plt.axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.general.Hist`；`hotpot/plugins/plots/plots/general.py:34`

本类字段与枚举值：

- `x_name` = `kwargs.pop('x_name')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.general.Hist`；`hotpot/plugins/plots/plots/general.py:31`
- `kwargs` = `kwargs` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.general.Hist`；`hotpot/plugins/plots/plots/general.py:32`


<a id="hotpot-plugins-plots-plots-ml"></a>
## `hotpot.plugins.plots.plots.ml`

源码 `hotpot/plugins/plots/plots/ml.py`；SHA-256 `7a654cca1bb000920077a86eb2c3efb4f7d25a3c6c0de0cbbac98143f34e7bdd`。

显式 __all__：`('R2Regression', 'FeatureImportance', 'HierarchicalTree', 'PearsonMatrix', 'Pearson', 'SciPlotter', 'EmbeddingDataTo2dMap', 'ConfusionMatrix', 'ROCCurve', 'DETCurve', 'PrecisionRecallCurve', 'MultiClassROCCurve', 'SHAPlot')`。

显式导出绑定：

- `R2Regression` → `hotpot.plugins.plots.plots.ml.R2Regression`；构造声明 `__init__(self, xy1, xy2=None, unit: str=None, xy_lim: Union[Sequence, np.ndarray]=None, s1=None, s2=None, c1=None, c2=None, ch=None, marker1=None, marker2=None, err1=None, err2=None, show_mae=False, show_rmse=False, *args, target_name='Target', prediction_name='Predicted', sample1_name: str=None, sample2_name: str=None, sampleh_name: str=None, xy1_highlight_indices=None, xy2_highlight_indices=None, **kwargs)`
- `FeatureImportance` → `hotpot.plugins.plots.plots.ml.FeatureImportance`；构造声明 `__init__(self, feature_name, imp, force_show_label=False)`
- `HierarchicalTree` → `hotpot.plugins.plots.plots.ml.HierarchicalTree`；构造声明 `__init__(self, x: np.ndarray, xlabels: Sequence[Union[str, str]], color_threshold=0.6)`
- `PearsonMatrix` → `hotpot.plugins.plots.plots.ml.PearsonMatrix`；构造声明 `__init__(self, x: np.ndarray, xlabels: list[str], is_matrix: bool=False, show_values=False, num_round=3, colorbar=True)`
- `Pearson` → `hotpot.plugins.plots.plots.ml.Pearson`；构造声明 `__init__(self, xy, xlabel, ylabel)`
- `SciPlotter` → `hotpot.plugins.plots.plotter.SciPlotter`；构造声明 `__init__(self, plotters: Union['Plot', Callable, np.ndarray[Union['Plot', Callable]], Sequence[Callable]], **kwargs)`
- `EmbeddingDataTo2dMap` → `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；构造声明 `__init__(self, X_train, X_test, y_train, y_test, p_train, p_test, embedding=TSNE(), feature_weight=None, color_data: Union[str, np.ndarray]='error')`
- `ConfusionMatrix` → `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；构造声明 `__init__(self, pred: np.ndarray, target: np.ndarray, norm='true', threshold: Optional[float]=None, show_values: bool=False) -> None`
- `ROCCurve` → `hotpot.plugins.plots.plots.ml.ROCCurve`；构造声明 `__init__(self, pred: np.ndarray, target: np.ndarray) -> None`
- `DETCurve` → `hotpot.plugins.plots.plots.ml.DETCurve`；构造声明 `__init__(self, pred: np.ndarray[int], target: np.ndarray[int]) -> None`
- `PrecisionRecallCurve` → `hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`；构造声明 `__init__(self, pred: np.ndarray[int], target: np.ndarray[int]) -> None`
- `MultiClassROCCurve` → `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；构造声明 `__init__(self, pred: np.ndarray[int], target: np.ndarray[int], class_names: Iterable[str]=None) -> None`
- `SHAPlot` → `hotpot.plugins.plots.plots.ml.SHAPlot`；构造声明 `__init__(self, exp: shap.Explanation, plot_type: _plot_type='bar', *args, **kwargs)`

<a id="hotpot-plugins-plots-plots-ml-confusionmatrix"></a>
### `ConfusionMatrix`

定义 owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；源码 `hotpot/plugins/plots/plots/ml.py:65`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.ConfusionMatrix`。

本类属性与方法：

- `__init__(self, pred: np.ndarray, target: np.ndarray, norm='true', threshold: Optional[float]=None, show_values: bool=False) -> None`；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:66`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:94`

本类字段与枚举值：

- `show_values` = `show_values` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:74`
- `target` = `np.asarray(target).flatten()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:90`
- `confusion_matrix` = `confusion_matrix(self.target, self.pred, normalize=norm, labels=self.labels)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:92`
- `threshold` = `threshold` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:76`
- `labels` = `np.array([0, 1])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:84`
- `pred` = `np.asarray(pred > self.threshold, dtype=int).flatten()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.ConfusionMatrix`；`hotpot/plugins/plots/plots/ml.py:85`

<a id="hotpot-plugins-plots-plots-ml-roccurve"></a>
### `ROCCurve`

定义 owner `hotpot.plugins.plots.plots.ml.ROCCurve`；源码 `hotpot/plugins/plots/plots/ml.py:109`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.ROCCurve`。

本类属性与方法：

- `__init__(self, pred: np.ndarray, target: np.ndarray) -> None`；owner `hotpot.plugins.plots.plots.ml.ROCCurve`；`hotpot/plugins/plots/plots/ml.py:110`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.ROCCurve`；`hotpot/plugins/plots/plots/ml.py:121`

本类字段与枚举值：

- `roc_auc` = `auc(self.fpr, self.tpr)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.ROCCurve`；`hotpot/plugins/plots/plots/ml.py:119`

<a id="hotpot-plugins-plots-plots-ml-detcurve"></a>
### `DETCurve`

定义 owner `hotpot.plugins.plots.plots.ml.DETCurve`；源码 `hotpot/plugins/plots/plots/ml.py:134`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.DETCurve`。

本类属性与方法：

- `__init__(self, pred: np.ndarray[int], target: np.ndarray[int]) -> None`；owner `hotpot.plugins.plots.plots.ml.DETCurve`；`hotpot/plugins/plots/plots/ml.py:135`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.DETCurve`；`hotpot/plugins/plots/plots/ml.py:138`

<a id="hotpot-plugins-plots-plots-ml-precisionrecallcurve"></a>
### `PrecisionRecallCurve`

定义 owner `hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`；源码 `hotpot/plugins/plots/plots/ml.py:147`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`。

本类属性与方法：

- `__init__(self, pred: np.ndarray[int], target: np.ndarray[int]) -> None`；owner `hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`；`hotpot/plugins/plots/plots/ml.py:148`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`；`hotpot/plugins/plots/plots/ml.py:155`

本类字段与枚举值：

- `f1_score` = `2 * self.precision * self.recall / (self.precision + self.recall)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`；`hotpot/plugins/plots/plots/ml.py:150`
- `thresh` = `np.concatenate(([np.min(thresh) - 1e-12], thresh))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`；`hotpot/plugins/plots/plots/ml.py:152`
- `max_thresh` = `self.thresh[self._f1_argmax]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PrecisionRecallCurve`；`hotpot/plugins/plots/plots/ml.py:153`

<a id="hotpot-plugins-plots-plots-ml-multiclassroccurve"></a>
### `MultiClassROCCurve`

定义 owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；源码 `hotpot/plugins/plots/plots/ml.py:166`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.MultiClassROCCurve`。

本类属性与方法：

- `__init__(self, pred: np.ndarray[int], target: np.ndarray[int], class_names: Iterable[str]=None) -> None`；owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；`hotpot/plugins/plots/plots/ml.py:167`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；`hotpot/plugins/plots/plots/ml.py:194`

本类字段与枚举值：

- `pred` = `np.asarray(pred)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；`hotpot/plugins/plots/plots/ml.py:177`
- `target` = `np.asarray(target, dtype=int).flatten()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；`hotpot/plugins/plots/plots/ml.py:178`
- `n_classes` = `pred.shape[1]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；`hotpot/plugins/plots/plots/ml.py:179`
- `class_names` = `list(class_names) if class_names is not None else None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；`hotpot/plugins/plots/plots/ml.py:180`
- `roc_auc` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.MultiClassROCCurve`；`hotpot/plugins/plots/plots/ml.py:187`

<a id="hotpot-plugins-plots-plots-ml-r2regression"></a>
### `R2Regression`

定义 owner `hotpot.plugins.plots.plots.ml.R2Regression`；源码 `hotpot/plugins/plots/plots/ml.py:208`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.R2Regression`。

本类属性与方法：

- `__init__(self, xy1, xy2=None, unit: str=None, xy_lim: Union[Sequence, np.ndarray]=None, s1=None, s2=None, c1=None, c2=None, ch=None, marker1=None, marker2=None, err1=None, err2=None, show_mae=False, show_rmse=False, *args, target_name='Target', prediction_name='Predicted', sample1_name: str=None, sample2_name: str=None, sampleh_name: str=None, xy1_highlight_indices=None, xy2_highlight_indices=None, **kwargs)`；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:209`
- `add_diagonal(ax: plt.Axes)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:321`
- `add_metric_info(self, ax: plt.Axes)`；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:345`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:409`

本类字段与枚举值：

- `xy1` = `xy1 if isinstance(xy1, np.ndarray) else np.array(xy1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:248`
- `xy2` = `xy2 if xy2 is None or isinstance(xy2, np.ndarray) else np.array(xy2)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:249`
- `sample1_name` = `''` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:250`
- `sample2_name` = `''` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:251`
- `sampleh_name` = `''` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:252`
- `target_name` = `target_name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:253`
- `prediction_name` = `prediction_name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:254`
- `sample_num` = `len(self.xy1) + (len(self.xy2) if isinstance(self.xy2, np.ndarray) else 0)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:256`
- `is_small_sample` = `self.sample_num < 1000` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:257`
- `unit` = `f' ({unit})' if unit else ''` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:259`
- `s1` = `s1` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:274`
- `s2` = `s2 or s1` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:275`
- `c1` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:276`
- `c2` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:277`
- `ch` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:278`
- `marker1` = `marker1 or 'o'` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:279`
- `marker2` = `marker2 or 'x'` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:280`
- `err1` = `err1` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:281`
- `err2` = `err2` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:282`
- `args` = `args` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:284`
- `kwargs` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:286`
- `show_mae` = `show_mae` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:292`
- `show_rmse` = `show_rmse` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:293`
- `metrics` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:294`
- `xy_highlight` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:296`
- `xy1_highlight_indices` = `np.array(xy1_highlight_indices).tolist()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:297`
- `xy2_highlight_indices` = `np.array(xy2_highlight_indices).tolist()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:298`
- `xy_lim` = `xy_lim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.R2Regression`；`hotpot/plugins/plots/plots/ml.py:263`

<a id="hotpot-plugins-plots-plots-ml-featureimportance"></a>
### `FeatureImportance`

定义 owner `hotpot.plugins.plots.plots.ml.FeatureImportance`；源码 `hotpot/plugins/plots/plots/ml.py:464`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.FeatureImportance`。

本类属性与方法：

- `__init__(self, feature_name, imp, force_show_label=False)`；owner `hotpot.plugins.plots.plots.ml.FeatureImportance`；`hotpot/plugins/plots/plots/ml.py:465`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.FeatureImportance`；`hotpot/plugins/plots/plots/ml.py:470`

本类字段与枚举值：

- `feature_name` = `feature_name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.FeatureImportance`；`hotpot/plugins/plots/plots/ml.py:466`
- `imp` = `imp` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.FeatureImportance`；`hotpot/plugins/plots/plots/ml.py:467`
- `force_show_label` = `force_show_label` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.FeatureImportance`；`hotpot/plugins/plots/plots/ml.py:468`

<a id="hotpot-plugins-plots-plots-ml-hierarchicaltree"></a>
### `HierarchicalTree`

定义 owner `hotpot.plugins.plots.plots.ml.HierarchicalTree`；源码 `hotpot/plugins/plots/plots/ml.py:487`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.HierarchicalTree`。

本类属性与方法：

- `__init__(self, x: np.ndarray, xlabels: Sequence[Union[str, str]], color_threshold=0.6)`；owner `hotpot.plugins.plots.plots.ml.HierarchicalTree`；`hotpot/plugins/plots/plots/ml.py:488`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.HierarchicalTree`；`hotpot/plugins/plots/plots/ml.py:496`

本类字段与枚举值：

- `x` = `x` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.HierarchicalTree`；`hotpot/plugins/plots/plots/ml.py:492`
- `xlabels` = `np.array(xlabels)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.HierarchicalTree`；`hotpot/plugins/plots/plots/ml.py:493`
- `threshold` = `color_threshold` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.HierarchicalTree`；`hotpot/plugins/plots/plots/ml.py:494`

<a id="hotpot-plugins-plots-plots-ml-pearson"></a>
### `Pearson`

定义 owner `hotpot.plugins.plots.plots.ml.Pearson`；源码 `hotpot/plugins/plots/plots/ml.py:513`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.Pearson`。

本类属性与方法：

- `__init__(self, xy, xlabel, ylabel)`；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:515`
- `fit_line(x: np.ndarray, y: np.ndarray, ax: plt.Axes)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:523`
- `add_line(self, ax: plt.Axes)`；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:530`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:534`

本类字段与枚举值：

- `xy` = `xy` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:516`
- `c` = `utils.calculate_scatter_density(xy)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:517`
- `xlabel` = `xlabel` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:519`
- `ylabel` = `ylabel` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.Pearson`；`hotpot/plugins/plots/plots/ml.py:520`

<a id="hotpot-plugins-plots-plots-ml-pearsonmatrix"></a>
### `PearsonMatrix`

定义 owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；源码 `hotpot/plugins/plots/plots/ml.py:546`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.PearsonMatrix`。

本类属性与方法：

- `__init__(self, x: np.ndarray, xlabels: list[str], is_matrix: bool=False, show_values=False, num_round=3, colorbar=True)`；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:547`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:566`

本类字段与枚举值：

- `x` = `x` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:559`
- `xlabels` = `xlabels` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:560`
- `is_matrix` = `is_matrix` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:561`
- `round` = `num_round` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:562`
- `show_values` = `show_values` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:563`
- `colorbar` = `colorbar` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.PearsonMatrix`；`hotpot/plugins/plots/plots/ml.py:564`

<a id="hotpot-plugins-plots-plots-ml-shapairplot"></a>
### `SHAPairPlot`

定义 owner `hotpot.plugins.plots.plots.ml.SHAPairPlot`；源码 `hotpot/plugins/plots/plots/ml.py:598`。

```python
SHAPairPlot(exp: shap.Explanation, *args, **kwargs)
```

<a id="hotpot-plugins-plots-plots-ml-shaplot"></a>
### `SHAPlot`

定义 owner `hotpot.plugins.plots.plots.ml.SHAPlot`；源码 `hotpot/plugins/plots/plots/ml.py:602`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.SHAPlot`。

本类属性与方法：

- `__init__(self, exp: shap.Explanation, plot_type: _plot_type='bar', *args, **kwargs)`；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:606`
- `__call__(self, ax: plt.Axes, sciplot: SciPlotter=None)`；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:613`
- `set_yticklabels(self, ax)`；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:673`
- `autowrap(text: str, max_len=12)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:687`
- `add_superscript(fig: plt.Figure, axs: np.ndarray[plt.Axes])` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:710`

本类字段与枚举值：

- `exp` = `exp` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:608`
- `plot_type` = `plot_type` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:609`
- `args` = `args` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:610`
- `kwargs` = `kwargs` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.SHAPlot`；`hotpot/plugins/plots/plots/ml.py:611`

<a id="hotpot-plugins-plots-plots-ml-embeddingdatato2dmap"></a>
### `EmbeddingDataTo2dMap`

定义 owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；源码 `hotpot/plugins/plots/plots/ml.py:765`。

基类原始声明：`Plot`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`。

本类属性与方法：

- `__init__(self, X_train, X_test, y_train, y_test, p_train, p_test, embedding=TSNE(), feature_weight=None, color_data: Union[str, np.ndarray]='error')`；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:766`
- `__call__(self, ax: plt.Axes, *args, **kwargs)`；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:809`

本类字段与枚举值：

- `color_data` = `color_data` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:790`
- `X_train` = `X_train` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:792`
- `X_test` = `X_test` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:793`
- `y_train` = `y_train` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:794`
- `y_test` = `y_test` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:795`
- `p_train` = `p_train` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:796`
- `p_test` = `p_test` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:797`
- `embedding` = `embedding` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:798`
- `feature_weight` = `feature_weight if feature_weight else np.ones(X_train.shape[1])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:799`
- `total_X` = `np.vstack((X_train, X_test))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:801`
- `total_y` = `np.hstack((y_train, y_test))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:802`
- `weighted_X` = `self.total_X * self.feature_weight` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:804`
- `emb_X` = `embedding.fit_transform(self.weighted_X).T` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.ml.EmbeddingDataTo2dMap`；`hotpot/plugins/plots/plots/ml.py:806`


<a id="hotpot-plugins-plots-plots-perodotable"></a>
## `hotpot.plugins.plots.plots.perodotable`

源码 `hotpot/plugins/plots/plots/perodotable.py`；SHA-256 `23345288cba5fb49a973e70d69e383cdf6bea27851683c6519c3fb8606f37a8e`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `PeriodicTable` → `hotpot.plugins.plots.plots.perodotable.PeriodicTable`；构造声明 `__init__(self, element_values: dict[str, float])`

<a id="hotpot-plugins-plots-plots-perodotable-periodictable"></a>
### `PeriodicTable`

定义 owner `hotpot.plugins.plots.plots.perodotable.PeriodicTable`；源码 `hotpot/plugins/plots/plots/perodotable.py:4`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, element_values: dict[str, float])`；owner `hotpot.plugins.plots.plots.perodotable.PeriodicTable`；`hotpot/plugins/plots/plots/perodotable.py:6`

本类字段与枚举值：

- `element_values` = `element_values` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plots.perodotable.PeriodicTable`；`hotpot/plugins/plots/plots/perodotable.py:7`


<a id="hotpot-plugins-plots-plotter"></a>
## `hotpot.plugins.plots.plotter`

源码 `hotpot/plugins/plots/plotter.py`；SHA-256 `a83ef97ab7f35b2e121842cd38f5020f2c47693d55d0785ef9053ae4043a0ea6`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `axes_set_method` → `hotpot.plugins.plots.plotter.axes_set_method`；`axes_set_method(method: Callable)`
- `SciPlotter` → `hotpot.plugins.plots.plotter.SciPlotter`；构造声明 `__init__(self, plotters: Union['Plot', Callable, np.ndarray[Union['Plot', Callable]], Sequence[Callable]], **kwargs)`
- `LegendOrganizer` → `hotpot.plugins.plots.plotter.LegendOrganizer`；构造声明 `__init__(self)`

<a id="hotpot-plugins-plots-plotter-axes-set-method"></a>
### `axes_set_method`

定义 owner `hotpot.plugins.plots.plotter.axes_set_method`；源码 `hotpot/plugins/plots/plotter.py:24`。

```python
axes_set_method(method: Callable)
```

<a id="hotpot-plugins-plots-plotter-sciplotter"></a>
### `SciPlotter`

定义 owner `hotpot.plugins.plots.plotter.SciPlotter`；源码 `hotpot/plugins/plots/plotter.py:40`。

基类原始声明：`Plotter`；装饰器：无。

显式导出位置：`hotpot.plugins.plots.plots.ml.SciPlotter`。

本类属性与方法：

- `__init__(self, plotters: Union['Plot', Callable, np.ndarray[Union['Plot', Callable]], Sequence[Callable]], **kwargs)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:41`
- `init_figure_size(self, nrows, ncols)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:96`
- `set_arguments(self, **kwargs)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:112`
- `print_settings(self)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:116`
- `__call__(self, *args, **kwargs)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:120`
- `set_spline(self, main_ax: plt.Axes, *twin_axs: plt.Axes)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:142`
- `set_xylabel_font(self, main_ax: plt.Axes, *twin_axs: plt.Axes)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:152`
- `set_spline(self, main_ax: plt.Axes, *twin_axs: plt.Axes)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:170`
- `set_ticks(self, main_ax: plt.Axes, *twin_axs: plt.Axes)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:180`
- `set_axes_position(self, main_ax, *twin_axs)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:210`
- `add_superscript(self, ax, i, font_dict: dict=None)`；装饰器 `axes_set_method`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:219`
- `add_colorbar(self, ax: plt.Axes, mappable=None, value_lim: Sequence=None, colorbar_label: str=None, colorbar_pos: Sequence[float]=None, orientation: Literal['horizontal', 'vertical']='vertical', **kwargs)`；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:223`
- `rows_interval(self)` [读；只读]；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:85`
- `cols_interval(self)` [读；只读]；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:91`

本类字段与枚举值：

- `settings` = `Settings()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:47`
- `plotters` = `plotters` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:64`
- `fig` = `fig` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:69`
- `axs` = `np.array([[axs]])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.SciPlotter`；`hotpot/plugins/plots/plotter.py:71`

继承成员（逐项保留声明 owner）：

- `add_axes_modifier(self, which: Literal['fig', 'ax'], modifier: 'FigAxesModifier')`；owner `hotpot.plugins.plots.base.Plotter`；`hotpot/plugins/plots/base.py:40`；继承经 `hotpot.plugins.plots.base.Plotter`
- `post_process_axes(self) -> None`；owner `hotpot.plugins.plots.base.Plotter`；`hotpot/plugins/plots/base.py:44`；继承经 `hotpot.plugins.plots.base.Plotter`

<a id="hotpot-plugins-plots-plotter-legendorganizer"></a>
### `LegendOrganizer`

定义 owner `hotpot.plugins.plots.plotter.LegendOrganizer`；源码 `hotpot/plugins/plots/plotter.py:239`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self)`；owner `hotpot.plugins.plots.plotter.LegendOrganizer`；`hotpot/plugins/plots/plotter.py:240`
- `add_plot(self, obj_name, plot_func, *args, **kwargs)`；owner `hotpot.plugins.plots.plotter.LegendOrganizer`；`hotpot/plugins/plots/plotter.py:245`
- `add_legend(self, ax: plt.Axes, *args, **kwargs)`；owner `hotpot.plugins.plots.plotter.LegendOrganizer`；`hotpot/plugins/plots/plotter.py:263`

本类字段与枚举值：

- `legend_names` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.plots.plotter.LegendOrganizer`；`hotpot/plugins/plots/plotter.py:242`


<a id="hotpot-plugins-plots-utils"></a>
## `hotpot.plugins.plots.utils`

源码 `hotpot/plugins/plots/utils.py`；SHA-256 `4abf75be5bf232a8191fa8e4bd70ae4513de60dc93761b0c61a7b8cbe0d27384`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `calc_active_span` → `hotpot.plugins.plots.utils.calc_active_span`；`calc_active_span(nrows, ncols, i, j)`
- `abs2rela_pos` → `hotpot.plugins.plots.utils.abs2rela_pos`；`abs2rela_pos(ax: plt.Axes, left: float, bottom: float, width: float, height: float, rows_interval: float=None, cols_interval: float=None)`
- `insert_axes` → `hotpot.plugins.plots.utils.insert_axes`；`insert_axes(main_ax, ins_ax, relative_pos: Sequence[float]=None)`
- `specify_colorbar_pos` → `hotpot.plugins.plots.utils.specify_colorbar_pos`；`specify_colorbar_pos(main_ax: plt.Axes, orientation: Literal['horizontal', 'vertical']='vertical')`
- `add_colorbar` → `hotpot.plugins.plots.utils.add_colorbar`；`add_colorbar(ax: plt.Axes, mappable=None, value_lim: Sequence=None, colorbar_label: str=None, colorbar_pos: Sequence[float]=None, orientation: Literal['horizontal', 'vertical']='vertical', norm: mpl.colors.Normalize=None, cmap: Union[str, mpl.colors.Colormap]=None, **kwargs)`
- `add_text` → `hotpot.plugins.plots.utils.add_text`；`add_text(ax, xfrac: float, yfrac: float, s: str, fontdict: dict=None)`
- `sample_density` → `hotpot.plugins.plots.utils.sample_density`；`sample_density(a, b, ratio=0.4)`
- `later_call` → `hotpot.plugins.plots.utils.later_call`；`later_call(func: Callable)`
- `calculate_scatter_density` → `hotpot.plugins.plots.utils.calculate_scatter_density`；`calculate_scatter_density(xy: np.ndarray)`
- `scatter_to_coutourf` → `hotpot.plugins.plots.utils.scatter_to_coutourf`；`scatter_to_coutourf(x, y, c, mesh_num=50)`

<a id="hotpot-plugins-plots-utils-calc-active-span"></a>
### `calc_active_span`

定义 owner `hotpot.plugins.plots.utils.calc_active_span`；源码 `hotpot/plugins/plots/utils.py:24`。

```python
calc_active_span(nrows, ncols, i, j)
```

<a id="hotpot-plugins-plots-utils-abs2rela-pos"></a>
### `abs2rela_pos`

定义 owner `hotpot.plugins.plots.utils.abs2rela_pos`；源码 `hotpot/plugins/plots/utils.py:33`。

```python
abs2rela_pos(ax: plt.Axes, left: float, bottom: float, width: float, height: float, rows_interval: float=None, cols_interval: float=None)
```

<a id="hotpot-plugins-plots-utils-insert-axes"></a>
### `insert_axes`

定义 owner `hotpot.plugins.plots.utils.insert_axes`；源码 `hotpot/plugins/plots/utils.py:77`。

```python
insert_axes(main_ax, ins_ax, relative_pos: Sequence[float]=None)
```

<a id="hotpot-plugins-plots-utils-specify-colorbar-pos"></a>
### `specify_colorbar_pos`

定义 owner `hotpot.plugins.plots.utils.specify_colorbar_pos`；源码 `hotpot/plugins/plots/utils.py:85`。

```python
specify_colorbar_pos(main_ax: plt.Axes, orientation: Literal['horizontal', 'vertical']='vertical')
```

<a id="hotpot-plugins-plots-utils-add-colorbar"></a>
### `add_colorbar`

定义 owner `hotpot.plugins.plots.utils.add_colorbar`；源码 `hotpot/plugins/plots/utils.py:96`。

```python
add_colorbar(ax: plt.Axes, mappable=None, value_lim: Sequence=None, colorbar_label: str=None, colorbar_pos: Sequence[float]=None, orientation: Literal['horizontal', 'vertical']='vertical', norm: mpl.colors.Normalize=None, cmap: Union[str, mpl.colors.Colormap]=None, **kwargs)
```

<a id="hotpot-plugins-plots-utils-add-text"></a>
### `add_text`

定义 owner `hotpot.plugins.plots.utils.add_text`；源码 `hotpot/plugins/plots/utils.py:146`。

```python
add_text(ax, xfrac: float, yfrac: float, s: str, fontdict: dict=None)
```

<a id="hotpot-plugins-plots-utils-sample-density"></a>
### `sample_density`

定义 owner `hotpot.plugins.plots.utils.sample_density`；源码 `hotpot/plugins/plots/utils.py:155`。

```python
sample_density(a, b, ratio=0.4)
```

<a id="hotpot-plugins-plots-utils-later-call"></a>
### `later_call`

定义 owner `hotpot.plugins.plots.utils.later_call`；源码 `hotpot/plugins/plots/utils.py:160`。

```python
later_call(func: Callable)
```

<a id="hotpot-plugins-plots-utils-calculate-scatter-density"></a>
### `calculate_scatter_density`

定义 owner `hotpot.plugins.plots.utils.calculate_scatter_density`；源码 `hotpot/plugins/plots/utils.py:171`。

```python
calculate_scatter_density(xy: np.ndarray)
```

<a id="hotpot-plugins-plots-utils-scatter-to-coutourf"></a>
### `scatter_to_coutourf`

定义 owner `hotpot.plugins.plots.utils.scatter_to_coutourf`；源码 `hotpot/plugins/plots/utils.py:178`。

```python
scatter_to_coutourf(x, y, c, mesh_num=50)
```


<a id="hotpot-plugins-pubchem"></a>
## `hotpot.plugins.pubchem`

源码 `hotpot/plugins/pubchem/__init__.py`；SHA-256 `e22e5483614dea690e7ff84a6acb3b2ce7d1017839a4c3c8eb6e6d7e26953384`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-pubchem-get-data"></a>
## `hotpot.plugins.pubchem.get_data`

源码 `hotpot/plugins/pubchem/get_data.py`；SHA-256 `add87f2b5a351ce32c70fc0c666ab13f4648d465f5846ce2a113880caa1cb14f`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `download_pubchem_sdf_files` → `hotpot.plugins.pubchem.get_data.download_pubchem_sdf_files`；`download_pubchem_sdf_files(output_dir, base_url='https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/')`
- `download_file` → `hotpot.plugins.pubchem.get_data.download_file`；`download_file(url, dest_path)`
- `out_dir` → `hotpot.plugins.pubchem.get_data.out_dir`

<a id="hotpot-plugins-pubchem-get-data-download-pubchem-sdf-files"></a>
### `download_pubchem_sdf_files`

定义 owner `hotpot.plugins.pubchem.get_data.download_pubchem_sdf_files`；源码 `hotpot/plugins/pubchem/get_data.py:16`。

```python
download_pubchem_sdf_files(output_dir, base_url='https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/')
```

<a id="hotpot-plugins-pubchem-get-data-download-file"></a>
### `download_file`

定义 owner `hotpot.plugins.pubchem.get_data.download_file`；源码 `hotpot/plugins/pubchem/get_data.py:48`。

```python
download_file(url, dest_path)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `out_dir` = `'D:\\pubchem'`；`hotpot/plugins/pubchem/get_data.py:92`


<a id="hotpot-plugins-pubchem-utils"></a>
## `hotpot.plugins.pubchem.utils`

源码 `hotpot/plugins/pubchem/utils.py`；SHA-256 `e92d5ed180688d5b282707d17cdee9b90a5a5a6ae95df5b251ff4e8289a52ca7`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `get_compounds_cid` → `hotpot.plugins.pubchem.utils.get_compounds_cid`；`get_compounds_cid(identifier: str, id_type: str='name')`
- `async_get_compounds_cid` → `hotpot.plugins.pubchem.utils.async_get_compounds_cid`；`async_get_compounds_cid(identifier: str, id_type: str='name')`

<a id="hotpot-plugins-pubchem-utils-get-compounds-cid"></a>
### `get_compounds_cid`

定义 owner `hotpot.plugins.pubchem.utils.get_compounds_cid`；源码 `hotpot/plugins/pubchem/utils.py:20`。

```python
get_compounds_cid(identifier: str, id_type: str='name')
```

<a id="hotpot-plugins-pubchem-utils-async-get-compounds-cid"></a>
### `async_get_compounds_cid`

定义 owner `hotpot.plugins.pubchem.utils.async_get_compounds_cid`；源码 `hotpot/plugins/pubchem/utils.py:28`。

```python
async_get_compounds_cid(identifier: str, id_type: str='name')
```


<a id="hotpot-plugins-qm"></a>
## `hotpot.plugins.qm`

源码 `hotpot/plugins/qm/__init__.py`；SHA-256 `fb7fe7f0eeca3141e3434c439cb05a10b0d4316266adc7cfda7e802d083baa5e`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-qm-gaussian"></a>
## `hotpot.plugins.qm.gaussian`

源码 `hotpot/plugins/qm/gaussian/__init__.py`；SHA-256 `40296d5f692bf564a943151a0f673f77b96df0125e9d94effc9a9e7d22e2eaac`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-qm-gaussian--works"></a>
## `hotpot.plugins.qm.gaussian._works`

源码 `hotpot/plugins/qm/gaussian/_works.py`；SHA-256 `12ebef2625e21aae63da41ca1ad23934e9d3589add2cb93dc8c2069324b5e4cb`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `run_gaussian` → `hotpot.plugins.qm.gaussian._works.run_gaussian`；`run_gaussian(mol: Molecule, link0: str=None, route: str=None, g16root: Union[str, PathLike]=None, gjf_save_path: Union[str, PathLike]=None, log_save_path: Union[str, PathLike]=None, err_save_path: Union[str, PathLike]=None, report_set_resource_error: bool=False, options: Options=None, test: bool=True, **kwargs)`
- `export_results` → `hotpot.plugins.qm.gaussian._works.export_results`；`export_results(*log_file_path: Union[str, PathLike], skip_errors: bool=True, retrieve_mol: bool=True, nproc: int=None, timeout: float=None)`
- `parse_gjf` → `hotpot.plugins.qm.gaussian._works.parse_gjf`；`parse_gjf(path_gjf: Union[str, Path])`
- `reorganize_gjf` → `hotpot.plugins.qm.gaussian._works.reorganize_gjf`；`reorganize_gjf(parsed_gjf: dict)`
- `ladder_opti` → `hotpot.plugins.qm.gaussian._works.ladder_opti`；`ladder_opti(mol: Molecule, ladder: list[str], *args, **kwargs)`
- `update_gjf_coordinates` → `hotpot.plugins.qm.gaussian._works.update_gjf_coordinates`；`update_gjf_coordinates(old_gjf_file: Union[str, PathLike], log_file: Union[str, PathLike])`
- `update_gjf` → `hotpot.plugins.qm.gaussian._works.update_gjf`；`update_gjf(old_gjf_file: Union[str, PathLike], update_dict: dict)`
- `parse_route` → `hotpot.plugins.qm.gaussian._works.parse_route`；`parse_route(route: str) -> dict`
- `ResultsExtract` → `hotpot.plugins.qm.gaussian._works.ResultsExtract`；构造声明 `__init__(self, dir_log: Union[str, PathLike], pass_error: bool=False)`

<a id="hotpot-plugins-qm-gaussian--works-run-gaussian"></a>
### `run_gaussian`

定义 owner `hotpot.plugins.qm.gaussian._works.run_gaussian`；源码 `hotpot/plugins/qm/gaussian/_works.py:29`。

```python
run_gaussian(mol: Molecule, link0: str=None, route: str=None, g16root: Union[str, PathLike]=None, gjf_save_path: Union[str, PathLike]=None, log_save_path: Union[str, PathLike]=None, err_save_path: Union[str, PathLike]=None, report_set_resource_error: bool=False, options: Options=None, test: bool=True, **kwargs)
```

<a id="hotpot-plugins-qm-gaussian--works-export-results"></a>
### `export_results`

定义 owner `hotpot.plugins.qm.gaussian._works.export_results`；源码 `hotpot/plugins/qm/gaussian/_works.py:60`。

```python
export_results(*log_file_path: Union[str, PathLike], skip_errors: bool=True, retrieve_mol: bool=True, nproc: int=None, timeout: float=None)
```

<a id="hotpot-plugins-qm-gaussian--works-parse-gjf"></a>
### `parse_gjf`

定义 owner `hotpot.plugins.qm.gaussian._works.parse_gjf`；源码 `hotpot/plugins/qm/gaussian/_works.py:112`。

```python
parse_gjf(path_gjf: Union[str, Path])
```

<a id="hotpot-plugins-qm-gaussian--works-reorganize-gjf"></a>
### `reorganize_gjf`

定义 owner `hotpot.plugins.qm.gaussian._works.reorganize_gjf`；源码 `hotpot/plugins/qm/gaussian/_works.py:182`。

```python
reorganize_gjf(parsed_gjf: dict)
```

<a id="hotpot-plugins-qm-gaussian--works-ladder-opti"></a>
### `ladder_opti`

定义 owner `hotpot.plugins.qm.gaussian._works.ladder_opti`；源码 `hotpot/plugins/qm/gaussian/_works.py:200`。

```python
ladder_opti(mol: Molecule, ladder: list[str], *args, **kwargs)
```

<a id="hotpot-plugins-qm-gaussian--works-update-gjf-coordinates"></a>
### `update_gjf_coordinates`

定义 owner `hotpot.plugins.qm.gaussian._works.update_gjf_coordinates`；源码 `hotpot/plugins/qm/gaussian/_works.py:204`。

```python
update_gjf_coordinates(old_gjf_file: Union[str, PathLike], log_file: Union[str, PathLike])
```

<a id="hotpot-plugins-qm-gaussian--works-update-gjf"></a>
### `update_gjf`

定义 owner `hotpot.plugins.qm.gaussian._works.update_gjf`；源码 `hotpot/plugins/qm/gaussian/_works.py:213`。

```python
update_gjf(old_gjf_file: Union[str, PathLike], update_dict: dict)
```

<a id="hotpot-plugins-qm-gaussian--works-parse-route"></a>
### `parse_route`

定义 owner `hotpot.plugins.qm.gaussian._works.parse_route`；源码 `hotpot/plugins/qm/gaussian/_works.py:219`。

```python
parse_route(route: str) -> dict
```

<a id="hotpot-plugins-qm-gaussian--works-resultsextract"></a>
### `ResultsExtract`

定义 owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；源码 `hotpot/plugins/qm/gaussian/_works.py:293`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, dir_log: Union[str, PathLike], pass_error: bool=False)`；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:300`
- `extract(self) -> (pd.DataFrame, Molecule)`；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:310`
- `rewrite_route(parsed_route: dict)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:332`
- `extract_log_info(log_path) -> dict` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:358`
- `to_gjf(self, addition: Union[str, list[str]]='\n') -> list[str]`；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:395`

本类字段与枚举值：

- `dir_log` = `Path(dir_log)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:305`
- `attrs` = `['energy', 'zero_point', 'free_energy', 'enthalpy', 'entropy', 'thermal_energy', 'capacity']` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:307`
- `pass_error` = `pass_error` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian._works.ResultsExtract`；`hotpot/plugins/qm/gaussian/_works.py:308`


<a id="hotpot-plugins-qm-gaussian-gauss"></a>
## `hotpot.plugins.qm.gaussian.gauss`

源码 `hotpot/plugins/qm/gaussian/gauss.py`；SHA-256 `92d8d6c7d7506ac62c8f0ed9ecb856c3d7ba4b99de4f39b13f926005abb9c7ea`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `extract_times_from_gaussian_log` → `hotpot.plugins.qm.gaussian.gauss.extract_times_from_gaussian_log`；`extract_times_from_gaussian_log(log_content)`
- `OptionPath` → `hotpot.plugins.qm.gaussian.gauss.OptionPath`；构造声明 `__init__(self, path: str)`
- `Options` → `hotpot.plugins.qm.gaussian.gauss.Options`；构造声明 `__init__(self, path: Union[str, OptionPath], root=None)`
- `Gaussian` → `hotpot.plugins.qm.gaussian.gauss.Gaussian`；构造声明 `__init__(self, g16root: Union[str, os.PathLike]=None, path_gjf: Union[str, os.PathLike]=None, path_log: Union[str, os.PathLike]=None, path_err: Union[str, os.PathLike]=None, report_set_resource_error: bool=False, options: Options=None)`
- `GaussOut` → `hotpot.plugins.qm.gaussian.gauss.GaussOut`；构造声明 `__init__(self, stdout: str, stderr: str=None)`
- `root_dir` → `hotpot.plugins.qm.gaussian.gauss.root_dir`
- `settings` → `hotpot.plugins.qm.gaussian.gauss.settings`

<a id="hotpot-plugins-qm-gaussian-gauss-extract-times-from-gaussian-log"></a>
### `extract_times_from_gaussian_log`

定义 owner `hotpot.plugins.qm.gaussian.gauss.extract_times_from_gaussian_log`；源码 `hotpot/plugins/qm/gaussian/gauss.py:61`。

```python
extract_times_from_gaussian_log(log_content)
```

<a id="hotpot-plugins-qm-gaussian-gauss-optionpath"></a>
### `OptionPath`

定义 owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；源码 `hotpot/plugins/qm/gaussian/gauss.py:89`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, path: str)`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:91`
- `__repr__(self)`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:95`
- `__hash__(self)`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:98`
- `__eq__(self, other)`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:101`
- `__lt__(self, other: 'OptionPath')`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:104`
- `__gt__(self, other: 'OptionPath')`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:107`
- `__len__(self)`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:110`
- `create_from_brief_path(cls, path: str) -> 'OptionPath'` [classmethod]；装饰器 `classmethod`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:148`
- `get_child(self, child_name: str)`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:199`
- `get_normalize_path(path: Union[str, 'OptionPath']) -> str` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:217`
- `brief_nodes(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:121`
- `brief_path(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:125`
- `children(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:140`
- `descendants(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:177`
- `end_name(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:193`
- `is_leaf(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:334`
- `is_root(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:338`
- `is_selection(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:342`
- `nodes(self) -> list[str]` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:346`
- `parent(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:350`
- `pre_selection_path(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:357`
- `subtree(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:369`

本类字段与枚举值：

- `path` = `path` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.OptionPath`；`hotpot/plugins/qm/gaussian/gauss.py:93`

<a id="hotpot-plugins-qm-gaussian-gauss-options"></a>
### `Options`

定义 owner `hotpot.plugins.qm.gaussian.gauss.Options`；源码 `hotpot/plugins/qm/gaussian/gauss.py:381`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, path: Union[str, OptionPath], root=None)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:382`
- `__bool__(self)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:395`
- `__repr__(self)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:398`
- `__hash__(self)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:401`
- `__eq__(self, other)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:404`
- `__dir__(self) -> Iterable[str]`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:407`
- `__getattr__(self, item)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:419`
- `__len__(self)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:427`
- `__call__(self, value=None, *args, **kwargs)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:430`
- `add(self, op: 'Options')`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:457`
- `clear(self)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:476`
- `get_option_dict(self)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:483`
- `parsed_input_to_options(self, parsed_input: dict)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:504`
- `update_parsed_input(self, parsed_input: dict)`；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:525`

本类字段与枚举值：

- `root` = `root if root is not None else self` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:390`
- `value` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:392`
- `path` = `path` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Options`；`hotpot/plugins/qm/gaussian/gauss.py:384`

<a id="hotpot-plugins-qm-gaussian-gauss-gaussian"></a>
### `Gaussian`

定义 owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；源码 `hotpot/plugins/qm/gaussian/gauss.py:533`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, g16root: Union[str, os.PathLike]=None, path_gjf: Union[str, os.PathLike]=None, path_log: Union[str, os.PathLike]=None, path_err: Union[str, os.PathLike]=None, report_set_resource_error: bool=False, options: Options=None)`；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:541`
- `run(self, script: str=None, test: bool=False)`；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:647`
- `extract_g16root_from_bashrc(bashrc_path='~/.bashrc')` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:691`
- `full_option_values(self, title: str, kwd: str=None, op: Any=None, value: Optional=None)`；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:1068`
- `to_conformer(self, cfm_idx=-1, fail_raise=False)`；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:1094`

本类字段与枚举值：

- `envs` = `self._set_environs()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:577`
- `path_gjf` = `path_gjf` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:580`
- `path_log` = `path_log` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:581`
- `path_err` = `path_err` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:582`
- `path_chk` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:585`
- `path_rwf` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:586`
- `parsed_input` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:588`
- `g16process` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:599`
- `stdin` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:601`
- `output` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:602`
- `stdout` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:603`
- `stderr` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:604`
- `g16root` = `Path(g16root)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:569`
- `op` = `Options('')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.Gaussian`；`hotpot/plugins/qm/gaussian/gauss.py:597`

<a id="hotpot-plugins-qm-gaussian-gauss-gaussout"></a>
### `GaussOut`

定义 owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；源码 `hotpot/plugins/qm/gaussian/gauss.py:1126`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, stdout: str, stderr: str=None)`；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1143`
- `read_file(cls, filename: Union[Path, str]) -> 'GaussOut'` [classmethod]；装饰器 `classmethod`；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1148`
- `get_times(self, unit: Literal['day', 'hour', 'minute', 'second']='second')`；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1152`
- `parse(self)`；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1165`
- `update_mol(self, mol)`；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1189`
- `export_mol(self)`；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1232`
- `export_pandas_series(self, name='mol')`；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1235`
- `is_error(self) -> bool` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1240`
- `error_link(self) -> str` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1244`
- `is_hangup_error(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1253`
- `is_opti_convergence_error(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1259`
- `is_scf_convergence_error(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1266`
- `is_scrf_Vdw_cage_error(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1273`
- `is_ZMatrix_error(self)` [读；只读]；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1280`

本类字段与枚举值：

- `stdout` = `stdout` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1144`
- `stderr` = `stderr` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.qm.gaussian.gauss.GaussOut`；`hotpot/plugins/qm/gaussian/gauss.py:1145`

模块值/类型别名（声明存在不构成事实写许可）：

- `root_dir` = `os.path.dirname(__file__)`；`hotpot/plugins/qm/gaussian/gauss.py:27`
- `settings: dict` = `json.load(open(opj(root_dir, 'settings.json')))`；`hotpot/plugins/qm/gaussian/gauss.py:28`


<a id="hotpot-plugins-zeoxx"></a>
## `hotpot.plugins.zeoxx`

源码 `hotpot/plugins/zeoxx/__init__.py`；SHA-256 `b8682dd613412b836ffcf3fcb993385a86d57f8e5ac1cb89ffb49f02d16dfce6`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-zeoxx-core"></a>
## `hotpot.plugins.zeoxx.core`

源码 `hotpot/plugins/zeoxx/core.py`；SHA-256 `879a1e2a1c00570f13158a9db98e32d59cc8a4963fed8e240b889a67254ea6a6`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `ZeoppRunner` → `hotpot.plugins.zeoxx.core.ZeoppRunner`；构造声明 `__init__(self, network_bin: str='network', out_dir: Optional[Union[str, Path]]=None, ha: bool=True, nor: bool=False, radii_file: Optional[Union[str, Path]]=None, mass_file: Optional[Union[str, Path]]=None, strip_atom_names: bool=False, extra_args: Optional[Sequence[str]]=None, timeout: Optional[int]=None, enabled_tasks: Optional[Sequence[str]]=None, units_as_rows: bool=False, cleanup_outputs: bool=False)`
- `concat_results` → `hotpot.plugins.zeoxx.core.concat_results`；`concat_results(results: Dict[str, pd.DataFrame], tasks: Optional[Sequence[str]]=None, keep_units: bool=True, chan_mode: str='expand', chan_agg: str='max', include_context: bool=False, add_task_suffix_on_collision: bool=True, sort_rows: bool=True) -> pd.DataFrame`
- `NON_PSD_TASKS` → `hotpot.plugins.zeoxx.core.NON_PSD_TASKS`
- `KEY_COLS` → `hotpot.plugins.zeoxx.core.KEY_COLS`
- `CONTEXT_COLS` → `hotpot.plugins.zeoxx.core.CONTEXT_COLS`

<a id="hotpot-plugins-zeoxx-core-zeopprunner"></a>
### `ZeoppRunner`

定义 owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；源码 `hotpot/plugins/zeoxx/core.py:232`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, network_bin: str='network', out_dir: Optional[Union[str, Path]]=None, ha: bool=True, nor: bool=False, radii_file: Optional[Union[str, Path]]=None, mass_file: Optional[Union[str, Path]]=None, strip_atom_names: bool=False, extra_args: Optional[Sequence[str]]=None, timeout: Optional[int]=None, enabled_tasks: Optional[Sequence[str]]=None, units_as_rows: bool=False, cleanup_outputs: bool=False)`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:238`
- `run_res(self, structure: Union[str, Path], _apply_units: bool=True) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:395`
- `run_chan(self, structure: Union[str, Path], probe_radius: float, _apply_units: bool=True) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:408`
- `run_sa(self, structure: Union[str, Path], probe_radius: float, samples_per_atom: int, chan_radius: Optional[float]=None, _apply_units: bool=True) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:441`
- `run_vol(self, structure: Union[str, Path], probe_radius: float, samples_per_uc: int, chan_radius: Optional[float]=None, _apply_units: bool=True) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:469`
- `run_volpo(self, structure: Union[str, Path], probe_radius: float, samples_per_uc: int, chan_radius: Optional[float]=None, _apply_units: bool=True) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:497`
- `run_psd(self, structure: Union[str, Path], probe_radius: float, samples_per_uc: int, chan_radius: Optional[float]=None, _apply_units: bool=True) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:525`
- `batch_res(self, structures: Sequence[Union[str, Path]]) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:559`
- `batch_chan(self, structures: Sequence[Union[str, Path]], probe_radii: Sequence[float]) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:564`
- `batch_sa(self, structures: Sequence[Union[str, Path]], probe_radii: Sequence[float], samples_per_atom: int, chan_radius: Optional[float]=None) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:572`
- `batch_vol(self, structures: Sequence[Union[str, Path]], probe_radii: Sequence[float], samples_per_uc: int, chan_radius: Optional[float]=None) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:586`
- `batch_volpo(self, structures: Sequence[Union[str, Path]], probe_radii: Sequence[float], samples_per_uc: int, chan_radius: Optional[float]=None) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:600`
- `batch_psd(self, structures: Sequence[Union[str, Path]], probe_radii: Sequence[float], samples_per_uc: int, chan_radius: Optional[float]=None) -> pd.DataFrame`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:614`
- `concat_list_results(self, lst_res: List[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:631`
- `summarize(self, structures: Sequence[Union[str, Path]], probe_radii: Sequence[float], sa_samples_per_atom: int=2000, vol_samples_per_uc: int=50000, psd_samples_per_uc: Optional[int]=None, chan_radius_for_sa_vol: Optional[float]=None, nproc: Optional[int]=1) -> Dict[str, pd.DataFrame]`；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:714`

本类字段与枚举值：

- `network_bin` = `resolved` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:278`
- `out_dir` = `_ensure_out_dir(out_dir)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:279`
- `ha` = `ha` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:280`
- `nor` = `nor` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:281`
- `radii_file` = `Path(radii_file) if radii_file else None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:282`
- `mass_file` = `Path(mass_file) if mass_file else None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:283`
- `strip_atom_names` = `strip_atom_names` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:284`
- `extra_args` = `list(extra_args) if extra_args else []` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:285`
- `timeout` = `timeout` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:286`
- `units_as_rows` = `bool(units_as_rows)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:299`
- `cleanup_outputs` = `bool(cleanup_outputs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:300`
- `enabled_tasks` = `set(default_tasks)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.zeoxx.core.ZeoppRunner`；`hotpot/plugins/zeoxx/core.py:291`

<a id="hotpot-plugins-zeoxx-core-concat-results"></a>
### `concat_results`

定义 owner `hotpot.plugins.zeoxx.core.concat_results`；源码 `hotpot/plugins/zeoxx/core.py:1033`。

```python
concat_results(results: Dict[str, pd.DataFrame], tasks: Optional[Sequence[str]]=None, keep_units: bool=True, chan_mode: str='expand', chan_agg: str='max', include_context: bool=False, add_task_suffix_on_collision: bool=True, sort_rows: bool=True) -> pd.DataFrame
```

模块值/类型别名（声明存在不构成事实写许可）：

- `NON_PSD_TASKS` = `('res', 'chan', 'sa', 'vol', 'volpo')`；`hotpot/plugins/zeoxx/core.py:745`
- `KEY_COLS` = `('structure', 'probe_A')`；`hotpot/plugins/zeoxx/core.py:748`
- `CONTEXT_COLS` = `('samples_per_atom', 'samples_per_uc', 'chan_radius_A', 'dimensionality')`；`hotpot/plugins/zeoxx/core.py:751`
