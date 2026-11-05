# 辅助 ComplexFormer：完整静态声明

库存基准提交 `2c2ef889abb6adcceba043b606da2e718dbea109`；日期 `2026-11-05`。逐模块 SHA-256 保留如下。

仅按现有 AST 库存展开非私有本地定义、显式导出目标、声明字段、property、常用 magic 方法及可静态解析的本地继承。参数拼写、默认值与注解采用库存中的源码 AST 表达式，未经运行时求值。

这份参考不声称辅助模块已在 Plan20 中重新科学验收。私有模块内的非私有名称仍可能是内部实现；直接模块可访问面与显式 __all__ 分开列出，不扩大批准的公共合同。

property 可写指源码声明 setter；只读 getter 不保证返回对象不可变。类声明值、self 赋值字段和 frozen dataclass 权限分别标示，均不自动授予权威化学事实写权限。

未穷尽第三方继承成员、运行时安装属性、元类生成行为或可选依赖实际可用性。静态解析的基类名称不保证是可导入的 owner，特别是内置类及泛型表达式；原始基类拼写另列。

[辅助入口与类索引](auxiliary_index.md) · [主交付树](../delivery_api_tree.md) · [完整机器库存](README.md#raw-inventory) · [主内核动态权限](dynamic_permissions.md)

## 模块导航

- [`hotpot.plugins.ComplexFormer`](#hotpot-plugins-complexformer)
- [`hotpot.plugins.ComplexFormer.callbacks`](#hotpot-plugins-complexformer-callbacks)
- [`hotpot.plugins.ComplexFormer.check`](#hotpot-plugins-complexformer-check)
- [`hotpot.plugins.ComplexFormer.config_task`](#hotpot-plugins-complexformer-config-task)
- [`hotpot.plugins.ComplexFormer.configs`](#hotpot-plugins-complexformer-configs)
- [`hotpot.plugins.ComplexFormer.data`](#hotpot-plugins-complexformer-data)
- [`hotpot.plugins.ComplexFormer.data.collate`](#hotpot-plugins-complexformer-data-collate)
- [`hotpot.plugins.ComplexFormer.data.data`](#hotpot-plugins-complexformer-data-data)
- [`hotpot.plugins.ComplexFormer.data.data_module`](#hotpot-plugins-complexformer-data-data-module)
- [`hotpot.plugins.ComplexFormer.data.dataset`](#hotpot-plugins-complexformer-data-dataset)
- [`hotpot.plugins.ComplexFormer.data.loader`](#hotpot-plugins-complexformer-data-loader)
- [`hotpot.plugins.ComplexFormer.data_process`](#hotpot-plugins-complexformer-data-process)
- [`hotpot.plugins.ComplexFormer.data_process.ccdc`](#hotpot-plugins-complexformer-data-process-ccdc)
- [`hotpot.plugins.ComplexFormer.data_process.normalize_data`](#hotpot-plugins-complexformer-data-process-normalize-data)
- [`hotpot.plugins.ComplexFormer.data_process.sc_logk`](#hotpot-plugins-complexformer-data-process-sc-logk)
- [`hotpot.plugins.ComplexFormer.data_process.sc_logk.core`](#hotpot-plugins-complexformer-data-process-sc-logk-core)
- [`hotpot.plugins.ComplexFormer.data_process.sc_logk.data`](#hotpot-plugins-complexformer-data-process-sc-logk-data)
- [`hotpot.plugins.ComplexFormer.models`](#hotpot-plugins-complexformer-models)
- [`hotpot.plugins.ComplexFormer.models._core`](#hotpot-plugins-complexformer-models--core)
- [`hotpot.plugins.ComplexFormer.models.attn`](#hotpot-plugins-complexformer-models-attn)
- [`hotpot.plugins.ComplexFormer.models.cm`](#hotpot-plugins-complexformer-models-cm)
- [`hotpot.plugins.ComplexFormer.models.core`](#hotpot-plugins-complexformer-models-core)
- [`hotpot.plugins.ComplexFormer.models.core._utils`](#hotpot-plugins-complexformer-models-core--utils)
- [`hotpot.plugins.ComplexFormer.models.core.attn_core`](#hotpot-plugins-complexformer-models-core-attn-core)
- [`hotpot.plugins.ComplexFormer.models.core.base`](#hotpot-plugins-complexformer-models-core-base)
- [`hotpot.plugins.ComplexFormer.models.core.cloud`](#hotpot-plugins-complexformer-models-core-cloud)
- [`hotpot.plugins.ComplexFormer.models.core.envs_encoder`](#hotpot-plugins-complexformer-models-core-envs-encoder)
- [`hotpot.plugins.ComplexFormer.models.core.graph`](#hotpot-plugins-complexformer-models-core-graph)
- [`hotpot.plugins.ComplexFormer.models.core.graph_core`](#hotpot-plugins-complexformer-models-core-graph-core)
- [`hotpot.plugins.ComplexFormer.models.core.node_processor`](#hotpot-plugins-complexformer-models-core-node-processor)
- [`hotpot.plugins.ComplexFormer.models.core.se3net`](#hotpot-plugins-complexformer-models-core-se3net)
- [`hotpot.plugins.ComplexFormer.models.graph`](#hotpot-plugins-complexformer-models-graph)
- [`hotpot.plugins.ComplexFormer.models.loss`](#hotpot-plugins-complexformer-models-loss)
- [`hotpot.plugins.ComplexFormer.models.metrics`](#hotpot-plugins-complexformer-models-metrics)
- [`hotpot.plugins.ComplexFormer.models.plot_makers`](#hotpot-plugins-complexformer-models-plot-makers)
- [`hotpot.plugins.ComplexFormer.models.predictor`](#hotpot-plugins-complexformer-models-predictor)
- [`hotpot.plugins.ComplexFormer.models.utils`](#hotpot-plugins-complexformer-models-utils)
- [`hotpot.plugins.ComplexFormer.optim_config`](#hotpot-plugins-complexformer-optim-config)
- [`hotpot.plugins.ComplexFormer.run`](#hotpot-plugins-complexformer-run)
- [`hotpot.plugins.ComplexFormer.run.cv`](#hotpot-plugins-complexformer-run-cv)
- [`hotpot.plugins.ComplexFormer.run.datacls`](#hotpot-plugins-complexformer-run-datacls)
- [`hotpot.plugins.ComplexFormer.run.run`](#hotpot-plugins-complexformer-run-run)
- [`hotpot.plugins.ComplexFormer.run.run_tools`](#hotpot-plugins-complexformer-run-run-tools)
- [`hotpot.plugins.ComplexFormer.run.train`](#hotpot-plugins-complexformer-run-train)
- [`hotpot.plugins.ComplexFormer.tasks`](#hotpot-plugins-complexformer-tasks)
- [`hotpot.plugins.ComplexFormer.tasks.tasks`](#hotpot-plugins-complexformer-tasks-tasks)
- [`hotpot.plugins.ComplexFormer.tools`](#hotpot-plugins-complexformer-tools)
- [`hotpot.plugins.ComplexFormer.types`](#hotpot-plugins-complexformer-types)

<a id="hotpot-plugins-complexformer"></a>
## `hotpot.plugins.ComplexFormer`

源码 `hotpot/plugins/ComplexFormer/__init__.py`；SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-complexformer-callbacks"></a>
## `hotpot.plugins.ComplexFormer.callbacks`

源码 `hotpot/plugins/ComplexFormer/callbacks.py`；SHA-256 `7e6a9607f1c55ba24ae2c3b8d703607e5287e7e01b40e57a3c14508eed7d7591`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `get_inspect_table` → `hotpot.plugins.ComplexFormer.callbacks.get_inspect_table`；`get_inspect_table(inspect_dict, table_kw: dict=None, title: str=None)`
- `Pbar` → `hotpot.plugins.ComplexFormer.callbacks.Pbar`；构造声明 `__init__(self)`
- `Debugger` → `hotpot.plugins.ComplexFormer.callbacks.Debugger`

<a id="hotpot-plugins-complexformer-callbacks-get-inspect-table"></a>
### `get_inspect_table`

定义 owner `hotpot.plugins.ComplexFormer.callbacks.get_inspect_table`；源码 `hotpot/plugins/ComplexFormer/callbacks.py:18`。

```python
get_inspect_table(inspect_dict, table_kw: dict=None, title: str=None)
```

<a id="hotpot-plugins-complexformer-callbacks-pbar"></a>
### `Pbar`

定义 owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；源码 `hotpot/plugins/ComplexFormer/callbacks.py:47`。

基类原始声明：`ProgressBar`；装饰器：无。

本类属性与方法：

- `__init__(self)`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:48`
- `disable(self)`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:56`
- `init_train_tqdm(self)`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:59`
- `end_liver(self)`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:70`
- `on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:73`
- `on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:83`
- `on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:103`
- `on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:119`
- `on_sanity_check_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:124`
- `on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:128`
- `on_sanity_check_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:145`

本类字段与枚举值：

- `buf` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:50`
- `train_pbar` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:51`
- `layout` = `Layout()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:52`
- `liver` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.callbacks.Pbar`；`hotpot/plugins/ComplexFormer/callbacks.py:54`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`lightning.pytorch.callbacks.ProgressBar`。

<a id="hotpot-plugins-complexformer-callbacks-debugger"></a>
### `Debugger`

定义 owner `hotpot.plugins.ComplexFormer.callbacks.Debugger`；源码 `hotpot/plugins/ComplexFormer/callbacks.py:149`。

基类原始声明：`Callback`；装饰器：无。

本类属性与方法：

- `on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`；owner `hotpot.plugins.ComplexFormer.callbacks.Debugger`；`hotpot/plugins/ComplexFormer/callbacks.py:150`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`lightning.pytorch.callbacks.Callback`。


<a id="hotpot-plugins-complexformer-check"></a>
## `hotpot.plugins.ComplexFormer.check`

源码 `hotpot/plugins/ComplexFormer/check.py`；SHA-256 `a47e29b5251af099b30150b45246f35446c835e6856a8fcba749ea45e901eb1d`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `check_trainable_parameters` → `hotpot.plugins.ComplexFormer.check.check_trainable_parameters`；`check_trainable_parameters(model: nn.Module)`
- `check_all_parameters_status` → `hotpot.plugins.ComplexFormer.check.check_all_parameters_status`；`check_all_parameters_status(model: nn.Module)`
- `check_gradient_values` → `hotpot.plugins.ComplexFormer.check.check_gradient_values`；`check_gradient_values(model: nn.Module)`
- `NAME_LEN` → `hotpot.plugins.ComplexFormer.check.NAME_LEN`

<a id="hotpot-plugins-complexformer-check-check-trainable-parameters"></a>
### `check_trainable_parameters`

定义 owner `hotpot.plugins.ComplexFormer.check.check_trainable_parameters`；源码 `hotpot/plugins/ComplexFormer/check.py:29`。

```python
check_trainable_parameters(model: nn.Module)
```

<a id="hotpot-plugins-complexformer-check-check-all-parameters-status"></a>
### `check_all_parameters_status`

定义 owner `hotpot.plugins.ComplexFormer.check.check_all_parameters_status`；源码 `hotpot/plugins/ComplexFormer/check.py:64`。

```python
check_all_parameters_status(model: nn.Module)
```

<a id="hotpot-plugins-complexformer-check-check-gradient-values"></a>
### `check_gradient_values`

定义 owner `hotpot.plugins.ComplexFormer.check.check_gradient_values`；源码 `hotpot/plugins/ComplexFormer/check.py:86`。

```python
check_gradient_values(model: nn.Module)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `NAME_LEN` = `100`；`hotpot/plugins/ComplexFormer/check.py:12`


<a id="hotpot-plugins-complexformer-config-task"></a>
## `hotpot.plugins.ComplexFormer.config_task`

源码 `hotpot/plugins/ComplexFormer/config_task.py`；SHA-256 `0b5a9843edd070803e08c7746aa61037e3228c6fe441d7814afc1c619af9503f`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `config_task` → `hotpot.plugins.ComplexFormer.config_task.config_task`；`config_task(cfg: ConfigArgs)`
- `wrap_loss_fn` → `hotpot.plugins.ComplexFormer.config_task.wrap_loss_fn`；`wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable`
- `TaskComponents` → `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；构造声明 `TaskComponents(task_names: list[str], target_getter: dict[str, tools.TargetGetter], feature_extractor: dict[str, M.AttnExtractor], predictors: dict[str, M.Predictor], loss_fn: dict[str, tp.LossFn], primary_metrics: dict[str, str], metrics: dict[str, M.Metrics], plot_makers: dict, x_masker: Optional[Callable], mask_need_task: Sequence[str], loss_weight_calculator: Optional[Callable], inputs_preprocessor: Callable, batch_preprocessor: Any, inputs_getter: Any, xyz_index: Iterable[int], xyz_perturb_mode: str, xyz_perturb_sigma: float, hypers: ParamSets, with_sol: bool, with_med: bool, with_env: bool, sol_graph_inputs: Iterable[str], med_graph_inputs: Iterable[str], kwargs: dict)`
- `TaskBuilder` → `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；构造声明 `__init__(self, cfg: ConfigArgs)`
- `TargetTypeOption` → `hotpot.plugins.ComplexFormer.config_task.TargetTypeOption`
- `MetricType` → `hotpot.plugins.ComplexFormer.config_task.MetricType`
- `metrics_name_convert` → `hotpot.plugins.ComplexFormer.config_task.metrics_name_convert`
- `metrics_options` → `hotpot.plugins.ComplexFormer.config_task.metrics_options`
- `predictor_plot_maker_map` → `hotpot.plugins.ComplexFormer.config_task.predictor_plot_maker_map`
- `loss_options` → `hotpot.plugins.ComplexFormer.config_task.loss_options`
- `x_masker_options` → `hotpot.plugins.ComplexFormer.config_task.x_masker_options`
- `INPUT_X_ATTR` → `hotpot.plugins.ComplexFormer.config_task.INPUT_X_ATTR`
- `COORD_X_ATTR` → `hotpot.plugins.ComplexFormer.config_task.COORD_X_ATTR`

<a id="hotpot-plugins-complexformer-config-task-config-task"></a>
### `config_task`

定义 owner `hotpot.plugins.ComplexFormer.config_task.config_task`；源码 `hotpot/plugins/ComplexFormer/config_task.py:79`。

```python
config_task(cfg: ConfigArgs)
```

<a id="hotpot-plugins-complexformer-config-task-wrap-loss-fn"></a>
### `wrap_loss_fn`

定义 owner `hotpot.plugins.ComplexFormer.config_task.wrap_loss_fn`；源码 `hotpot/plugins/ComplexFormer/config_task.py:248`。

```python
wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable
```

<a id="hotpot-plugins-complexformer-config-task-taskcomponents"></a>
### `TaskComponents`

定义 owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；源码 `hotpot/plugins/ComplexFormer/config_task.py:258`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
TaskComponents(task_names: list[str], target_getter: dict[str, tools.TargetGetter], feature_extractor: dict[str, M.AttnExtractor], predictors: dict[str, M.Predictor], loss_fn: dict[str, tp.LossFn], primary_metrics: dict[str, str], metrics: dict[str, M.Metrics], plot_makers: dict, x_masker: Optional[Callable], mask_need_task: Sequence[str], loss_weight_calculator: Optional[Callable], inputs_preprocessor: Callable, batch_preprocessor: Any, inputs_getter: Any, xyz_index: Iterable[int], xyz_perturb_mode: str, xyz_perturb_sigma: float, hypers: ParamSets, with_sol: bool, with_med: bool, with_env: bool, sol_graph_inputs: Iterable[str], med_graph_inputs: Iterable[str], kwargs: dict)
```

本类属性与方法：

- `to_dict(self) -> dict[str, Any]`；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:288`

本类字段与枚举值：

- `task_names: list[str]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:263`
- `target_getter: dict[str, tools.TargetGetter]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:264`
- `feature_extractor: dict[str, M.AttnExtractor]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:265`
- `predictors: dict[str, M.Predictor]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:266`
- `loss_fn: dict[str, tp.LossFn]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:267`
- `primary_metrics: dict[str, str]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:268`
- `metrics: dict[str, M.Metrics]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:269`
- `plot_makers: dict` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:270`
- `x_masker: Optional[Callable]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:271`
- `mask_need_task: Sequence[str]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:272`
- `loss_weight_calculator: Optional[Callable]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:273`
- `inputs_preprocessor: Callable` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:274`
- `batch_preprocessor: Any` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:275`
- `inputs_getter: Any` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:276`
- `xyz_index: Iterable[int]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:277`
- `xyz_perturb_mode: str` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:278`
- `xyz_perturb_sigma: float` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:279`
- `hypers: ParamSets` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:280`
- `with_sol: bool` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:281`
- `with_med: bool` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:282`
- `with_env: bool` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:283`
- `sol_graph_inputs: Iterable[str]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:284`
- `med_graph_inputs: Iterable[str]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:285`
- `kwargs: dict` [可写值字段]；owner `hotpot.plugins.ComplexFormer.config_task.TaskComponents`；`hotpot/plugins/ComplexFormer/config_task.py:286`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-complexformer-config-task-taskbuilder"></a>
### `TaskBuilder`

定义 owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；源码 `hotpot/plugins/ComplexFormer/config_task.py:636`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, cfg: ConfigArgs)`；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:640`
- `build_task(self) -> tuple[M.CoreBase, tasks.BaseTask, dict[str, M.Predictor]]`；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:802`

本类字段与枚举值：

- `INPUT_X_ATTR` = `('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:637`
- `COORD_X_ATTR` = `('x', 'y', 'z')` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:638`
- `core` = `self._init_core(cfg.hypers)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:642`
- `task_type` = `self._specify_task_type(cfg)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:643`
- `cfg` = `cfg` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:644`
- `first_data` = `cfg.dataModule.first_data` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:646`
- `xyz_index` = `self._specify_xyz_index(self.first_data, cfg.with_xyz)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:647`
- `builder` = `_Builder(cfg, self.core, self.first_data, self.xyz_index)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.config_task.TaskBuilder`；`hotpot/plugins/ComplexFormer/config_task.py:651`

模块值/类型别名（声明存在不构成事实写许可）：

- `TargetTypeOption` = `('xyz', 'onehot', 'binary', 'num')`；`hotpot/plugins/ComplexFormer/config_task.py:33`
- `MetricType` = `Literal['r2score', 'rmse', 'mse', 'mae', 'accuracy', 'binary_accuracy', 'metal_accuracy']`；`hotpot/plugins/ComplexFormer/config_task.py:34`
- `metrics_name_convert` = `{'r2score': 'r2', 'accuracy': 'acc', 'metal_accuracy': 'macc', 'binary_accuracy': 'bacc'}`；`hotpot/plugins/ComplexFormer/config_task.py:35`
- `metrics_options` = <长声明表达式；完整值见库存及源码位置>；`hotpot/plugins/ComplexFormer/config_task.py:41`
- `predictor_plot_maker_map` = `{'num': ('r2',), 'onehot': ('conf', 'mroc'), 'binary': ('bconf', 'roc', 'det', 'prc'), 'xyz': ('hist',)}`；`hotpot/plugins/ComplexFormer/config_task.py:56`
- `loss_options` = `{'mse': F.mse_loss, 'cross_entropy': M.LossMethods.calc_atom_type_loss, 'binary_cross_entropy': F.binary_cross_entropy_with_logits, 'amd': M.LossMethods.average_maximum_displacement}`；`hotpot/plugins/ComplexFormer/config_task.py:63`
- `x_masker_options` = `{'atom': M.mask_atom_type, 'metal': M.mask_atom_type}`；`hotpot/plugins/ComplexFormer/config_task.py:69`
- `INPUT_X_ATTR` = `('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')`；`hotpot/plugins/ComplexFormer/config_task.py:75`
- `COORD_X_ATTR` = `('x', 'y', 'z')`；`hotpot/plugins/ComplexFormer/config_task.py:76`


<a id="hotpot-plugins-complexformer-configs"></a>
## `hotpot.plugins.ComplexFormer.configs`

源码 `hotpot/plugins/ComplexFormer/configs.py`；SHA-256 `191e7de615e6e0ce96e3b3de60329b7ffe52146073332ac6945da8998855252e`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `wrap_loss_fn` → `hotpot.plugins.ComplexFormer.configs.wrap_loss_fn`；`wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable`
- `config_tasks_from_multi_datasets` → `hotpot.plugins.ComplexFormer.configs.config_tasks_from_multi_datasets`；`config_tasks_from_multi_datasets(work_name: str, dataset_counts: int, inputs_getter: Callable, first_data: list[Data], core: M.CoreBase, predictor: M.Predictor, feature_extractor: list[dict[str, Callable]], target_getter: list[dict[str, tp.TargetGetter]], loss_fn: list[dict[str, tp.LossFn]], primary_metrics: list[dict[str, str]], other_metrics: Union[dict[str, Union[str, Iterable[str], tp.MetricFn]], list[tp.OtherMetricConfig]], hypers: Union[ParamSets, list[ParamSets]], task_names: Optional[list[Sequence[str]]]=None, batch_preprocessor: Optional[list[tp.BatchPreProcessor]]=None, inputs_preprocessor: Optional[list[Callable]]=None, xyz_index: Optional[list[Iterable[int]]]=None, with_sol: Optional[Union[bool, Iterable[bool]]]=None, with_med: Optional[Union[bool, Iterable[bool]]]=None, with_env: Optional[Union[bool, Iterable[bool]]]=None, xyz_perturb_sigma: Optional[list[float]]=None, xyz_perturb_mode: M.PerturbMode='uniform', extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]]=None, loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]]=None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='inverse-count', loss_fn_wrap_tasks: Optional[list[str, set[str]]]=None, onehot_types: Optional[list[dict[str, int]]]=None, x_masker: Optional[list[tp.XMasker]]=None, mask_need_task: Optional[list[list[str]]]=None, **kwargs)`
- `config` → `hotpot.plugins.ComplexFormer.configs.config`；`config(work_name: str, task_names: Union[list[list], list[str]], task_type: type, dataModule: DataModule, inputs_getter: Callable, core: M.CoreBase, predictor: M.Predictor, feature_extractor, target_getter, loss_fn, primary_metrics, other_metrics: Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]], hypers: ParamSets, batch_preprocessor: Optional[list[tp.BatchPreProcessor]]=None, inputs_preprocessor: Optional[list[Callable]]=None, with_xyz: Optional[Union[bool, Iterable[bool]]]=None, with_sol: Optional[Union[bool, Iterable[bool]]]=None, with_med: Optional[Union[bool, Iterable[bool]]]=None, with_env: Optional[Union[bool, Iterable[bool]]]=None, xyz_perturb_sigma: Optional[list[float]]=None, xyz_perturb_mode: M.PerturbMode='uniform', extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]]=None, loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]]=None, loss_fn_wrap_tasks: Optional[Union[bool, set[str], Iterable[set[str]]]]=None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='inverse-count', onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]]=None, x_masker: Optional[list[tp.XMasker]]=None, mask_need_task: Optional[list[list[str]]]=None, **kwargs)`
- `TargetTypeOption` → `hotpot.plugins.ComplexFormer.configs.TargetTypeOption`
- `MetricType` → `hotpot.plugins.ComplexFormer.configs.MetricType`
- `metrics_name_convert` → `hotpot.plugins.ComplexFormer.configs.metrics_name_convert`
- `metrics_options` → `hotpot.plugins.ComplexFormer.configs.metrics_options`
- `predictor_plot_maker_map` → `hotpot.plugins.ComplexFormer.configs.predictor_plot_maker_map`
- `loss_options` → `hotpot.plugins.ComplexFormer.configs.loss_options`
- `x_masker_options` → `hotpot.plugins.ComplexFormer.configs.x_masker_options`
- `INPUT_X_ATTR` → `hotpot.plugins.ComplexFormer.configs.INPUT_X_ATTR`
- `COORD_X_ATTR` → `hotpot.plugins.ComplexFormer.configs.COORD_X_ATTR`

<a id="hotpot-plugins-complexformer-configs-wrap-loss-fn"></a>
### `wrap_loss_fn`

定义 owner `hotpot.plugins.ComplexFormer.configs.wrap_loss_fn`；源码 `hotpot/plugins/ComplexFormer/configs.py:465`。

```python
wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable
```

<a id="hotpot-plugins-complexformer-configs-config-tasks-from-multi-datasets"></a>
### `config_tasks_from_multi_datasets`

定义 owner `hotpot.plugins.ComplexFormer.configs.config_tasks_from_multi_datasets`；源码 `hotpot/plugins/ComplexFormer/configs.py:897`。

```python
config_tasks_from_multi_datasets(work_name: str, dataset_counts: int, inputs_getter: Callable, first_data: list[Data], core: M.CoreBase, predictor: M.Predictor, feature_extractor: list[dict[str, Callable]], target_getter: list[dict[str, tp.TargetGetter]], loss_fn: list[dict[str, tp.LossFn]], primary_metrics: list[dict[str, str]], other_metrics: Union[dict[str, Union[str, Iterable[str], tp.MetricFn]], list[tp.OtherMetricConfig]], hypers: Union[ParamSets, list[ParamSets]], task_names: Optional[list[Sequence[str]]]=None, batch_preprocessor: Optional[list[tp.BatchPreProcessor]]=None, inputs_preprocessor: Optional[list[Callable]]=None, xyz_index: Optional[list[Iterable[int]]]=None, with_sol: Optional[Union[bool, Iterable[bool]]]=None, with_med: Optional[Union[bool, Iterable[bool]]]=None, with_env: Optional[Union[bool, Iterable[bool]]]=None, xyz_perturb_sigma: Optional[list[float]]=None, xyz_perturb_mode: M.PerturbMode='uniform', extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]]=None, loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]]=None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='inverse-count', loss_fn_wrap_tasks: Optional[list[str, set[str]]]=None, onehot_types: Optional[list[dict[str, int]]]=None, x_masker: Optional[list[tp.XMasker]]=None, mask_need_task: Optional[list[list[str]]]=None, **kwargs)
```

<a id="hotpot-plugins-complexformer-configs-config"></a>
### `config`

定义 owner `hotpot.plugins.ComplexFormer.configs.config`；源码 `hotpot/plugins/ComplexFormer/configs.py:1019`。

```python
config(work_name: str, task_names: Union[list[list], list[str]], task_type: type, dataModule: DataModule, inputs_getter: Callable, core: M.CoreBase, predictor: M.Predictor, feature_extractor, target_getter, loss_fn, primary_metrics, other_metrics: Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]], hypers: ParamSets, batch_preprocessor: Optional[list[tp.BatchPreProcessor]]=None, inputs_preprocessor: Optional[list[Callable]]=None, with_xyz: Optional[Union[bool, Iterable[bool]]]=None, with_sol: Optional[Union[bool, Iterable[bool]]]=None, with_med: Optional[Union[bool, Iterable[bool]]]=None, with_env: Optional[Union[bool, Iterable[bool]]]=None, xyz_perturb_sigma: Optional[list[float]]=None, xyz_perturb_mode: M.PerturbMode='uniform', extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]]=None, loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]]=None, loss_fn_wrap_tasks: Optional[Union[bool, set[str], Iterable[set[str]]]]=None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='inverse-count', onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]]=None, x_masker: Optional[list[tp.XMasker]]=None, mask_need_task: Optional[list[list[str]]]=None, **kwargs)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `TargetTypeOption` = `('xyz', 'onehot', 'binary', 'num')`；`hotpot/plugins/ComplexFormer/configs.py:23`
- `MetricType` = `Literal['r2score', 'rmse', 'mse', 'mae', 'accuracy', 'binary_accuracy', 'metal_accuracy']`；`hotpot/plugins/ComplexFormer/configs.py:24`
- `metrics_name_convert` = `{'r2score': 'r2', 'accuracy': 'acc', 'metal_accuracy': 'macc', 'binary_accuracy': 'bacc'}`；`hotpot/plugins/ComplexFormer/configs.py:25`
- `metrics_options` = <长声明表达式；完整值见库存及源码位置>；`hotpot/plugins/ComplexFormer/configs.py:31`
- `predictor_plot_maker_map` = `{'num': ('r2',), 'onehot': ('conf', 'mroc'), 'binary': ('bconf', 'roc', 'det', 'prc'), 'xyz': ('hist',)}`；`hotpot/plugins/ComplexFormer/configs.py:46`
- `loss_options` = `{'mse': F.mse_loss, 'cross_entropy': M.LossMethods.calc_atom_type_loss, 'binary_cross_entropy': F.binary_cross_entropy_with_logits, 'amd': M.LossMethods.average_maximum_displacement}`；`hotpot/plugins/ComplexFormer/configs.py:53`
- `x_masker_options` = `{'atom': M.mask_atom_type, 'metal': M.mask_atom_type}`；`hotpot/plugins/ComplexFormer/configs.py:59`
- `INPUT_X_ATTR` = `('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')`；`hotpot/plugins/ComplexFormer/configs.py:65`
- `COORD_X_ATTR` = `('x', 'y', 'z')`；`hotpot/plugins/ComplexFormer/configs.py:66`


<a id="hotpot-plugins-complexformer-data"></a>
## `hotpot.plugins.ComplexFormer.data`

源码 `hotpot/plugins/ComplexFormer/data/__init__.py`；SHA-256 `09962e1657a4e11d870cc4bf2243c40fdf0c612bc3198e288840abc86b7c6999`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-complexformer-data-collate"></a>
## `hotpot.plugins.ComplexFormer.data.collate`

源码 `hotpot/plugins/ComplexFormer/data/collate.py`；SHA-256 `a1b7e7281332136933c78e94dd324e9f3a7a3f8825f70a60d730c86d9173604c`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Collater` → `hotpot.plugins.ComplexFormer.data.collate.Collater`；构造声明 `__init__(self, dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter], follow_batch: Optional[List[str]]=None, exclude_keys: Optional[List[str]]=None)`

<a id="hotpot-plugins-complexformer-data-collate-collater"></a>
### `Collater`

定义 owner `hotpot.plugins.ComplexFormer.data.collate.Collater`；源码 `hotpot/plugins/ComplexFormer/data/collate.py:10`。

基类原始声明：`PyGCollater`；装饰器：无。

本类属性与方法：

- `__init__(self, dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter], follow_batch: Optional[List[str]]=None, exclude_keys: Optional[List[str]]=None)`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.data.collate.Collater`；`hotpot/plugins/ComplexFormer/data/collate.py:12`
- `__call__(self, batch: List[Any]) -> Any`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.data.collate.Collater`；`hotpot/plugins/ComplexFormer/data/collate.py:33`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch_geometric.loader.dataloader.Collater`。


<a id="hotpot-plugins-complexformer-data-data"></a>
## `hotpot.plugins.ComplexFormer.data.data`

源码 `hotpot/plugins/ComplexFormer/data/data.py`；SHA-256 `bb51529caf1fce4b6d45cb461b5f3606da7be4caab32d621241676b5fcb0a8d1`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `ExtractionData` → `hotpot.plugins.ComplexFormer.data.data.ExtractionData`

<a id="hotpot-plugins-complexformer-data-data-extractiondata"></a>
### `ExtractionData`

定义 owner `hotpot.plugins.ComplexFormer.data.data.ExtractionData`；源码 `hotpot/plugins/ComplexFormer/data/data.py:22`。

基类原始声明：`Data`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch_geometric.data.Data`。


<a id="hotpot-plugins-complexformer-data-data-module"></a>
## `hotpot.plugins.ComplexFormer.data.data_module`

源码 `hotpot/plugins/ComplexFormer/data/data_module.py`；SHA-256 `66b2461f44152bc3bfed817db37a4249315398b004d5f488aa16a8c909c200ce`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `get_first_data` → `hotpot.plugins.ComplexFormer.data.data_module.get_first_data`；`get_first_data(dir_datasets: str) -> dict[str, Data]`
- `DataModule` → `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；构造声明 `__init__(self, args_cfg: datacls.DataModuleArgs)`

<a id="hotpot-plugins-complexformer-data-data-module-get-first-data"></a>
### `get_first_data`

定义 owner `hotpot.plugins.ComplexFormer.data.data_module.get_first_data`；源码 `hotpot/plugins/ComplexFormer/data/data_module.py:99`。

```python
get_first_data(dir_datasets: str) -> dict[str, Data]
```

<a id="hotpot-plugins-complexformer-data-data-module-datamodule"></a>
### `DataModule`

定义 owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；源码 `hotpot/plugins/ComplexFormer/data/data_module.py:124`。

基类原始声明：`L.LightningDataModule`；装饰器：无。

本类属性与方法：

- `__init__(self, args_cfg: datacls.DataModuleArgs)`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:220`
- `train_val_test_split(self)`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:347`
- `cross_val_split(self, cv: int=5)`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:360`
- `setup(self, stage: Optional[str]=None)`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:375`
- `get_loader(self, dataset, batch_size: int=1, shuffle: bool=False)`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:395`
- `train_dataloader(self) -> DataLoader`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:423`
- `val_dataloader(self) -> DataLoader`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:427`
- `test_dataloader(self) -> DataLoader`；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:431`
- `dataset_counts(self) -> int` [读；只读]；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:278`
- `first_data(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:283`
- `is_multi_datasets(self) -> bool` [读；只读]；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:302`

本类字段与枚举值：

- `dir_datasets` = `args_cfg.dir_datasets` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:225`
- `test_only` = `args_cfg.test_only` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:226`
- `batch_num` = `args_cfg.batch_num` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:251`
- `seed` = `args_cfg.seed` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:255`
- `ratios` = `[r / sum(args_cfg.ratios) for r in args_cfg.ratios]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:256`
- `batch_size` = `args_cfg.batch_size` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:258`
- `shuffle` = `args_cfg.shuffle` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:259`
- `num_replicas` = `args_cfg.num_replicas` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:260`
- `list_dataset_names` = `sorted(os.listdir(args_cfg.dir_datasets))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:230`
- `device_count` = `torch.cuda.device_count()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:263`
- `train_dataset` = `MConcatDataset(_train_datasets)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:387`
- `val_dataset` = `MConcatDataset(_val_datasets)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:388`
- `test_dataset` = `MConcatDataset(_test_datasets)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.data_module.DataModule`；`hotpot/plugins/ComplexFormer/data/data_module.py:389`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.data.data_module.L.LightningDataModule`。


<a id="hotpot-plugins-complexformer-data-dataset"></a>
## `hotpot.plugins.ComplexFormer.data.dataset`

源码 `hotpot/plugins/ComplexFormer/data/dataset.py`；SHA-256 `43d3e219e702a490ef5e0dda460fbebe5f831fcd55f200492e114abb56dbc20d`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `torch_load_data` → `hotpot.plugins.ComplexFormer.data.dataset.torch_load_data`；`torch_load_data(p: str) -> Data`
- `load_data` → `hotpot.plugins.ComplexFormer.data.dataset.load_data`；`load_data(dataset: Union[Iterable[Data], Sequence[Data]], sample_num: int=None)`
- `DataWrapper` → `hotpot.plugins.ComplexFormer.data.dataset.DataWrapper`；构造声明 `__init__(self, data_iter: Iterable[Data]) -> object`
- `MConcatDataset` → `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`
- `OnFlyLoadingDataset` → `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；构造声明 `__init__(self, data_files: list['str'])`
- `PathStoredDataset` → `hotpot.plugins.ComplexFormer.data.dataset.PathStoredDataset`；构造声明 `__init__(self, data_files: list['str'])`

<a id="hotpot-plugins-complexformer-data-dataset-torch-load-data"></a>
### `torch_load_data`

定义 owner `hotpot.plugins.ComplexFormer.data.dataset.torch_load_data`；源码 `hotpot/plugins/ComplexFormer/data/dataset.py:17`。

```python
torch_load_data(p: str) -> Data
```

<a id="hotpot-plugins-complexformer-data-dataset-load-data"></a>
### `load_data`

定义 owner `hotpot.plugins.ComplexFormer.data.dataset.load_data`；源码 `hotpot/plugins/ComplexFormer/data/dataset.py:27`。

```python
load_data(dataset: Union[Iterable[Data], Sequence[Data]], sample_num: int=None)
```

<a id="hotpot-plugins-complexformer-data-dataset-datawrapper"></a>
### `DataWrapper`

定义 owner `hotpot.plugins.ComplexFormer.data.dataset.DataWrapper`；源码 `hotpot/plugins/ComplexFormer/data/dataset.py:46`。

基类原始声明：`Dataset`；装饰器：无。

本类属性与方法：

- `__init__(self, data_iter: Iterable[Data]) -> object`；owner `hotpot.plugins.ComplexFormer.data.dataset.DataWrapper`；`hotpot/plugins/ComplexFormer/data/dataset.py:47`
- `__getitem__(self, idx: Union[int, slice]) -> Data`；owner `hotpot.plugins.ComplexFormer.data.dataset.DataWrapper`；`hotpot/plugins/ComplexFormer/data/dataset.py:50`
- `__len__(self) -> int`；owner `hotpot.plugins.ComplexFormer.data.dataset.DataWrapper`；`hotpot/plugins/ComplexFormer/data/dataset.py:55`
- `__iter__(self) -> Iterable[Data]`；owner `hotpot.plugins.ComplexFormer.data.dataset.DataWrapper`；`hotpot/plugins/ComplexFormer/data/dataset.py:62`

本类字段与枚举值：

- `data_iter` = `data_iter` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.dataset.DataWrapper`；`hotpot/plugins/ComplexFormer/data/dataset.py:48`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch.utils.data.Dataset`。

<a id="hotpot-plugins-complexformer-data-dataset-mconcatdataset"></a>
### `MConcatDataset`

定义 owner `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`；源码 `hotpot/plugins/ComplexFormer/data/dataset.py:65`。

基类原始声明：`ConcatDataset`；装饰器：无。

本类属性与方法：

- `__repr__(self)`；owner `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:66`
- `get_with_ds_idx(self, idx)`；owner `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:69`
- `load_data(self, sample_num: Optional[int]=None)`；owner `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:74`
- `__getitem__(self, idx: Union[int, slice]) -> Data`；owner `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:79`

本类字段与枚举值：

- `datasets` = `[load_data(dataset, sample_num) for dataset in self.datasets]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:75`
- `cumulative_sizes` = `self.cumsum(self.datasets)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.dataset.MConcatDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:76`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch.utils.data.ConcatDataset`。

<a id="hotpot-plugins-complexformer-data-dataset-onflyloadingdataset"></a>
### `OnFlyLoadingDataset`

定义 owner `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；源码 `hotpot/plugins/ComplexFormer/data/dataset.py:88`。

基类原始声明：`Dataset`；装饰器：无。

本类属性与方法：

- `__init__(self, data_files: list['str'])`；owner `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:89`
- `__len__(self) -> int`；owner `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:93`
- `__getitem__(self, idx: int) -> Data`；owner `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:96`
- `data_number(self) -> int` [读；只读]；owner `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:105`

本类字段与枚举值：

- `data_files` = `data_files` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:90`
- `data` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.dataset.OnFlyLoadingDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:91`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch.utils.data.Dataset`。

<a id="hotpot-plugins-complexformer-data-dataset-pathstoreddataset"></a>
### `PathStoredDataset`

定义 owner `hotpot.plugins.ComplexFormer.data.dataset.PathStoredDataset`；源码 `hotpot/plugins/ComplexFormer/data/dataset.py:109`。

基类原始声明：`Dataset`；装饰器：无。

本类属性与方法：

- `__init__(self, data_files: list['str'])`；owner `hotpot.plugins.ComplexFormer.data.dataset.PathStoredDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:110`
- `__len__(self) -> int`；owner `hotpot.plugins.ComplexFormer.data.dataset.PathStoredDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:114`
- `__getitem__(self, idx: int) -> Data`；owner `hotpot.plugins.ComplexFormer.data.dataset.PathStoredDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:117`

本类字段与枚举值：

- `data_files` = `data_files` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.dataset.PathStoredDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:111`
- `data` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.dataset.PathStoredDataset`；`hotpot/plugins/ComplexFormer/data/dataset.py:112`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch.utils.data.Dataset`。


<a id="hotpot-plugins-complexformer-data-loader"></a>
## `hotpot.plugins.ComplexFormer.data.loader`

源码 `hotpot/plugins/ComplexFormer/data/loader.py`；SHA-256 `66402941334ec22fd811e2ac8d5e1b524b15997c8ddf8f146a8271c9d9e95f60`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `DistConcatBatchSampler` → `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；构造声明 `__init__(self, dataset: Dataset, batch_size: Optional[int]=None, *, num_replicas: Optional[int]=None, rank: Optional[int]=None, shuffle: bool=True, seed: int=0, drop_last: bool=False)`
- `CDataLoader` → `hotpot.plugins.ComplexFormer.data.loader.CDataLoader`；构造声明 `__init__(self, dataset: Union[D.MConcatDataset, Iterable[Iterable[BaseData]]], batch_size: int=1, shuffle: bool=False, follow_batch: Optional[List[str]]=None, exclude_keys: Optional[List[str]]=None, **kwargs)`
- `DistConcatLoader` → `hotpot.plugins.ComplexFormer.data.loader.DistConcatLoader`；构造声明 `__init__(self, dataset: Union[D.MConcatDataset, Iterable[Iterable[BaseData]]], batch_size: int=1, shuffle: bool=False, follow_batch: Optional[List[str]]=None, exclude_keys: Optional[List[str]]=None, **kwargs)`

<a id="hotpot-plugins-complexformer-data-loader-distconcatbatchsampler"></a>
### `DistConcatBatchSampler`

定义 owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；源码 `hotpot/plugins/ComplexFormer/data/loader.py:78`。

基类原始声明：`Sampler`；装饰器：无。

本类属性与方法：

- `__init__(self, dataset: Dataset, batch_size: Optional[int]=None, *, num_replicas: Optional[int]=None, rank: Optional[int]=None, shuffle: bool=True, seed: int=0, drop_last: bool=False)`；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:79`
- `__len__(self) -> int`；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:165`
- `__iter__(self) -> Iterable[list[int]]`；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:168`
- `datasets(self) -> list[Dataset]` [读；只读]；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:162`

本类字段与枚举值：

- `dataset` = `_check_concat_dataset(dataset)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:103`
- `cunsum_size` = `_cumsum_datasets(self.dataset)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:104`
- `batch_size` = `batch_size` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:105`
- `num_replicas` = `num_replicas` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:106`
- `rank` = `rank` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:107`
- `epoch` = `0` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:108`
- `drop_last` = `drop_last` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:109`
- `split_size` = `self.batch_size * self.num_replicas` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:110`
- `total_size` = `self.num_samples * self.num_replicas` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:142`
- `shuffle` = `shuffle` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:143`
- `seed` = `seed` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:144`
- `num_samples` = `0` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatBatchSampler`；`hotpot/plugins/ComplexFormer/data/loader.py:119`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch.utils.data.Sampler`。

<a id="hotpot-plugins-complexformer-data-loader-cdataloader"></a>
### `CDataLoader`

定义 owner `hotpot.plugins.ComplexFormer.data.loader.CDataLoader`；源码 `hotpot/plugins/ComplexFormer/data/loader.py:406`。

基类原始声明：`DataLoader`；装饰器：无。

本类属性与方法：

- `__init__(self, dataset: Union[D.MConcatDataset, Iterable[Iterable[BaseData]]], batch_size: int=1, shuffle: bool=False, follow_batch: Optional[List[str]]=None, exclude_keys: Optional[List[str]]=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.data.loader.CDataLoader`；`hotpot/plugins/ComplexFormer/data/loader.py:407`

本类字段与枚举值：

- `collate_fn` = `collate.Collater(dataset, follow_batch, exclude_keys)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.CDataLoader`；`hotpot/plugins/ComplexFormer/data/loader.py:439`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch_geometric.loader.DataLoader`。

<a id="hotpot-plugins-complexformer-data-loader-distconcatloader"></a>
### `DistConcatLoader`

定义 owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatLoader`；源码 `hotpot/plugins/ComplexFormer/data/loader.py:441`。

基类原始声明：`DataLoader`；装饰器：无。

本类属性与方法：

- `__init__(self, dataset: Union[D.MConcatDataset, Iterable[Iterable[BaseData]]], batch_size: int=1, shuffle: bool=False, follow_batch: Optional[List[str]]=None, exclude_keys: Optional[List[str]]=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatLoader`；`hotpot/plugins/ComplexFormer/data/loader.py:442`

本类字段与枚举值：

- `collate_fn` = `collate.Collater(dataset, follow_batch, exclude_keys)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data.loader.DistConcatLoader`；`hotpot/plugins/ComplexFormer/data/loader.py:476`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch_geometric.loader.DataLoader`。


<a id="hotpot-plugins-complexformer-data-process"></a>
## `hotpot.plugins.ComplexFormer.data_process`

源码 `hotpot/plugins/ComplexFormer/data_process/__init__.py`；SHA-256 `4bd70e22c604447d41f8be116d315c43fa9a4dc5a25d6b4464b64b8748936792`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-complexformer-data-process-ccdc"></a>
## `hotpot.plugins.ComplexFormer.data_process.ccdc`

源码 `hotpot/plugins/ComplexFormer/data_process/ccdc.py`；SHA-256 `677ed532dcfd54d808a95f61f882834061f5de1ccabfab1d375934cb98272eec`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `ccdc_struct_to_data` → `hotpot.plugins.ComplexFormer.data_process.ccdc.ccdc_struct_to_data`；`ccdc_struct_to_data(struct_dir: str, data_dir: str)`
- `convert_ml_pairs_to_cbond_broken_data` → `hotpot.plugins.ComplexFormer.data_process.ccdc.convert_ml_pairs_to_cbond_broken_data`；`convert_ml_pairs_to_cbond_broken_data(path_struct: str, data_dir: str, catoms_elements=None) -> Optional[set[str]]`
- `default_catoms_elements` → `hotpot.plugins.ComplexFormer.data_process.ccdc.default_catoms_elements`

<a id="hotpot-plugins-complexformer-data-process-ccdc-ccdc-struct-to-data"></a>
### `ccdc_struct_to_data`

定义 owner `hotpot.plugins.ComplexFormer.data_process.ccdc.ccdc_struct_to_data`；源码 `hotpot/plugins/ComplexFormer/data_process/ccdc.py:31`。

```python
ccdc_struct_to_data(struct_dir: str, data_dir: str)
```

<a id="hotpot-plugins-complexformer-data-process-ccdc-convert-ml-pairs-to-cbond-broken-data"></a>
### `convert_ml_pairs_to_cbond_broken_data`

定义 owner `hotpot.plugins.ComplexFormer.data_process.ccdc.convert_ml_pairs_to_cbond_broken_data`；源码 `hotpot/plugins/ComplexFormer/data_process/ccdc.py:94`。

```python
convert_ml_pairs_to_cbond_broken_data(path_struct: str, data_dir: str, catoms_elements=None) -> Optional[set[str]]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `default_catoms_elements` = `{'O', 'N', 'S', 'P', 'Si', 'B'}`；`hotpot/plugins/ComplexFormer/data_process/ccdc.py:93`


<a id="hotpot-plugins-complexformer-data-process-normalize-data"></a>
## `hotpot.plugins.ComplexFormer.data_process.normalize_data`

源码 `hotpot/plugins/ComplexFormer/data_process/normalize_data.py`；SHA-256 `bf3b2499292cb25942a827c7275c76d5cca5f0f2b928659a1aec71464b4599b2`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Method` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.Method`
- `DataProfile` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；构造声明 `DataProfile(path: str, method: Method, data: Optional[np.ndarray] = None, mean: float = 0.0, std: float = 0.0, min_val: float = 0.0, max_val: float = 1.0, shift: float = 0.0)`
- `DatasetAnalyzer` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；构造声明 `__init__(self, datasets_root: Union[str, os.PathLike], profile_dir: Optional[os.PathLike]=None, strict_mode: bool=False)`
- `TensorNormalizer` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；构造声明 `__init__(self, path: str, profile: DataProfile)`
- `SKEWNESS_THRESHOLD` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.SKEWNESS_THRESHOLD`
- `VARIANCE_THRESHOLD` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.VARIANCE_THRESHOLD`
- `EPSILON` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.EPSILON`
- `WorkItemDict` → `hotpot.plugins.ComplexFormer.data_process.normalize_data.WorkItemDict`

<a id="hotpot-plugins-complexformer-data-process-normalize-data-method"></a>
### `Method`

定义 owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.Method`；源码 `hotpot/plugins/ComplexFormer/data_process/normalize_data.py:41`。

基类原始声明：`Enum`；装饰器：无。

本类字段与枚举值：

- `IDENTITY` = `'identity'` [枚举常量；只读]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.Method`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:42`
- `MIN_MAX` = `'min_max'` [枚举常量；只读]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.Method`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:43`
- `Z_SCORE` = `'z_score'` [枚举常量；只读]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.Method`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:44`
- `LOG` = `'log'` [枚举常量；只读]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.Method`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:45`

Enum members: read-only .name/.value; class iteration, name lookup C[name], value construction C(value), equality/hash. Standard-library generated members are not individually copied into local method counts.

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`enum.Enum`。

<a id="hotpot-plugins-complexformer-data-process-normalize-data-dataprofile"></a>
### `DataProfile`

定义 owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；源码 `hotpot/plugins/ComplexFormer/data_process/normalize_data.py:54`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
DataProfile(path: str, method: Method, data: Optional[np.ndarray] = None, mean: float = 0.0, std: float = 0.0, min_val: float = 0.0, max_val: float = 1.0, shift: float = 0.0)
```

本类属性与方法：

- `to_dict(self)`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:72`

本类字段与枚举值：

- `path: str` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:56`
- `method: Method` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:57`
- `data: Optional[np.ndarray]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:58`
- `mean: float` = `0.0` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:59`
- `std: float` = `0.0` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:60`
- `min_val: float` = `0.0` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:61`
- `max_val: float` = `1.0` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:62`
- `shift: float` = `0.0` [可写值字段]；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DataProfile`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:63`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-complexformer-data-process-normalize-data-datasetanalyzer"></a>
### `DatasetAnalyzer`

定义 owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；源码 `hotpot/plugins/ComplexFormer/data_process/normalize_data.py:78`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, datasets_root: Union[str, os.PathLike], profile_dir: Optional[os.PathLike]=None, strict_mode: bool=False)`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:79`
- `compute_stats(self, path: str, data: np.ndarray) -> DataProfile`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:368`
- `analyze_datasets(self, paths: Optional[Union[str, Iterable[str]]]=None)`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:374`
- `batch_transform(self, batch: Batch) -> Batch`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:386`

本类字段与枚举值：

- `datasets_root` = `Path(datasets_root)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:85`
- `profile_dir` = `Path(profile_dir) if profile_dir else self.datasets_root / '.profile'` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:87`
- `strict_mode` = `strict_mode` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:91`
- `profiles` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.DatasetAnalyzer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:92`

<a id="hotpot-plugins-complexformer-data-process-normalize-data-tensornormalizer"></a>
### `TensorNormalizer`

定义 owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；源码 `hotpot/plugins/ComplexFormer/data_process/normalize_data.py:389`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, path: str, profile: DataProfile)`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:390`
- `transform(self, tensor: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:396`
- `inverse(self, normalized_tensor: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:410`

本类字段与枚举值：

- `path` = `path` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:391`
- `profile` = `profile` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:392`
- `trans_logger` = `RateLimitLogger(interval_count=100)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:393`
- `inv_logger` = `RateLimitLogger(interval_count=100)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.data_process.normalize_data.TensorNormalizer`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:394`

模块值/类型别名（声明存在不构成事实写许可）：

- `SKEWNESS_THRESHOLD` = `1.0`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:36`
- `VARIANCE_THRESHOLD` = `1e-06`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:37`
- `EPSILON` = `1e-08`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:38`
- `WorkItemDict` = `dict[str, Union[None, list[str]]]`；`hotpot/plugins/ComplexFormer/data_process/normalize_data.py:47`


<a id="hotpot-plugins-complexformer-data-process-sc-logk"></a>
## `hotpot.plugins.ComplexFormer.data_process.sc_logk`

源码 `hotpot/plugins/ComplexFormer/data_process/sc_logk/__init__.py`；SHA-256 `902af907b42ee87eb1192c1bd3fbee8c1e21f37dd44f99d6e14146c0f794548c`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-complexformer-data-process-sc-logk-core"></a>
## `hotpot.plugins.ComplexFormer.data_process.sc_logk.core`

源码 `hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py`；SHA-256 `0008b87c4cafe7c0f90e7a5208b98183d16f0b8be2f4e19e91e3ddfed5a0186e`。

显式 __all__：`('mp_process_SclogK',)`。

显式导出绑定：

- `mp_process_SclogK` → `hotpot.plugins.ComplexFormer.data_process.sc_logk.core.mp_process_SclogK`；`mp_process_SclogK(path_raw: str, data_dir: str, store_metal_cluster: bool=False)`

<a id="hotpot-plugins-complexformer-data-process-sc-logk-core-split-metal"></a>
### `split_metal`

定义 owner `hotpot.plugins.ComplexFormer.data_process.sc_logk.core.split_metal`；源码 `hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py:61`。

```python
split_metal(metal_info: str)
```

<a id="hotpot-plugins-complexformer-data-process-sc-logk-core-mp-process-sclogk"></a>
### `mp_process_SclogK`

定义 owner `hotpot.plugins.ComplexFormer.data_process.sc_logk.core.mp_process_SclogK`；源码 `hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py:227`。

```python
mp_process_SclogK(path_raw: str, data_dir: str, store_metal_cluster: bool=False)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `sc_cols` = `['W', 'number', 'Tech.', 'SMILES', 'Metal', 'Medium', 'Med_cid', 'Sol1', 'Sol1_cid', 'Sol2', 'Sol2_cid', 'Sol1Ratio', 'Sol2Ratio', 'RatioMetric', 't', 'I-str', 'pH', 'P/bar', 'logK1']`；`hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py:37`
- `med_info_names` = `['names', 'CASs', 'formulas', 'smiless', 'Cid']`；`hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py:42`
- `med_attr_names` = `['MW', 'similarity_variables', 'Vmg_STPs', 'rhog_STPs', 'rhog_STPs_mass', 'similarity_variable', 'Cpg', 'Cpgm', 'Cpl', 'Cplm', 'Cps', 'Cpsm', 'Cvg', 'Cvgm', 'isentropic_exponent', 'JTg', 'kl', 'rhog', 'SGg', 'Density_medium (kg/m3)', 'Molar Mass_medium (g/mol)', 'Tm_medium (K)']`；`hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py:43`
- `sol_info_names` = `['names', 'CASs', 'formulas', 'Cid', 'smiless']`；`hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py:50`
- `sol_attr_names` = <长声明表达式；完整值见库存及源码位置>；`hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py:51`


<a id="hotpot-plugins-complexformer-data-process-sc-logk-data"></a>
## `hotpot.plugins.ComplexFormer.data_process.sc_logk.data`

源码 `hotpot/plugins/ComplexFormer/data_process/sc_logk/data.py`；SHA-256 `bb51529caf1fce4b6d45cb461b5f3606da7be4caab32d621241676b5fcb0a8d1`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `ExtractionData` → `hotpot.plugins.ComplexFormer.data_process.sc_logk.data.ExtractionData`

<a id="hotpot-plugins-complexformer-data-process-sc-logk-data-extractiondata"></a>
### `ExtractionData`

定义 owner `hotpot.plugins.ComplexFormer.data_process.sc_logk.data.ExtractionData`；源码 `hotpot/plugins/ComplexFormer/data_process/sc_logk/data.py:22`。

基类原始声明：`Data`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch_geometric.data.Data`。


<a id="hotpot-plugins-complexformer-models"></a>
## `hotpot.plugins.ComplexFormer.models`

源码 `hotpot/plugins/ComplexFormer/models/__init__.py`；SHA-256 `01e2bfa03bfa0f47026e0112e0b724e0ce346c298bbdd1015290bbab389d1730`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-complexformer-models--core"></a>
## `hotpot.plugins.ComplexFormer.models._core`

源码 `hotpot/plugins/ComplexFormer/models/_core.py`；SHA-256 `59678de61712934a5752d4a37944ad0b17f519179e78ea6d5c84b8d8868f0155`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `complete_graph_generator` → `hotpot.plugins.ComplexFormer.models._core.complete_graph_generator`；`complete_graph_generator(ptr)`
- `CloudGraph` → `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；构造声明 `__init__(self, vec_dim: int)`
- `NodeProcessor` → `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；构造声明 `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None)`
- `CoreBase` → `hotpot.plugins.ComplexFormer.models._core.CoreBase`；构造声明 `__init__(self, vec_dim: int, x_label_nums: Optional[int]=None)`
- `CompleteGraphExtractor` → `hotpot.plugins.ComplexFormer.models._core.CompleteGraphExtractor`
- `CompleteGraphCore` → `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；构造声明 `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, **kwargs)`
- `AttnExtractor` → `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`
- `AttnCore` → `hotpot.plugins.ComplexFormer.models._core.AttnCore`；构造声明 `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, **kwargs)`
- `Core` → `hotpot.plugins.ComplexFormer.models._core.AttnCore`；构造声明 `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, **kwargs)`

<a id="hotpot-plugins-complexformer-models--core-complete-graph-generator"></a>
### `complete_graph_generator`

定义 owner `hotpot.plugins.ComplexFormer.models._core.complete_graph_generator`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:16`。

```python
complete_graph_generator(ptr)
```

<a id="hotpot-plugins-complexformer-models--core-cloudgraph"></a>
### `CloudGraph`

定义 owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:54`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int)`；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:56`
- `forward(self, x, xyz, ptr)`；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:67`

本类字段与枚举值：

- `xyz_proj` = `nn.Linear(3, vec_dim, bias=False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:58`
- `xyz_proj_norm` = `nn.BatchNorm1d(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:59`
- `lin1` = `nn.Linear(vec_dim, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:61`
- `norm1` = `nn.LayerNorm(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:62`
- `lin2` = `nn.Linear(vec_dim, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:64`
- `norm2` = `nn.LayerNorm(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CloudGraph`；`hotpot/plugins/ComplexFormer/models/_core.py:65`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models._core.nn.Module`。

<a id="hotpot-plugins-complexformer-models--core-nodeprocessor"></a>
### `NodeProcessor`

定义 owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:93`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None)`；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:94`
- `forward(self, x, edge_index, ptr, xyz=None)`；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:125`

本类字段与枚举值：

- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:102`
- `x_label_nums` = `x_label_nums` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:103`
- `cloud_graph` = `CloudGraph(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:112`
- `lin` = `nn.Linear(vec_dim, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:113`
- `norm` = `nn.BatchNorm1d(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:114`
- `x_emb` = `nn.Embedding(x_label_nums + 1, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:106`
- `x_mask_vec` = `self.x_emb.weight[0]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:107`
- `x_proj` = `nn.Linear(x_dim, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:109`
- `graph` = `graph_model` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/_core.py:117`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models._core.nn.Module`。

<a id="hotpot-plugins-complexformer-models--core-corebase"></a>
### `CoreBase`

定义 owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:143`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int, x_label_nums: Optional[int]=None)`；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:155`
- `x_mask_vec(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:168`

本类字段与枚举值：

- `extractor_class` = `None` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:145`
- `extractor_keys` = `{'atom': 'extract_atom_vec', 'bond': 'extract_bond_vec', 'pair': 'extract_pair_vec', 'ring': 'extract_ring_vec', 'mol': 'extract_mol_vec', 'cbond': 'extract_cbond_pair', 'metal': 'extract_metal_vec'}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:146`
- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:157`
- `x_label_nums` = `x_label_nums` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:158`
- `feature_extractor` = `{key: getattr(self.extractor_class, method_name) for key, method_name in self.extractor_keys.items() if hasattr(self.extractor_class, method_name)}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:160`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models._core.nn.Module`。

<a id="hotpot-plugins-complexformer-models--core-completegraphextractor"></a>
### `CompleteGraphExtractor`

定义 owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphExtractor`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:171`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `extract_atom_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:173`
- `extract_cbond_pair(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:177`
- `extract_pair_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:190`
- `extract_ring_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:201`
- `extract_mol_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:205`

<a id="hotpot-plugins-complexformer-models--core-completegraphcore"></a>
### `CompleteGraphCore`

定义 owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:209`。

基类原始声明：`CoreBase`；装饰器：无。

本类属性与方法：

- `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/_core.py:211`
- `forward(self, x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr, *, xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]]=None) -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/_core.py:237`
- `x_mask_vec(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/_core.py:250`

本类字段与枚举值：

- `extractor_class` = `CompleteGraphExtractor` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/_core.py:210`
- `node_processor` = `NodeProcessor(x_dim, vec_dim, x_label_nums=x_label_nums, graph_model=graph_model)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/_core.py:232`
- `ring_encoder` = `CompleteGraph(vec_dim, ring_layers, ring_nheads)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/_core.py:234`
- `mol_encoder` = `CompleteGraph(vec_dim, mol_layers, mol_nheads)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/_core.py:235`

继承成员（逐项保留声明 owner）：

- `extractor_keys` = `{'atom': 'extract_atom_vec', 'bond': 'extract_bond_vec', 'pair': 'extract_pair_vec', 'ring': 'extract_ring_vec', 'mol': 'extract_mol_vec', 'cbond': 'extract_cbond_pair', 'metal': 'extract_metal_vec'}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:146`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`
- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:157`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`
- `x_label_nums` = `x_label_nums` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:158`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`
- `feature_extractor` = `{key: getattr(self.extractor_class, method_name) for key, method_name in self.extractor_keys.items() if hasattr(self.extractor_class, method_name)}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:160`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`

本类名称别名：`extractor_class` → `CompleteGraphExtractor`。

<a id="hotpot-plugins-complexformer-models--core-attnextractor"></a>
### `AttnExtractor`

定义 owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:254`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:256`
- `extract_bond_vec(seq, X_mask, R_mask, batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:265`
- `extract_metal_vec(seq, X_mask, R_mask, batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:275`
- `extract_cbond_pair(seq, X_mask, R_mask, batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:281`
- `extract_pair_vec(seq, X_mask, R_mask, batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:296`
- `extract_ring_vec(seq, X_mask, R_mask, batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:307`
- `extract_mol_vec(seq, X_mask, R_mask, batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models._core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/_core.py:319`

<a id="hotpot-plugins-complexformer-models--core-attncore"></a>
### `AttnCore`

定义 owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；源码 `hotpot/plugins/ComplexFormer/models/_core.py:323`。

基类原始声明：`CoreBase`；装饰器：无。

本类属性与方法：

- `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:326`
- `forward(self, x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr, *, xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]]=None)`；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:402`
- `x_mask_vec(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:370`

本类字段与枚举值：

- `extractor_class` = `AttnExtractor` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:325`
- `node_processor` = `NodeProcessor(x_dim, vec_dim, x_label_nums=x_label_nums, graph_model=graph_model)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:347`
- `ring_encoder` = `nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=vec_dim, nhead=ring_nheads, dim_feedforward=1024, batch_first=True), num_layers=ring_layers)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:348`
- `mol_encoder` = `nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=vec_dim, nhead=mol_nheads, dim_feedforward=1024, batch_first=True), num_layers=mol_layers)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:356`
- `CLS` = `nn.Parameter(torch.randn(1, vec_dim))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:365`
- `RING` = `nn.Parameter(torch.randn(1, vec_dim))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:366`
- `END` = `nn.Parameter(torch.randn(1, vec_dim))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:367`

继承成员（逐项保留声明 owner）：

- `extractor_keys` = `{'atom': 'extract_atom_vec', 'bond': 'extract_bond_vec', 'pair': 'extract_pair_vec', 'ring': 'extract_ring_vec', 'mol': 'extract_mol_vec', 'cbond': 'extract_cbond_pair', 'metal': 'extract_metal_vec'}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:146`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`
- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:157`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`
- `x_label_nums` = `x_label_nums` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:158`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`
- `feature_extractor` = `{key: getattr(self.extractor_class, method_name) for key, method_name in self.extractor_keys.items() if hasattr(self.extractor_class, method_name)}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models._core.CoreBase`；`hotpot/plugins/ComplexFormer/models/_core.py:160`；继承经 `hotpot.plugins.ComplexFormer.models._core.CoreBase`

本类名称别名：`extractor_class` → `AttnExtractor`。

模块值/类型别名（声明存在不构成事实写许可）：

- `Core` = `AttnCore`；`hotpot/plugins/ComplexFormer/models/_core.py:413`


<a id="hotpot-plugins-complexformer-models-attn"></a>
## `hotpot.plugins.ComplexFormer.models.attn`

源码 `hotpot/plugins/ComplexFormer/models/attn.py`；SHA-256 `8dddc4cef8e5cb3a2de73cd3c4a4049a210c2aa2d90146f4a7efd620919a18b8`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Encoder` → `hotpot.plugins.ComplexFormer.models.attn.Encoder`；构造声明 `__init__(self, n_layers: int, d_model: int, nheads: int, dim_feedforward: int=1024, dropout: float=0.0, activation: Union[str, Callable[[Tensor], Tensor]]=F.relu, layer_norm_eps: float=1e-05, bias=False, device=None, dtype=None, norm_first: bool=True, E_q: int=None, E_k: int=None, E_v: int=None, E_total: int=None, norm: Optional[nn.Module]=None)`
- `EncoderLayer` → `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；构造声明 `__init__(self, d_model: int=None, E_q: int=None, E_k: int=None, E_v: int=None, E_total: int=None, nheads: int=1, dim_feedforward: int=1024, dropout: float=0.0, activation: Union[str, Callable[[Tensor], Tensor]]=F.relu, layer_norm_eps: float=1e-05, bias=True, device=None, dtype=None, norm_first: bool=True)`
- `MultiHeadAttention` → `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；构造声明 `__init__(self, E_q: int, E_k: int, E_v: int, E_total: int, nheads: int, dropout: float=0.0, bias=False, device=None, dtype=None)`
- `scaled_dot_product_attention` → `hotpot.plugins.ComplexFormer.models.attn.scaled_dot_product_attention`；`scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor`

<a id="hotpot-plugins-complexformer-models-attn-encoder"></a>
### `Encoder`

定义 owner `hotpot.plugins.ComplexFormer.models.attn.Encoder`；源码 `hotpot/plugins/ComplexFormer/models/attn.py:29`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, n_layers: int, d_model: int, nheads: int, dim_feedforward: int=1024, dropout: float=0.0, activation: Union[str, Callable[[Tensor], Tensor]]=F.relu, layer_norm_eps: float=1e-05, bias=False, device=None, dtype=None, norm_first: bool=True, E_q: int=None, E_k: int=None, E_v: int=None, E_total: int=None, norm: Optional[nn.Module]=None)`；owner `hotpot.plugins.ComplexFormer.models.attn.Encoder`；`hotpot/plugins/ComplexFormer/models/attn.py:31`
- `forward(self, src: Tensor, mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, is_causal: bool=False) -> Tensor`；owner `hotpot.plugins.ComplexFormer.models.attn.Encoder`；`hotpot/plugins/ComplexFormer/models/attn.py:73`

本类字段与枚举值：

- `layers` = `_get_clones(n_layers, EncoderLayer(d_model, E_q, E_k, E_v, E_total, nheads, dim_feedforward, dropout, activation, layer_norm_eps, bias, device, dtype, norm_first))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.Encoder`；`hotpot/plugins/ComplexFormer/models/attn.py:51`
- `norm` = `norm` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.Encoder`；`hotpot/plugins/ComplexFormer/models/attn.py:71`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.attn.nn.Module`。

<a id="hotpot-plugins-complexformer-models-attn-encoderlayer"></a>
### `EncoderLayer`

定义 owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；源码 `hotpot/plugins/ComplexFormer/models/attn.py:95`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, d_model: int=None, E_q: int=None, E_k: int=None, E_v: int=None, E_total: int=None, nheads: int=1, dim_feedforward: int=1024, dropout: float=0.0, activation: Union[str, Callable[[Tensor], Tensor]]=F.relu, layer_norm_eps: float=1e-05, bias=True, device=None, dtype=None, norm_first: bool=True)`；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:96`
- `forward(self, src: Tensor, src_mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, is_causal: bool=False) -> Tensor`；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:163`

本类字段与枚举值：

- `E_q` = `E_q or d_model` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:116`
- `E_k` = `E_k or d_model` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:117`
- `E_v` = `E_v or d_model` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:118`
- `E_total` = `E_total or d_model` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:119`
- `md_attn` = `MultiHeadAttention(self.E_q, self.E_k, self.E_v, self.E_total, nheads, dropout, bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:127`
- `linear1` = `nn.Linear(self.E_v, dim_feedforward, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:139`
- `dropout` = `nn.Dropout(dropout)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:140`
- `linear2` = `nn.Linear(dim_feedforward, self.E_v, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:141`
- `norm_first` = `norm_first` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:143`
- `norm1` = `nn.LayerNorm(self.E_v, eps=layer_norm_eps, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:144`
- `norm2` = `nn.LayerNorm(self.E_v, eps=layer_norm_eps, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:145`
- `dropout1` = `nn.Dropout(dropout)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:146`
- `dropout2` = `nn.Dropout(dropout)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:147`
- `activation` = `activation` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:161`
- `activation_relu_or_gelu` = `1` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.EncoderLayer`；`hotpot/plugins/ComplexFormer/models/attn.py:156`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.attn.nn.Module`。

<a id="hotpot-plugins-complexformer-models-attn-multiheadattention"></a>
### `MultiHeadAttention`

定义 owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；源码 `hotpot/plugins/ComplexFormer/models/attn.py:214`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, E_q: int, E_k: int, E_v: int, E_total: int, nheads: int, dropout: float=0.0, bias=False, device=None, dtype=None)`；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:229`
- `forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal=False) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:258`

本类字段与枚举值：

- `nheads` = `nheads` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:243`
- `dropout` = `dropout` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:244`
- `out_proj` = `nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:253`
- `E_head` = `E_total // nheads` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:255`
- `bias` = `bias` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:256`
- `packed_proj` = `nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:247`
- `q_proj` = `nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:249`
- `k_proj` = `nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:250`
- `v_proj` = `nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.attn.MultiHeadAttention`；`hotpot/plugins/ComplexFormer/models/attn.py:251`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.attn.nn.Module`。

<a id="hotpot-plugins-complexformer-models-attn-scaled-dot-product-attention"></a>
### `scaled_dot_product_attention`

定义 owner `hotpot.plugins.ComplexFormer.models.attn.scaled_dot_product_attention`；源码 `hotpot/plugins/ComplexFormer/models/attn.py:369`。

```python
scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor
```


<a id="hotpot-plugins-complexformer-models-cm"></a>
## `hotpot.plugins.ComplexFormer.models.cm`

源码 `hotpot/plugins/ComplexFormer/models/cm.py`；SHA-256 `dc4d5a3d6c4fd0a02ca89af40e05effd0f17d84e8fef82d33c487ccaaa35b701`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Model` → `hotpot.plugins.ComplexFormer.models.cm.Model`；构造声明 `__init__(self, core: Union[nn.Module, str], predictor: nn.Module)`

<a id="hotpot-plugins-complexformer-models-cm-model"></a>
### `Model`

定义 owner `hotpot.plugins.ComplexFormer.models.cm.Model`；源码 `hotpot/plugins/ComplexFormer/models/cm.py:9`。

基类原始声明：`L.LightningModule`；装饰器：无。

本类属性与方法：

- `__init__(self, core: Union[nn.Module, str], predictor: nn.Module)`；owner `hotpot.plugins.ComplexFormer.models.cm.Model`；`hotpot/plugins/ComplexFormer/models/cm.py:10`
- `forward(self, *args, **kw)`；owner `hotpot.plugins.ComplexFormer.models.cm.Model`；`hotpot/plugins/ComplexFormer/models/cm.py:19`
- `x_label_nums(self) -> Optional[int]` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.cm.Model`；`hotpot/plugins/ComplexFormer/models/cm.py:23`
- `x_mask_vec(self) -> Optional[torch.Tensor]` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.cm.Model`；`hotpot/plugins/ComplexFormer/models/cm.py:27`

本类字段与枚举值：

- `core` = `core` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.cm.Model`；`hotpot/plugins/ComplexFormer/models/cm.py:16`
- `predictor` = `predictor` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.cm.Model`；`hotpot/plugins/ComplexFormer/models/cm.py:17`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.cm.L.LightningModule`。


<a id="hotpot-plugins-complexformer-models-core"></a>
## `hotpot.plugins.ComplexFormer.models.core`

源码 `hotpot/plugins/ComplexFormer/models/core/__init__.py`；SHA-256 `57dd04b1367ac0283935e64170d2ffa18e8d71fe1b7aee4442fe4ee765c40186`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Core` → `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；构造声明 `__init__(self, vec_dim: int=512, emb_type: Literal['proj', 'atom', 'orbital']='proj', x_label_nums: Optional[int]=None, graph_layer: int=6, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, with_sol_encoder: bool=False, sol_props_nums: Optional[int]=None, sol_props_net_layers: int=2, sol_gnn_layers: int=3, sol_gnn_kw: Optional[dict]=None, with_med_encoder: bool=False, med_props_nums: Optional[int]=None, med_props_net_layers: int=2, med_gnn_layers: int=3, med_gnn_kw: Optional[dict]=None, mol_level_net: Optional[Union[nn.Module, Sequence[int]]]=None, **kwargs)`

模块值/类型别名（声明存在不构成事实写许可）：

- `Core` = `AttnCore`；`hotpot/plugins/ComplexFormer/models/core/__init__.py:27`


<a id="hotpot-plugins-complexformer-models-core--utils"></a>
## `hotpot.plugins.ComplexFormer.models.core._utils`

源码 `hotpot/plugins/ComplexFormer/models/core/_utils.py`；SHA-256 `b01792217fd490a78eaf3aad8a98e70fe0945499c8ec1e6d84171cbd571224fe`。

显式 __all__：`('complete_graph_generator', 'split_padding', 'split_padding_deploy', 'seq_absmax_pooling')`。

显式导出绑定：

- `complete_graph_generator` → `hotpot.plugins.ComplexFormer.models.core._utils.complete_graph_generator`；`complete_graph_generator(ptr)`
- `split_padding` → `hotpot.plugins.ComplexFormer.models.core._utils.split_padding`；`split_padding(x: torch.Tensor, nums: torch.Tensor)`
- `split_padding_deploy` → `hotpot.plugins.ComplexFormer.models.core._utils.split_padding_deploy`；`split_padding_deploy(x: torch.Tensor, nums: torch.Tensor)`
- `seq_absmax_pooling` → `hotpot.plugins.ComplexFormer.models.core._utils.seq_absmax_pooling`；`seq_absmax_pooling(seq)`

<a id="hotpot-plugins-complexformer-models-core--utils-complete-graph-generator"></a>
### `complete_graph_generator`

定义 owner `hotpot.plugins.ComplexFormer.models.core._utils.complete_graph_generator`；源码 `hotpot/plugins/ComplexFormer/models/core/_utils.py:26`。

```python
complete_graph_generator(ptr)
```

<a id="hotpot-plugins-complexformer-models-core--utils-split-padding-deploy"></a>
### `split_padding_deploy`

定义 owner `hotpot.plugins.ComplexFormer.models.core._utils.split_padding_deploy`；源码 `hotpot/plugins/ComplexFormer/models/core/_utils.py:32`。

```python
split_padding_deploy(x: torch.Tensor, nums: torch.Tensor)
```

<a id="hotpot-plugins-complexformer-models-core--utils-split-padding"></a>
### `split_padding`

定义 owner `hotpot.plugins.ComplexFormer.models.core._utils.split_padding`；源码 `hotpot/plugins/ComplexFormer/models/core/_utils.py:100`。

```python
split_padding(x: torch.Tensor, nums: torch.Tensor)
```

<a id="hotpot-plugins-complexformer-models-core--utils-seq-absmax-pooling"></a>
### `seq_absmax_pooling`

定义 owner `hotpot.plugins.ComplexFormer.models.core._utils.seq_absmax_pooling`；源码 `hotpot/plugins/ComplexFormer/models/core/_utils.py:217`。

```python
seq_absmax_pooling(seq)
```


<a id="hotpot-plugins-complexformer-models-core-attn-core"></a>
## `hotpot.plugins.ComplexFormer.models.core.attn_core`

源码 `hotpot/plugins/ComplexFormer/models/core/attn_core.py`；SHA-256 `60952ccad7c5f7ef6361361f7c2f9b717744c9d7584850ea55b036f24154e43f`。

显式 __all__：`('AttnExtractor', 'AttnCore')`。

显式导出绑定：

- `AttnExtractor` → `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`
- `AttnCore` → `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；构造声明 `__init__(self, vec_dim: int=512, emb_type: Literal['proj', 'atom', 'orbital']='proj', x_label_nums: Optional[int]=None, graph_layer: int=6, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, with_sol_encoder: bool=False, sol_props_nums: Optional[int]=None, sol_props_net_layers: int=2, sol_gnn_layers: int=3, sol_gnn_kw: Optional[dict]=None, with_med_encoder: bool=False, med_props_nums: Optional[int]=None, med_props_net_layers: int=2, med_gnn_layers: int=3, med_gnn_kw: Optional[dict]=None, mol_level_net: Optional[Union[nn.Module, Sequence[int]]]=None, **kwargs)`

<a id="hotpot-plugins-complexformer-models-core-attn-core-attnextractor"></a>
### `AttnExtractor`

定义 owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；源码 `hotpot/plugins/ComplexFormer/models/core/attn_core.py:36`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`。

本类属性与方法：

- `extract_atom_vec(seq, X_mask, R_mask, batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:38`
- `extract_bond_vec(seq, X_mask, R_mask, batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:47`
- `extract_metal_vec(seq, X_mask, R_mask, batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:57`
- `extract_cbond_pair(seq, X_mask, R_mask, batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:63`
- `extract_pair_vec(seq, X_mask, R_mask, batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:74`
- `extract_ring_vec(seq, X_mask, R_mask, batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:85`
- `extract_mol_vec(seq, X_mask, R_mask, batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnExtractor`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:97`

<a id="hotpot-plugins-complexformer-models-core-attn-core-attncore"></a>
### `AttnCore`

定义 owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；源码 `hotpot/plugins/ComplexFormer/models/core/attn_core.py:101`。

基类原始声明：`CoreBase`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`。

本类属性与方法：

- `__init__(self, vec_dim: int=512, emb_type: Literal['proj', 'atom', 'orbital']='proj', x_label_nums: Optional[int]=None, graph_layer: int=6, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, with_sol_encoder: bool=False, sol_props_nums: Optional[int]=None, sol_props_net_layers: int=2, sol_gnn_layers: int=3, sol_gnn_kw: Optional[dict]=None, with_med_encoder: bool=False, med_props_nums: Optional[int]=None, med_props_net_layers: int=2, med_gnn_layers: int=3, med_gnn_kw: Optional[dict]=None, mol_level_net: Optional[Union[nn.Module, Sequence[int]]]=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:104`
- `env_encoder(self, net_name: str, batch_size: int, graph_inputs: Optional[Union[dict[str, torch.Tensor], Iterable[dict[str, torch.Tensor]]]]=None, prop_inputs: Optional[Union[torch.Tensor, Iterable[torch.Tensor]]]=None, sol_ratios: torch.Tensor=None)`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:291`
- `mol_level_encoder(self, mol_level_info: Optional[torch.Tensor]=None)`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:358`
- `forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, rings_node_index: torch.Tensor, rings_node_nums: torch.Tensor, mol_rings_nums: torch.Tensor, batch: torch.Tensor, ptr: torch.Tensor, *, xyz: Optional[torch.Tensor]=None, sol_graph: Optional[Iterable[torch.Tensor]]=None, sol_props: Optional[torch.Tensor]=None, med_graph: Optional[Iterable[torch.Tensor]]=None, med_props: Optional[torch.Tensor]=None, mol_level_info: Optional[torch.Tensor]=None)`；装饰器 `overload`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:364`
- `forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, rings_node_index: torch.Tensor, rings_node_nums: torch.Tensor, mol_rings_nums: torch.Tensor, batch: torch.Tensor, ptr: torch.Tensor, *, xyz: Optional[torch.Tensor]=None, sol_ratios: torch.Tensor=None, sol_graph: Optional[Iterable[Iterable[torch.Tensor]]]=None, sol_props: Optional[Iterable[torch.Tensor]]=None, med_graph: Optional[Iterable[torch.Tensor]]=None, med_props: Optional[torch.Tensor]=None, mol_level_info: Optional[torch.Tensor]=None)`；装饰器 `overload`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:385`
- `forward(self, x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr, *, xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]]=None, sol_graph: Optional[Union[dict[str, torch.Tensor], Iterable[dict[str, torch.Tensor]]]]=None, sol_props: Optional[Union[torch.Tensor, Iterable[torch.Tensor]]]=None, sol_ratios: Optional[torch.Tensor]=None, med_graph: Optional[Iterable[torch.Tensor]]=None, med_props: Optional[torch.Tensor]=None, med_ratios: Optional[torch.Tensor]=None, mol_level_info: Optional[Union[torch.Tensor, torch.nested.nested_tensor]]=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:406`
- `labeled_elements(self) -> Optional[bool]` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:210`
- `x_mask_vec(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:219`
- `node_mask(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:223`
- `elem_emb_vec(self) -> Optional[torch.Tensor]` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:227`

本类字段与枚举值：

- `extractor_class` = `AttnExtractor` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:103`
- `emb_type` = `emb_type` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:141`
- `node_processor` = `NodeProcessor(vec_dim, emb_type=emb_type, graph_layer=graph_layer)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:143`
- `ring_encoder` = `nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=vec_dim, nhead=ring_nheads, dim_feedforward=ring_encoder_kw.get('dim_feedforward', 1024) if isinstance(ring_encoder_kw, dict) else 1024, batch_first=True), num_layers=ring_layers, **ring_encoder_block_kw or {})` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:149`
- `mol_encoder` = `nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=vec_dim, nhead=mol_nheads, dim_feedforward=mol_encoder_kw.get('dim_feedforward', 1024) if isinstance(mol_encoder_kw, dict) else 1024, batch_first=True), num_layers=mol_layers, **ring_encoder_block_kw or {})` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:157`
- `CLS` = `nn.Parameter(torch.randn(1, vec_dim))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:166`
- `RING` = `nn.Parameter(torch.randn(1, vec_dim))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:167`
- `END` = `nn.Parameter(torch.randn(1, vec_dim))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:168`
- `sol_encoder` = `SolventNet(vec_dim=vec_dim, node_embedder=self.node_processor.node_embedder, props_nums=sol_props_nums, props_net_layers=sol_props_net_layers, gnn_layers=sol_gnn_layers, gnn_kw=sol_gnn_kw)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:172`
- `med_encoder` = `SolventNet(vec_dim=vec_dim, node_embedder=self.node_processor.node_embedder, props_nums=med_props_nums, props_net_layers=med_props_net_layers, gnn_layers=med_gnn_layers, gnn_kw=med_gnn_kw)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:185`
- `mol_info_net` = `pygnn.MLP(mol_level_net)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:202`

继承成员（逐项保留声明 owner）：

- `extractor_keys` = `{'atom': 'extract_atom_vec', 'bond': 'extract_bond_vec', 'pair': 'extract_pair_vec', 'ring': 'extract_ring_vec', 'mol': 'extract_mol_vec', 'cbond': 'extract_cbond_pair', 'metal': 'extract_metal_vec'}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:22`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`
- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:33`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`
- `x_label_nums` = `x_label_nums` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:34`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`
- `feature_extractor` = `{key: getattr(self.extractor_class, method_name) for key, method_name in self.extractor_keys.items() if hasattr(self.extractor_class, method_name)}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:36`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`

本类名称别名：`extractor_class` → `AttnExtractor`。

模块值/类型别名（声明存在不构成事实写许可）：

- `NodeProcessorType` = `Literal['graph', 'se3']`；`hotpot/plugins/ComplexFormer/models/core/attn_core.py:100`


<a id="hotpot-plugins-complexformer-models-core-base"></a>
## `hotpot.plugins.ComplexFormer.models.core.base`

源码 `hotpot/plugins/ComplexFormer/models/core/base.py`；SHA-256 `cb2017fb610835789aa90918a5ed02a8d8d158f2a187db79534a46a0559aa1f3`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `CoreBase` → `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；构造声明 `__init__(self, vec_dim: int, x_label_nums: Optional[int]=None)`

<a id="hotpot-plugins-complexformer-models-core-base-corebase"></a>
### `CoreBase`

定义 owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；源码 `hotpot/plugins/ComplexFormer/models/core/base.py:19`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int, x_label_nums: Optional[int]=None)`；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:31`
- `x_mask_vec(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:44`

本类字段与枚举值：

- `extractor_class` = `None` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:21`
- `extractor_keys` = `{'atom': 'extract_atom_vec', 'bond': 'extract_bond_vec', 'pair': 'extract_pair_vec', 'ring': 'extract_ring_vec', 'mol': 'extract_mol_vec', 'cbond': 'extract_cbond_pair', 'metal': 'extract_metal_vec'}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:22`
- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:33`
- `x_label_nums` = `x_label_nums` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:34`
- `feature_extractor` = `{key: getattr(self.extractor_class, method_name) for key, method_name in self.extractor_keys.items() if hasattr(self.extractor_class, method_name)}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:36`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.base.nn.Module`。


<a id="hotpot-plugins-complexformer-models-core-cloud"></a>
## `hotpot.plugins.ComplexFormer.models.core.cloud`

源码 `hotpot/plugins/ComplexFormer/models/core/cloud.py`；SHA-256 `9eab03f916671edc8e070be4ff0af6b009d4f0aa42b15ef1f331f73196914de8`。

显式 __all__：`('CloudGraph',)`。

显式导出绑定：

- `CloudGraph` → `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；构造声明 `__init__(self, vec_dim: int)`

<a id="hotpot-plugins-complexformer-models-core-cloud-cloudgraph"></a>
### `CloudGraph`

定义 owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；源码 `hotpot/plugins/ComplexFormer/models/core/cloud.py:29`。

基类原始声明：`nn.Module`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`。

本类属性与方法：

- `__init__(self, vec_dim: int)`；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:31`
- `forward(self, x, xyz, batch)`；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:42`

本类字段与枚举值：

- `xyz_proj` = `nn.Linear(3, vec_dim, bias=False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:33`
- `xyz_proj_norm` = `nn.BatchNorm1d(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:34`
- `lin1` = `nn.Linear(vec_dim, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:36`
- `norm1` = `nn.LayerNorm(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:37`
- `lin2` = `nn.Linear(vec_dim, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:39`
- `norm2` = `nn.LayerNorm(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.cloud.CloudGraph`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:40`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.cloud.nn.Module`。

<a id="hotpot-plugins-complexformer-models-core-cloud-se3cloud"></a>
### `SE3Cloud`

定义 owner `hotpot.plugins.ComplexFormer.models.core.cloud.SE3Cloud`；源码 `hotpot/plugins/ComplexFormer/models/core/cloud.py:70`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.core.cloud.SE3Cloud`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:71`

本类字段与枚举值：

- `se3net` = `SE3Net(vec_dim, **kwargs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.cloud.SE3Cloud`；`hotpot/plugins/ComplexFormer/models/core/cloud.py:73`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.cloud.nn.Module`。


<a id="hotpot-plugins-complexformer-models-core-envs-encoder"></a>
## `hotpot.plugins.ComplexFormer.models.core.envs_encoder`

源码 `hotpot/plugins/ComplexFormer/models/core/envs_encoder.py`；SHA-256 `78e2a4a9b440c654466af1fb1d7d424ae37ce6fa6b85e51e29bd8b84a33dab30`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `SolventNet` → `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；构造声明 `__init__(self, vec_dim: int, node_embedder: nn.Module, props_nums: Optional[int]=None, props_net_layers: int=2, gnn_layers: int=3, gnn_kw: Optional[dict]=None)`

<a id="hotpot-plugins-complexformer-models-core-envs-encoder-solventnet"></a>
### `SolventNet`

定义 owner `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；源码 `hotpot/plugins/ComplexFormer/models/core/envs_encoder.py:23`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int, node_embedder: nn.Module, props_nums: Optional[int]=None, props_net_layers: int=2, gnn_layers: int=3, gnn_kw: Optional[dict]=None)`；owner `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；`hotpot/plugins/ComplexFormer/models/core/envs_encoder.py:24`
- `forward(self, graph_repr: Optional[dict[str, torch.Tensor]]=None, props_vec: torch.Tensor=None) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；`hotpot/plugins/ComplexFormer/models/core/envs_encoder.py:50`

本类字段与枚举值：

- `vec_dim` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；`hotpot/plugins/ComplexFormer/models/core/envs_encoder.py:33`
- `node_embedder` = `node_embedder` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；`hotpot/plugins/ComplexFormer/models/core/envs_encoder.py:44`
- `gnn` = `pyg_nn.GIN(vec_dim, vec_dim, gnn_layers, **gnn_kw)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；`hotpot/plugins/ComplexFormer/models/core/envs_encoder.py:48`
- `props_net` = `pyg_nn.MLP(mlp_nums)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.envs_encoder.SolventNet`；`hotpot/plugins/ComplexFormer/models/core/envs_encoder.py:39`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.envs_encoder.nn.Module`。


<a id="hotpot-plugins-complexformer-models-core-graph"></a>
## `hotpot.plugins.ComplexFormer.models.core.graph`

源码 `hotpot/plugins/ComplexFormer/models/core/graph.py`；SHA-256 `3110853c25d8d3947521a91ce47681b80d29b016f284c391b53014f304007189`。

显式 __all__：`('CompleteGraph',)`。

显式导出绑定：

- `CompleteGraph` → `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；构造声明 `__init__(self, vec_dim: int, n_layers: int, nheads: int)`

<a id="hotpot-plugins-complexformer-models-core-graph-completegraph"></a>
### `CompleteGraph`

定义 owner `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；源码 `hotpot/plugins/ComplexFormer/models/core/graph.py:11`。

基类原始声明：`nn.Module`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`。

本类属性与方法：

- `__init__(self, vec_dim: int, n_layers: int, nheads: int)`；owner `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/core/graph.py:12`
- `ring_forward(self, x, rings_node_index, rings_node_nums)`；owner `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/core/graph.py:71`
- `mol_forward(self, x, xr, mol_ring_nums, mol_ptr, mol_batch) -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/core/graph.py:78`
- `forward(self, *args, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/core/graph.py:93`

本类字段与枚举值：

- `graph` = `pyg_nn.GAT(vec_dim, vec_dim, n_layers, vec_dim, 0.1, norm=pyg_nn.LayerNorm(vec_dim), edge_dim=vec_dim, v2=True, heads=nheads)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/core/graph.py:19`
- `seq_mol_pooling` = `pyg_nn.SAGPooling(vec_dim, 1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/core/graph.py:25`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.graph.nn.Module`。


<a id="hotpot-plugins-complexformer-models-core-graph-core"></a>
## `hotpot.plugins.ComplexFormer.models.core.graph_core`

源码 `hotpot/plugins/ComplexFormer/models/core/graph_core.py`；SHA-256 `cdfbd1240e6a3276c9dfc874c9bb4485099491481dccc264f03d251bad44b0df`。

显式 __all__：`('CompleteGraphExtractor', 'CompleteGraphCore')`。

显式导出绑定：

- `CompleteGraphExtractor` → `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`
- `CompleteGraphCore` → `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；构造声明 `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, **kwargs)`

<a id="hotpot-plugins-complexformer-models-core-graph-core-completegraphextractor"></a>
### `CompleteGraphExtractor`

定义 owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`；源码 `hotpot/plugins/ComplexFormer/models/core/graph_core.py:31`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`。

本类属性与方法：

- `extract_atom_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:33`
- `extract_cbond_pair(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:37`
- `extract_pair_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None) -> Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:50`
- `extract_ring_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:61`
- `extract_mol_vec(mol_vec: Tensor, node_vec: list[Tensor], ring_vec: list[Tensor], batch, batch_getter=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphExtractor`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:65`

<a id="hotpot-plugins-complexformer-models-core-graph-core-completegraphcore"></a>
### `CompleteGraphCore`

定义 owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；源码 `hotpot/plugins/ComplexFormer/models/core/graph_core.py:69`。

基类原始声明：`CoreBase`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`。

本类属性与方法：

- `__init__(self, x_dim: int, vec_dim: int=512, x_label_nums: Optional[int]=None, graph_model: nn.Module=None, ring_layers: int=1, ring_nheads: int=2, ring_encoder_kw: dict=None, ring_encoder_block_kw: dict=None, mol_layers: int=1, mol_nheads: int=4, mol_encoder_kw: dict=None, mol_encoder_block_kw: dict=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:71`
- `forward(self, x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr, *, xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]]=None) -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:97`
- `x_mask_vec(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:110`

本类字段与枚举值：

- `extractor_class` = `CompleteGraphExtractor` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:70`
- `node_processor` = `NodeProcessor(x_dim, vec_dim, x_label_nums=x_label_nums, graph_model=graph_model)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:92`
- `ring_encoder` = `CompleteGraph(vec_dim, ring_layers, ring_nheads)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:94`
- `mol_encoder` = `CompleteGraph(vec_dim, mol_layers, mol_nheads)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.graph_core.CompleteGraphCore`；`hotpot/plugins/ComplexFormer/models/core/graph_core.py:95`

继承成员（逐项保留声明 owner）：

- `extractor_keys` = `{'atom': 'extract_atom_vec', 'bond': 'extract_bond_vec', 'pair': 'extract_pair_vec', 'ring': 'extract_ring_vec', 'mol': 'extract_mol_vec', 'cbond': 'extract_cbond_pair', 'metal': 'extract_metal_vec'}` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:22`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`
- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:33`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`
- `x_label_nums` = `x_label_nums` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:34`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`
- `feature_extractor` = `{key: getattr(self.extractor_class, method_name) for key, method_name in self.extractor_keys.items() if hasattr(self.extractor_class, method_name)}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`；`hotpot/plugins/ComplexFormer/models/core/base.py:36`；继承经 `hotpot.plugins.ComplexFormer.models.core.base.CoreBase`

本类名称别名：`extractor_class` → `CompleteGraphExtractor`。


<a id="hotpot-plugins-complexformer-models-core-node-processor"></a>
## `hotpot.plugins.ComplexFormer.models.core.node_processor`

源码 `hotpot/plugins/ComplexFormer/models/core/node_processor.py`；SHA-256 `b505f5230377e0e5e24edd30ed478cf96ff8a0c8787e39d84e68650bbe97f41e`。

显式 __all__：`('NodeProcessor',)`。

显式导出绑定：

- `NodeProcessor` → `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；构造声明 `__init__(self, vec_dim: int=512, emb_type: Literal['proj', 'atom', 'orbital']='atom', graph_layer: int=6)`

<a id="hotpot-plugins-complexformer-models-core-node-processor-graphcloud"></a>
### `GraphCloud`

定义 owner `hotpot.plugins.ComplexFormer.models.core.node_processor.GraphCloud`；源码 `hotpot/plugins/ComplexFormer/models/core/node_processor.py:28`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int=256, graph_layer: int=6, cloud_layer: int=4, max_l: int=1, parity: Parity=-1, weights_mlp: Sequence[int]=(64,), max_radius: float=4.5, num_radius_basis: int=10)`；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.GraphCloud`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:29`
- `forward(self, x, edge_index=None, batch=None, xyz=None)`；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.GraphCloud`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:52`

本类字段与枚举值：

- `graph` = `pygnn.GAT(vec_dim, vec_dim, graph_layer, vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim), edge_dim=vec_dim, v2=True)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.GraphCloud`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:41`
- `cloud` = `SE3Net(vec_dim, max_l=max_l, parity=parity, num_layers=cloud_layer, max_radius=max_radius, num_radius_basis=num_radius_basis)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.GraphCloud`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:46`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.node_processor.nn.Module`。

<a id="hotpot-plugins-complexformer-models-core-node-processor-nodeemb"></a>
### `NodeEmb`

定义 owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；源码 `hotpot/plugins/ComplexFormer/models/core/node_processor.py:76`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int=256, emb_type: Literal['proj', 'atom', 'orbital']='atom')`；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:77`
- `forward(self, x)`；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:104`
- `node_mask(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:117`

本类字段与枚举值：

- `emb_type` = `emb_type` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:83`
- `x_proj` = `pygnn.MLP([7, 1024, vec_dim])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:85`
- `x_mask_vec` = `nn.Parameter(torch.randn(7))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:86`
- `x_emb` = `nn.Embedding(120, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:88`
- `emb_dim` = `vec_dim // 4` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:91`
- `emb_segment` = `[0, 120, 8, 3, 7, 11, 15, 19]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:93`
- `emb_proj` = `pygnn.MLP([self.emb_dim * 7, vec_dim])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeEmb`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:99`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.node_processor.nn.Module`。

<a id="hotpot-plugins-complexformer-models-core-node-processor-nodeprocessor"></a>
### `NodeProcessor`

定义 owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；源码 `hotpot/plugins/ComplexFormer/models/core/node_processor.py:128`。

基类原始声明：`nn.Module`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`。

本类属性与方法：

- `__init__(self, vec_dim: int=512, emb_type: Literal['proj', 'atom', 'orbital']='atom', graph_layer: int=6)`；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:129`
- `forward(self, x, edge_index=None, batch=None, xyz=None)`；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:153`
- `node_mask(self)` [读；只读]；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:181`

本类字段与枚举值：

- `vec_size` = `vec_dim` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:136`
- `node_embedder` = `NodeEmb(vec_dim, emb_type)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:137`
- `graph` = `pygnn.GAT(vec_dim, vec_dim, graph_layer, vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim), edge_dim=vec_dim, v2=True)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:139`
- `cloud` = `SE3Net(vec_dim, max_l=1, parity=-1, num_layers=4, max_radius=4.5, num_radius_basis=10)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:145`
- `lin` = `nn.Linear(vec_dim, vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:150`
- `norm` = `nn.BatchNorm1d(vec_dim)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.node_processor.NodeProcessor`；`hotpot/plugins/ComplexFormer/models/core/node_processor.py:151`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.node_processor.nn.Module`。


<a id="hotpot-plugins-complexformer-models-core-se3net"></a>
## `hotpot.plugins.ComplexFormer.models.core.se3net`

源码 `hotpot/plugins/ComplexFormer/models/core/se3net.py`；SHA-256 `2e6668dbbc4716ae739cc2b5cf3e26eaeaee3e842298391b370c2ac030f79562`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `auto_irreps` → `hotpot.plugins.ComplexFormer.models.core.se3net.auto_irreps`；`auto_irreps(emb_dim: int, max_l: int, parity: Parity=-1) -> Irreps`
- `SE3EquivLayer` → `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；构造声明 `__init__(self, emb_dim: int, in_irreps: Optional[str]=None, out_irreps: Optional[str]=None, max_l: Optional[int]=1, parity: Parity=-1, weights_mlp: Sequence[int]=(64,), max_radius: float=4.5, num_radius_basis: int=10)`
- `SE3Net` → `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；构造声明 `__init__(self, emb_dim: int=64, in_irreps: Optional[Irreps]=None, out_irreps: Optional[Irreps]=None, max_l: int=1, parity: Parity=-1, weights_mlp: Sequence[int]=(16, 64), num_layers: int=2, max_radius: float=4.5, num_radius_basis: int=10)`
- `Parity` → `hotpot.plugins.ComplexFormer.models.core.se3net.Parity`
- `N` → `hotpot.plugins.ComplexFormer.models.core.se3net.N`
- `emb_dim` → `hotpot.plugins.ComplexFormer.models.core.se3net.emb_dim`
- `max_l` → `hotpot.plugins.ComplexFormer.models.core.se3net.max_l`
- `xyz` → `hotpot.plugins.ComplexFormer.models.core.se3net.xyz`
- `h` → `hotpot.plugins.ComplexFormer.models.core.se3net.h`
- `rot_mat` → `hotpot.plugins.ComplexFormer.models.core.se3net.rot_mat`
- `euler_angles` → `hotpot.plugins.ComplexFormer.models.core.se3net.euler_angles`
- `net` → `hotpot.plugins.ComplexFormer.models.core.se3net.net`
- `D_in` → `hotpot.plugins.ComplexFormer.models.core.se3net.D_in`
- `D_out` → `hotpot.plugins.ComplexFormer.models.core.se3net.D_out`
- `y_rot_before` → `hotpot.plugins.ComplexFormer.models.core.se3net.y_rot_before`
- `y_rot_after` → `hotpot.plugins.ComplexFormer.models.core.se3net.y_rot_after`
- `res` → `hotpot.plugins.ComplexFormer.models.core.se3net.res`
- `fig` → `hotpot.plugins.ComplexFormer.models.core.se3net.fig`
- `axs` → `hotpot.plugins.ComplexFormer.models.core.se3net.axs`

<a id="hotpot-plugins-complexformer-models-core-se3net-auto-irreps"></a>
### `auto_irreps`

定义 owner `hotpot.plugins.ComplexFormer.models.core.se3net.auto_irreps`；源码 `hotpot/plugins/ComplexFormer/models/core/se3net.py:71`。

```python
auto_irreps(emb_dim: int, max_l: int, parity: Parity=-1) -> Irreps
```

<a id="hotpot-plugins-complexformer-models-core-se3net-se3equivlayer"></a>
### `SE3EquivLayer`

定义 owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；源码 `hotpot/plugins/ComplexFormer/models/core/se3net.py:103`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, emb_dim: int, in_irreps: Optional[str]=None, out_irreps: Optional[str]=None, max_l: Optional[int]=1, parity: Parity=-1, weights_mlp: Sequence[int]=(64,), max_radius: float=4.5, num_radius_basis: int=10)`；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:104`
- `forward(self, x, src, dst, edge_vec)`；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:130`

本类字段与枚举值：

- `in_irreps` = `Irreps(_get_irreps(in_irreps, emb_dim, max_l, parity))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:117`
- `out_irreps` = `Irreps(_get_irreps(out_irreps, emb_dim, max_l, parity))` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:118`
- `lmax` = `self.in_irreps.lmax * self.out_irreps.lmax` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:120`
- `sh_irreps` = `Irreps.spherical_harmonics(lmax=self.lmax, p=parity)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:121`
- `tp` = `FullyConnectedTensorProduct(self.in_irreps, self.sh_irreps, self.out_irreps, shared_weights=False)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:123`
- `radius_mlp_channels` = `[num_radius_basis] + list(weights_mlp) + [self.tp.weight_numel]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:125`
- `radius_mlp` = `FullyConnectedNet(self.radius_mlp_channels, torch.relu)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:126`
- `max_radius` = `max_radius` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:127`
- `num_radius_basis` = `num_radius_basis` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3EquivLayer`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:128`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.se3net.nn.Module`。

<a id="hotpot-plugins-complexformer-models-core-se3net-se3net"></a>
### `SE3Net`

定义 owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；源码 `hotpot/plugins/ComplexFormer/models/core/se3net.py:149`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, emb_dim: int=64, in_irreps: Optional[Irreps]=None, out_irreps: Optional[Irreps]=None, max_l: int=1, parity: Parity=-1, weights_mlp: Sequence[int]=(16, 64), num_layers: int=2, max_radius: float=4.5, num_radius_basis: int=10)`；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:150`
- `D_in(self, rot: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:173`
- `D_out(self, rot: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:177`
- `forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor]=None) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:181`

本类字段与枚举值：

- `max_radius` = `max_radius` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:163`
- `layers` = `nn.ModuleList([SE3EquivLayer(emb_dim, in_irreps, out_irreps, max_l, parity, weights_mlp, max_radius, num_radius_basis) for _ in range(num_layers)])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.core.se3net.SE3Net`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:168`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.core.se3net.nn.Module`。

模块值/类型别名（声明存在不构成事实写许可）：

- `Parity` = `Enum('Parity', [('Even', 1), ('Odd', -1)])`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:57`
- `N` = `(16, 64, 2)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:201`
- `emb_dim` = `(16, 64, 2)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:201`
- `max_l` = `(16, 64, 2)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:201`
- `xyz` = `torch.randn(N, 3)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:202`
- `h` = `torch.ones(N, emb_dim)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:203`
- `rot_mat` = `o3.rand_matrix()`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:205`
- `euler_angles` = `Rot.from_matrix(rot_mat).as_euler('zyx', degrees=True)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:206`
- `net` = `SE3Net(emb_dim=emb_dim, max_l=max_l)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:211`
- `D_in` = `net.D_in(rot_mat).numpy()`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:213`
- `D_out` = `net.D_out(rot_mat).numpy()`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:214`
- `y_rot_before` = `net(h @ net.D_out(rot_mat).T, xyz @ rot_mat.T)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:215`
- `y_rot_after` = `net(h, xyz) @ net.D_out(rot_mat).T`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:216`
- `res` = `torch.allclose(y_rot_after, y_rot_before, rtol=0.0001, atol=0.0001)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:218`
- `fig` = `plt.subplots(2, 2)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:222`
- `axs` = `plt.subplots(2, 2)`；`hotpot/plugins/ComplexFormer/models/core/se3net.py:222`


<a id="hotpot-plugins-complexformer-models-graph"></a>
## `hotpot.plugins.ComplexFormer.models.graph`

源码 `hotpot/plugins/ComplexFormer/models/graph.py`；SHA-256 `d58df76f7e6fdddddde02a919a9541ac33afe129c96873a06e2a7c9ef29fe21e`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `CompleteGraph` → `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；构造声明 `__init__(self, vec_dim: int, n_layers: int, nheads: int)`

<a id="hotpot-plugins-complexformer-models-graph-completegraph"></a>
### `CompleteGraph`

定义 owner `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；源码 `hotpot/plugins/ComplexFormer/models/graph.py:10`。

基类原始声明：`nn.Module`；装饰器：无。

本类属性与方法：

- `__init__(self, vec_dim: int, n_layers: int, nheads: int)`；owner `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/graph.py:11`
- `ring_forward(self, x, rings_node_index, rings_node_nums)`；owner `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/graph.py:70`
- `mol_forward(self, x, xr, mol_ring_nums, mol_ptr, mol_batch) -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/graph.py:77`
- `forward(self, *args, **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/graph.py:92`

本类字段与枚举值：

- `graph` = `pyg_nn.GAT(vec_dim, vec_dim, n_layers, vec_dim, 0.1, norm=pyg_nn.LayerNorm(vec_dim), edge_dim=vec_dim, v2=True, heads=nheads)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/graph.py:18`
- `seq_mol_pooling` = `pyg_nn.SAGPooling(vec_dim, 1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.graph.CompleteGraph`；`hotpot/plugins/ComplexFormer/models/graph.py:24`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.graph.nn.Module`。


<a id="hotpot-plugins-complexformer-models-loss"></a>
## `hotpot.plugins.ComplexFormer.models.loss`

源码 `hotpot/plugins/ComplexFormer/models/loss.py`；SHA-256 `bd74d6dd6386cff078f7df6a5d4386079e3d92a6754e5a2ade2007b3372a93aa`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `LossMethods` → `hotpot.plugins.ComplexFormer.models.loss.LossMethods`
- `label_weights_calculator` → `hotpot.plugins.ComplexFormer.models.loss.label_weights_calculator`；`label_weights_calculator(onehot_types: int, weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='cross-entropy')`
- `atl_calculator` → `hotpot.plugins.ComplexFormer.models.loss.atl_calculator`；`atl_calculator(metrics: dict[str, float], p: Union[int, float]=3, eps: float=0.0001)`

<a id="hotpot-plugins-complexformer-models-loss-lossmethods"></a>
### `LossMethods`

定义 owner `hotpot.plugins.ComplexFormer.models.loss.LossMethods`；源码 `hotpot/plugins/ComplexFormer/models/loss.py:9`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `calc_atom_type_loss(pred, target, weight=None, acc=torch.tensor(1.0))` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.loss.LossMethods`；`hotpot/plugins/ComplexFormer/models/loss.py:12`
- `average_maximum_displacement(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], *args, **kwargs) -> Union[torch.Tensor, np.ndarray, float]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.loss.LossMethods`；`hotpot/plugins/ComplexFormer/models/loss.py:21`

<a id="hotpot-plugins-complexformer-models-loss-label-weights-calculator"></a>
### `label_weights_calculator`

定义 owner `hotpot.plugins.ComplexFormer.models.loss.label_weights_calculator`；源码 `hotpot/plugins/ComplexFormer/models/loss.py:37`。

```python
label_weights_calculator(onehot_types: int, weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='cross-entropy')
```

<a id="hotpot-plugins-complexformer-models-loss-atl-calculator"></a>
### `atl_calculator`

定义 owner `hotpot.plugins.ComplexFormer.models.loss.atl_calculator`；源码 `hotpot/plugins/ComplexFormer/models/loss.py:63`。

```python
atl_calculator(metrics: dict[str, float], p: Union[int, float]=3, eps: float=0.0001)
```


<a id="hotpot-plugins-complexformer-models-metrics"></a>
## `hotpot.plugins.ComplexFormer.models.metrics`

源码 `hotpot/plugins/ComplexFormer/models/metrics.py`；SHA-256 `a782843895fa5b61014a829834e9ff66513ed8b435b2599f524d3e86521e6e9c`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Metrics` → `hotpot.plugins.ComplexFormer.models.metrics.Metrics`

<a id="hotpot-plugins-complexformer-models-metrics-metrics"></a>
### `Metrics`

定义 owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；源码 `hotpot/plugins/ComplexFormer/models/metrics.py:10`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `average_inverse_distance(pred, target)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:13`
- `calc_oh_accuracy(pred, target)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:28`
- `metal_oh_accuracy(pred, target)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:38`
- `binary_accuracy(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray])` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:53`
- `r2_score(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:59`
- `rmse(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:93`
- `mse(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:103`
- `mae(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:113`
- `precision(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:123`
- `recall(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:132`
- `f1_score(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:147`
- `mf1_score(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor])` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:157`
- `auc(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> float` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:174`
- `roc(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> (Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], float)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.models.metrics.Metrics`；`hotpot/plugins/ComplexFormer/models/metrics.py:183`


<a id="hotpot-plugins-complexformer-models-plot-makers"></a>
## `hotpot.plugins.ComplexFormer.models.plot_makers`

源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py`；SHA-256 `287bcbf0adb7ff4e6eaacf656d33122d1b5864fde61d5425dc0be671a5e63d6d`。

显式 __all__：`('plots_options',)`。

显式导出绑定：

- `plots_options` → `hotpot.plugins.ComplexFormer.models.plot_makers.plots_options`

<a id="hotpot-plugins-complexformer-models-plot-makers-confusion-matrix"></a>
### `confusion_matrix`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.confusion_matrix`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:51`。

```python
confusion_matrix(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

<a id="hotpot-plugins-complexformer-models-plot-makers-binary-confusion-metrix"></a>
### `binary_confusion_metrix`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.binary_confusion_metrix`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:54`。

```python
binary_confusion_metrix(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

<a id="hotpot-plugins-complexformer-models-plot-makers-r2-regression"></a>
### `r2_regression`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.r2_regression`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:57`。

```python
r2_regression(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

<a id="hotpot-plugins-complexformer-models-plot-makers-roc-curve"></a>
### `roc_curve`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.roc_curve`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:60`。

```python
roc_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

<a id="hotpot-plugins-complexformer-models-plot-makers-det-curve"></a>
### `det_curve`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.det_curve`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:63`。

```python
det_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

<a id="hotpot-plugins-complexformer-models-plot-makers-precision-recall-curve"></a>
### `precision_recall_curve`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.precision_recall_curve`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:66`。

```python
precision_recall_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

<a id="hotpot-plugins-complexformer-models-plot-makers-multiclass-roc-curve"></a>
### `multiclass_roc_curve`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.multiclass_roc_curve`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:69`。

```python
multiclass_roc_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

<a id="hotpot-plugins-complexformer-models-plot-makers-hist"></a>
### `hist`

定义 owner `hotpot.plugins.ComplexFormer.models.plot_makers.hist`；源码 `hotpot/plugins/ComplexFormer/models/plot_makers.py:72`。

```python
hist(pred: np.ndarray, target: np.ndarray) -> plt.Figure
```

模块值/类型别名（声明存在不构成事实写许可）：

- `plots_options` = `{'r2': r2_regression, 'bconf': binary_confusion_metrix, 'roc': roc_curve, 'det': det_curve, 'prc': precision_recall_curve, 'conf': confusion_matrix, 'mroc': multiclass_roc_curve, 'hist': hist}`；`hotpot/plugins/ComplexFormer/models/plot_makers.py:81`


<a id="hotpot-plugins-complexformer-models-predictor"></a>
## `hotpot.plugins.ComplexFormer.models.predictor`

源码 `hotpot/plugins/ComplexFormer/models/predictor.py`；SHA-256 `2a23a104d4370608af1f58e3f0b25e466a8909822284d504a7f486f3fc17e3a3`。

显式 __all__：`('TargetTypeName', 'Predictor', 'apply_lora_to_predictor')`。

显式导出绑定：

- `TargetTypeName` → `hotpot.plugins.ComplexFormer.models.predictor.TargetTypeName`
- `Predictor` → `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；构造声明 `__init__(self, in_size: int, target_type: TargetTypeName, hidden_dim: int=256, num_layers: int=1, out_size: int=1024, dropout: float=0.1, out_act: Type[nn.Module]=nn.ReLU, name: str='', **kwargs)`
- `apply_lora_to_predictor` → `hotpot.plugins.ComplexFormer.models.predictor.apply_lora_to_predictor`；`apply_lora_to_predictor(predictor_model, target_modules=None, rank=4, alpha=16, dropout=0.3, **kwargs)`

<a id="hotpot-plugins-complexformer-models-predictor-replace-pyg-linear-in-mlp"></a>
### `replace_pyg_linear_in_mlp`

定义 owner `hotpot.plugins.ComplexFormer.models.predictor.replace_pyg_linear_in_mlp`；源码 `hotpot/plugins/ComplexFormer/models/predictor.py:19`。

```python
replace_pyg_linear_in_mlp(mlp_module)
```

<a id="hotpot-plugins-complexformer-models-predictor-predictor"></a>
### `Predictor`

定义 owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；源码 `hotpot/plugins/ComplexFormer/models/predictor.py:59`。

基类原始声明：`nn.Module`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.models.predictor.Predictor`。

本类属性与方法：

- `__init__(self, in_size: int, target_type: TargetTypeName, hidden_dim: int=256, num_layers: int=1, out_size: int=1024, dropout: float=0.1, out_act: Type[nn.Module]=nn.ReLU, name: str='', **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:61`
- `forward(self, z)`；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:106`
- `convert_to_standard_linear(self)`；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:117`
- `apply_lora(self, rank=4, alpha=4, dropout=0.3, force: bool=False, target_modules='lins\\.\\d+', **kwargs)`；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:122`

本类字段与枚举值：

- `define_types` = `('num', 'xyz', 'onehot', 'binary')` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:60`
- `name` = `name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:74`
- `num_layers` = `num_layers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:75`
- `in_layers` = `pygnn.MLP([in_size, hidden_dim])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:76`
- `represent_layer` = `pygnn.MLP([hidden_dim, out_size], dropout=dropout, norm=None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:81`
- `target_type` = `target_type` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:83`
- `hidden_layers` = `pygnn.MLP((num_layers + 1) * [hidden_dim], dropout=dropout, norm=None)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:78`
- `out_layer` = `nn.Linear(out_size, 1)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:86`
- `out_act` = `nn.LeakyReLU()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:87`
- `onehot_type` = `kwargs['onehot_type']` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.models.predictor.Predictor`；`hotpot/plugins/ComplexFormer/models/predictor.py:94`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.models.predictor.nn.Module`。

<a id="hotpot-plugins-complexformer-models-predictor-apply-lora-to-predictor"></a>
### `apply_lora_to_predictor`

定义 owner `hotpot.plugins.ComplexFormer.models.predictor.apply_lora_to_predictor`；源码 `hotpot/plugins/ComplexFormer/models/predictor.py:132`。

```python
apply_lora_to_predictor(predictor_model, target_modules=None, rank=4, alpha=16, dropout=0.3, **kwargs)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `TargetTypeName` = `Literal['num', 'xyz', 'onehot', 'binary']`；`hotpot/plugins/ComplexFormer/models/predictor.py:58`


<a id="hotpot-plugins-complexformer-models-utils"></a>
## `hotpot.plugins.ComplexFormer.models.utils`

源码 `hotpot/plugins/ComplexFormer/models/utils.py`；SHA-256 `c6f456691f361d5d30e2eb71650a766a19d5ad36e16b55d8bc9ea9090fc682cb`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `norm_binary_to_zero_one` → `hotpot.plugins.ComplexFormer.models.utils.norm_binary_to_zero_one`；`norm_binary_to_zero_one(inp_vec: Union[torch.Tensor, np.ndarray]) -> Union[np.ndarray, torch.Tensor]`
- `oh2label` → `hotpot.plugins.ComplexFormer.models.utils.oh2label`；`oh2label(inp_vec: Union[torch.Tensor, np.ndarray])`
- `inverse_onehot` → `hotpot.plugins.ComplexFormer.models.utils.inverse_onehot`；`inverse_onehot(is_onehot, *onehot_vecs: Union[torch.Tensor, np.ndarray])`
- `where_metal` → `hotpot.plugins.ComplexFormer.models.utils.where_metal`；`where_metal(type_labels: Union[torch.Tensor, np.ndarray])`
- `weight_labels` → `hotpot.plugins.ComplexFormer.models.utils.weight_labels`；`weight_labels(labels: torch.Tensor, num_types: int=119, weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='cross-entropy') -> torch.Tensor`
- `weight_binary` → `hotpot.plugins.ComplexFormer.models.utils.weight_binary`；`weight_binary(labels: torch.Tensor, eps: float=1e-07)`
- `get_x_input_attrs` → `hotpot.plugins.ComplexFormer.models.utils.get_x_input_attrs`；`get_x_input_attrs(inputs: Union[dict, list, tuple], input_x_index: Union[list, torch.Tensor], dtype: Optional[torch.dtype]=None)`
- `get_labeled_x_input_attrs` → `hotpot.plugins.ComplexFormer.models.utils.get_labeled_x_input_attrs`；`get_labeled_x_input_attrs(inputs, input_x_index: Union[list, torch.Tensor]=None)`
- `get_masked_input_and_labels` → `hotpot.plugins.ComplexFormer.models.utils.get_masked_input_and_labels`；`get_masked_input_and_labels(inp_vec: torch.Tensor, masked_idx: torch.Tensor, mask_vec: torch.Tensor, inp_atom_labels: torch.Tensor, label_mask: bool=False, to_mask_label: int=0)`
- `mask_atom_type` → `hotpot.plugins.ComplexFormer.models.utils.mask_atom_type`；`mask_atom_type(inputs: tuple, node_mask: Union[int, torch.Tensor])`
- `mask_metal_type` → `hotpot.plugins.ComplexFormer.models.utils.mask_metal_type`；`mask_metal_type(inputs: tuple, node_mask: Union[int, torch.Tensor])`
- `x_masker_func` → `hotpot.plugins.ComplexFormer.models.utils.x_masker_func`；`x_masker_func(inputs: tuple, masked_vec: torch.Tensor)`
- `metal_masker_func` → `hotpot.plugins.ComplexFormer.models.utils.metal_masker_func`；`metal_masker_func(inputs: tuple, masked_vec: torch.Tensor)`
- `torch_numpy_exchanger` → `hotpot.plugins.ComplexFormer.models.utils.torch_numpy_exchanger`；`torch_numpy_exchanger(nf: Callable, **kw)`
- `perturb_xyz` → `hotpot.plugins.ComplexFormer.models.utils.perturb_xyz`；`perturb_xyz(xyz: torch.Tensor, sigma: float=1.0, mode: PerturbMode='uniform')`
- `electron_config_tensor` → `hotpot.plugins.ComplexFormer.models.utils.electron_config_tensor`
- `metal_index` → `hotpot.plugins.ComplexFormer.models.utils.metal_index`
- `PerturbMode` → `hotpot.plugins.ComplexFormer.models.utils.PerturbMode`

<a id="hotpot-plugins-complexformer-models-utils-norm-binary-to-zero-one"></a>
### `norm_binary_to_zero_one`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.norm_binary_to_zero_one`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:15`。

```python
norm_binary_to_zero_one(inp_vec: Union[torch.Tensor, np.ndarray]) -> Union[np.ndarray, torch.Tensor]
```

<a id="hotpot-plugins-complexformer-models-utils-oh2label"></a>
### `oh2label`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.oh2label`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:24`。

```python
oh2label(inp_vec: Union[torch.Tensor, np.ndarray])
```

<a id="hotpot-plugins-complexformer-models-utils-inverse-onehot"></a>
### `inverse_onehot`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.inverse_onehot`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:36`。

```python
inverse_onehot(is_onehot, *onehot_vecs: Union[torch.Tensor, np.ndarray])
```

<a id="hotpot-plugins-complexformer-models-utils-where-metal"></a>
### `where_metal`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.where_metal`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:53`。

```python
where_metal(type_labels: Union[torch.Tensor, np.ndarray])
```

<a id="hotpot-plugins-complexformer-models-utils-weight-labels"></a>
### `weight_labels`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.weight_labels`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:60`。

```python
weight_labels(labels: torch.Tensor, num_types: int=119, weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='cross-entropy') -> torch.Tensor
```

<a id="hotpot-plugins-complexformer-models-utils-weight-binary"></a>
### `weight_binary`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.weight_binary`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:86`。

```python
weight_binary(labels: torch.Tensor, eps: float=1e-07)
```

<a id="hotpot-plugins-complexformer-models-utils-get-x-input-attrs"></a>
### `get_x_input_attrs`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.get_x_input_attrs`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:109`。

```python
get_x_input_attrs(inputs: Union[dict, list, tuple], input_x_index: Union[list, torch.Tensor], dtype: Optional[torch.dtype]=None)
```

<a id="hotpot-plugins-complexformer-models-utils-get-labeled-x-input-attrs"></a>
### `get_labeled_x_input_attrs`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.get_labeled_x_input_attrs`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:122`。

```python
get_labeled_x_input_attrs(inputs, input_x_index: Union[list, torch.Tensor]=None)
```

<a id="hotpot-plugins-complexformer-models-utils-get-masked-input-and-labels"></a>
### `get_masked_input_and_labels`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.get_masked_input_and_labels`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:175`。

```python
get_masked_input_and_labels(inp_vec: torch.Tensor, masked_idx: torch.Tensor, mask_vec: torch.Tensor, inp_atom_labels: torch.Tensor, label_mask: bool=False, to_mask_label: int=0)
```

<a id="hotpot-plugins-complexformer-models-utils-mask-atom-type"></a>
### `mask_atom_type`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.mask_atom_type`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:225`。

```python
mask_atom_type(inputs: tuple, node_mask: Union[int, torch.Tensor])
```

装饰器：`_masker_func(_select_atom_type_index)`。

<a id="hotpot-plugins-complexformer-models-utils-mask-metal-type"></a>
### `mask_metal_type`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.mask_metal_type`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:238`。

```python
mask_metal_type(inputs: tuple, node_mask: Union[int, torch.Tensor])
```

装饰器：`_masker_func(_select_metal_type_index)`。

<a id="hotpot-plugins-complexformer-models-utils-x-masker-func"></a>
### `x_masker_func`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.x_masker_func`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:251`。

```python
x_masker_func(inputs: tuple, masked_vec: torch.Tensor)
```

<a id="hotpot-plugins-complexformer-models-utils-metal-masker-func"></a>
### `metal_masker_func`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.metal_masker_func`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:260`。

```python
metal_masker_func(inputs: tuple, masked_vec: torch.Tensor)
```

<a id="hotpot-plugins-complexformer-models-utils-torch-numpy-exchanger"></a>
### `torch_numpy_exchanger`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.torch_numpy_exchanger`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:265`。

```python
torch_numpy_exchanger(nf: Callable, **kw)
```

<a id="hotpot-plugins-complexformer-models-utils-perturb-xyz"></a>
### `perturb_xyz`

定义 owner `hotpot.plugins.ComplexFormer.models.utils.perturb_xyz`；源码 `hotpot/plugins/ComplexFormer/models/utils.py:275`。

```python
perturb_xyz(xyz: torch.Tensor, sigma: float=1.0, mode: PerturbMode='uniform')
```

模块值/类型别名（声明存在不构成事实写许可）：

- `electron_config_tensor` = `torch.cat((torch.arange(0, 120).unsqueeze(-1), torch.tensor(elements.electron_configs)), dim=1)`；`hotpot/plugins/ComplexFormer/models/utils.py:12`
- `metal_index` = `torch.tensor(list(elements.metal | elements.metalloid_2nd))`；`hotpot/plugins/ComplexFormer/models/utils.py:162`
- `PerturbMode` = `Literal['norm', 'uniform']`；`hotpot/plugins/ComplexFormer/models/utils.py:274`


<a id="hotpot-plugins-complexformer-optim-config"></a>
## `hotpot.plugins.ComplexFormer.optim_config`

源码 `hotpot/plugins/ComplexFormer/optim_config.py`；SHA-256 `4390ffe691c23c10759b647e80c601c7bf28dc309c112aa8c4ba0e491c9858e4`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `OptimizerConfigure` → `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；构造声明 `OptimizerConfigure(hypers: ParamSets, optimizer: Optional[Type[Optimizer]] = None, lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None, constant_lr: bool = False, lr_scheduler_frequency: int = 2, lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict), primary_monitor: Optional[str] = None, task: Optional[Any] = None, required_hypers: list[str] = ('lr', 'weight_decay'))`

<a id="hotpot-plugins-complexformer-optim-config-optimizerconfigure"></a>
### `OptimizerConfigure`

定义 owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；源码 `hotpot/plugins/ComplexFormer/optim_config.py:22`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
OptimizerConfigure(hypers: ParamSets, optimizer: Optional[Type[Optimizer]] = None, lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None, constant_lr: bool = False, lr_scheduler_frequency: int = 2, lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict), primary_monitor: Optional[str] = None, task: Optional[Any] = None, required_hypers: list[str] = ('lr', 'weight_decay'))
```

本类属性与方法：

- `parse_primary_monitor_from_task_type(task)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:46`
- `__call__(self, pl_module: L.LightningModule) -> Dict[str, Any]`；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:78`

本类字段与枚举值：

- `hypers: ParamSets` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:26`
- `optimizer: Optional[Type[Optimizer]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:30`
- `lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:31`
- `constant_lr: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:34`
- `lr_scheduler_frequency: int` = `2` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:35`
- `lr_scheduler_kwargs: Dict[str, Any]` = `field(default_factory=dict)` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:37`
- `primary_monitor: Optional[str]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:40`
- `task: Optional[Any]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:41`
- `required_hypers: list[str]` = `('lr', 'weight_decay')` [可写值字段]；owner `hotpot.plugins.ComplexFormer.optim_config.OptimizerConfigure`；`hotpot/plugins/ComplexFormer/optim_config.py:43`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.


<a id="hotpot-plugins-complexformer-run"></a>
## `hotpot.plugins.ComplexFormer.run`

源码 `hotpot/plugins/ComplexFormer/run/__init__.py`；SHA-256 `d3636c28d975ec7067af5d28c390997f3aa7025c4428d18a28a28edfd7f82fc6`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-complexformer-run-cv"></a>
## `hotpot.plugins.ComplexFormer.run.cv`

源码 `hotpot/plugins/ComplexFormer/run/cv.py`；SHA-256 `b63e3a3b28fd0e8a1864be3afcacb85ba73fdb253c0648152132bd40c2dc9112`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `cross_validation` → `hotpot.plugins.ComplexFormer.run.cv.cross_validation`；`cross_validation(trainer: pl.Trainer, pl_module: LightPretrain, datamodule: DataModule, n_splits: int=5, lora_cfg: LoraConfig=None)`
- `run_cv` → `hotpot.plugins.ComplexFormer.run.cv.run_cv`；`run_cv(log_dir: os.PathLike, run_args: datacls.RunArgs, n_splits: int=5)`

<a id="hotpot-plugins-complexformer-run-cv-cross-validation"></a>
### `cross_validation`

定义 owner `hotpot.plugins.ComplexFormer.run.cv.cross_validation`；源码 `hotpot/plugins/ComplexFormer/run/cv.py:30`。

```python
cross_validation(trainer: pl.Trainer, pl_module: LightPretrain, datamodule: DataModule, n_splits: int=5, lora_cfg: LoraConfig=None)
```

<a id="hotpot-plugins-complexformer-run-cv-run-cv"></a>
### `run_cv`

定义 owner `hotpot.plugins.ComplexFormer.run.cv.run_cv`；源码 `hotpot/plugins/ComplexFormer/run/cv.py:62`。

```python
run_cv(log_dir: os.PathLike, run_args: datacls.RunArgs, n_splits: int=5)
```


<a id="hotpot-plugins-complexformer-run-datacls"></a>
## `hotpot.plugins.ComplexFormer.run.datacls`

源码 `hotpot/plugins/ComplexFormer/run/datacls.py`；SHA-256 `936ba0f801be1db90c209cdc1310db68e983ede755f1ab88de49904c76fe8b38`。

显式 __all__：`('merge_dataclass', 'RunArgs', 'DataModuleArgs', 'ConfigArgs')`。

显式导出绑定：

- `merge_dataclass` → `hotpot.plugins.ComplexFormer.run.datacls.merge_dataclass`；`merge_dataclass(target: T, *sources: Any, exclude_none: bool=False, **kwargs)`
- `RunArgs` → `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；构造声明 `RunArgs(work_name: str, work_dir: str, hypers: Union['ParamSets', 'ParamSpace'], dir_datasets: str, dataset_names: Optional[Union[str, Sequence[str]]] = None, exclude_datasets: Optional[Union[str, Sequence[str]]] = None, shuffle_dataset: bool = True, dataModule_seed: int = 315, data_split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), external_datasets: Optional[Union[str, Sequence[str]]] = None, stages: Optional[Union[tp.Stages, list[tp.Stages]]] = None, eval_each_step: Optional[int] = 1, epochs: int = 100, batch_size: int = 512, early_stopping: bool = True, early_stop_step: int = 10, target_metrics: Union[str, List[str], Callable[[dict], float], None] = None, num_trials: int = 10, checkpoint_path: Union[str, int, None] = None, optimizer: Optional[Type['Optimizer']] = None, constant_lr: bool = False, lr_scheduler: Optional[Callable] = None, lr_scheduler_frequency: int = 2, lr_scheduler_kwargs: Optional[dict] = None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count', inputs_getter: Optional[Any] = INPUT_GETTER, xyz_perturb_sigma: Optional[float] = None, xyz_perturb_mode: 'M.PerturbMode' = 'uniform', batch_preprocessor: Optional[Union['tp.BatchPreProcessor', List['tp.BatchPreProcessor']]] = None, inputs_preprocessor: Optional[Union[Callable, List[Callable]]] = None, x_masker: Optional[Union[str, Callable]] = None, mask_need_task: Optional[List[str]] = None, with_xyz: Optional[Union[bool, Iterable[bool]]] = None, with_sol: Optional[Union[bool, Iterable[bool]]] = None, with_med: Optional[Union[bool, Iterable[bool]]] = None, with_env: Optional[Union[bool, Iterable[bool]]] = None, task_names: Optional[Union[List[str], List[List[str]]]] = None, target_getter: 'tp.TargetGetterInput' = None, feature_extractor: Optional['tp.FeatureExtractorInput'] = None, predictor: Optional['tp.PredictorInput'] = None, loss_fn: Optional['tp.LossFnInput'] = None, primary_metrics: Optional['tp.MetricType'] = None, other_metrics: Optional[Union['tp.OtherMetricConfig', List['tp.OtherMetricConfig']]] = None, extractor_attr_getter: Optional[Union[Callable, Dict[str, Callable], List[Union[Dict, Callable]]]] = None, minimize_metric: bool = False, onehot_types: Optional[Union[int, Dict[str, int], List[Dict[str, int]]]] = None, loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], List[bool]]] = None, save_model: bool = True, devices: Optional[int] = None, precision: str = 'bf16-mixed', accelerator: str = 'cuda' if torch.cuda.is_available() else 'cpu', float32_matmul_precision: str = 'medium', profiler: str = 'simple', show_pbar: bool = True, debug: bool = False, overfit_test: bool = False, use_debugger: bool = False, warning_allowed: bool = True, lofn_wrap_tasks: Optional[Union[str, Sequence[str]]] = None, lofn_wrap_exclude_tasks: Optional[Union[str, Sequence[str]]] = None, lofn_wrap_metric_names: Optional[Union[str, Dict[str, str]]] = None, lofn_wrap_metric_weights: Optional[Union[float, Dict[str, float]]] = None, extra_kwargs: Dict[str, Any] = field(default_factory=dict))`
- `DataModuleArgs` → `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；构造声明 `DataModuleArgs(dir_datasets: str, dataset_names: Optional[Union[str, Sequence[str]]] = None, exclude_datasets: Optional[Union[str, Sequence[str]]] = None, seed: int = 315, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), batch_size: int = 1, shuffle: bool = True, load_data_memory: bool = True, devices: Optional[int] = None, num_replicas: Optional[int] = None, batch_num: Optional[int] = None, stages: list[tp.Stages] = None, test_only: bool = False)`
- `ConfigArgs` → `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；构造声明 `ConfigArgs(hypers: Union[ParamSets, ParamSpace], batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]] = None, constant_lr: bool = False, dataModule: Optional[Any] = None, extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[Union[dict, Callable]]]] = None, feature_extractor: Optional[tp.FeatureExtractorInput] = None, inputs_getter: Optional[Any] = INPUT_GETTER, inputs_preprocessor: Optional[Union[Callable, list[Callable]]] = None, loss_fn: Optional[tp.LossFnInput] = None, lofn_wrap_tasks: Optional[Union[bool, float, dict[str, float]]] = None, loss_weight_method: Optional[Union[Callable, bool]] = None, lr_scheduler: Optional[Callable] = None, lr_scheduler_frequency: int = 2, lr_scheduler_kwargs: Optional[dict[str, Any]] = None, mask_need_task: Optional[list[str]] = None, onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None, optimizer: Optional[Type[Optimizer]] = None, other_metrics: Optional[Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]]] = None, predictor: Optional[tp.PredictorInput] = None, primary_metrics: Optional[dict[str, str]] = None, target_getter: tp.TargetGetterInput = None, task_names: Optional[Union[list[str], list[list[str]]]] = None, with_med: Optional[Union[bool, Iterable[bool]]] = None, with_sol: Optional[Union[bool, Iterable[bool]]] = None, with_xyz: Optional[Union[bool, Iterable[bool]]] = None, with_env: Optional[Union[bool, Iterable[bool]]] = None, sol_graph_inputs: Optional[Iterable[str]] = None, med_graph_inputs: Optional[Iterable[str]] = None, work_name: str = 'default_work', x_masker: Optional[Union[str, Callable]] = None, xyz_perturb_sigma: Optional[float] = None, xyz_perturb_mode: M.PerturbMode = 'uniform', show_pbar: bool = True, onehot_type: Optional[Union[int, dict[str, int]]] = None, kwargs: dict[str, Any] = field(default_factory=dict))`

<a id="hotpot-plugins-complexformer-run-datacls-merge-dataclass"></a>
### `merge_dataclass`

定义 owner `hotpot.plugins.ComplexFormer.run.datacls.merge_dataclass`；源码 `hotpot/plugins/ComplexFormer/run/datacls.py:44`。

```python
merge_dataclass(target: T, *sources: Any, exclude_none: bool=False, **kwargs)
```

<a id="hotpot-plugins-complexformer-run-datacls-runargs"></a>
### `RunArgs`

定义 owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；源码 `hotpot/plugins/ComplexFormer/run/datacls.py:70`。

基类原始声明：无显式基类；装饰器：`dataclass`。

显式导出位置：`hotpot.plugins.ComplexFormer.run.datacls.RunArgs`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
RunArgs(work_name: str, work_dir: str, hypers: Union['ParamSets', 'ParamSpace'], dir_datasets: str, dataset_names: Optional[Union[str, Sequence[str]]] = None, exclude_datasets: Optional[Union[str, Sequence[str]]] = None, shuffle_dataset: bool = True, dataModule_seed: int = 315, data_split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), external_datasets: Optional[Union[str, Sequence[str]]] = None, stages: Optional[Union[tp.Stages, list[tp.Stages]]] = None, eval_each_step: Optional[int] = 1, epochs: int = 100, batch_size: int = 512, early_stopping: bool = True, early_stop_step: int = 10, target_metrics: Union[str, List[str], Callable[[dict], float], None] = None, num_trials: int = 10, checkpoint_path: Union[str, int, None] = None, optimizer: Optional[Type['Optimizer']] = None, constant_lr: bool = False, lr_scheduler: Optional[Callable] = None, lr_scheduler_frequency: int = 2, lr_scheduler_kwargs: Optional[dict] = None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count', inputs_getter: Optional[Any] = INPUT_GETTER, xyz_perturb_sigma: Optional[float] = None, xyz_perturb_mode: 'M.PerturbMode' = 'uniform', batch_preprocessor: Optional[Union['tp.BatchPreProcessor', List['tp.BatchPreProcessor']]] = None, inputs_preprocessor: Optional[Union[Callable, List[Callable]]] = None, x_masker: Optional[Union[str, Callable]] = None, mask_need_task: Optional[List[str]] = None, with_xyz: Optional[Union[bool, Iterable[bool]]] = None, with_sol: Optional[Union[bool, Iterable[bool]]] = None, with_med: Optional[Union[bool, Iterable[bool]]] = None, with_env: Optional[Union[bool, Iterable[bool]]] = None, task_names: Optional[Union[List[str], List[List[str]]]] = None, target_getter: 'tp.TargetGetterInput' = None, feature_extractor: Optional['tp.FeatureExtractorInput'] = None, predictor: Optional['tp.PredictorInput'] = None, loss_fn: Optional['tp.LossFnInput'] = None, primary_metrics: Optional['tp.MetricType'] = None, other_metrics: Optional[Union['tp.OtherMetricConfig', List['tp.OtherMetricConfig']]] = None, extractor_attr_getter: Optional[Union[Callable, Dict[str, Callable], List[Union[Dict, Callable]]]] = None, minimize_metric: bool = False, onehot_types: Optional[Union[int, Dict[str, int], List[Dict[str, int]]]] = None, loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], List[bool]]] = None, save_model: bool = True, devices: Optional[int] = None, precision: str = 'bf16-mixed', accelerator: str = 'cuda' if torch.cuda.is_available() else 'cpu', float32_matmul_precision: str = 'medium', profiler: str = 'simple', show_pbar: bool = True, debug: bool = False, overfit_test: bool = False, use_debugger: bool = False, warning_allowed: bool = True, lofn_wrap_tasks: Optional[Union[str, Sequence[str]]] = None, lofn_wrap_exclude_tasks: Optional[Union[str, Sequence[str]]] = None, lofn_wrap_metric_names: Optional[Union[str, Dict[str, str]]] = None, lofn_wrap_metric_weights: Optional[Union[float, Dict[str, float]]] = None, extra_kwargs: Dict[str, Any] = field(default_factory=dict))
```

本类字段与枚举值：

- `work_name: str` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:149`
- `work_dir: str` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:150`
- `hypers: Union['ParamSets', 'ParamSpace']` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:151`
- `dir_datasets: str` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:154`
- `dataset_names: Optional[Union[str, Sequence[str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:155`
- `exclude_datasets: Optional[Union[str, Sequence[str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:156`
- `shuffle_dataset: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:157`
- `dataModule_seed: int` = `315` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:158`
- `data_split_ratios: tuple[float, float, float]` = `(0.8, 0.1, 0.1)` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:159`
- `external_datasets: Optional[Union[str, Sequence[str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:160`
- `stages: Optional[Union[tp.Stages, list[tp.Stages]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:163`
- `eval_each_step: Optional[int]` = `1` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:164`
- `epochs: int` = `100` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:167`
- `batch_size: int` = `512` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:168`
- `early_stopping: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:169`
- `early_stop_step: int` = `10` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:170`
- `target_metrics: Union[str, List[str], Callable[[dict], float], None]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:171`
- `num_trials: int` = `10` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:172`
- `checkpoint_path: Union[str, int, None]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:175`
- `optimizer: Optional[Type['Optimizer']]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:178`
- `constant_lr: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:179`
- `lr_scheduler: Optional[Callable]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:180`
- `lr_scheduler_frequency: int` = `2` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:181`
- `lr_scheduler_kwargs: Optional[dict]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:182`
- `loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']` = `'inverse-count'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:183`
- `inputs_getter: Optional[Any]` = `INPUT_GETTER` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:186`
- `xyz_perturb_sigma: Optional[float]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:187`
- `xyz_perturb_mode: 'M.PerturbMode'` = `'uniform'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:188`
- `batch_preprocessor: Optional[Union['tp.BatchPreProcessor', List['tp.BatchPreProcessor']]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:189`
- `inputs_preprocessor: Optional[Union[Callable, List[Callable]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:190`
- `x_masker: Optional[Union[str, Callable]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:191`
- `mask_need_task: Optional[List[str]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:192`
- `with_xyz: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:195`
- `with_sol: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:196`
- `with_med: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:197`
- `with_env: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:198`
- `task_names: Optional[Union[List[str], List[List[str]]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:201`
- `target_getter: 'tp.TargetGetterInput'` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:202`
- `feature_extractor: Optional['tp.FeatureExtractorInput']` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:203`
- `predictor: Optional['tp.PredictorInput']` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:204`
- `loss_fn: Optional['tp.LossFnInput']` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:205`
- `primary_metrics: Optional['tp.MetricType']` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:206`
- `other_metrics: Optional[Union['tp.OtherMetricConfig', List['tp.OtherMetricConfig']]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:207`
- `extractor_attr_getter: Optional[Union[Callable, Dict[str, Callable], List[Union[Dict, Callable]]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:208`
- `minimize_metric: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:209`
- `onehot_types: Optional[Union[int, Dict[str, int], List[Dict[str, int]]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:210`
- `loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], List[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:211`
- `save_model: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:214`
- `devices: Optional[int]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:217`
- `precision: str` = `'bf16-mixed'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:218`
- `accelerator: str` = `'cuda' if torch.cuda.is_available() else 'cpu'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:219`
- `strategy` = `'auto'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:220`
- `float32_matmul_precision: str` = `'medium'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:221`
- `profiler: str` = `'simple'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:222`
- `show_pbar: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:223`
- `debug: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:224`
- `overfit_test: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:225`
- `use_debugger: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:226`
- `warning_allowed: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:227`
- `lofn_wrap_tasks: Optional[Union[str, Sequence[str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:230`
- `lofn_wrap_exclude_tasks: Optional[Union[str, Sequence[str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:231`
- `lofn_wrap_metric_names: Optional[Union[str, Dict[str, str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:232`
- `lofn_wrap_metric_weights: Optional[Union[float, Dict[str, float]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:233`
- `extra_kwargs: Dict[str, Any]` = `field(default_factory=dict)` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.RunArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:237`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-complexformer-run-datacls-datamoduleargs"></a>
### `DataModuleArgs`

定义 owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；源码 `hotpot/plugins/ComplexFormer/run/datacls.py:274`。

基类原始声明：无显式基类；装饰器：`dataclass`。

显式导出位置：`hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
DataModuleArgs(dir_datasets: str, dataset_names: Optional[Union[str, Sequence[str]]] = None, exclude_datasets: Optional[Union[str, Sequence[str]]] = None, seed: int = 315, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), batch_size: int = 1, shuffle: bool = True, load_data_memory: bool = True, devices: Optional[int] = None, num_replicas: Optional[int] = None, batch_num: Optional[int] = None, stages: list[tp.Stages] = None, test_only: bool = False)
```

本类字段与枚举值：

- `dir_datasets: str` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:296`
- `dataset_names: Optional[Union[str, Sequence[str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:297`
- `exclude_datasets: Optional[Union[str, Sequence[str]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:298`
- `seed: int` = `315` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:301`
- `ratios: Tuple[float, float, float]` = `(0.8, 0.1, 0.1)` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:302`
- `batch_size: int` = `1` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:305`
- `shuffle: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:306`
- `load_data_memory: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:307`
- `devices: Optional[int]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:310`
- `num_replicas: Optional[int]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:311`
- `batch_num: Optional[int]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:314`
- `stages: list[tp.Stages]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:315`
- `test_only: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.DataModuleArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:316`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-complexformer-run-datacls-configargs"></a>
### `ConfigArgs`

定义 owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；源码 `hotpot/plugins/ComplexFormer/run/datacls.py:324`。

基类原始声明：无显式基类；装饰器：`dataclass`。

显式导出位置：`hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
ConfigArgs(hypers: Union[ParamSets, ParamSpace], batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]] = None, constant_lr: bool = False, dataModule: Optional[Any] = None, extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[Union[dict, Callable]]]] = None, feature_extractor: Optional[tp.FeatureExtractorInput] = None, inputs_getter: Optional[Any] = INPUT_GETTER, inputs_preprocessor: Optional[Union[Callable, list[Callable]]] = None, loss_fn: Optional[tp.LossFnInput] = None, lofn_wrap_tasks: Optional[Union[bool, float, dict[str, float]]] = None, loss_weight_method: Optional[Union[Callable, bool]] = None, lr_scheduler: Optional[Callable] = None, lr_scheduler_frequency: int = 2, lr_scheduler_kwargs: Optional[dict[str, Any]] = None, mask_need_task: Optional[list[str]] = None, onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None, optimizer: Optional[Type[Optimizer]] = None, other_metrics: Optional[Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]]] = None, predictor: Optional[tp.PredictorInput] = None, primary_metrics: Optional[dict[str, str]] = None, target_getter: tp.TargetGetterInput = None, task_names: Optional[Union[list[str], list[list[str]]]] = None, with_med: Optional[Union[bool, Iterable[bool]]] = None, with_sol: Optional[Union[bool, Iterable[bool]]] = None, with_xyz: Optional[Union[bool, Iterable[bool]]] = None, with_env: Optional[Union[bool, Iterable[bool]]] = None, sol_graph_inputs: Optional[Iterable[str]] = None, med_graph_inputs: Optional[Iterable[str]] = None, work_name: str = 'default_work', x_masker: Optional[Union[str, Callable]] = None, xyz_perturb_sigma: Optional[float] = None, xyz_perturb_mode: M.PerturbMode = 'uniform', show_pbar: bool = True, onehot_type: Optional[Union[int, dict[str, int]]] = None, kwargs: dict[str, Any] = field(default_factory=dict))
```

本类字段与枚举值：

- `hypers: Union[ParamSets, ParamSpace]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:325`
- `batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:326`
- `constant_lr: bool` = `False` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:327`
- `dataModule: Optional[Any]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:328`
- `extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[Union[dict, Callable]]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:329`
- `feature_extractor: Optional[tp.FeatureExtractorInput]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:330`
- `inputs_getter: Optional[Any]` = `INPUT_GETTER` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:331`
- `inputs_preprocessor: Optional[Union[Callable, list[Callable]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:332`
- `loss_fn: Optional[tp.LossFnInput]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:333`
- `lofn_wrap_tasks: Optional[Union[bool, float, dict[str, float]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:334`
- `loss_weight_method: Optional[Union[Callable, bool]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:335`
- `lr_scheduler: Optional[Callable]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:336`
- `lr_scheduler_frequency: int` = `2` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:337`
- `lr_scheduler_kwargs: Optional[dict[str, Any]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:338`
- `mask_need_task: Optional[list[str]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:339`
- `onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:340`
- `optimizer: Optional[Type[Optimizer]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:341`
- `other_metrics: Optional[Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:342`
- `predictor: Optional[tp.PredictorInput]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:343`
- `primary_metrics: Optional[dict[str, str]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:344`
- `target_getter: tp.TargetGetterInput` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:345`
- `task_names: Optional[Union[list[str], list[list[str]]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:346`
- `with_med: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:347`
- `with_sol: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:348`
- `with_xyz: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:349`
- `with_env: Optional[Union[bool, Iterable[bool]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:350`
- `sol_graph_inputs: Optional[Iterable[str]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:351`
- `med_graph_inputs: Optional[Iterable[str]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:352`
- `work_name: str` = `'default_work'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:353`
- `x_masker: Optional[Union[str, Callable]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:354`
- `xyz_perturb_sigma: Optional[float]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:355`
- `xyz_perturb_mode: M.PerturbMode` = `'uniform'` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:356`
- `show_pbar: bool` = `True` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:357`
- `onehot_type: Optional[Union[int, dict[str, int]]]` = `None` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:358`
- `kwargs: dict[str, Any]` = `field(default_factory=dict)` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.ConfigArgs`；`hotpot/plugins/ComplexFormer/run/datacls.py:359`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-plugins-complexformer-run-datacls-callbackconfig"></a>
### `CallbackConfig`

定义 owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；源码 `hotpot/plugins/ComplexFormer/run/datacls.py:363`。

基类原始声明：无显式基类；装饰器：`dataclass`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
CallbackConfig(optim_configure: Any, stages: List[str], early_stop_step: int, early_stopping: int, minimize_metric: bool, show_pbar: bool, use_debugger: bool)
```

本类属性与方法：

- `build(self) -> Optional[List[Any]]`；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:399`

本类字段与枚举值：

- `optim_configure: Any` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:367`
- `stages: List[str]` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:368`
- `early_stop_step: int` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:369`
- `early_stopping: int` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:370`
- `minimize_metric: bool` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:371`
- `show_pbar: bool` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:372`
- `use_debugger: bool` [可写值字段]；owner `hotpot.plugins.ComplexFormer.run.datacls.CallbackConfig`；`hotpot/plugins/ComplexFormer/run/datacls.py:373`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

模块值/类型别名（声明存在不构成事实写许可）：

- `INPUT_GETTER` = `attrgetter('x', 'edge_index', 'edge_attr', 'rings_node_index', 'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')`；`hotpot/plugins/ComplexFormer/run/datacls.py:37`
- `T` = `TypeVar('T')`；`hotpot/plugins/ComplexFormer/run/datacls.py:43`


<a id="hotpot-plugins-complexformer-run-run"></a>
## `hotpot.plugins.ComplexFormer.run.run`

源码 `hotpot/plugins/ComplexFormer/run/run.py`；SHA-256 `1c8b2278dbf4eac340fd7167db4c0a93dc42fdfa9ea759e53e56a6a62216c8c3`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `run` → `hotpot.plugins.ComplexFormer.run.run.run`；`run(run_args: datacls.RunArgs)`
- `INPUT_X_ATTR` → `hotpot.plugins.ComplexFormer.run.run.INPUT_X_ATTR`
- `COORD_X_ATTR` → `hotpot.plugins.ComplexFormer.run.run.COORD_X_ATTR`

<a id="hotpot-plugins-complexformer-run-run-run"></a>
### `run`

定义 owner `hotpot.plugins.ComplexFormer.run.run.run`；源码 `hotpot/plugins/ComplexFormer/run/run.py:154`。

```python
run(run_args: datacls.RunArgs)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `INPUT_X_ATTR` = `('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')`；`hotpot/plugins/ComplexFormer/run/run.py:32`
- `COORD_X_ATTR` = `('x', 'y', 'z')`；`hotpot/plugins/ComplexFormer/run/run.py:33`


<a id="hotpot-plugins-complexformer-run-run-tools"></a>
## `hotpot.plugins.ComplexFormer.run.run_tools`

源码 `hotpot/plugins/ComplexFormer/run/run_tools.py`；SHA-256 `a8eaa00e128e3b29776d15fd32ebe892503fa21b317312882ff4a13c6e538531`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `load_ckpt` → `hotpot.plugins.ComplexFormer.run.run_tools.load_ckpt`；`load_ckpt(work_dir, which: Optional[Union[int, str]]=-1)`
- `load_model_state_dict` → `hotpot.plugins.ComplexFormer.run.run_tools.load_model_state_dict`；`load_model_state_dict(model: nn.Module, ckpt: dict, extract_predictor: Optional[str]=None, strict_core_load: bool=True)`
- `init_model` → `hotpot.plugins.ComplexFormer.run.run_tools.init_model`；`init_model(core, task_kwargs: Union[dict, list[dict]], task: Union[tasks.SingleTask, tasks.MultiTask, tasks.MultiDataTask], optim_configure: optim_config.OptimizerConfigure)`
- `init_model_dir` → `hotpot.plugins.ComplexFormer.run.run_tools.init_model_dir`；`init_model_dir(work_dir, work_name, prefix: str='')`
- `prepare_trainer_pl_module` → `hotpot.plugins.ComplexFormer.run.run_tools.prepare_trainer_pl_module`；`prepare_trainer_pl_module(run_args: RunArgs, model_dir, task, core, logger, predictors: dict[str, M.Predictor])`

<a id="hotpot-plugins-complexformer-run-run-tools-load-ckpt"></a>
### `load_ckpt`

定义 owner `hotpot.plugins.ComplexFormer.run.run_tools.load_ckpt`；源码 `hotpot/plugins/ComplexFormer/run/run_tools.py:54`。

```python
load_ckpt(work_dir, which: Optional[Union[int, str]]=-1)
```

<a id="hotpot-plugins-complexformer-run-run-tools-load-model-state-dict"></a>
### `load_model_state_dict`

定义 owner `hotpot.plugins.ComplexFormer.run.run_tools.load_model_state_dict`；源码 `hotpot/plugins/ComplexFormer/run/run_tools.py:80`。

```python
load_model_state_dict(model: nn.Module, ckpt: dict, extract_predictor: Optional[str]=None, strict_core_load: bool=True)
```

<a id="hotpot-plugins-complexformer-run-run-tools-init-model"></a>
### `init_model`

定义 owner `hotpot.plugins.ComplexFormer.run.run_tools.init_model`；源码 `hotpot/plugins/ComplexFormer/run/run_tools.py:114`。

```python
init_model(core, task_kwargs: Union[dict, list[dict]], task: Union[tasks.SingleTask, tasks.MultiTask, tasks.MultiDataTask], optim_configure: optim_config.OptimizerConfigure)
```

<a id="hotpot-plugins-complexformer-run-run-tools-init-model-dir"></a>
### `init_model_dir`

定义 owner `hotpot.plugins.ComplexFormer.run.run_tools.init_model_dir`；源码 `hotpot/plugins/ComplexFormer/run/run_tools.py:130`。

```python
init_model_dir(work_dir, work_name, prefix: str='')
```

<a id="hotpot-plugins-complexformer-run-run-tools-prepare-trainer-pl-module"></a>
### `prepare_trainer_pl_module`

定义 owner `hotpot.plugins.ComplexFormer.run.run_tools.prepare_trainer_pl_module`；源码 `hotpot/plugins/ComplexFormer/run/run_tools.py:145`。

```python
prepare_trainer_pl_module(run_args: RunArgs, model_dir, task, core, logger, predictors: dict[str, M.Predictor])
```


<a id="hotpot-plugins-complexformer-run-train"></a>
## `hotpot.plugins.ComplexFormer.run.train`

源码 `hotpot/plugins/ComplexFormer/run/train.py`；SHA-256 `5f46f2402dc78873122636aec16e0e2cf35f6907b1434d4c5cc643c5ba5f561d`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `LightPretrain` → `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；构造声明 `__init__(self, core: Union[nn.Module, str], predictors: Union[nn.Module, dict[str, nn.Module]], tasks: Union[Task, Iterable[Task]], optim_configure: OptimizerConfigure)`

<a id="hotpot-plugins-complexformer-run-train-lightpretrain"></a>
### `LightPretrain`

定义 owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；源码 `hotpot/plugins/ComplexFormer/run/train.py:18`。

基类原始声明：`L.LightningModule`；装饰器：无。

本类属性与方法：

- `__init__(self, core: Union[nn.Module, str], predictors: Union[nn.Module, dict[str, nn.Module]], tasks: Union[Task, Iterable[Task]], optim_configure: OptimizerConfigure)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:19`
- `predict_step(self, batch) -> Any`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:56`
- `encode(self, batch)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:67`
- `f(self, batch)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:101`
- `get_target(self, batch: Batch, masked_idx: Optional[torch.Tensor]=None, norm: bool=False, **kwargs)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:109`
- `on_fit_start(self)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:149`
- `on_test_start(self) -> None`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:152`
- `training_step(self, batch, batch_idx)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:155`
- `on_train_end(self) -> None`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:170`
- `on_after_backward(self) -> None`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:173`
- `validation_step(self, batch, batch_idx)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:182`
- `on_validation_epoch_end(self) -> None`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:188`
- `configure_optimizers(self)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:191`
- `test_step(self, batch, batch_idx)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:194`
- `on_test_epoch_end(self) -> None`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:200`
- `on_test_end(self) -> None`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:204`
- `freeze_core(self)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:207`
- `freeze_(self)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:211`
- `refresh_predictors(self) -> None`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:225`
- `apply_lora_to_predictors(self, which: Union[str, Iterable[str]]=None, rank=4, alpha=4, dropout=0.3, force: bool=False, **kwargs)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:303`
- `lr(self) -> float` [读；可写]；setter `lr(self, lr: float)`；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:49`

本类字段与枚举值：

- `core` = `core` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:27`
- `tasks` = `tasks` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:34`
- `optim_configure` = `optim_configure` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:35`
- `train_metrics` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:37`
- `val_metrics` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:38`
- `test_metrics` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:39`
- `pred_inspect` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:40`
- `get_features: bool` = `False` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:42`
- `predictors` = `predictors` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.run.train.LightPretrain`；`hotpot/plugins/ComplexFormer/run/train.py:29`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.run.train.L.LightningModule`。


<a id="hotpot-plugins-complexformer-tasks"></a>
## `hotpot.plugins.ComplexFormer.tasks`

源码 `hotpot/plugins/ComplexFormer/tasks/__init__.py`；SHA-256 `6f858663ddbabd88148d038a4a4390c44c621a52ca1ce2b48d21a020ed4bd5f6`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-plugins-complexformer-tasks-tasks"></a>
## `hotpot.plugins.ComplexFormer.tasks.tasks`

源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py`；SHA-256 `4a634592b979c1bcf3c9801d9807b8f858bd5ec7aeb30676ad966f0b64b21119`。

显式 __all__：`('specify_task_types', 'specify_single_dataset_task', 'NaNMetricError', 'BaseTask', 'Task', 'SingleTask', 'MultiTask', 'MultiDataTask')`。

显式导出绑定：

- `specify_task_types` → `hotpot.plugins.ComplexFormer.tasks.tasks.specify_task_types`；`specify_task_types(is_multi_datasets: bool, target_getter: Callable)`
- `specify_single_dataset_task` → `hotpot.plugins.ComplexFormer.tasks.tasks.specify_single_dataset_task`；`specify_single_dataset_task(target_getter: Union[Callable, dict])`
- `NaNMetricError` → `hotpot.plugins.ComplexFormer.tasks.tasks.NaNMetricError`
- `BaseTask` → `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `Task` → `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；构造声明 `__init__(self, inputs_getter: Callable, feature_extractor: Union[Callable, dict[str, Callable]], target_getter: Union[tp.TargetGetter, dict[str, tp.TargetGetter]], loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor], primary_metrics: Union[str, dict[str, str]], metrics: dict[str, Union[tp.MetricFn, dict[str, tp.MetricFn]]], hypers: hotpot.plugins.opti.params_space.ParamSets, batch_preprocessor: Optional[tp.BatchPreProcessor]=None, inputs_preprocessor: Optional[Callable]=None, xyz_index: Optional[Iterable[int]]=None, xyz_perturb_sigma: Optional[float]=None, xyz_perturb_mode: M.PerturbMode='uniform', loss_weight_calculator: Optional[Union[tp.LossWeightCalculator, dict[str, tp.LossWeightCalculator]]]=None, x_masker: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, mask_need_task: Optional[list[str]]=None, pred_inspect: Union[bool, Iterable[str]]=False, plot_makers: Optional[dict[str, Union[tp.PlotMaker, tp.PlotMakerDict]]]=None, target_normalizer=None, with_sol: bool=False, with_med: bool=False, with_env: bool=False, sol_graph_inputs: Iterable[str]=None, med_graph_inputs: Iterable[str]=None, **kwargs)`
- `SingleTask` → `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；构造声明 `__init__(self, *args, **kwargs) -> None`
- `MultiTask` → `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；构造声明 `__init__(self, *args, **kwargs) -> None`
- `MultiDataTask` → `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；构造声明 `__init__(self, *tasks: MultiTask, list_kwargs: list[dict[str, Any]]=None)`

<a id="hotpot-plugins-complexformer-tasks-tasks-add-mpl-font"></a>
### `add_mpl_font`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.add_mpl_font`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:47`。

```python
add_mpl_font()
```

<a id="hotpot-plugins-complexformer-tasks-tasks-specify-single-dataset-task"></a>
### `specify_single_dataset_task`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.specify_single_dataset_task`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:59`。

```python
specify_single_dataset_task(target_getter: Union[Callable, dict])
```

<a id="hotpot-plugins-complexformer-tasks-tasks-specify-task-types"></a>
### `specify_task_types`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.specify_task_types`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:67`。

```python
specify_task_types(is_multi_datasets: bool, target_getter: Callable)
```

<a id="hotpot-plugins-complexformer-tasks-tasks-nanmetricerror"></a>
### `NaNMetricError`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.NaNMetricError`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:82`。

基类原始声明：`Exception`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.tasks.tasks.NaNMetricError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.plugins.ComplexFormer.tasks.tasks.Exception`。

<a id="hotpot-plugins-complexformer-tasks-tasks-basetask"></a>
### `BaseTask`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:85`。

基类原始声明：`ABC`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`。

本类属性与方法：

- `batch_dtype_preprocessor(batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:102`
- `batch_preprocessor(self, batch: Batch) -> Batch`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:114`
- `inputs_getter(self, batch: Batch) -> torch.Tensor`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:118`
- `get_xyz(self, inputs: tuple, batch: Batch) -> Optional[torch.Tensor]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:122`
- `get_sol_info(self, batch: Batch) -> Optional[tuple[torch.Tensor, ...]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:126`
- `get_med_info(self, batch: Batch) -> Optional[tuple[torch.Tensor, ...]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:130`
- `perturb_xyz(self, xyz, batch: Batch) -> torch.Tensor`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:134`
- `inputs_preprocessor(self, inputs: tuple[torch.Tensor, ...], **kwargs) -> tuple[torch.Tensor, ...]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:138`
- `x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], torch.Tensor)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:142`
- `feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:149`
- `inverse_pred(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:153`
- `target_getter(self, batch: Batch, norm: bool=False) -> Union[torch.Tensor, dict[str, torch.Tensor]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:157`
- `predict(predictor: Union[nn.Module, dict[str, nn.Module]], features: Union[torch.Tensor, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]` [staticmethod]；装饰器 `staticmethod`, `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:162`
- `loss_weight_calculator(self, target)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:169`
- `loss_fn(self, pred, target, loss_weight)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:173`
- `label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:177`
- `peel_unmaksed_obj(feature_target: Union[torch.Tensor, dict[str, torch.Tensor]], mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]]=None)` [staticmethod]；装饰器 `staticmethod`, `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:182`
- `log_on_train_batch_end(self, pl_module: L.LightningModule, loss: Union[torch.Tensor, dict[str, torch.Tensor]], pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]]) -> None`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:189`
- `add_val_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:199`
- `add_test_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:207`
- `summary_metrics(self, pl_module: L.LightningModule, stages: Literal['val', 'test']='val') -> tuple[dict[str, Union[float, dict]], dict[str, float]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:215`
- `eval_on_val_end(self, pl_module: L.LightningModule)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:275`
- `make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:280`
- `retrieve_path_from_log_dir(pl_module, subdir: Optional[str]=None) -> str` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:284`
- `log_plots(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:295`
- `store_save_metrics_table(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:308`
- `export_hparams(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:321`

本类字段与枚举值：

- `hypers: hotpot.plugins.opti.params_space.ParamSets` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:89`
- `show_pbar: bool` = `True` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:90`
- `test_primary_metrics: Union[float, dict[str, float]]` = `None` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:91`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`abc.ABC`。

<a id="hotpot-plugins-complexformer-tasks-tasks-task"></a>
### `Task`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:328`。

基类原始声明：`BaseTask`, `ABC`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.tasks.tasks.Task`。

本类属性与方法：

- `__init__(self, inputs_getter: Callable, feature_extractor: Union[Callable, dict[str, Callable]], target_getter: Union[tp.TargetGetter, dict[str, tp.TargetGetter]], loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor], primary_metrics: Union[str, dict[str, str]], metrics: dict[str, Union[tp.MetricFn, dict[str, tp.MetricFn]]], hypers: hotpot.plugins.opti.params_space.ParamSets, batch_preprocessor: Optional[tp.BatchPreProcessor]=None, inputs_preprocessor: Optional[Callable]=None, xyz_index: Optional[Iterable[int]]=None, xyz_perturb_sigma: Optional[float]=None, xyz_perturb_mode: M.PerturbMode='uniform', loss_weight_calculator: Optional[Union[tp.LossWeightCalculator, dict[str, tp.LossWeightCalculator]]]=None, x_masker: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=None, mask_need_task: Optional[list[str]]=None, pred_inspect: Union[bool, Iterable[str]]=False, plot_makers: Optional[dict[str, Union[tp.PlotMaker, tp.PlotMakerDict]]]=None, target_normalizer=None, with_sol: bool=False, with_med: bool=False, with_env: bool=False, sol_graph_inputs: Iterable[str]=None, med_graph_inputs: Iterable[str]=None, **kwargs)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:333`
- `batch_preprocessor(self, batch: Batch) -> Batch`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:433`
- `inputs_getter(self, batch: Batch) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:438`
- `get_xyz(self, inputs: tuple, batch: Batch) -> Optional[torch.Tensor]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:441`
- `envs_graph_postprocessing(self, graph_dict, attrs, ratios)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:578`
- `get_sol_info(self, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:591`
- `get_med_info(self, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:602`
- `perturb_xyz(self, xyz, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:610`
- `inputs_preprocessor(self, inputs: Union[dict, list, tuple], **kwargs) -> tuple[torch.Tensor, ...]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:618`
- `x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], Optional[torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:623`
- `concat_pred_target(self, stage: Literal['val', 'test']) -> (Union[dict, np.ndarray], Optional[np.ndarray])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:629`
- `calc_metrics(self, pred, target, get_primary=False) -> (dict[str, float], dict[str, float])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:633`
- `calc_train_batch_loss_metrics(self, pl_module: L.LightningModule, loss: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:651`
- `log_on_train_batch_end(self, pl_module: L.LightningModule, loss: Union[torch.Tensor, dict[str, torch.Tensor]], pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]]) -> None`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:660`
- `eval_on_val_end(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:671`
- `dict_fmt_print(dict_: dict[str, float], print_func=fmt_print.bold_magenta, prefix='')` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:676`

本类字段与枚举值：

- `xyz_index` = `xyz_index` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:363`
- `atl_weights: Optional[dict[str, float]]` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:382`
- `primary_metric` = `primary_metrics` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:385`
- `pred_inspect` = `pred_inspect` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:389`
- `plot_makers` = `plot_makers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:392`
- `plots` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:393`
- `hypers` = `hypers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:396`
- `with_sol` = `with_sol` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:401`
- `sol_graph_inputs` = `sol_graph_inputs or DEFAULT_SOL_GRAPH_INPUTS` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:402`
- `with_med` = `with_med` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:404`
- `med_graph_inputs` = `med_graph_inputs or DEFAULT_MED_GRAPH_INPUTS` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:405`
- `with_env` = `with_env` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:407`
- `info_logger` = `LoggerDict(interval_count=25)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:413`

继承成员（逐项保留声明 owner）：

- `batch_dtype_preprocessor(batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:102`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:149`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `inverse_pred(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:153`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `target_getter(self, batch: Batch, norm: bool=False) -> Union[torch.Tensor, dict[str, torch.Tensor]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:157`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `predict(predictor: Union[nn.Module, dict[str, nn.Module]], features: Union[torch.Tensor, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]` [staticmethod]；装饰器 `staticmethod`, `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:162`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `loss_weight_calculator(self, target)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:169`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `loss_fn(self, pred, target, loss_weight)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:173`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:177`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `peel_unmaksed_obj(feature_target: Union[torch.Tensor, dict[str, torch.Tensor]], mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]]=None)` [staticmethod]；装饰器 `staticmethod`, `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:182`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `add_val_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:199`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `add_test_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:207`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `summary_metrics(self, pl_module: L.LightningModule, stages: Literal['val', 'test']='val') -> tuple[dict[str, Union[float, dict]], dict[str, float]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:215`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:280`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `retrieve_path_from_log_dir(pl_module, subdir: Optional[str]=None) -> str` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:284`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `log_plots(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:295`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `store_save_metrics_table(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:308`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `export_hparams(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:321`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `show_pbar: bool` = `True` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:90`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `test_primary_metrics: Union[float, dict[str, float]]` = `None` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:91`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`abc.ABC`。

<a id="hotpot-plugins-complexformer-tasks-tasks-singletask"></a>
### `SingleTask`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:682`。

基类原始声明：`Task`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`。

本类属性与方法：

- `__init__(self, *args, **kwargs) -> None`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:694`
- `feature_extractor(self, *args, **kwargs) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:707`
- `predict(predictor: nn.Module, features: torch.Tensor) -> torch.Tensor` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:711`
- `label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]])`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:714`
- `peel_unmaksed_obj(feature_target: torch.Tensor, mask_idx: Optional[torch.Tensor]=None)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:718`
- `target_getter(self, batch: Batch, norm=False) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:729`
- `loss_weight_calculator(self, target) -> Optional[torch.Tensor]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:732`
- `loss_fn(self, pred, target, loss_weight)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:737`
- `inverse_pred(self, pred: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:742`
- `calc_train_batch_loss_metrics(self, pl_module: L.LightningModule, loss: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:745`
- `add_val_pred_target(self, pred: torch.Tensor, target: torch.Tensor)`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:758`
- `add_test_pred_target(self, pred: torch.Tensor, target: torch.Tensor)`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:767`
- `concat_pred_target(self, stage: Literal['val', 'test']) -> (np.ndarray, np.ndarray)`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:776`
- `calc_primary_metrics(self, pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> dict[str, float]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:784`
- `calc_metrics(self, pred: torch.Tensor, target: torch.Tensor, get_primary: bool=False) -> tuple[dict[str, float], dict[str, float]]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:791`
- `summary_metrics(self, pl_module: L.LightningModule, stages='val') -> tuple[dict[str, float], dict[str, float]]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:804`
- `make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:814`
- `eval_on_val_end(self, pl_module: L.LightningModule)`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:819`

本类字段与枚举值：

- `val_pred` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:696`
- `val_target` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:697`
- `test_pred` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:699`
- `test_target` = `[]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.SingleTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:700`

继承成员（逐项保留声明 owner）：

- `batch_preprocessor(self, batch: Batch) -> Batch`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:433`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `inputs_getter(self, batch: Batch) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:438`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `get_xyz(self, inputs: tuple, batch: Batch) -> Optional[torch.Tensor]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:441`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `envs_graph_postprocessing(self, graph_dict, attrs, ratios)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:578`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `get_sol_info(self, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:591`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `get_med_info(self, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:602`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `perturb_xyz(self, xyz, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:610`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `inputs_preprocessor(self, inputs: Union[dict, list, tuple], **kwargs) -> tuple[torch.Tensor, ...]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:618`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], Optional[torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:623`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `log_on_train_batch_end(self, pl_module: L.LightningModule, loss: Union[torch.Tensor, dict[str, torch.Tensor]], pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]]) -> None`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:660`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `dict_fmt_print(dict_: dict[str, float], print_func=fmt_print.bold_magenta, prefix='')` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:676`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `xyz_index` = `xyz_index` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:363`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `atl_weights: Optional[dict[str, float]]` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:382`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `primary_metric` = `primary_metrics` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:385`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `pred_inspect` = `pred_inspect` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:389`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `plot_makers` = `plot_makers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:392`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `plots` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:393`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `hypers` = `hypers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:396`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `with_sol` = `with_sol` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:401`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `sol_graph_inputs` = `sol_graph_inputs or DEFAULT_SOL_GRAPH_INPUTS` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:402`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `with_med` = `with_med` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:404`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `med_graph_inputs` = `med_graph_inputs or DEFAULT_MED_GRAPH_INPUTS` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:405`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `with_env` = `with_env` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:407`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `info_logger` = `LoggerDict(interval_count=25)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:413`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `batch_dtype_preprocessor(batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:102`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `retrieve_path_from_log_dir(pl_module, subdir: Optional[str]=None) -> str` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:284`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `log_plots(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:295`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `store_save_metrics_table(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:308`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `export_hparams(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:321`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `show_pbar: bool` = `True` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:90`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `test_primary_metrics: Union[float, dict[str, float]]` = `None` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:91`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`

<a id="hotpot-plugins-complexformer-tasks-tasks-multitask"></a>
### `MultiTask`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:828`。

基类原始声明：`Task`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`。

本类属性与方法：

- `__init__(self, *args, **kwargs) -> None`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:840`
- `feature_extractor(self, *args, **kwargs) -> dict[str, torch.Tensor]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:863`
- `predict(predictor: dict[str, nn.Module], features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:867`
- `target_getter(self, batch: Batch, norm=False) -> dict[str, torch.Tensor]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:870`
- `inverse_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:873`
- `label2oh_conversion(self, target: dict[str, torch.Tensor])`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:877`
- `peel_unmaksed_obj(self, feature_target: dict[str, torch.Tensor], mask_idx: Union[torch.Tensor]=None)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:880`
- `loss_weight_calculator(self, target) -> Optional[dict[str, torch.Tensor]]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:894`
- `loss_fn(self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], loss_weight: dict[str, torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:903`
- `calc_train_batch_loss_metrics(self, pl_module: L.LightningModule, loss: torch.Tensor, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, float]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:935`
- `add_val_pred_target(self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor])`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:971`
- `add_test_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:981`
- `concat_pred_target(self, stages: Literal['val', 'test']) -> (dict[str, np.ndarray], dict[str, np.ndarray])`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:991`
- `calc_metrics(self, pred: dict[str, Union[torch.Tensor, np.ndarray]], target: dict[str, torch.Tensor, np.ndarray], get_primary=False) -> tuple[dict[str, dict[str, float]], dict[str, float]]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1003`
- `summary_metrics(self, pl_module: L.LightningModule, stages='val') -> tuple[dict[str, dict[str, float]], dict[str, float]]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1024`
- `eval_on_val_end(self, pl_module: L.LightningModule)`；装饰器 `override`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1043`
- `make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1050`

本类字段与枚举值：

- `loss_dict` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:842`
- `val_pred` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:843`
- `val_target` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:844`
- `test_pred` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:845`
- `test_target` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:846`
- `atl_weights_calculators` = `kwargs['atl_weights_calculators']` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:849`

继承成员（逐项保留声明 owner）：

- `batch_preprocessor(self, batch: Batch) -> Batch`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:433`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `inputs_getter(self, batch: Batch) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:438`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `get_xyz(self, inputs: tuple, batch: Batch) -> Optional[torch.Tensor]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:441`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `envs_graph_postprocessing(self, graph_dict, attrs, ratios)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:578`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `get_sol_info(self, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:591`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `get_med_info(self, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:602`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `perturb_xyz(self, xyz, batch: Batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:610`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `inputs_preprocessor(self, inputs: Union[dict, list, tuple], **kwargs) -> tuple[torch.Tensor, ...]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:618`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], Optional[torch.Tensor])`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:623`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `log_on_train_batch_end(self, pl_module: L.LightningModule, loss: Union[torch.Tensor, dict[str, torch.Tensor]], pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]]) -> None`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:660`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `dict_fmt_print(dict_: dict[str, float], print_func=fmt_print.bold_magenta, prefix='')` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:676`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `xyz_index` = `xyz_index` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:363`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `atl_weights: Optional[dict[str, float]]` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:382`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `primary_metric` = `primary_metrics` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:385`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `pred_inspect` = `pred_inspect` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:389`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `plot_makers` = `plot_makers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:392`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `plots` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:393`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `hypers` = `hypers` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:396`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `with_sol` = `with_sol` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:401`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `sol_graph_inputs` = `sol_graph_inputs or DEFAULT_SOL_GRAPH_INPUTS` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:402`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `with_med` = `with_med` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:404`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `med_graph_inputs` = `med_graph_inputs or DEFAULT_MED_GRAPH_INPUTS` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:405`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `with_env` = `with_env` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:407`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `info_logger` = `LoggerDict(interval_count=25)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.Task`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:413`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `batch_dtype_preprocessor(batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:102`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `retrieve_path_from_log_dir(pl_module, subdir: Optional[str]=None) -> str` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:284`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `log_plots(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:295`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `store_save_metrics_table(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:308`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `export_hparams(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:321`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `show_pbar: bool` = `True` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:90`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`
- `test_primary_metrics: Union[float, dict[str, float]]` = `None` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:91`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.Task`

<a id="hotpot-plugins-complexformer-tasks-tasks-multidatatask"></a>
### `MultiDataTask`

定义 owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；源码 `hotpot/plugins/ComplexFormer/tasks/tasks.py:1076`。

基类原始声明：`BaseTask`；装饰器：无。

显式导出位置：`hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`。

本类属性与方法：

- `__init__(self, *tasks: MultiTask, list_kwargs: list[dict[str, Any]]=None)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1079`
- `__getitem__(self, item: int)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1090`
- `__repr__(self)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1099`
- `init_from_args(cls, tasks_counts: int, inputs_getter: Callable, feature_extractor: list[dict[str, Callable]], target_getter: list[dict[str, tp.TargetGetter]], loss_fn: list[dict[str, tp.LossFn]], primary_metric: list[dict[str, str]], metrics: list[dict[str, tp.MetricFn]], batch_preprocessor: Optional[list[tp.BatchPreProcessor]]=None, inputs_preprocessor: Optional[list[Callable]]=None, xyz_index: Optional[list[Iterable[int]]]=None, xyz_perturb_sigma: Optional[list[float]]=None, extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]]=None, loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]]=None, onehot_types: Optional[list[dict[str, int]]]=None, x_masker: Optional[list[tp.XMasker]]=None, mask_need_task: Optional[list[list[str]]]=None, **kwargs)` [classmethod]；装饰器 `classmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1103`
- `choose_task(self, batch)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1174`
- `batch_preprocessor(self, batch: Batch) -> Batch`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1181`
- `log_on_train_batch_end(self, pl_module: L.LightningModule, loss: Union[torch.Tensor, dict[str, torch.Tensor]], pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]]) -> None`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1185`
- `summary_metrics(self, pl_module: L.LightningModule, stages: Literal['val', 'test']='val') -> tuple[dict[str, dict[str, float]], dict[str, float]]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1204`
- `eval_on_val_end(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1229`
- `make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1233`
- `current_task_index(self) -> Optional[int]` [读；只读]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1094`

本类字段与枚举值：

- `lr_matcher` = `re.compile('\\d+-lr')` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1077`
- `current_task` = `None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1087`
- `metrics_dict` = `{}` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tasks.tasks.MultiDataTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:1088`

继承成员（逐项保留声明 owner）：

- `batch_dtype_preprocessor(batch)` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:102`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `inputs_getter(self, batch: Batch) -> torch.Tensor`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:118`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `get_xyz(self, inputs: tuple, batch: Batch) -> Optional[torch.Tensor]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:122`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `get_sol_info(self, batch: Batch) -> Optional[tuple[torch.Tensor, ...]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:126`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `get_med_info(self, batch: Batch) -> Optional[tuple[torch.Tensor, ...]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:130`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `perturb_xyz(self, xyz, batch: Batch) -> torch.Tensor`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:134`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `inputs_preprocessor(self, inputs: tuple[torch.Tensor, ...], **kwargs) -> tuple[torch.Tensor, ...]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:138`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], torch.Tensor)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:142`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:149`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `inverse_pred(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:153`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `target_getter(self, batch: Batch, norm: bool=False) -> Union[torch.Tensor, dict[str, torch.Tensor]]`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:157`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `predict(predictor: Union[nn.Module, dict[str, nn.Module]], features: Union[torch.Tensor, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]` [staticmethod]；装饰器 `staticmethod`, `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:162`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `loss_weight_calculator(self, target)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:169`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `loss_fn(self, pred, target, loss_weight)`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:173`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:177`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `peel_unmaksed_obj(feature_target: Union[torch.Tensor, dict[str, torch.Tensor]], mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]]=None)` [staticmethod]；装饰器 `staticmethod`, `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:182`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `add_val_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:199`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `add_test_pred_target(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]], target: Union[torch.Tensor, dict[str, torch.Tensor]])`；装饰器 `abstractmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:207`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `retrieve_path_from_log_dir(pl_module, subdir: Optional[str]=None) -> str` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:284`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `log_plots(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:295`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `store_save_metrics_table(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:308`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `export_hparams(self, pl_module: L.LightningModule)`；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:321`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `hypers: hotpot.plugins.opti.params_space.ParamSets` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:89`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `show_pbar: bool` = `True` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:90`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`
- `test_primary_metrics: Union[float, dict[str, float]]` = `None` [字段声明；写许可依具体合同]；owner `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:91`；继承经 `hotpot.plugins.ComplexFormer.tasks.tasks.BaseTask`

模块值/类型别名（声明存在不构成事实写许可）：

- `DEFAULT_SOL_GRAPH_INPUTS` = `('x', 'edge_index', 'batch')`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:325`
- `DEFAULT_MED_GRAPH_INPUTS` = `('x', 'edge_index', 'batch')`；`hotpot/plugins/ComplexFormer/tasks/tasks.py:326`


<a id="hotpot-plugins-complexformer-tools"></a>
## `hotpot.plugins.ComplexFormer.tools`

源码 `hotpot/plugins/ComplexFormer/tools.py`；SHA-256 `40e1ea7638929005cab58e1df20902023d4cb6af8707b3874a188f09f52c9f70`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `specify_target_getter` → `hotpot.plugins.ComplexFormer.tools.specify_target_getter`；`specify_target_getter(first_data: Data, data_item: str, attrs: str=None, condition_fn: str=None)`
- `TargetGetter` → `hotpot.plugins.ComplexFormer.tools.TargetGetter`；构造声明 `__init__(self, first_data, data_item: str, attrs: Union[str, Iterable[str]]=None, condition_fn: Callable=None, normalizer: Optional[TensorNormalizer]=None)`
- `get_index` → `hotpot.plugins.ComplexFormer.tools.get_index`；`get_index(first_data, data_item: str, attrs: Union[str, Iterable[str]]=None) -> Union[int, list[int]]`
- `target_getter_condition_fns` → `hotpot.plugins.ComplexFormer.tools.target_getter_condition_fns`

<a id="hotpot-plugins-complexformer-tools-specify-target-getter"></a>
### `specify_target_getter`

定义 owner `hotpot.plugins.ComplexFormer.tools.specify_target_getter`；源码 `hotpot/plugins/ComplexFormer/tools.py:18`。

```python
specify_target_getter(first_data: Data, data_item: str, attrs: str=None, condition_fn: str=None)
```

<a id="hotpot-plugins-complexformer-tools-targetgetter"></a>
### `TargetGetter`

定义 owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；源码 `hotpot/plugins/ComplexFormer/tools.py:31`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, first_data, data_item: str, attrs: Union[str, Iterable[str]]=None, condition_fn: Callable=None, normalizer: Optional[TensorNormalizer]=None)`；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:32`
- `__call__(self, batch: Batch, norm=False) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:82`
- `get_target(self, batch: Batch) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:93`
- `normalize(self, tensor: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:104`
- `inverse(self, tensor: torch.Tensor) -> torch.Tensor`；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:109`

本类字段与枚举值：

- `data_item` = `data_item` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:40`
- `attrs` = `attrs` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:41`
- `data_idx` = `get_index(first_data, self.data_item, attrs)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:42`
- `condition_fn` = `condition_fn` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:43`
- `normalizer` = `self._specify_normalizer(first_data, normalizer)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.plugins.ComplexFormer.tools.TargetGetter`；`hotpot/plugins/ComplexFormer/tools.py:44`

<a id="hotpot-plugins-complexformer-tools-get-index"></a>
### `get_index`

定义 owner `hotpot.plugins.ComplexFormer.tools.get_index`；源码 `hotpot/plugins/ComplexFormer/tools.py:116`。

```python
get_index(first_data, data_item: str, attrs: Union[str, Iterable[str]]=None) -> Union[int, list[int]]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `target_getter_condition_fns` = `{'where_metal': where_metal}`；`hotpot/plugins/ComplexFormer/tools.py:13`


<a id="hotpot-plugins-complexformer-types"></a>
## `hotpot.plugins.ComplexFormer.types`

源码 `hotpot/plugins/ComplexFormer/types.py`；SHA-256 `16dfa47fa328a21bc7508c707f94d2d3046a26cf83ff8aa133cd256bbef51794`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `Stages` → `hotpot.plugins.ComplexFormer.types.Stages`
- `TargetType` → `hotpot.plugins.ComplexFormer.types.TargetType`
- `TensorArray` → `hotpot.plugins.ComplexFormer.types.TensorArray`
- `BatchPreProcessor` → `hotpot.plugins.ComplexFormer.types.BatchPreProcessor`
- `ExtractorAttrGetter` → `hotpot.plugins.ComplexFormer.types.ExtractorAttrGetter`
- `FeatureExtractorName` → `hotpot.plugins.ComplexFormer.types.FeatureExtractorName`
- `TargetGetter` → `hotpot.plugins.ComplexFormer.types.TargetGetter`
- `LossWeightCalculator` → `hotpot.plugins.ComplexFormer.types.LossWeightCalculator`
- `LossWeightMethods` → `hotpot.plugins.ComplexFormer.types.LossWeightMethods`
- `LossFn` → `hotpot.plugins.ComplexFormer.types.LossFn`
- `MetricFn` → `hotpot.plugins.ComplexFormer.types.MetricFn`
- `XMasker` → `hotpot.plugins.ComplexFormer.types.XMasker`
- `OtherMetricConfig` → `hotpot.plugins.ComplexFormer.types.OtherMetricConfig`
- `WorkNameInput` → `hotpot.plugins.ComplexFormer.types.WorkNameInput`
- `TargetGetterInput` → `hotpot.plugins.ComplexFormer.types.TargetGetterInput`
- `FeatureExtractorInput` → `hotpot.plugins.ComplexFormer.types.FeatureExtractorInput`
- `PredictorInput` → `hotpot.plugins.ComplexFormer.types.PredictorInput`
- `LossFnInput` → `hotpot.plugins.ComplexFormer.types.LossFnInput`
- `MetricType` → `hotpot.plugins.ComplexFormer.types.MetricType`
- `PlotMaker` → `hotpot.plugins.ComplexFormer.types.PlotMaker`
- `PlotMakerDict` → `hotpot.plugins.ComplexFormer.types.PlotMakerDict`

模块值/类型别名（声明存在不构成事实写许可）：

- `Stages` = `Literal['train', 'test']`；`hotpot/plugins/ComplexFormer/types.py:11`
- `TargetType` = `Literal['xyz', 'onehot', 'binary', 'num']`；`hotpot/plugins/ComplexFormer/types.py:12`
- `TensorArray` = `Union[torch.Tensor, np.ndarray]`；`hotpot/plugins/ComplexFormer/types.py:13`
- `BatchPreProcessor` = `Callable[[Batch], Batch]`；`hotpot/plugins/ComplexFormer/types.py:14`
- `ExtractorAttrGetter` = `Callable[[Batch], Union[tuple, torch.Tensor]]`；`hotpot/plugins/ComplexFormer/types.py:15`
- `FeatureExtractorName` = `Literal['atom', 'bond', 'pair', 'ring', 'mol', 'cbond', 'metal']`；`hotpot/plugins/ComplexFormer/types.py:16`
- `TargetGetter` = `Callable[[Batch], torch.Tensor]`；`hotpot/plugins/ComplexFormer/types.py:17`
- `LossWeightCalculator` = `Callable[[torch.Tensor], torch.Tensor]`；`hotpot/plugins/ComplexFormer/types.py:18`
- `LossWeightMethods` = `Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']`；`hotpot/plugins/ComplexFormer/types.py:19`
- `LossFn` = `Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor]`；`hotpot/plugins/ComplexFormer/types.py:20`
- `MetricFn` = `Callable[[TensorArray, TensorArray], Union[float, TensorArray]]`；`hotpot/plugins/ComplexFormer/types.py:21`
- `XMasker` = `Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]]`；`hotpot/plugins/ComplexFormer/types.py:22`
- `OtherMetricConfig` = `Union[str, Iterable[str], dict[str, Union[str, Iterable[str], MetricFn]]]`；`hotpot/plugins/ComplexFormer/types.py:23`
- `WorkNameInput` = `Union[str, Sequence[str]]`；`hotpot/plugins/ComplexFormer/types.py:27`
- `TargetGetterInput` = `Union[str, Callable, Sequence[Callable], dict[str, Callable]]`；`hotpot/plugins/ComplexFormer/types.py:28`
- `FeatureExtractorInput` = `Union[str, Callable, Sequence[Callable], dict[str, Callable]]`；`hotpot/plugins/ComplexFormer/types.py:29`
- `PredictorInput` = `Union[str, M.Predictor, Sequence[M.Predictor], dict[str, M.Predictor]]`；`hotpot/plugins/ComplexFormer/types.py:30`
- `LossFnInput` = `Union[str, Callable, Sequence[Union[str, Callable]], dict[str, Callable]]`；`hotpot/plugins/ComplexFormer/types.py:31`
- `MetricType` = `Literal['r2score', 'rmse', 'mse', 'mae', 'accuracy', 'binary_accuracy', 'metal_accuracy']`；`hotpot/plugins/ComplexFormer/types.py:32`
- `PlotMaker` = `Callable[[np.ndarray, np.ndarray], Figure]`；`hotpot/plugins/ComplexFormer/types.py:35`
- `PlotMakerDict` = `dict[str, PlotMaker]`；`hotpot/plugins/ComplexFormer/types.py:36`
