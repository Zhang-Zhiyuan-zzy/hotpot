# 命令入口的 Python 声明：全签名目录

来源：当前静态源码；显式 __all__ 与直接模块定义分别列明。类型定义 owner 与公开导出路径分别保留。
只读 property、动态 owner setter、不可变值字段与可重绑定类声明值不是同一权限概念。外部继承保持可归属的边界，未声称第三方运行时属性穷尽。
[完整树](../delivery_api_tree.md) · [动态权限](dynamic_permissions.md) · [native 声明](native_contracts.md)

## 模块导航

- [`hotpot.__main__`](#hotpot---main--)
- [`hotpot._cli`](#hotpot--cli)
- [`hotpot.commands`](#hotpot-commands)
- [`hotpot.commands.cbond`](#hotpot-commands-cbond)
- [`hotpot.commands.convert`](#hotpot-commands-convert)
- [`hotpot.commands.ff`](#hotpot-commands-ff)
- [`hotpot.commands.mca`](#hotpot-commands-mca)
- [`hotpot.commands.ml_train`](#hotpot-commands-ml-train)
- [`hotpot.commands.models`](#hotpot-commands-models)
- [`hotpot.commands.optimize`](#hotpot-commands-optimize)

<a id="hotpot---main--"></a>
## `hotpot.__main__`

`hotpot/__main__.py`；显式 __all__: 无


<a id="hotpot---main---is-running-in-foreground"></a>
### `is_running_in_foreground`

```python
is_running_in_foreground() -> bool
```

<a id="hotpot---main---show-version"></a>
### `show_version`

```python
show_version() -> None
```

<a id="hotpot---main---build-parser"></a>
### `build_parser`

```python
build_parser(selected_command: Optional[str]=None) -> argparse.ArgumentParser
```

<a id="hotpot---main---run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot---main---main"></a>
### `main`

```python
main(argv: Optional[Sequence[str]]=None) -> int
```

模块值/类型别名（声明存在不构成事实写许可）：

- `PROJECT_DESCRIPTION` = `'A cheminformatics harness for coordination chemistry'`


<a id="hotpot--cli"></a>
## `hotpot._cli`

`hotpot/_cli.py`；显式 __all__: ('MarkdownDocumentationAction',)


<a id="hotpot--cli-markdowndocumentationaction"></a>
### `MarkdownDocumentationAction`

定义 owner `hotpot._cli.MarkdownDocumentationAction`；基类 `argparse.Action`。

- `__init__(self, option_strings: Sequence[str], dest: str, nargs: Optional[int]=None, *, document_loader: Callable[[], str], **kwargs: object) -> None`
- `__call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: object, option_string: Optional[str]=None) -> None`


<a id="hotpot-commands"></a>
## `hotpot.commands`

`hotpot/commands/__init__.py`；显式 __all__: 无



<a id="hotpot-commands-cbond"></a>
## `hotpot.commands.cbond`

`hotpot/commands/cbond.py`；显式 __all__: 无


<a id="hotpot-commands-cbond-load-cli-documentation"></a>
### `load_cli_documentation`

```python
load_cli_documentation() -> str
```

<a id="hotpot-commands-cbond-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-commands-cbond-format-single-result"></a>
### `format_single_result`

```python
format_single_result(result, bond_detail: bool=False) -> str
```

<a id="hotpot-commands-cbond-format-all-results"></a>
### `format_all_results`

```python
format_all_results(results, bond_detail: bool=False) -> str
```

<a id="hotpot-commands-cbond-run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot-commands-cbond-build-parser"></a>
### `build_parser`

```python
build_parser() -> argparse.ArgumentParser
```

<a id="hotpot-commands-cbond-main"></a>
### `main`

```python
main(argv: list[str] | None=None) -> int
```


<a id="hotpot-commands-convert"></a>
## `hotpot.commands.convert`

`hotpot/commands/convert.py`；显式 __all__: 无


<a id="hotpot-commands-convert-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-commands-convert-convert"></a>
### `convert`

```python
convert(input_file, output_file, out_fmt: str, in_fmt=None)
```

<a id="hotpot-commands-convert-run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot-commands-convert-conversionfailure"></a>
### `ConversionFailure`

定义 owner `hotpot.commands.convert.ConversionFailure`；基类 无。

```python
ConversionFailure(name: Any, kind: Literal['worker_error', 'protocol_error', 'nonzero_exit', 'timeout'], exitcode: Optional[int], build_timeout: float, output_path: str, error_type: Optional[str] = None, error_message: Optional[str] = None, worker_traceback: Optional[str] = None)
```

- `name: Any` [只读值字段]
- `kind: Literal['worker_error', 'protocol_error', 'nonzero_exit', 'timeout']` [只读值字段]
- `exitcode: Optional[int]` [只读值字段]
- `build_timeout: float` [只读值字段]
- `output_path: str` [只读值字段]
- `error_type: Optional[str]` = `None` [只读值字段]
- `error_message: Optional[str]` = `None` [只读值字段]
- `worker_traceback: Optional[str]` = `None` [只读值字段]

<a id="hotpot-commands-convert-conversionworkerresult"></a>
### `ConversionWorkerResult`

定义 owner `hotpot.commands.convert.ConversionWorkerResult`；基类 无。

```python
ConversionWorkerResult(status: Literal['ok', 'error'], error_type: Optional[str] = None, error_message: Optional[str] = None, traceback: Optional[str] = None)
```

- `status: Literal['ok', 'error']` [只读值字段]
- `error_type: Optional[str]` = `None` [只读值字段]
- `error_message: Optional[str]` = `None` [只读值字段]
- `traceback: Optional[str]` = `None` [只读值字段]

<a id="hotpot-commands-convert-conversionbatcherror"></a>
### `ConversionBatchError`

定义 owner `hotpot.commands.convert.ConversionBatchError`；基类 `RuntimeError`。

- `__init__(self, failures)`
- `failures` = `tuple(failures)` [可写实例字段；非自动批准事实写许可]（从 self 赋值提取，非固定 schema）

<a id="hotpot-commands-convert-convert-sdf-to-smiles"></a>
### `convert_sdf_to_smiles`

```python
convert_sdf_to_smiles(sdf_dir: [str, Path], smiles_dir: Union[str, Path], split_number: int=1000000) -> None
```

<a id="hotpot-commands-convert-convert-smiles-to-3dmol"></a>
### `convert_smiles_to_3dmol`

```python
convert_smiles_to_3dmol(list_smi: list[str], save_dir: str, alone_ligand_save_dir: str=None, sdf_save_dir: str=None, file_names: list[str]=None, fmt: str='gjf', nproc: Optional[int]=None, timeout: int=1000, **kwargs)
```

<a id="hotpot-commands-convert-convert-g16log-to-gjf"></a>
### `convert_g16log_to_gjf`

```python
convert_g16log_to_gjf(g16log_dir: str, g16gjf_dir: str, link0=None, route=None)
```


<a id="hotpot-commands-ff"></a>
## `hotpot.commands.ff`

`hotpot/commands/ff.py`；显式 __all__: ('add_arguments', 'build_parser', 'load_cli_documentation', 'main', 'run')


<a id="hotpot-commands-ff-load-cli-documentation"></a>
### `load_cli_documentation`

```python
load_cli_documentation() -> str
```

<a id="hotpot-commands-ff-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-commands-ff-run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot-commands-ff-build-parser"></a>
### `build_parser`

```python
build_parser() -> argparse.ArgumentParser
```

<a id="hotpot-commands-ff-main"></a>
### `main`

```python
main(argv: Optional[Sequence[str]]=None) -> int
```


<a id="hotpot-commands-mca"></a>
## `hotpot.commands.mca`

`hotpot/commands/mca.py`；显式 __all__: 无


<a id="hotpot-commands-mca-load-cli-documentation"></a>
### `load_cli_documentation`

```python
load_cli_documentation() -> str
```

<a id="hotpot-commands-mca-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-commands-mca-save-plot"></a>
### `save_plot`

```python
save_plot(molecules, predictions, path: str, all_sites: bool) -> None
```

<a id="hotpot-commands-mca-run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot-commands-mca-build-parser"></a>
### `build_parser`

```python
build_parser() -> argparse.ArgumentParser
```

<a id="hotpot-commands-mca-main"></a>
### `main`

```python
main(argv: list[str] | None=None) -> int
```


<a id="hotpot-commands-ml-train"></a>
## `hotpot.commands.ml_train`

`hotpot/commands/ml_train.py`；显式 __all__: 无


<a id="hotpot-commands-ml-train-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser)
```

<a id="hotpot-commands-ml-train-train"></a>
### `train`

```python
train(args: argparse.Namespace)
```


<a id="hotpot-commands-models"></a>
## `hotpot.commands.models`

`hotpot/commands/models.py`；显式 __all__: 无


<a id="hotpot-commands-models-add-arguments"></a>
### `add_arguments`

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-commands-models-run"></a>
### `run`

```python
run(args: argparse.Namespace) -> int
```


<a id="hotpot-commands-optimize"></a>
## `hotpot.commands.optimize`

`hotpot/commands/optimize.py`；显式 __all__: 无


<a id="hotpot-commands-optimize-read-excel"></a>
### `read_excel`

```python
read_excel(excel_file, args, *, input_field, exclude_field)
```

<a id="hotpot-commands-optimize-add-arguments"></a>
### `add_arguments`

```python
add_arguments(optimize_parser: argparse.ArgumentParser)
```

<a id="hotpot-commands-optimize-optimize"></a>
### `optimize`

```python
optimize(excel_file: Union[Path, str], out_file: Union[Path, str], args)
```

<a id="hotpot-commands-optimize-example-cof-params"></a>
### `example_cof_params`

```python
example_cof_params(out_file, args)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `RED` = `'\x1b[91m'`
- `RESET` = `'\x1b[0m'`
