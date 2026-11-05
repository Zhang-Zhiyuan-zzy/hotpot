# 辅助 dataset/ds 与通用工具：完整静态声明

库存基准提交 `2c2ef889abb6adcceba043b606da2e718dbea109`；日期 `2026-11-05`。逐模块 SHA-256 保留如下。

仅按现有 AST 库存展开非私有本地定义、显式导出目标、声明字段、property、常用 magic 方法及可静态解析的本地继承。参数拼写、默认值与注解采用库存中的源码 AST 表达式，未经运行时求值。

这份参考不声称辅助模块已在 Plan20 中重新科学验收。私有模块内的非私有名称仍可能是内部实现；直接模块可访问面与显式 __all__ 分开列出，不扩大批准的公共合同。

property 可写指源码声明 setter；只读 getter 不保证返回对象不可变。类声明值、self 赋值字段和 frozen dataclass 权限分别标示，均不自动授予权威化学事实写权限。

未穷尽第三方继承成员、运行时安装属性、元类生成行为或可选依赖实际可用性。静态解析的基类名称不保证是可导入的 owner，特别是内置类及泛型表达式；原始基类拼写另列。

[辅助入口与类索引](auxiliary_index.md) · [主交付树](../delivery_api_tree.md) · [完整机器库存](README.md#raw-inventory) · [主内核动态权限](dynamic_permissions.md)

## 模块导航

- [`hotpot.dataset`](#hotpot-dataset)
- [`hotpot.ds`](#hotpot-ds)
- [`hotpot.ds._bundle_cleanup`](#hotpot-ds--bundle-cleanup)
- [`hotpot.ds._communication`](#hotpot-ds--communication)
- [`hotpot.ds._export`](#hotpot-ds--export)
- [`hotpot.ds._paths`](#hotpot-ds--paths)
- [`hotpot.ds._store_copy`](#hotpot-ds--store-copy)
- [`hotpot.ds.cli`](#hotpot-ds-cli)
- [`hotpot.ds.gates`](#hotpot-ds-gates)
- [`hotpot.ds.gates._base`](#hotpot-ds-gates--base)
- [`hotpot.ds.gates._checker`](#hotpot-ds-gates--checker)
- [`hotpot.ds.gates._gates`](#hotpot-ds-gates--gates)
- [`hotpot.ds.gates._persistence`](#hotpot-ds-gates--persistence)
- [`hotpot.ds.gates.benchmark`](#hotpot-ds-gates-benchmark)
- [`hotpot.ds.gates.evidence`](#hotpot-ds-gates-evidence)
- [`hotpot.ds.gates.quality`](#hotpot-ds-gates-quality)
- [`hotpot.ds.governance`](#hotpot-ds-governance)
- [`hotpot.ds.providers`](#hotpot-ds-providers)
- [`hotpot.ds.providers._io`](#hotpot-ds-providers--io)
- [`hotpot.ds.providers.lnqm`](#hotpot-ds-providers-lnqm)
- [`hotpot.ds.providers.lnqm.pyg_data`](#hotpot-ds-providers-lnqm-pyg-data)
- [`hotpot.ds.providers.tmqm`](#hotpot-ds-providers-tmqm)
- [`hotpot.ds.providers.tmqm.download`](#hotpot-ds-providers-tmqm-download)
- [`hotpot.ds.public`](#hotpot-ds-public)
- [`hotpot.ds.public._assessment`](#hotpot-ds-public--assessment)
- [`hotpot.ds.public._bundle_state`](#hotpot-ds-public--bundle-state)
- [`hotpot.ds.public._generation`](#hotpot-ds-public--generation)
- [`hotpot.ds.public._implementation`](#hotpot-ds-public--implementation)
- [`hotpot.ds.public.bundle`](#hotpot-ds-public-bundle)
- [`hotpot.ds.public.case`](#hotpot-ds-public-case)
- [`hotpot.ds.public.catalog`](#hotpot-ds-public-catalog)
- [`hotpot.ds.public.certificate`](#hotpot-ds-public-certificate)
- [`hotpot.ds.public.dataset`](#hotpot-ds-public-dataset)
- [`hotpot.ds.public.search`](#hotpot-ds-public-search)
- [`hotpot.ds.public.structures`](#hotpot-ds-public-structures)
- [`hotpot.ds.storage`](#hotpot-ds-storage)
- [`hotpot.ds.tools`](#hotpot-ds-tools)
- [`hotpot.ds.tools.analyze_structure_composition`](#hotpot-ds-tools-analyze-structure-composition)
- [`hotpot.ds.tools.electronic_state`](#hotpot-ds-tools-electronic-state)
- [`hotpot.ds.tools.electronic_state.assess_quality`](#hotpot-ds-tools-electronic-state-assess-quality)
- [`hotpot.ds.tools.electronic_state.build_database`](#hotpot-ds-tools-electronic-state-build-database)
- [`hotpot.ds.tools.oxidation`](#hotpot-ds-tools-oxidation)
- [`hotpot.ds.tools.oxidation.validate`](#hotpot-ds-tools-oxidation-validate)
- [`hotpot.ds.tools.spin`](#hotpot-ds-tools-spin)
- [`hotpot.ds.tools.spin.validate`](#hotpot-ds-tools-spin-validate)
- [`hotpot.utils`](#hotpot-utils)
- [`hotpot.utils.chem`](#hotpot-utils-chem)
- [`hotpot.utils.chem.atom`](#hotpot-utils-chem-atom)
- [`hotpot.utils.configs`](#hotpot-utils-configs)
- [`hotpot.utils.configs.logging_config`](#hotpot-utils-configs-logging-config)
- [`hotpot.utils.fmt_print`](#hotpot-utils-fmt-print)
- [`hotpot.utils.mp`](#hotpot-utils-mp)
- [`hotpot.utils.tools`](#hotpot-utils-tools)
- [`hotpot.utils.types`](#hotpot-utils-types)

<a id="hotpot-dataset"></a>
## `hotpot.dataset`

源码 `hotpot/dataset/__init__.py`；SHA-256 `a5f2c414c5864a247736dd5bbad04afa2d010344852cfe322339ba1844cbc704`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds"></a>
## `hotpot.ds`

源码 `hotpot/ds/__init__.py`；SHA-256 `38a9ea95f579c1370379f9da867d76fabdbba39b8fefde69cb2966b0456bbb60`。

显式 __all__：`('datasets_list', 'load', 'Dataset', 'Data', 'BaseSearcher', 'DataSearcher', 'DatasetSearcher', 'SmartsSearcher', 'load_bundle', 'Bundle', 'write_certificate')`。

显式导出绑定：

- `datasets_list` → `hotpot.ds.public.catalog.datasets_list`
- `load` → `hotpot.ds.public.catalog.load`；`load(ds_name: str, version: Optional[Union[str, int]]=None, *, db: Optional[Union[str, Path]]=None, archive_class: Optional[Union[str, Iterable[str]]]=None) -> Dataset`
- `Dataset` → `hotpot.ds.public.dataset.Dataset`；构造声明 `__init__(self, store: Any, dataset_info: Mapping[str, Any], release: Mapping[str, Any], headers: Iterable[Mapping[str, Any]], *, archive_classes: Iterable[str]) -> None`
- `Data` → `hotpot.ds.public.dataset.Data`；构造声明 `__init__(self, dataset: Dataset, value: Mapping[str, Any]) -> None`
- `BaseSearcher` → `hotpot.ds.public.search.BaseSearcher`
- `DataSearcher` → `hotpot.ds.public.search.DataSearcher`
- `DatasetSearcher` → `hotpot.ds.public.search.DatasetSearcher`
- `SmartsSearcher` → `hotpot.ds.public.search.SmartsSearcher`；构造声明 `__init__(self, smarts: str, *, semantics=None)`
- `load_bundle` → `hotpot.ds.public.bundle.load_bundle`；`load_bundle(root: Union[str, Path], ds_name: str, *, db: Optional[Union[str, Path]]=None) -> Bundle`
- `Bundle` → `hotpot.ds.public.bundle.Bundle`；构造声明 `__init__(self, root: Union[str, Path], ds_name: str, *, db: Optional[Union[str, Path]]=None, gates: Optional[Iterable[Any]]=None) -> None`
- `write_certificate` → `hotpot.ds.public.bundle.write_certificate`；`write_certificate(bundle: Bundle) -> Path`

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds--bundle-cleanup"></a>
## `hotpot.ds._bundle_cleanup`

源码 `hotpot/ds/_bundle_cleanup.py`；SHA-256 `0416bb8cbcc409cb912d0e0bffd6a828ba1fa49cc57c85652ae8ef0a9b14771d`。

显式 __all__：`('BundleCleanupError', 'remove_bundle_intermediates')`。

显式导出绑定：

- `BundleCleanupError` → `hotpot.ds._bundle_cleanup.BundleCleanupError`
- `remove_bundle_intermediates` → `hotpot.ds._bundle_cleanup.remove_bundle_intermediates`；`remove_bundle_intermediates(root: Path, dataset: str, database: Optional[Path]=None) -> dict[str, Any]`

<a id="hotpot-ds--bundle-cleanup-bundlecleanuperror"></a>
### `BundleCleanupError`

定义 owner `hotpot.ds._bundle_cleanup.BundleCleanupError`；源码 `hotpot/ds/_bundle_cleanup.py:63`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds._bundle_cleanup.BundleCleanupError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds._bundle_cleanup.RuntimeError`。

<a id="hotpot-ds--bundle-cleanup-remove-bundle-intermediates"></a>
### `remove_bundle_intermediates`

定义 owner `hotpot.ds._bundle_cleanup.remove_bundle_intermediates`；源码 `hotpot/ds/_bundle_cleanup.py:859`。

```python
remove_bundle_intermediates(root: Path, dataset: str, database: Optional[Path]=None) -> dict[str, Any]
```


<a id="hotpot-ds--communication"></a>
## `hotpot.ds._communication`

源码 `hotpot/ds/_communication.py`；SHA-256 `5c4663b4d21f5882e4c590c6d0b6cf4c54fa7eeba7eedae17092bab59db766e5`。

显式 __all__：`('CommunicationError', 'emit_envelope', 'extract_path', 'load_protocol', 'make_envelope', 'protocol_sha256', 'read_input')`。

显式导出绑定：

- `CommunicationError` → `hotpot.ds._communication.CommunicationError`
- `emit_envelope` → `hotpot.ds._communication.emit_envelope`；`emit_envelope(key: str, payload: Any, *, channel: str='default', stream: TextIO | None=None) -> dict[str, Any]`
- `extract_path` → `hotpot.ds._communication.extract_path`；`extract_path(document: Mapping[str, Any], path: str) -> Any`
- `load_protocol` → `hotpot.ds._communication.load_protocol`；`load_protocol() -> dict[str, Any]`
- `make_envelope` → `hotpot.ds._communication.make_envelope`；`make_envelope(key: str, payload: Any, *, channel: str='default') -> dict[str, Any]`
- `protocol_sha256` → `hotpot.ds._communication.protocol_sha256`；`protocol_sha256() -> str`
- `read_input` → `hotpot.ds._communication.read_input`；`read_input(consumer: str, stream: TextIO | None=None) -> dict[str, Any]`

<a id="hotpot-ds--communication-communicationerror"></a>
### `CommunicationError`

定义 owner `hotpot.ds._communication.CommunicationError`；源码 `hotpot/ds/_communication.py:21`。

基类原始声明：`ValueError`；装饰器：无。

显式导出位置：`hotpot.ds._communication.CommunicationError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds._communication.ValueError`。

<a id="hotpot-ds--communication-load-protocol"></a>
### `load_protocol`

定义 owner `hotpot.ds._communication.load_protocol`；源码 `hotpot/ds/_communication.py:110`。

```python
load_protocol() -> dict[str, Any]
```

装饰器：`lru_cache(maxsize=1)`。

<a id="hotpot-ds--communication-protocol-sha256"></a>
### `protocol_sha256`

定义 owner `hotpot.ds._communication.protocol_sha256`；源码 `hotpot/ds/_communication.py:189`。

```python
protocol_sha256() -> str
```

装饰器：`lru_cache(maxsize=1)`。

<a id="hotpot-ds--communication-extract-path"></a>
### `extract_path`

定义 owner `hotpot.ds._communication.extract_path`；源码 `hotpot/ds/_communication.py:300`。

```python
extract_path(document: Mapping[str, Any], path: str) -> Any
```

<a id="hotpot-ds--communication-make-envelope"></a>
### `make_envelope`

定义 owner `hotpot.ds._communication.make_envelope`；源码 `hotpot/ds/_communication.py:334`。

```python
make_envelope(key: str, payload: Any, *, channel: str='default') -> dict[str, Any]
```

<a id="hotpot-ds--communication-emit-envelope"></a>
### `emit_envelope`

定义 owner `hotpot.ds._communication.emit_envelope`；源码 `hotpot/ds/_communication.py:350`。

```python
emit_envelope(key: str, payload: Any, *, channel: str='default', stream: TextIO | None=None) -> dict[str, Any]
```

<a id="hotpot-ds--communication-read-input"></a>
### `read_input`

定义 owner `hotpot.ds._communication.read_input`；源码 `hotpot/ds/_communication.py:365`。

```python
read_input(consumer: str, stream: TextIO | None=None) -> dict[str, Any]
```


<a id="hotpot-ds--export"></a>
## `hotpot.ds._export`

源码 `hotpot/ds/_export.py`；SHA-256 `a8a1742ceb0198a2eb04f940282a992e7048bd40053f3898a40901dc9c6a7c71`。

显式 __all__：`()`。

<a id="hotpot-ds--export-exporterror"></a>
### `ExportError`

定义 owner `hotpot.ds._export.ExportError`；源码 `hotpot/ds/_export.py:20`。

基类原始声明：`ValueError`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds._export.ValueError`。

<a id="hotpot-ds--export-normalize-format"></a>
### `normalize_format`

定义 owner `hotpot.ds._export.normalize_format`；源码 `hotpot/ds/_export.py:24`。

```python
normalize_format(fmt: str) -> str
```

<a id="hotpot-ds--export-render-structure-record"></a>
### `render_structure_record`

定义 owner `hotpot.ds._export.render_structure_record`；源码 `hotpot/ds/_export.py:132`。

```python
render_structure_record(data: Any, fmt: str, *, structure_index: Optional[int]=None, structure_role: Optional[str]=None, structure_id: Optional[str]=None) -> dict[str, Any]
```

<a id="hotpot-ds--export-export-records"></a>
### `export_records`

定义 owner `hotpot.ds._export.export_records`；源码 `hotpot/ds/_export.py:186`。

```python
export_records(dataset: Any, output: Path, *, fmt: Optional[str], selection: Mapping[str, Any]) -> dict[str, Any]
```


<a id="hotpot-ds--paths"></a>
## `hotpot.ds._paths`

源码 `hotpot/ds/_paths.py`；SHA-256 `bb120eff431c1eba7010afdde8ba9d043e31283f52c3f05b0f1165f91f67b7ce`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `HOTPOT_PACKAGE_ROOT` → `hotpot.ds._paths.HOTPOT_PACKAGE_ROOT`
- `DATASET_ROOT` → `hotpot.ds._paths.DATASET_ROOT`
- `DATA_ROOT` → `hotpot.ds._paths.DATA_ROOT`
- `STORE_ROOT` → `hotpot.ds._paths.STORE_ROOT`
- `STORE_GENERATIONS_ROOT` → `hotpot.ds._paths.STORE_GENERATIONS_ROOT`
- `ACTIVE_STORE_ROOT` → `hotpot.ds._paths.ACTIVE_STORE_ROOT`
- `SOURCES_ROOT` → `hotpot.ds._paths.SOURCES_ROOT`
- `STRUCTURES_ROOT` → `hotpot.ds._paths.STRUCTURES_ROOT`
- `AUDITS_ROOT` → `hotpot.ds._paths.AUDITS_ROOT`
- `DEFAULT_STORE_PATH` → `hotpot.ds._paths.DEFAULT_STORE_PATH`

模块值/类型别名（声明存在不构成事实写许可）：

- `HOTPOT_PACKAGE_ROOT` = `Path(__file__).resolve().parents[1]`；`hotpot/ds/_paths.py:6`
- `DATASET_ROOT` = `HOTPOT_PACKAGE_ROOT / 'dataset'`；`hotpot/ds/_paths.py:7`
- `DATA_ROOT` = `DATASET_ROOT / 'data'`；`hotpot/ds/_paths.py:8`
- `STORE_ROOT` = `DATA_ROOT / 'store'`；`hotpot/ds/_paths.py:9`
- `STORE_GENERATIONS_ROOT` = `STORE_ROOT / 'generations'`；`hotpot/ds/_paths.py:10`
- `ACTIVE_STORE_ROOT` = `STORE_ROOT / 'active'`；`hotpot/ds/_paths.py:11`
- `SOURCES_ROOT` = `DATA_ROOT / 'sources'`；`hotpot/ds/_paths.py:12`
- `STRUCTURES_ROOT` = `DATA_ROOT / 'structures'`；`hotpot/ds/_paths.py:13`
- `AUDITS_ROOT` = `DATA_ROOT / 'audits'`；`hotpot/ds/_paths.py:14`
- `DEFAULT_STORE_PATH` = `ACTIVE_STORE_ROOT / 'datasets.sqlite3'`；`hotpot/ds/_paths.py:15`


<a id="hotpot-ds--store-copy"></a>
## `hotpot.ds._store_copy`

源码 `hotpot/ds/_store_copy.py`；SHA-256 `0310d68978cf0939f42a2dd6a47b5795adabc1fbba9f3d1d1b28415064c061f0`。

显式 __all__：`('StoreCopyError', 'copy_store')`。

显式导出绑定：

- `StoreCopyError` → `hotpot.ds._store_copy.StoreCopyError`
- `copy_store` → `hotpot.ds._store_copy.copy_store`；`copy_store(source_db: str | Path, target_package: str | Path) -> dict[str, Any]`

<a id="hotpot-ds--store-copy-storecopyerror"></a>
### `StoreCopyError`

定义 owner `hotpot.ds._store_copy.StoreCopyError`；源码 `hotpot/ds/_store_copy.py:39`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds._store_copy.StoreCopyError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds._store_copy.RuntimeError`。

<a id="hotpot-ds--store-copy-copy-store"></a>
### `copy_store`

定义 owner `hotpot.ds._store_copy.copy_store`；源码 `hotpot/ds/_store_copy.py:492`。

```python
copy_store(source_db: str | Path, target_package: str | Path) -> dict[str, Any]
```


<a id="hotpot-ds-cli"></a>
## `hotpot.ds.cli`

源码 `hotpot/ds/cli.py`；SHA-256 `8a9553bd0280d10040cdde0dcbb870408b8f54ef787ff396d21ca2d6d0707602`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `load_cli_documentation` → `hotpot.ds.cli.load_cli_documentation`；`load_cli_documentation(topic: Optional[str]=None) -> str`
- `add_arguments` → `hotpot.ds.cli.add_arguments`；`add_arguments(parser: argparse.ArgumentParser) -> None`
- `run` → `hotpot.ds.cli.run`；`run(args: argparse.Namespace) -> int`
- `build_parser` → `hotpot.ds.cli.build_parser`；`build_parser() -> argparse.ArgumentParser`
- `main` → `hotpot.ds.cli.main`；`main() -> int`
- `HELP_FORMATTER` → `argparse.ArgumentDefaultsHelpFormatter`

<a id="hotpot-ds-cli-load-cli-documentation"></a>
### `load_cli_documentation`

定义 owner `hotpot.ds.cli.load_cli_documentation`；源码 `hotpot/ds/cli.py:105`。

```python
load_cli_documentation(topic: Optional[str]=None) -> str
```

<a id="hotpot-ds-cli-add-arguments"></a>
### `add_arguments`

定义 owner `hotpot.ds.cli.add_arguments`；源码 `hotpot/ds/cli.py:225`。

```python
add_arguments(parser: argparse.ArgumentParser) -> None
```

<a id="hotpot-ds-cli-run"></a>
### `run`

定义 owner `hotpot.ds.cli.run`；源码 `hotpot/ds/cli.py:1038`。

```python
run(args: argparse.Namespace) -> int
```

<a id="hotpot-ds-cli-build-parser"></a>
### `build_parser`

定义 owner `hotpot.ds.cli.build_parser`；源码 `hotpot/ds/cli.py:1452`。

```python
build_parser() -> argparse.ArgumentParser
```

<a id="hotpot-ds-cli-main"></a>
### `main`

定义 owner `hotpot.ds.cli.main`；源码 `hotpot/ds/cli.py:1458`。

```python
main() -> int
```

模块值/类型别名（声明存在不构成事实写许可）：

- `HELP_FORMATTER` = `argparse.ArgumentDefaultsHelpFormatter`；`hotpot/ds/cli.py:39`


<a id="hotpot-ds-gates"></a>
## `hotpot.ds.gates`

源码 `hotpot/ds/gates/__init__.py`；SHA-256 `15050e07e5fe35039b371387204c046fca04534943ef20dc4ec2fff296b06f67`。

显式 __all__：`('EvidenceGate', 'QualityGate', 'BenchmarkGate')`。

显式导出绑定：

- `EvidenceGate` → `hotpot.ds.gates._gates.EvidenceGate`；构造声明 `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, require_locator: bool=True)`
- `QualityGate` → `hotpot.ds.gates._gates.QualityGate`；构造声明 `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, structure_checks: Optional[Mapping[str, Union[bool, Checker]]]=None)`
- `BenchmarkGate` → `hotpot.ds.gates._gates.BenchmarkGate`；构造声明 `__init__(self, checklist: Optional[Mapping[str, Union[bool, Checker]]]=None, direct_entry: Optional[Mapping[str, Union[bool, Checker]]]=None, *, vetoes: Optional[Iterable[str]]=None, numeric_tolerances: Optional[Mapping[str, Mapping[str, float]]]=None)`

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-gates--base"></a>
## `hotpot.ds.gates._base`

源码 `hotpot/ds/gates/_base.py`；SHA-256 `98ddc67e37ed9714f882af942fe49129d9cb5ae4906459cf619ea614bca61d31`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-gates--checker"></a>
## `hotpot.ds.gates._checker`

源码 `hotpot/ds/gates/_checker.py`；SHA-256 `db54f00911d82e3e0e66f1fa346298b60611762c1f9af554fc19a2487a2bd497`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `CheckerIsolationUnavailableError` → `hotpot.ds.gates._checker.CheckerIsolationUnavailableError`
- `CheckerExecutionError` → `hotpot.ds.gates._checker.CheckerExecutionError`
- `OSIsolatedCheckerRunner` → `hotpot.ds.gates._checker.OSIsolatedCheckerRunner`；构造声明 `__init__(self, forbidden_roots: Tuple[Path, ...]=()) -> None`
- `evaluate_configured_check` → `hotpot.ds.gates._checker.evaluate_configured_check`；`evaluate_configured_check(checker: _CheckerDefinition, case: Mapping[str, Any], bundle: Mapping[str, Any], *, builtin_result: Optional[Tuple[bool, str]]=None, isolated_runner: Optional[Any]=None) -> Tuple[bool, str]`
- `Checker` → `hotpot.ds.gates._checker.Checker`
- `CHECKER_PROTOCOL_DIGEST` → `hotpot.ds.gates._checker.CHECKER_PROTOCOL_DIGEST`

<a id="hotpot-ds-gates--checker-checkerisolationunavailableerror"></a>
### `CheckerIsolationUnavailableError`

定义 owner `hotpot.ds.gates._checker.CheckerIsolationUnavailableError`；源码 `hotpot/ds/gates/_checker.py:252`。

基类原始声明：`RuntimeError`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.gates._checker.RuntimeError`。

<a id="hotpot-ds-gates--checker-checkerexecutionerror"></a>
### `CheckerExecutionError`

定义 owner `hotpot.ds.gates._checker.CheckerExecutionError`；源码 `hotpot/ds/gates/_checker.py:256`。

基类原始声明：`RuntimeError`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.gates._checker.RuntimeError`。

<a id="hotpot-ds-gates--checker-osisolatedcheckerrunner"></a>
### `OSIsolatedCheckerRunner`

定义 owner `hotpot.ds.gates._checker.OSIsolatedCheckerRunner`；源码 `hotpot/ds/gates/_checker.py:287`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, forbidden_roots: Tuple[Path, ...]=()) -> None`；owner `hotpot.ds.gates._checker.OSIsolatedCheckerRunner`；`hotpot/ds/gates/_checker.py:292`
- `run(self, checker: Mapping[str, Any], case: Mapping[str, Any], bundle: Mapping[str, Any]) -> Tuple[bool, str]`；owner `hotpot.ds.gates._checker.OSIsolatedCheckerRunner`；`hotpot/ds/gates/_checker.py:453`

本类字段与枚举值：

- `os_isolation_verified` = `True` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.gates._checker.OSIsolatedCheckerRunner`；`hotpot/ds/gates/_checker.py:290`

<a id="hotpot-ds-gates--checker-evaluate-configured-check"></a>
### `evaluate_configured_check`

定义 owner `hotpot.ds.gates._checker.evaluate_configured_check`；源码 `hotpot/ds/gates/_checker.py:716`。

```python
evaluate_configured_check(checker: _CheckerDefinition, case: Mapping[str, Any], bundle: Mapping[str, Any], *, builtin_result: Optional[Tuple[bool, str]]=None, isolated_runner: Optional[Any]=None) -> Tuple[bool, str]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `Checker` = `Callable[[Mapping[str, Any], Mapping[str, Any]], Tuple[bool, str]]`；`hotpot/ds/gates/_checker.py:25`
- `CHECKER_PROTOCOL_DIGEST` = `hashlib.sha256(_canonical_bytes(_CHECKER_PROTOCOL)).hexdigest()`；`hotpot/ds/gates/_checker.py:247`


<a id="hotpot-ds-gates--gates"></a>
## `hotpot.ds.gates._gates`

源码 `hotpot/ds/gates/_gates.py`；SHA-256 `b320401bfaaf965d73fb9f45768ab72db99d3ca97032be0470e4286dbda39d20`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `EvidenceGate` → `hotpot.ds.gates._gates.EvidenceGate`；构造声明 `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, require_locator: bool=True)`
- `QualityGate` → `hotpot.ds.gates._gates.QualityGate`；构造声明 `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, structure_checks: Optional[Mapping[str, Union[bool, Checker]]]=None)`
- `BenchmarkGate` → `hotpot.ds.gates._gates.BenchmarkGate`；构造声明 `__init__(self, checklist: Optional[Mapping[str, Union[bool, Checker]]]=None, direct_entry: Optional[Mapping[str, Union[bool, Checker]]]=None, *, vetoes: Optional[Iterable[str]]=None, numeric_tolerances: Optional[Mapping[str, Mapping[str, float]]]=None)`

<a id="hotpot-ds-gates--gates-evidencegate"></a>
### `EvidenceGate`

定义 owner `hotpot.ds.gates._gates.EvidenceGate`；源码 `hotpot/ds/gates/_gates.py:116`。

基类原始声明：`_ImmutableGate`；装饰器：无。

显式导出位置：`hotpot.ds.gates.EvidenceGate`, `hotpot.ds.gates.evidence.EvidenceGate`。

本类属性与方法：

- `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, require_locator: bool=True)`；owner `hotpot.ds.gates._gates.EvidenceGate`；`hotpot/ds/gates/_gates.py:119`

继承成员（逐项保留声明 owner）：

- `__setattr__(self, name: str, value: Any) -> None`；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:31`；继承经 `hotpot.ds.gates._base._ImmutableGate`
- `info(self) -> dict` [读；只读]；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:35`；继承经 `hotpot.ds.gates._base._ImmutableGate`
- `digest(self) -> str` [读；只读]；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:39`；继承经 `hotpot.ds.gates._base._ImmutableGate`

<a id="hotpot-ds-gates--gates-qualitygate"></a>
### `QualityGate`

定义 owner `hotpot.ds.gates._gates.QualityGate`；源码 `hotpot/ds/gates/_gates.py:177`。

基类原始声明：`_ImmutableGate`；装饰器：无。

显式导出位置：`hotpot.ds.gates.QualityGate`, `hotpot.ds.gates.quality.QualityGate`。

本类属性与方法：

- `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, structure_checks: Optional[Mapping[str, Union[bool, Checker]]]=None)`；owner `hotpot.ds.gates._gates.QualityGate`；`hotpot/ds/gates/_gates.py:180`

继承成员（逐项保留声明 owner）：

- `__setattr__(self, name: str, value: Any) -> None`；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:31`；继承经 `hotpot.ds.gates._base._ImmutableGate`
- `info(self) -> dict` [读；只读]；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:35`；继承经 `hotpot.ds.gates._base._ImmutableGate`
- `digest(self) -> str` [读；只读]；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:39`；继承经 `hotpot.ds.gates._base._ImmutableGate`

<a id="hotpot-ds-gates--gates-benchmarkgate"></a>
### `BenchmarkGate`

定义 owner `hotpot.ds.gates._gates.BenchmarkGate`；源码 `hotpot/ds/gates/_gates.py:234`。

基类原始声明：`_ImmutableGate`；装饰器：无。

显式导出位置：`hotpot.ds.gates.BenchmarkGate`, `hotpot.ds.gates.benchmark.BenchmarkGate`。

本类属性与方法：

- `__init__(self, checklist: Optional[Mapping[str, Union[bool, Checker]]]=None, direct_entry: Optional[Mapping[str, Union[bool, Checker]]]=None, *, vetoes: Optional[Iterable[str]]=None, numeric_tolerances: Optional[Mapping[str, Mapping[str, float]]]=None)`；owner `hotpot.ds.gates._gates.BenchmarkGate`；`hotpot/ds/gates/_gates.py:237`

继承成员（逐项保留声明 owner）：

- `__setattr__(self, name: str, value: Any) -> None`；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:31`；继承经 `hotpot.ds.gates._base._ImmutableGate`
- `info(self) -> dict` [读；只读]；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:35`；继承经 `hotpot.ds.gates._base._ImmutableGate`
- `digest(self) -> str` [读；只读]；owner `hotpot.ds.gates._base._ImmutableGate`；`hotpot/ds/gates/_base.py:39`；继承经 `hotpot.ds.gates._base._ImmutableGate`


<a id="hotpot-ds-gates--persistence"></a>
## `hotpot.ds.gates._persistence`

源码 `hotpot/ds/gates/_persistence.py`；SHA-256 `a6a4d8dae90a340fadb65db96b2a76f76eb5cdcb98f23380abf92e102b4a2373`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `default_gates` → `hotpot.ds.gates._persistence.default_gates`；`default_gates() -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]`
- `serialize_gate_configs` → `hotpot.ds.gates._persistence.serialize_gate_configs`；`serialize_gate_configs(gates: Iterable[object]) -> list`
- `deserialize_gate_configs` → `hotpot.ds.gates._persistence.deserialize_gate_configs`；`deserialize_gate_configs(configurations: Iterable[Mapping[str, object]]) -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]`
- `materialize_checker_sources` → `hotpot.ds.gates._persistence.materialize_checker_sources`；`materialize_checker_sources(gates: Iterable[object], bundle_root: Path) -> Tuple[Path, ...]`

<a id="hotpot-ds-gates--persistence-default-gates"></a>
### `default_gates`

定义 owner `hotpot.ds.gates._persistence.default_gates`；源码 `hotpot/ds/gates/_persistence.py:21`。

```python
default_gates() -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]
```

<a id="hotpot-ds-gates--persistence-serialize-gate-configs"></a>
### `serialize_gate_configs`

定义 owner `hotpot.ds.gates._persistence.serialize_gate_configs`；源码 `hotpot/ds/gates/_persistence.py:25`。

```python
serialize_gate_configs(gates: Iterable[object]) -> list
```

<a id="hotpot-ds-gates--persistence-deserialize-gate-configs"></a>
### `deserialize_gate_configs`

定义 owner `hotpot.ds.gates._persistence.deserialize_gate_configs`；源码 `hotpot/ds/gates/_persistence.py:30`。

```python
deserialize_gate_configs(configurations: Iterable[Mapping[str, object]]) -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]
```

<a id="hotpot-ds-gates--persistence-materialize-checker-sources"></a>
### `materialize_checker_sources`

定义 owner `hotpot.ds.gates._persistence.materialize_checker_sources`；源码 `hotpot/ds/gates/_persistence.py:43`。

```python
materialize_checker_sources(gates: Iterable[object], bundle_root: Path) -> Tuple[Path, ...]
```


<a id="hotpot-ds-gates-benchmark"></a>
## `hotpot.ds.gates.benchmark`

源码 `hotpot/ds/gates/benchmark.py`；SHA-256 `d8a1fa9c3df10d99878e3ca9950cdd847e4fcbb4c57d11b3a85335253ae72aad`。

显式 __all__：`('BenchmarkGate',)`。

显式导出绑定：

- `BenchmarkGate` → `hotpot.ds.gates._gates.BenchmarkGate`；构造声明 `__init__(self, checklist: Optional[Mapping[str, Union[bool, Checker]]]=None, direct_entry: Optional[Mapping[str, Union[bool, Checker]]]=None, *, vetoes: Optional[Iterable[str]]=None, numeric_tolerances: Optional[Mapping[str, Mapping[str, float]]]=None)`

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-gates-evidence"></a>
## `hotpot.ds.gates.evidence`

源码 `hotpot/ds/gates/evidence.py`；SHA-256 `306d80396d246b473bfe740ad5a9f219fd337eb0a5dfa6b1a61272e79553e792`。

显式 __all__：`('EvidenceGate',)`。

显式导出绑定：

- `EvidenceGate` → `hotpot.ds.gates._gates.EvidenceGate`；构造声明 `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, require_locator: bool=True)`

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-gates-quality"></a>
## `hotpot.ds.gates.quality`

源码 `hotpot/ds/gates/quality.py`；SHA-256 `ccaf3829844c028988d54e0cb34056b79eefdd90a3ba6a9834b1348e916c1b8a`。

显式 __all__：`('QualityGate',)`。

显式导出绑定：

- `QualityGate` → `hotpot.ds.gates._gates.QualityGate`；构造声明 `__init__(self, checks: Optional[Mapping[str, Union[bool, Checker]]]=None, *, level_checks: Optional[Mapping[str, Iterable[str]]]=None, structure_checks: Optional[Mapping[str, Union[bool, Checker]]]=None)`

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-governance"></a>
## `hotpot.ds.governance`

源码 `hotpot/ds/governance.py`；SHA-256 `88365dfd0baef09febc4a5c1c7f48ec5a827ca34000fec9e12f1e6567198b250`。

显式 __all__：`('GovernanceAuthorizationError', 'GovernanceError', 'GovernanceVerificationError', 'GovernanceKey', 'PinnedRoot', 'VerifiedRegistry', 'load_pinned_root', 'verify_registry', 'verify_hotpot_attestation')`。

显式导出绑定：

- `GovernanceAuthorizationError` → `hotpot.ds.governance.GovernanceAuthorizationError`
- `GovernanceError` → `hotpot.ds.governance.GovernanceError`
- `GovernanceVerificationError` → `hotpot.ds.governance.GovernanceVerificationError`
- `GovernanceKey` → `hotpot.ds.governance.GovernanceKey`；构造声明 `GovernanceKey(key_id: str, public_key: str, predecessor_key_id: Optional[str], activated_at: str, revoked_at: Optional[str])`
- `PinnedRoot` → `hotpot.ds.governance.PinnedRoot`；构造声明 `PinnedRoot(schema: str, signature_schema: str, pinned_genesis_key_id: str, public_key: str, created_at: str, digest: str)`
- `VerifiedRegistry` → `hotpot.ds.governance.VerifiedRegistry`；构造声明 `__init__(self, snapshot: Mapping[str, Any], keys: Mapping[str, GovernanceKey]) -> None`
- `load_pinned_root` → `hotpot.ds.governance.load_pinned_root`；`load_pinned_root() -> PinnedRoot`
- `verify_registry` → `hotpot.ds.governance.verify_registry`；`verify_registry(snapshot: Mapping[str, Any]) -> VerifiedRegistry`
- `verify_hotpot_attestation` → `hotpot.ds.governance.verify_hotpot_attestation`；`verify_hotpot_attestation(attestation: Mapping[str, Any], dataset: str, release: str, case_id: str, registry: Mapping[str, Any] | VerifiedRegistry) -> tuple[bool, str]`

<a id="hotpot-ds-governance-governanceerror"></a>
### `GovernanceError`

定义 owner `hotpot.ds.governance.GovernanceError`；源码 `hotpot/ds/governance.py:87`。

基类原始声明：`ValueError`；装饰器：无。

显式导出位置：`hotpot.ds.governance.GovernanceError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.governance.ValueError`。

<a id="hotpot-ds-governance-governanceverificationerror"></a>
### `GovernanceVerificationError`

定义 owner `hotpot.ds.governance.GovernanceVerificationError`；源码 `hotpot/ds/governance.py:91`。

基类原始声明：`GovernanceError`；装饰器：无。

显式导出位置：`hotpot.ds.governance.GovernanceVerificationError`。

<a id="hotpot-ds-governance-governanceauthorizationerror"></a>
### `GovernanceAuthorizationError`

定义 owner `hotpot.ds.governance.GovernanceAuthorizationError`；源码 `hotpot/ds/governance.py:95`。

基类原始声明：`GovernanceError`；装饰器：无。

显式导出位置：`hotpot.ds.governance.GovernanceAuthorizationError`。

<a id="hotpot-ds-governance-pinnedroot"></a>
### `PinnedRoot`

定义 owner `hotpot.ds.governance.PinnedRoot`；源码 `hotpot/ds/governance.py:189`。

基类原始声明：无显式基类；装饰器：`dataclass(frozen=True)`。

显式导出位置：`hotpot.ds.governance.PinnedRoot`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
PinnedRoot(schema: str, signature_schema: str, pinned_genesis_key_id: str, public_key: str, created_at: str, digest: str)
```

本类属性与方法：

- `to_dict(self) -> dict[str, Any]`；owner `hotpot.ds.governance.PinnedRoot`；`hotpot/ds/governance.py:197`

本类字段与枚举值：

- `schema: str` [只读值字段]；owner `hotpot.ds.governance.PinnedRoot`；`hotpot/ds/governance.py:190`
- `signature_schema: str` [只读值字段]；owner `hotpot.ds.governance.PinnedRoot`；`hotpot/ds/governance.py:191`
- `pinned_genesis_key_id: str` [只读值字段]；owner `hotpot.ds.governance.PinnedRoot`；`hotpot/ds/governance.py:192`
- `public_key: str` [只读值字段]；owner `hotpot.ds.governance.PinnedRoot`；`hotpot/ds/governance.py:193`
- `created_at: str` [只读值字段]；owner `hotpot.ds.governance.PinnedRoot`；`hotpot/ds/governance.py:194`
- `digest: str` [只读值字段]；owner `hotpot.ds.governance.PinnedRoot`；`hotpot/ds/governance.py:195`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-ds-governance-governancekey"></a>
### `GovernanceKey`

定义 owner `hotpot.ds.governance.GovernanceKey`；源码 `hotpot/ds/governance.py:208`。

基类原始声明：无显式基类；装饰器：`dataclass(frozen=True)`。

显式导出位置：`hotpot.ds.governance.GovernanceKey`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
GovernanceKey(key_id: str, public_key: str, predecessor_key_id: Optional[str], activated_at: str, revoked_at: Optional[str])
```

本类字段与枚举值：

- `key_id: str` [只读值字段]；owner `hotpot.ds.governance.GovernanceKey`；`hotpot/ds/governance.py:209`
- `public_key: str` [只读值字段]；owner `hotpot.ds.governance.GovernanceKey`；`hotpot/ds/governance.py:210`
- `predecessor_key_id: Optional[str]` [只读值字段]；owner `hotpot.ds.governance.GovernanceKey`；`hotpot/ds/governance.py:211`
- `activated_at: str` [只读值字段]；owner `hotpot.ds.governance.GovernanceKey`；`hotpot/ds/governance.py:212`
- `revoked_at: Optional[str]` [只读值字段]；owner `hotpot.ds.governance.GovernanceKey`；`hotpot/ds/governance.py:213`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-ds-governance-verifiedregistry"></a>
### `VerifiedRegistry`

定义 owner `hotpot.ds.governance.VerifiedRegistry`；源码 `hotpot/ds/governance.py:216`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.governance.VerifiedRegistry`。

本类属性与方法：

- `__init__(self, snapshot: Mapping[str, Any], keys: Mapping[str, GovernanceKey]) -> None`；owner `hotpot.ds.governance.VerifiedRegistry`；`hotpot/ds/governance.py:221`
- `to_dict(self) -> dict[str, Any]`；owner `hotpot.ds.governance.VerifiedRegistry`；`hotpot/ds/governance.py:237`
- `key_is_active(self, key_id: str, at: str) -> bool`；owner `hotpot.ds.governance.VerifiedRegistry`；`hotpot/ds/governance.py:240`
- `keys(self) -> Mapping[str, GovernanceKey]` [读；只读]；owner `hotpot.ds.governance.VerifiedRegistry`；`hotpot/ds/governance.py:234`

本类字段与枚举值：

- `revision` = `int(copied['revision'])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.governance.VerifiedRegistry`；`hotpot/ds/governance.py:227`
- `digest` = `str(copied['registry_digest'])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.governance.VerifiedRegistry`；`hotpot/ds/governance.py:228`
- `pinned_genesis_key_id` = `str(copied['pinned_genesis_key_id'])` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.governance.VerifiedRegistry`；`hotpot/ds/governance.py:229`

<a id="hotpot-ds-governance-load-pinned-root"></a>
### `load_pinned_root`

定义 owner `hotpot.ds.governance.load_pinned_root`；源码 `hotpot/ds/governance.py:252`。

```python
load_pinned_root() -> PinnedRoot
```

<a id="hotpot-ds-governance-default-registry-snapshot"></a>
### `default_registry_snapshot`

定义 owner `hotpot.ds.governance.default_registry_snapshot`；源码 `hotpot/ds/governance.py:280`。

```python
default_registry_snapshot() -> dict[str, Any]
```

<a id="hotpot-ds-governance-verify-registry"></a>
### `verify_registry`

定义 owner `hotpot.ds.governance.verify_registry`；源码 `hotpot/ds/governance.py:347`。

```python
verify_registry(snapshot: Mapping[str, Any]) -> VerifiedRegistry
```

<a id="hotpot-ds-governance-verify-hotpot-attestation"></a>
### `verify_hotpot_attestation`

定义 owner `hotpot.ds.governance.verify_hotpot_attestation`；源码 `hotpot/ds/governance.py:501`。

```python
verify_hotpot_attestation(attestation: Mapping[str, Any], dataset: str, release: str, case_id: str, registry: Mapping[str, Any] | VerifiedRegistry) -> tuple[bool, str]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `ROOT_SCHEMA` = `'hotpot.ds.governance-root/current'`；`hotpot/ds/governance.py:27`
- `REGISTRY_SCHEMA` = `'hotpot.ds.governance-registry/current'`；`hotpot/ds/governance.py:28`
- `KEY_EVENT_SCHEMA` = `'hotpot.ds.governance-key/current'`；`hotpot/ds/governance.py:29`
- `ATTESTATION_SCHEMA` = `'hotpot.ds.attestation/current'`；`hotpot/ds/governance.py:30`
- `SIGNATURE_SCHEMA` = `'hotpot.ds.ed25519/current'`；`hotpot/ds/governance.py:31`
- `KEY_EVENT_DOMAIN` = `'hotpot.ds.governance-key'`；`hotpot/ds/governance.py:32`
- `ATTESTATION_DOMAIN` = `'hotpot.ds.attestation'`；`hotpot/ds/governance.py:33`


<a id="hotpot-ds-providers"></a>
## `hotpot.ds.providers`

源码 `hotpot/ds/providers/__init__.py`；SHA-256 `4420ee2575c99b8f1fb6dd733d735e1f1a856f7ad46d3482c3a28b7c5efea51e`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-providers--io"></a>
## `hotpot.ds.providers._io`

源码 `hotpot/ds/providers/_io.py`；SHA-256 `bcea74f8aff0bda659a51dff3df3e7c208ba8a533b48a44e0b85c5d131e491c2`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `download_file` → `hotpot.ds.providers._io.download_file`；`download_file(fp, _url)`
- `uncompress_all_gz` → `hotpot.ds.providers._io.uncompress_all_gz`；`uncompress_all_gz(file_path)`
- `uncompress_zip` → `hotpot.ds.providers._io.uncompress_zip`；`uncompress_zip(zip_path, extract_dir)`

<a id="hotpot-ds-providers--io-download-file"></a>
### `download_file`

定义 owner `hotpot.ds.providers._io.download_file`；源码 `hotpot/ds/providers/_io.py:11`。

```python
download_file(fp, _url)
```

<a id="hotpot-ds-providers--io-uncompress-all-gz"></a>
### `uncompress_all_gz`

定义 owner `hotpot.ds.providers._io.uncompress_all_gz`；源码 `hotpot/ds/providers/_io.py:27`。

```python
uncompress_all_gz(file_path)
```

<a id="hotpot-ds-providers--io-uncompress-zip"></a>
### `uncompress_zip`

定义 owner `hotpot.ds.providers._io.uncompress_zip`；源码 `hotpot/ds/providers/_io.py:45`。

```python
uncompress_zip(zip_path, extract_dir)
```


<a id="hotpot-ds-providers-lnqm"></a>
## `hotpot.ds.providers.lnqm`

源码 `hotpot/ds/providers/lnqm/__init__.py`；SHA-256 `abb446a3e497728b9180978b4a94b95dec5d0c0e77dba805bbcdf2cc3ddf2db0`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `LnQmDataset` → `hotpot.ds.providers.lnqm.LnQmDataset`；构造声明 `__init__(self)`
- `SOURCE_ARCHIVE_URL` → `hotpot.ds.providers.lnqm.SOURCE_ARCHIVE_URL`

<a id="hotpot-ds-providers-lnqm-lnqmdataset"></a>
### `LnQmDataset`

定义 owner `hotpot.ds.providers.lnqm.LnQmDataset`；源码 `hotpot/ds/providers/lnqm/__init__.py:19`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self)`；owner `hotpot.ds.providers.lnqm.LnQmDataset`；`hotpot/ds/providers/lnqm/__init__.py:20`
- `to_pyg_dataset(self)`；owner `hotpot.ds.providers.lnqm.LnQmDataset`；`hotpot/ds/providers/lnqm/__init__.py:43`

模块值/类型别名（声明存在不构成事实写许可）：

- `SOURCE_ARCHIVE_URL` = `'https://zenodo.org/api/records/10406124/files-archive'`；`hotpot/ds/providers/lnqm/__init__.py:9`


<a id="hotpot-ds-providers-lnqm-pyg-data"></a>
## `hotpot.ds.providers.lnqm.pyg_data`

源码 `hotpot/ds/providers/lnqm/pyg_data.py`；SHA-256 `505c87c44ca80b952978b62e895a10eb617886d695279152eaffe9a780ed3351`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `LnQM_Dataset` → `hotpot.ds.providers.lnqm.pyg_data.LnQM_Dataset`；构造声明 `__init__(self, path_to_hdf5: Union[str, Path], transform=None)`

<a id="hotpot-ds-providers-lnqm-pyg-data-lnqm-dataset"></a>
### `LnQM_Dataset`

定义 owner `hotpot.ds.providers.lnqm.pyg_data.LnQM_Dataset`；源码 `hotpot/ds/providers/lnqm/pyg_data.py:11`。

基类原始声明：`InMemoryDataset`；装饰器：无。

本类属性与方法：

- `__init__(self, path_to_hdf5: Union[str, Path], transform=None)`；owner `hotpot.ds.providers.lnqm.pyg_data.LnQM_Dataset`；`hotpot/ds/providers/lnqm/pyg_data.py:18`
- `from_hdf5(fp: Union[str, Path]) -> tuple[Data, defaultdict]` [staticmethod]；装饰器 `staticmethod`；owner `hotpot.ds.providers.lnqm.pyg_data.LnQM_Dataset`；`hotpot/ds/providers/lnqm/pyg_data.py:29`
- `to_hdf5(self, fp: Union[str, Path])`；owner `hotpot.ds.providers.lnqm.pyg_data.LnQM_Dataset`；`hotpot/ds/providers/lnqm/pyg_data.py:50`

本类字段与枚举值：

- `path_to_hdf5` = `path_to_hdf5` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.providers.lnqm.pyg_data.LnQM_Dataset`；`hotpot/ds/providers/lnqm/pyg_data.py:20`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`torch_geometric.data.InMemoryDataset`。


<a id="hotpot-ds-providers-tmqm"></a>
## `hotpot.ds.providers.tmqm`

源码 `hotpot/ds/providers/tmqm/__init__.py`；SHA-256 `db6b1e0197d1c5da3a20b4470a92d73feb28f1901a90cdbc0566fb03c66e8ae1`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `DataGenerator` → `hotpot.ds.providers.tmqm.DataGenerator`；构造声明 `__init__(self, data_dir, nproc=1)`
- `TmQmDataset` → `hotpot.ds.providers.tmqm.TmQmDataset`；构造声明 `__init__(self, nproc=None)`

<a id="hotpot-ds-providers-tmqm-datagenerator"></a>
### `DataGenerator`

定义 owner `hotpot.ds.providers.tmqm.DataGenerator`；源码 `hotpot/ds/providers/tmqm/__init__.py:15`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, data_dir, nproc=1)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:68`
- `get_xyz_generator(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:82`
- `get_csv_generator(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:99`
- `get_q_generator(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:108`
- `get_bo_generator(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:146`
- `mp_iter(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:270`
- `next_mol(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:295`
- `iter_mol(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:332`
- `__iter__(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:340`
- `__next__(self)`；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:346`

本类字段与枚举值：

- `XYZ_FILE` = `'tmQm.xyz'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:58`
- `CSV_FILE` = `'tmQm.csv'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:59`
- `Q_FILE` = `'tmQm.q'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:60`
- `BO_FILE` = `'tmQm.BO'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:61`
- `mol_attrs` = `('identifier', 'energy', 'dispersion', 'dipole', 'metal_q', 'Hl', 'HOMO', 'LUMO', 'polarizability')` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:63`
- `data_dir` = `data_dir` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:69`
- `nproc` = `nproc` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:71`
- `text2mol_queues` = `[mp.Queue() for _ in range(nproc)]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:72`
- `result_queues` = `[mp.Queue() for _ in range(nproc)]` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.providers.tmqm.DataGenerator`；`hotpot/ds/providers/tmqm/__init__.py:73`

<a id="hotpot-ds-providers-tmqm-tmqmdataset"></a>
### `TmQmDataset`

定义 owner `hotpot.ds.providers.tmqm.TmQmDataset`；源码 `hotpot/ds/providers/tmqm/__init__.py:351`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, nproc=None)`；owner `hotpot.ds.providers.tmqm.TmQmDataset`；`hotpot/ds/providers/tmqm/__init__.py:369`
- `__iter__(self)`；owner `hotpot.ds.providers.tmqm.TmQmDataset`；`hotpot/ds/providers/tmqm/__init__.py:394`
- `__next__(self)`；owner `hotpot.ds.providers.tmqm.TmQmDataset`；`hotpot/ds/providers/tmqm/__init__.py:398`

本类字段与枚举值：

- `mol_attrs` = `DataGenerator.mol_attrs` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.providers.tmqm.TmQmDataset`；`hotpot/ds/providers/tmqm/__init__.py:368`
- `queue` = `mp.Queue()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.providers.tmqm.TmQmDataset`；`hotpot/ds/providers/tmqm/__init__.py:391`
- `nproc` = `os.cpu_count()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.providers.tmqm.TmQmDataset`；`hotpot/ds/providers/tmqm/__init__.py:387`


<a id="hotpot-ds-providers-tmqm-download"></a>
## `hotpot.ds.providers.tmqm.download`

源码 `hotpot/ds/providers/tmqm/download.py`；SHA-256 `a22f39e13362b5ed470b797a9c019653945588a7ec69b95d7dbe3574f51ac16a`。

显式 __all__：`('urls', 'download_file', 'download_files', 'uncompress_all_gz')`。

显式导出绑定：

- `urls` → `hotpot.ds.providers.tmqm.download.urls`
- `download_file` → `hotpot.ds.providers._io.download_file`；`download_file(fp, _url)`
- `download_files` → `hotpot.ds.providers.tmqm.download.download_files`；`download_files()`
- `uncompress_all_gz` → `hotpot.ds.providers._io.uncompress_all_gz`；`uncompress_all_gz(file_path)`

<a id="hotpot-ds-providers-tmqm-download-download-files"></a>
### `download_files`

定义 owner `hotpot.ds.providers.tmqm.download.download_files`；源码 `hotpot/ds/providers/tmqm/download.py:26`。

```python
download_files()
```

模块值/类型别名（声明存在不构成事实写许可）：

- `urls` = <长声明表达式；完整值见库存及源码位置>；`hotpot/ds/providers/tmqm/download.py:18`


<a id="hotpot-ds-public"></a>
## `hotpot.ds.public`

源码 `hotpot/ds/public/__init__.py`；SHA-256 `7fe289d64a124e71e568c18449c8b68b81f5cf4ebac22d5f21f3ab1a910cd8b1`。

显式 __all__：`()`。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-public--assessment"></a>
## `hotpot.ds.public._assessment`

源码 `hotpot/ds/public/_assessment.py`；SHA-256 `217efb44920da3c49ea6ac6d93e32281d1cef0f59479f196378e0a69e6d13b4f`。

显式 __all__：`('B_LEVELS', 'E_LEVELS', 'assess_case', 'assess_evidence', 'canonical_digest', 'canonical_json', 'compare_observations', 'find_conflicts', 'identity_key', 'observation_context', 'observation_property', 'summarize_assessments', 'validate_observation', 'validate_property_policy')`。

显式导出绑定：

- `B_LEVELS` → `hotpot.ds.public._assessment.B_LEVELS`
- `E_LEVELS` → `hotpot.ds.public._assessment.E_LEVELS`
- `assess_case` → `hotpot.ds.public._assessment.assess_case`；`assess_case(case: Mapping[str, Any], *, core_labels: Sequence[str], core_info: Sequence[str], required_structure_roles: Sequence[str], asset_assessments: Sequence[Mapping[str, Any]], conflicting: bool, benchmark_policy: Mapping[str, Any], evidence_policy: Optional[Mapping[str, Any]]=None, quality_policy: Optional[Mapping[str, Any]]=None, conflict_assessments: Sequence[Mapping[str, Any]]=(), identity_reasons: Sequence[Mapping[str, Any]]=(), checker_context: Optional[Mapping[str, Any]]=None, isolated_runner: Optional[Any]=None) -> Dict[str, Any]`
- `assess_evidence` → `hotpot.ds.public._assessment.assess_evidence`；`assess_evidence(case: Mapping[str, Any], required_targets: Optional[Iterable[str]]=None, *, asset_assessments: Sequence[Mapping[str, Any]]=(), policy: Optional[Mapping[str, Any]]=None, checker_context: Optional[Mapping[str, Any]]=None, isolated_runner: Optional[Any]=None) -> Dict[str, Any]`
- `canonical_digest` → `hotpot.ds.public._assessment.canonical_digest`；`canonical_digest(value: Any) -> str`
- `canonical_json` → `hotpot.ds.public._assessment.canonical_json`；`canonical_json(value: Any) -> str`
- `compare_observations` → `hotpot.ds.public._assessment.compare_observations`；`compare_observations(left: Mapping[str, Any], right: Mapping[str, Any], tolerances: Mapping[str, Mapping[str, float]], *, property_definitions: Optional[Mapping[str, Any]]=None, property_comparators: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]`
- `find_conflicts` → `hotpot.ds.public._assessment.find_conflicts`；`find_conflicts(cases: Iterable[Mapping[str, Any]], identity_schema: Mapping[str, Any], tolerances: Mapping[str, Mapping[str, float]], *, property_definitions: Optional[Mapping[str, Any]]=None, property_comparators: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]`
- `identity_key` → `hotpot.ds.public._assessment.identity_key`；`identity_key(case: Mapping[str, Any], identity_schema: Mapping[str, Any]) -> Tuple[Optional[str], list[Dict[str, Any]]]`
- `observation_context` → `hotpot.ds.public._assessment.observation_context`；`observation_context(observation: Mapping[str, Any], definition: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]`
- `observation_property` → `hotpot.ds.public._assessment.observation_property`；`observation_property(observation: Mapping[str, Any]) -> Optional[str]`
- `summarize_assessments` → `hotpot.ds.public._assessment.summarize_assessments`；`summarize_assessments(assessments: Iterable[Mapping[str, Any]]) -> Dict[str, Any]`
- `validate_observation` → `hotpot.ds.public._assessment.validate_observation`；`validate_observation(observation: Mapping[str, Any], definitions: Mapping[str, Any], comparators: Mapping[str, Any]) -> None`
- `validate_property_policy` → `hotpot.ds.public._assessment.validate_property_policy`；`validate_property_policy(definitions: Mapping[str, Any], comparators: Mapping[str, Any]) -> None`

<a id="hotpot-ds-public--assessment-canonical-json"></a>
### `canonical_json`

定义 owner `hotpot.ds.public._assessment.canonical_json`；源码 `hotpot/ds/public/_assessment.py:150`。

```python
canonical_json(value: Any) -> str
```

<a id="hotpot-ds-public--assessment-canonical-digest"></a>
### `canonical_digest`

定义 owner `hotpot.ds.public._assessment.canonical_digest`；源码 `hotpot/ds/public/_assessment.py:154`。

```python
canonical_digest(value: Any) -> str
```

<a id="hotpot-ds-public--assessment-validate-identity-schema"></a>
### `validate_identity_schema`

定义 owner `hotpot.ds.public._assessment.validate_identity_schema`；源码 `hotpot/ds/public/_assessment.py:205`。

```python
validate_identity_schema(value: Mapping[str, Any]) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-observation-property"></a>
### `observation_property`

定义 owner `hotpot.ds.public._assessment.observation_property`；源码 `hotpot/ds/public/_assessment.py:300`。

```python
observation_property(observation: Mapping[str, Any]) -> Optional[str]
```

<a id="hotpot-ds-public--assessment-validate-property-policy"></a>
### `validate_property_policy`

定义 owner `hotpot.ds.public._assessment.validate_property_policy`；源码 `hotpot/ds/public/_assessment.py:318`。

```python
validate_property_policy(definitions: Mapping[str, Any], comparators: Mapping[str, Any]) -> None
```

<a id="hotpot-ds-public--assessment-validate-observation"></a>
### `validate_observation`

定义 owner `hotpot.ds.public._assessment.validate_observation`；源码 `hotpot/ds/public/_assessment.py:473`。

```python
validate_observation(observation: Mapping[str, Any], definitions: Mapping[str, Any], comparators: Mapping[str, Any]) -> None
```

<a id="hotpot-ds-public--assessment-assess-evidence"></a>
### `assess_evidence`

定义 owner `hotpot.ds.public._assessment.assess_evidence`；源码 `hotpot/ds/public/_assessment.py:1261`。

```python
assess_evidence(case: Mapping[str, Any], required_targets: Optional[Iterable[str]]=None, *, asset_assessments: Sequence[Mapping[str, Any]]=(), policy: Optional[Mapping[str, Any]]=None, checker_context: Optional[Mapping[str, Any]]=None, isolated_runner: Optional[Any]=None) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-identity-key"></a>
### `identity_key`

定义 owner `hotpot.ds.public._assessment.identity_key`；源码 `hotpot/ds/public/_assessment.py:1511`。

```python
identity_key(case: Mapping[str, Any], identity_schema: Mapping[str, Any]) -> Tuple[Optional[str], list[Dict[str, Any]]]
```

<a id="hotpot-ds-public--assessment-sample-match"></a>
### `sample_match`

定义 owner `hotpot.ds.public._assessment.sample_match`；源码 `hotpot/ds/public/_assessment.py:1533`。

```python
sample_match(left: Mapping[str, Any], right: Mapping[str, Any], identity_schema: Mapping[str, Any]) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-observation-context"></a>
### `observation_context`

定义 owner `hotpot.ds.public._assessment.observation_context`；源码 `hotpot/ds/public/_assessment.py:1611`。

```python
observation_context(observation: Mapping[str, Any], definition: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-observation-with-resolved-entities"></a>
### `observation_with_resolved_entities`

定义 owner `hotpot.ds.public._assessment.observation_with_resolved_entities`；源码 `hotpot/ds/public/_assessment.py:1633`。

```python
observation_with_resolved_entities(case: Mapping[str, Any], observation: Mapping[str, Any]) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-compare-observations"></a>
### `compare_observations`

定义 owner `hotpot.ds.public._assessment.compare_observations`；源码 `hotpot/ds/public/_assessment.py:1729`。

```python
compare_observations(left: Mapping[str, Any], right: Mapping[str, Any], tolerances: Mapping[str, Mapping[str, float]], *, property_definitions: Optional[Mapping[str, Any]]=None, property_comparators: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-find-conflicts"></a>
### `find_conflicts`

定义 owner `hotpot.ds.public._assessment.find_conflicts`；源码 `hotpot/ds/public/_assessment.py:2090`。

```python
find_conflicts(cases: Iterable[Mapping[str, Any]], identity_schema: Mapping[str, Any], tolerances: Mapping[str, Mapping[str, float]], *, property_definitions: Optional[Mapping[str, Any]]=None, property_comparators: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-assess-case"></a>
### `assess_case`

定义 owner `hotpot.ds.public._assessment.assess_case`；源码 `hotpot/ds/public/_assessment.py:2169`。

```python
assess_case(case: Mapping[str, Any], *, core_labels: Sequence[str], core_info: Sequence[str], required_structure_roles: Sequence[str], asset_assessments: Sequence[Mapping[str, Any]], conflicting: bool, benchmark_policy: Mapping[str, Any], evidence_policy: Optional[Mapping[str, Any]]=None, quality_policy: Optional[Mapping[str, Any]]=None, conflict_assessments: Sequence[Mapping[str, Any]]=(), identity_reasons: Sequence[Mapping[str, Any]]=(), checker_context: Optional[Mapping[str, Any]]=None, isolated_runner: Optional[Any]=None) -> Dict[str, Any]
```

<a id="hotpot-ds-public--assessment-summarize-assessments"></a>
### `summarize_assessments`

定义 owner `hotpot.ds.public._assessment.summarize_assessments`；源码 `hotpot/ds/public/_assessment.py:2681`。

```python
summarize_assessments(assessments: Iterable[Mapping[str, Any]]) -> Dict[str, Any]
```

模块值/类型别名（声明存在不构成事实写许可）：

- `B_LEVELS` = `tuple((f'B{index}' for index in range(5)))`；`hotpot/ds/public/_assessment.py:27`
- `E_LEVELS` = `tuple((f'E{index}' for index in range(5)))`；`hotpot/ds/public/_assessment.py:28`


<a id="hotpot-ds-public--bundle-state"></a>
## `hotpot.ds.public._bundle_state`

源码 `hotpot/ds/public/_bundle_state.py`；SHA-256 `7763755974e9ef563799742721061e7450c55089b3953571338ff936dd3ec026`。

显式 __all__：`('BUNDLE_SCHEMA', 'BundleAlreadyBuiltError', 'BundleNotBuiltError', 'BundleState', 'BundleStateError', 'ConcurrentStateError', 'DirtySourceStateError')`。

显式导出绑定：

- `BUNDLE_SCHEMA` → `hotpot.ds.public._bundle_state.BUNDLE_SCHEMA`
- `BundleAlreadyBuiltError` → `hotpot.ds.public._bundle_state.BundleAlreadyBuiltError`
- `BundleNotBuiltError` → `hotpot.ds.public._bundle_state.BundleNotBuiltError`
- `BundleState` → `hotpot.ds.public._bundle_state.BundleState`；构造声明 `__init__(self, root: Path, ds_name: str)`
- `BundleStateError` → `hotpot.ds.public._bundle_state.BundleStateError`
- `ConcurrentStateError` → `hotpot.ds.public._bundle_state.ConcurrentStateError`
- `DirtySourceStateError` → `hotpot.ds.public._bundle_state.DirtySourceStateError`

<a id="hotpot-ds-public--bundle-state-bundlestateerror"></a>
### `BundleStateError`

定义 owner `hotpot.ds.public._bundle_state.BundleStateError`；源码 `hotpot/ds/public/_bundle_state.py:124`。

基类原始声明：`ValueError`；装饰器：无。

显式导出位置：`hotpot.ds.public._bundle_state.BundleStateError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public._bundle_state.ValueError`。

<a id="hotpot-ds-public--bundle-state-bundlealreadybuilterror"></a>
### `BundleAlreadyBuiltError`

定义 owner `hotpot.ds.public._bundle_state.BundleAlreadyBuiltError`；源码 `hotpot/ds/public/_bundle_state.py:128`。

基类原始声明：`BundleStateError`；装饰器：无。

显式导出位置：`hotpot.ds.public._bundle_state.BundleAlreadyBuiltError`。

<a id="hotpot-ds-public--bundle-state-bundlenotbuilterror"></a>
### `BundleNotBuiltError`

定义 owner `hotpot.ds.public._bundle_state.BundleNotBuiltError`；源码 `hotpot/ds/public/_bundle_state.py:132`。

基类原始声明：`BundleStateError`；装饰器：无。

显式导出位置：`hotpot.ds.public._bundle_state.BundleNotBuiltError`。

<a id="hotpot-ds-public--bundle-state-concurrentstateerror"></a>
### `ConcurrentStateError`

定义 owner `hotpot.ds.public._bundle_state.ConcurrentStateError`；源码 `hotpot/ds/public/_bundle_state.py:136`。

基类原始声明：`BundleStateError`；装饰器：无。

显式导出位置：`hotpot.ds.public._bundle_state.ConcurrentStateError`。

<a id="hotpot-ds-public--bundle-state-dirtysourcestateerror"></a>
### `DirtySourceStateError`

定义 owner `hotpot.ds.public._bundle_state.DirtySourceStateError`；源码 `hotpot/ds/public/_bundle_state.py:140`。

基类原始声明：`BundleStateError`；装饰器：无。

显式导出位置：`hotpot.ds.public._bundle_state.DirtySourceStateError`。

<a id="hotpot-ds-public--bundle-state-bundlestate"></a>
### `BundleState`

定义 owner `hotpot.ds.public._bundle_state.BundleState`；源码 `hotpot/ds/public/_bundle_state.py:777`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.public._bundle_state.BundleState`。

本类属性与方法：

- `__init__(self, root: Path, ds_name: str)`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:780`
- `build(cls, root: Union[os.PathLike[str], str], ds_name: str, policy_json: Mapping[str, Any], target_context: Mapping[str, Any]) -> 'BundleState'` [classmethod]；装饰器 `classmethod`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:786`
- `open(cls, root: Union[os.PathLike[str], str], ds_name: str) -> 'BundleState'` [classmethod]；装饰器 `classmethod`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1002`
- `connect(self) -> Iterator[sqlite3.Connection]`；装饰器 `contextmanager`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1084`
- `__len__(self) -> int`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1124`
- `iter_cases(self) -> Iterator[Dict[str, Any]]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1128`
- `get_case(self, case_id: str) -> Dict[str, Any]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1136`
- `refresh_source_state(self, *, _source_observations: Optional[Dict[str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]]]=None) -> Dict[str, Any]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1253`
- `refresh_governance_state(self) -> Dict[str, Any]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1352`
- `prepare_synchronization(self) -> Dict[str, Any]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1436`
- `synchronization_snapshot(self) -> Iterator[_SynchronizationSnapshot]`；装饰器 `contextmanager`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1441`
- `mutate_case(self, case_id: str, action: str, mutation: Callable[[Dict[str, Any]], tuple[Mapping[str, Any], Any]]) -> Dict[str, Any]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1617`
- `mutate_policy(self, new_policy: Mapping[str, Any], *, action: str='gate_policy_changed', details: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1718`
- `publish_generation(self, generation_id: str, generation_digest: str, *, expected_source_revision: int, expected_source_digest: str, expected_policy_revision: int, expected_policy_digest: str, expected_governance_registry_revision: int, expected_governance_registry_digest: str) -> Dict[str, Any]`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1810`
- `refresh_audit_projection(self) -> Path`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1921`
- `write_policy_projection(self) -> Path`；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1951`
- `manifest(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1104`
- `policy(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1109`
- `target_context(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1117`
- `source_digest(self) -> str` [读；只读]；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1146`
- `policy_digest(self) -> str` [读；只读]；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1161`
- `status(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:1165`

本类字段与枚举值：

- `root` = `root.resolve()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:781`
- `ds_name` = `ds_name` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:782`
- `path` = `self.root / 'cases' / 'facts.sqlite3'` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.public._bundle_state.BundleState`；`hotpot/ds/public/_bundle_state.py:783`

模块值/类型别名（声明存在不构成事实写许可）：

- `BUNDLE_SCHEMA` = `'hotpot-ds-bundle/current'`；`hotpot/ds/public/_bundle_state.py:37`


<a id="hotpot-ds-public--generation"></a>
## `hotpot.ds.public._generation`

源码 `hotpot/ds/public/_generation.py`；SHA-256 `64e0c2e0c115495bf373c9da0c24b6cd0bf4df5134c6e6391075ea7ed90e1f7e`。

显式 __all__：`('AssessmentGeneration', 'AssessmentGenerationError', 'AssetSourceChangedError', 'GenerationIntegrityError')`。

显式导出绑定：

- `AssessmentGeneration` → `hotpot.ds.public._generation.AssessmentGeneration`；构造声明 `__init__(self, root: Path, directory: Path, manifest: Mapping[str, Any])`
- `AssessmentGenerationError` → `hotpot.ds.public._generation.AssessmentGenerationError`
- `AssetSourceChangedError` → `hotpot.ds.public._generation.AssetSourceChangedError`
- `GenerationIntegrityError` → `hotpot.ds.public._generation.GenerationIntegrityError`

<a id="hotpot-ds-public--generation-assessmentgenerationerror"></a>
### `AssessmentGenerationError`

定义 owner `hotpot.ds.public._generation.AssessmentGenerationError`；源码 `hotpot/ds/public/_generation.py:65`。

基类原始声明：`ValueError`；装饰器：无。

显式导出位置：`hotpot.ds.public._generation.AssessmentGenerationError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public._generation.ValueError`。

<a id="hotpot-ds-public--generation-generationintegrityerror"></a>
### `GenerationIntegrityError`

定义 owner `hotpot.ds.public._generation.GenerationIntegrityError`；源码 `hotpot/ds/public/_generation.py:69`。

基类原始声明：`AssessmentGenerationError`；装饰器：无。

显式导出位置：`hotpot.ds.public._generation.GenerationIntegrityError`。

<a id="hotpot-ds-public--generation-assetsourcechangederror"></a>
### `AssetSourceChangedError`

定义 owner `hotpot.ds.public._generation.AssetSourceChangedError`；源码 `hotpot/ds/public/_generation.py:73`。

基类原始声明：`AssessmentGenerationError`；装饰器：无。

显式导出位置：`hotpot.ds.public._generation.AssetSourceChangedError`。

<a id="hotpot-ds-public--generation-assessmentgeneration"></a>
### `AssessmentGeneration`

定义 owner `hotpot.ds.public._generation.AssessmentGeneration`；源码 `hotpot/ds/public/_generation.py:1819`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.public._generation.AssessmentGeneration`。

本类属性与方法：

- `__init__(self, root: Path, directory: Path, manifest: Mapping[str, Any])`；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:1822`
- `build(cls, root: Union[os.PathLike[str], str], state: BundleState, policy: Mapping[str, Any], governance_registry: Mapping[str, Any], *, input_snapshot: Mapping[str, Any], isolated_runner: Optional[Any]=None) -> 'AssessmentGeneration'` [classmethod]；装饰器 `classmethod`；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:1828`
- `open(cls, root: Union[os.PathLike[str], str], generation_id: str, expected_digest: Optional[str]=None) -> 'AssessmentGeneration'` [classmethod]；装饰器 `classmethod`；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2044`
- `lease(self) -> Iterator['AssessmentGeneration']`；装饰器 `contextmanager`；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2263`
- `get_case(self, case_id: str) -> Dict[str, Any]`；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2269`
- `iter_cases(self) -> Iterator[Dict[str, Any]]`；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2304`
- `iter_archive_ids(self, archive_class: str) -> Iterator[str]`；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2318`
- `id(self) -> str` [读；只读]；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2235`
- `digest(self) -> str` [读；只读]；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2239`
- `manifest(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2243`
- `manifest_path(self) -> Path` [读；只读]；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2247`
- `manifest_digest(self) -> str` [读；只读]；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2251`
- `summary(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public._generation.AssessmentGeneration`；`hotpot/ds/public/_generation.py:2256`

模块值/类型别名（声明存在不构成事实写许可）：

- `GENERATION_SCHEMA` = `'hotpot-ds-assessment-generation/current'`；`hotpot/ds/public/_generation.py:46`


<a id="hotpot-ds-public--implementation"></a>
## `hotpot.ds.public._implementation`

源码 `hotpot/ds/public/_implementation.py`；SHA-256 `1ab1998741ecf0c7e1d57d75198d3da1c6fd6c13d7ec5801b7a3fdde1d02d2cd`。

显式 __all__：`('PolicyProtocolError', 'parser_binding_for_format', 'require_current_runtime_policy', 'runtime_policy', 'runtime_policy_digest', 'runtime_policy_is_current', 'validate_frozen_runtime_policy')`。

显式导出绑定：

- `PolicyProtocolError` → `hotpot.ds.public._implementation.PolicyProtocolError`
- `parser_binding_for_format` → `hotpot.ds.public._implementation.parser_binding_for_format`；`parser_binding_for_format(format_name: str, policy: Mapping[str, Any] | None=None) -> dict[str, Any] | None`
- `require_current_runtime_policy` → `hotpot.ds.public._implementation.require_current_runtime_policy`；`require_current_runtime_policy(value: Mapping[str, Any]) -> None`
- `runtime_policy` → `hotpot.ds.public._implementation.runtime_policy`；`runtime_policy() -> dict[str, Any]`
- `runtime_policy_digest` → `hotpot.ds.public._implementation.runtime_policy_digest`；`runtime_policy_digest() -> str`
- `runtime_policy_is_current` → `hotpot.ds.public._implementation.runtime_policy_is_current`；`runtime_policy_is_current(value: Mapping[str, Any]) -> bool`
- `validate_frozen_runtime_policy` → `hotpot.ds.public._implementation.validate_frozen_runtime_policy`；`validate_frozen_runtime_policy(value: Mapping[str, Any]) -> dict[str, Any]`

<a id="hotpot-ds-public--implementation-policyprotocolerror"></a>
### `PolicyProtocolError`

定义 owner `hotpot.ds.public._implementation.PolicyProtocolError`；源码 `hotpot/ds/public/_implementation.py:239`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds.public._implementation.PolicyProtocolError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public._implementation.RuntimeError`。

<a id="hotpot-ds-public--implementation-runtime-policy"></a>
### `runtime_policy`

定义 owner `hotpot.ds.public._implementation.runtime_policy`；源码 `hotpot/ds/public/_implementation.py:435`。

```python
runtime_policy() -> dict[str, Any]
```

<a id="hotpot-ds-public--implementation-runtime-policy-digest"></a>
### `runtime_policy_digest`

定义 owner `hotpot.ds.public._implementation.runtime_policy_digest`；源码 `hotpot/ds/public/_implementation.py:441`。

```python
runtime_policy_digest() -> str
```

<a id="hotpot-ds-public--implementation-validate-frozen-runtime-policy"></a>
### `validate_frozen_runtime_policy`

定义 owner `hotpot.ds.public._implementation.validate_frozen_runtime_policy`；源码 `hotpot/ds/public/_implementation.py:665`。

```python
validate_frozen_runtime_policy(value: Mapping[str, Any]) -> dict[str, Any]
```

<a id="hotpot-ds-public--implementation-runtime-policy-is-current"></a>
### `runtime_policy_is_current`

定义 owner `hotpot.ds.public._implementation.runtime_policy_is_current`；源码 `hotpot/ds/public/_implementation.py:775`。

```python
runtime_policy_is_current(value: Mapping[str, Any]) -> bool
```

<a id="hotpot-ds-public--implementation-require-current-runtime-policy"></a>
### `require_current_runtime_policy`

定义 owner `hotpot.ds.public._implementation.require_current_runtime_policy`；源码 `hotpot/ds/public/_implementation.py:785`。

```python
require_current_runtime_policy(value: Mapping[str, Any]) -> None
```

<a id="hotpot-ds-public--implementation-parser-binding-for-format"></a>
### `parser_binding_for_format`

定义 owner `hotpot.ds.public._implementation.parser_binding_for_format`；源码 `hotpot/ds/public/_implementation.py:796`。

```python
parser_binding_for_format(format_name: str, policy: Mapping[str, Any] | None=None) -> dict[str, Any] | None
```

模块值/类型别名（声明存在不构成事实写许可）：

- `RUNTIME_POLICY_SCHEMA` = `'hotpot-ds-runtime-policy/current'`；`hotpot/ds/public/_implementation.py:29`
- `ASSESSMENT_RUNTIME_SCHEMA` = `'hotpot-ds-assessment-runtime/current'`；`hotpot/ds/public/_implementation.py:30`
- `ASSESSMENT_STORE_PROTOCOL_SCHEMA` = `'hotpot-ds-assessment-store/current'`；`hotpot/ds/public/_implementation.py:31`
- `CHECKER_PROTOCOL_SCHEMA` = `'hotpot-ds-checker/current'`；`hotpot/ds/public/_implementation.py:32`
- `PARSER_REGISTRY_SCHEMA` = `'hotpot-ds-parser-registry/current'`；`hotpot/ds/public/_implementation.py:33`
- `PARSER_BINDING_SCHEMA` = `'hotpot-ds-parser-binding/current'`；`hotpot/ds/public/_implementation.py:34`
- `PARSER_PROTOCOL_SCHEMA` = `'hotpot-ds-molecule-parser/current'`；`hotpot/ds/public/_implementation.py:35`
- `PARSER_SELECTION_SCHEMA` = `'hotpot-ds-parser-selection/current'`；`hotpot/ds/public/_implementation.py:36`


<a id="hotpot-ds-public-bundle"></a>
## `hotpot.ds.public.bundle`

源码 `hotpot/ds/public/bundle.py`；SHA-256 `b109537e2347e97a921418c0476962c9de19380626397dc2535c5feaac800796`。

显式 __all__：`('Bundle', 'load_bundle', 'write_certificate')`。

显式导出绑定：

- `Bundle` → `hotpot.ds.public.bundle.Bundle`；构造声明 `__init__(self, root: Union[str, Path], ds_name: str, *, db: Optional[Union[str, Path]]=None, gates: Optional[Iterable[Any]]=None) -> None`
- `load_bundle` → `hotpot.ds.public.bundle.load_bundle`；`load_bundle(root: Union[str, Path], ds_name: str, *, db: Optional[Union[str, Path]]=None) -> Bundle`
- `write_certificate` → `hotpot.ds.public.bundle.write_certificate`；`write_certificate(bundle: Bundle) -> Path`

<a id="hotpot-ds-public-bundle-existingdatasetpolicyerror"></a>
### `ExistingDatasetPolicyError`

定义 owner `hotpot.ds.public.bundle.ExistingDatasetPolicyError`；源码 `hotpot/ds/public/bundle.py:46`。

基类原始声明：`RuntimeError`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.bundle.RuntimeError`。

<a id="hotpot-ds-public-bundle-missingcertificateerror"></a>
### `MissingCertificateError`

定义 owner `hotpot.ds.public.bundle.MissingCertificateError`；源码 `hotpot/ds/public/bundle.py:50`。

基类原始声明：`RuntimeError`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.bundle.RuntimeError`。

<a id="hotpot-ds-public-bundle-sealedreleaseerror"></a>
### `SealedReleaseError`

定义 owner `hotpot.ds.public.bundle.SealedReleaseError`；源码 `hotpot/ds/public/bundle.py:54`。

基类原始声明：`RuntimeError`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.bundle.RuntimeError`。

<a id="hotpot-ds-public-bundle-sealedintegrityerror"></a>
### `SealedIntegrityError`

定义 owner `hotpot.ds.public.bundle.SealedIntegrityError`；源码 `hotpot/ds/public/bundle.py:58`。

基类原始声明：`RuntimeError`；装饰器：无。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.bundle.RuntimeError`。

<a id="hotpot-ds-public-bundle-bundle"></a>
### `Bundle`

定义 owner `hotpot.ds.public.bundle.Bundle`；源码 `hotpot/ds/public/bundle.py:149`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.Bundle`, `hotpot.ds.public.bundle.Bundle`。

本类属性与方法：

- `__init__(self, root: Union[str, Path], ds_name: str, *, db: Optional[Union[str, Path]]=None, gates: Optional[Iterable[Any]]=None) -> None`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:152`
- `from_certificate(cls, path: Union[str, Path], *, db: Optional[Union[str, Path]]=None) -> 'Bundle'` [classmethod]；装饰器 `classmethod`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:205`
- `__len__(self) -> int`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:440`
- `__repr__(self) -> str`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:443`
- `use_gate(self, gate: Any) -> None`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:449`
- `synchronize(self) -> Dict[str, Any]`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:475`
- `write_certificate(self) -> Path`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:533`
- `ingest(self, *, include_incomplete: bool=False, reason: Optional[str]=None) -> Dict[str, Any]`；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:558`
- `root(self) -> Path` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:232`
- `name(self) -> str` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:236`
- `mode(self) -> str` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:240`
- `info(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:251`
- `status(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:270`
- `is_synchronized(self) -> bool` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:362`
- `summary(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:377`
- `gates(self) -> Mapping[str, Any]` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:381`
- `certificate_path(self) -> Path` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:392`
- `certificate_stale(self) -> bool` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:396`
- `cases(self) -> CaseCollection` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:408`
- `incomplete(self) -> CaseView` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:412`
- `ordinary(self) -> CaseView` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:416`
- `benchmark(self) -> CaseView` [读；只读]；owner `hotpot.ds.public.bundle.Bundle`；`hotpot/ds/public/bundle.py:420`

<a id="hotpot-ds-public-bundle-load-bundle"></a>
### `load_bundle`

定义 owner `hotpot.ds.public.bundle.load_bundle`；源码 `hotpot/ds/public/bundle.py:1473`。

```python
load_bundle(root: Union[str, Path], ds_name: str, *, db: Optional[Union[str, Path]]=None) -> Bundle
```

<a id="hotpot-ds-public-bundle-write-certificate"></a>
### `write_certificate`

定义 owner `hotpot.ds.public.bundle.write_certificate`；源码 `hotpot/ds/public/bundle.py:1489`。

```python
write_certificate(bundle: Bundle) -> Path
```


<a id="hotpot-ds-public-case"></a>
## `hotpot.ds.public.case`

源码 `hotpot/ds/public/case.py`；SHA-256 `2371e066507b63fdb5cd7e7840b7a48efa30f399cc97d29b523228a4d518543a`。

显式 __all__：`('Case', 'CaseCollection', 'CaseOwnershipError', 'CaseView', 'UnsynchronizedBundleError')`。

显式导出绑定：

- `Case` → `hotpot.ds.public.case.Case`；构造声明 `__init__(self, owner: Any, case_id: str) -> None`
- `CaseCollection` → `hotpot.ds.public.case.CaseCollection`；构造声明 `__init__(self, owner: Any) -> None`
- `CaseOwnershipError` → `hotpot.ds.public.case.CaseOwnershipError`
- `CaseView` → `hotpot.ds.public.case.CaseView`；构造声明 `__init__(self, owner: Any, case_ids: Sequence[str]) -> None`
- `UnsynchronizedBundleError` → `hotpot.ds.public.case.UnsynchronizedBundleError`

<a id="hotpot-ds-public-case-caseownershiperror"></a>
### `CaseOwnershipError`

定义 owner `hotpot.ds.public.case.CaseOwnershipError`；源码 `hotpot/ds/public/case.py:148`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds.public.case.CaseOwnershipError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.case.RuntimeError`。

<a id="hotpot-ds-public-case-unsynchronizedbundleerror"></a>
### `UnsynchronizedBundleError`

定义 owner `hotpot.ds.public.case.UnsynchronizedBundleError`；源码 `hotpot/ds/public/case.py:152`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds.public.case.UnsynchronizedBundleError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.case.RuntimeError`。

<a id="hotpot-ds-public-case-case"></a>
### `Case`

定义 owner `hotpot.ds.public.case.Case`；源码 `hotpot/ds/public/case.py:156`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.public.case.Case`。

本类属性与方法：

- `__init__(self, owner: Any, case_id: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:161`
- `__setattr__(self, name: str, value: Any) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:165`
- `set_core_info(self, name: str, value: Any, *, reason: Optional[str]=None) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:269`
- `remove_core_info(self, name: str, *, reason: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:288`
- `add_structure(self, data: Mapping[str, Any]) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:297`
- `replace_structure(self, structure_id: str, data: Mapping[str, Any], *, reason: str) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:300`
- `withdraw_structure(self, structure_id: str, *, reason: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:311`
- `add_observation(self, data: Mapping[str, Any]) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:316`
- `replace_observation(self, observation_id: str, data: Mapping[str, Any], *, reason: str) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:319`
- `withdraw_observation(self, observation_id: str, *, reason: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:335`
- `add_claim(self, data: Mapping[str, Any]) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:344`
- `replace_claim(self, claim_id: str, data: Mapping[str, Any], *, reason: str) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:347`
- `withdraw_claim(self, claim_id: str, *, reason: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:358`
- `add_evidence(self, target: str, source: str, *, relation: str='supports', locator: Optional[str]=None, note: Optional[str]=None, details: Optional[Mapping[str, Any]]=None) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:361`
- `verify_evidence(self, evidence_id: str, *, method: str, verifier: str, note: Optional[str]=None, details: Optional[Mapping[str, Any]]=None) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:391`
- `withdraw_verification(self, verification_id: str, *, reason: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:424`
- `replace_evidence(self, evidence_id: str, source: str, *, reason: str, locator: Optional[str]=None, note: Optional[str]=None, details: Optional[Mapping[str, Any]]=None) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:433`
- `withdraw_evidence(self, evidence_id: str, *, reason: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:481`
- `attest_benchmark(self, approved: bool, *, authority: Literal['User', 'Other'], actor: str, reason: Optional[str]=None) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:486`
- `withdraw_attestation(self, attestation_id: str, *, reason: str) -> None`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:531`
- `__repr__(self) -> str`；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:652`
- `id(self) -> str` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:171`
- `info(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:194`
- `core_info(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:202`
- `structures(self) -> tuple[Dict[str, Any], ...]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:206`
- `labels(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:210`
- `observations(self) -> tuple[Dict[str, Any], ...]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:222`
- `claims(self) -> tuple[Dict[str, Any], ...]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:226`
- `evidence(self) -> tuple[Dict[str, Any], ...]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:230`
- `attestations(self) -> tuple[Dict[str, Any], ...]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:234`
- `evidence_levels(self) -> Dict[str, str]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:238`
- `evidence_level(self) -> Optional[str]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:242`
- `quality_level(self) -> str` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:246`
- `archive_class(self) -> str` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:250`
- `benchmark(self) -> bool` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:254`
- `benchmark_status(self) -> str` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:258`
- `reasons(self) -> list[Dict[str, Any]]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:262`
- `assessment(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.case.Case`；`hotpot/ds/public/case.py:266`

<a id="hotpot-ds-public-case-caseview"></a>
### `CaseView`

定义 owner `hotpot.ds.public.case.CaseView`；源码 `hotpot/ds/public/case.py:656`。

基类原始声明：`Sequence[Case]`；装饰器：无。

显式导出位置：`hotpot.ds.public.case.CaseView`。

本类属性与方法：

- `__init__(self, owner: Any, case_ids: Sequence[str]) -> None`；owner `hotpot.ds.public.case.CaseView`；`hotpot/ds/public/case.py:661`
- `__len__(self) -> int`；owner `hotpot.ds.public.case.CaseView`；`hotpot/ds/public/case.py:665`
- `__iter__(self) -> Iterator[Case]`；owner `hotpot.ds.public.case.CaseView`；`hotpot/ds/public/case.py:668`
- `__getitem__(self, index: Union[int, slice]) -> Union[Case, 'CaseView']`；owner `hotpot.ds.public.case.CaseView`；`hotpot/ds/public/case.py:671`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`collections.abc.Sequence`。

<a id="hotpot-ds-public-case-casecollection"></a>
### `CaseCollection`

定义 owner `hotpot.ds.public.case.CaseCollection`；源码 `hotpot/ds/public/case.py:677`。

基类原始声明：`CaseView`；装饰器：无。

显式导出位置：`hotpot.ds.public.case.CaseCollection`。

本类属性与方法：

- `__init__(self, owner: Any) -> None`；owner `hotpot.ds.public.case.CaseCollection`；`hotpot/ds/public/case.py:696`
- `__len__(self) -> int`；owner `hotpot.ds.public.case.CaseCollection`；`hotpot/ds/public/case.py:705`
- `__iter__(self) -> Iterator[Case]`；owner `hotpot.ds.public.case.CaseCollection`；`hotpot/ds/public/case.py:708`
- `__getitem__(self, index: Union[int, slice]) -> Union[Case, CaseView]`；owner `hotpot.ds.public.case.CaseCollection`；`hotpot/ds/public/case.py:714`
- `select(self, **criteria: Any) -> CaseView`；owner `hotpot.ds.public.case.CaseCollection`；`hotpot/ds/public/case.py:720`


<a id="hotpot-ds-public-catalog"></a>
## `hotpot.ds.public.catalog`

源码 `hotpot/ds/public/catalog.py`；SHA-256 `d9558132bc7e86af86af8380004198ba2f6c9938369fc0afa7e10e4d965aa4c5`。

显式 __all__：`('DEFAULT_DATABASE', 'DatasetCatalog', 'datasets_list', 'load')`。

显式导出绑定：

- `DEFAULT_DATABASE` → `hotpot.ds._paths.DEFAULT_STORE_PATH`
- `DatasetCatalog` → `hotpot.ds.public.catalog.DatasetCatalog`；构造声明 `__init__(self, db: Optional[Union[str, Path]]=None) -> None`
- `datasets_list` → `hotpot.ds.public.catalog.datasets_list`
- `load` → `hotpot.ds.public.catalog.load`；`load(ds_name: str, version: Optional[Union[str, int]]=None, *, db: Optional[Union[str, Path]]=None, archive_class: Optional[Union[str, Iterable[str]]]=None) -> Dataset`

<a id="hotpot-ds-public-catalog-datasetcatalog"></a>
### `DatasetCatalog`

定义 owner `hotpot.ds.public.catalog.DatasetCatalog`；源码 `hotpot/ds/public/catalog.py:27`。

基类原始声明：`Sequence[str]`；装饰器：无。

显式导出位置：`hotpot.ds.public.catalog.DatasetCatalog`。

本类属性与方法：

- `__init__(self, db: Optional[Union[str, Path]]=None) -> None`；owner `hotpot.ds.public.catalog.DatasetCatalog`；`hotpot/ds/public/catalog.py:30`
- `__call__(self, *, db: Optional[Union[str, Path]]=None) -> list[str]`；owner `hotpot.ds.public.catalog.DatasetCatalog`；`hotpot/ds/public/catalog.py:43`
- `__len__(self) -> int`；owner `hotpot.ds.public.catalog.DatasetCatalog`；`hotpot/ds/public/catalog.py:48`
- `__iter__(self) -> Iterator[str]`；owner `hotpot.ds.public.catalog.DatasetCatalog`；`hotpot/ds/public/catalog.py:51`
- `__getitem__(self, index)`；owner `hotpot.ds.public.catalog.DatasetCatalog`；`hotpot/ds/public/catalog.py:54`
- `__repr__(self) -> str`；owner `hotpot.ds.public.catalog.DatasetCatalog`；`hotpot/ds/public/catalog.py:57`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`collections.abc.Sequence`。

<a id="hotpot-ds-public-catalog-load"></a>
### `load`

定义 owner `hotpot.ds.public.catalog.load`；源码 `hotpot/ds/public/catalog.py:75`。

```python
load(ds_name: str, version: Optional[Union[str, int]]=None, *, db: Optional[Union[str, Path]]=None, archive_class: Optional[Union[str, Iterable[str]]]=None) -> Dataset
```

模块值/类型别名（声明存在不构成事实写许可）：

- `DEFAULT_DATABASE` = `DEFAULT_STORE_PATH`；`hotpot/ds/public/catalog.py:14`
- `datasets_list` = `DatasetCatalog()`；`hotpot/ds/public/catalog.py:124`


<a id="hotpot-ds-public-certificate"></a>
## `hotpot.ds.public.certificate`

源码 `hotpot/ds/public/certificate.py`；SHA-256 `64210a7b7867cc9bd3c4cd5a781c211096be5552bc8c6f642b23426d57c12e09`。

显式 __all__：`('CertificateError', 'CertificateFormatError', 'CertificateIntegrityError', 'CertificatePathError', 'certificate_digest', 'load_certificate_envelope', 'verify_bundle_certificate', 'write_bundle_certificate')`。

显式导出绑定：

- `CertificateError` → `hotpot.ds.public.certificate.CertificateError`
- `CertificateFormatError` → `hotpot.ds.public.certificate.CertificateFormatError`
- `CertificateIntegrityError` → `hotpot.ds.public.certificate.CertificateIntegrityError`
- `CertificatePathError` → `hotpot.ds.public.certificate.CertificatePathError`
- `certificate_digest` → `hotpot.ds.public.certificate.certificate_digest`；`certificate_digest(path: Union[str, Path]) -> str`
- `load_certificate_envelope` → `hotpot.ds.public.certificate.load_certificate_envelope`；`load_certificate_envelope(path: Union[str, Path]) -> Dict[str, Any]`
- `verify_bundle_certificate` → `hotpot.ds.public.certificate.verify_bundle_certificate`；`verify_bundle_certificate(bundle: Any, path: Optional[Union[str, Path]]=None) -> Dict[str, Any]`
- `write_bundle_certificate` → `hotpot.ds.public.certificate.write_bundle_certificate`；`write_bundle_certificate(bundle: Any) -> Path`

<a id="hotpot-ds-public-certificate-certificateerror"></a>
### `CertificateError`

定义 owner `hotpot.ds.public.certificate.CertificateError`；源码 `hotpot/ds/public/certificate.py:76`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds.public.certificate.CertificateError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.certificate.RuntimeError`。

<a id="hotpot-ds-public-certificate-certificateformaterror"></a>
### `CertificateFormatError`

定义 owner `hotpot.ds.public.certificate.CertificateFormatError`；源码 `hotpot/ds/public/certificate.py:80`。

基类原始声明：`CertificateError`；装饰器：无。

显式导出位置：`hotpot.ds.public.certificate.CertificateFormatError`。

<a id="hotpot-ds-public-certificate-certificateintegrityerror"></a>
### `CertificateIntegrityError`

定义 owner `hotpot.ds.public.certificate.CertificateIntegrityError`；源码 `hotpot/ds/public/certificate.py:84`。

基类原始声明：`CertificateError`；装饰器：无。

显式导出位置：`hotpot.ds.public.certificate.CertificateIntegrityError`。

<a id="hotpot-ds-public-certificate-certificatepatherror"></a>
### `CertificatePathError`

定义 owner `hotpot.ds.public.certificate.CertificatePathError`；源码 `hotpot/ds/public/certificate.py:88`。

基类原始声明：`CertificateIntegrityError`；装饰器：无。

显式导出位置：`hotpot.ds.public.certificate.CertificatePathError`。

<a id="hotpot-ds-public-certificate-load-certificate-envelope"></a>
### `load_certificate_envelope`

定义 owner `hotpot.ds.public.certificate.load_certificate_envelope`；源码 `hotpot/ds/public/certificate.py:1280`。

```python
load_certificate_envelope(path: Union[str, Path]) -> Dict[str, Any]
```

<a id="hotpot-ds-public-certificate-write-bundle-certificate"></a>
### `write_bundle_certificate`

定义 owner `hotpot.ds.public.certificate.write_bundle_certificate`；源码 `hotpot/ds/public/certificate.py:1541`。

```python
write_bundle_certificate(bundle: Any) -> Path
```

<a id="hotpot-ds-public-certificate-verify-bundle-certificate"></a>
### `verify_bundle_certificate`

定义 owner `hotpot.ds.public.certificate.verify_bundle_certificate`；源码 `hotpot/ds/public/certificate.py:1600`。

```python
verify_bundle_certificate(bundle: Any, path: Optional[Union[str, Path]]=None) -> Dict[str, Any]
```

<a id="hotpot-ds-public-certificate-certificate-digest"></a>
### `certificate_digest`

定义 owner `hotpot.ds.public.certificate.certificate_digest`；源码 `hotpot/ds/public/certificate.py:1618`。

```python
certificate_digest(path: Union[str, Path]) -> str
```

模块值/类型别名（声明存在不构成事实写许可）：

- `CERTIFICATE_SCHEMA` = `'hotpot-ds-certificate/current'`；`hotpot/ds/public/certificate.py:33`
- `ASSESSOR_VERSION` = `'hotpot-ds-assessor/current'`；`hotpot/ds/public/certificate.py:34`


<a id="hotpot-ds-public-dataset"></a>
## `hotpot.ds.public.dataset`

源码 `hotpot/ds/public/dataset.py`；SHA-256 `afeb4aac5ecb0f8dc0cfb8cf0815106ae91b9dce07e9706e0b3c592e71acb4c1`。

显式 __all__：`('Data', 'Dataset')`。

显式导出绑定：

- `Data` → `hotpot.ds.public.dataset.Data`；构造声明 `__init__(self, dataset: Dataset, value: Mapping[str, Any]) -> None`
- `Dataset` → `hotpot.ds.public.dataset.Dataset`；构造声明 `__init__(self, store: Any, dataset_info: Mapping[str, Any], release: Mapping[str, Any], headers: Iterable[Mapping[str, Any]], *, archive_classes: Iterable[str]) -> None`

<a id="hotpot-ds-public-dataset-identifierindex"></a>
### `IdentifierIndex`

定义 owner `hotpot.ds.public.dataset.IdentifierIndex`；源码 `hotpot/ds/public/dataset.py:42`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, dataset: 'Dataset', namespace: str) -> None`；owner `hotpot.ds.public.dataset.IdentifierIndex`；`hotpot/ds/public/dataset.py:45`
- `__getitem__(self, value: str) -> Union[Data, Dataset]`；owner `hotpot.ds.public.dataset.IdentifierIndex`；`hotpot/ds/public/dataset.py:49`

<a id="hotpot-ds-public-dataset-dataset"></a>
### `Dataset`

定义 owner `hotpot.ds.public.dataset.Dataset`；源码 `hotpot/ds/public/dataset.py:69`。

基类原始声明：`Sequence['Data']`；装饰器：无。

显式导出位置：`hotpot.ds.Dataset`, `hotpot.ds.public.dataset.Dataset`。

本类属性与方法：

- `__init__(self, store: Any, dataset_info: Mapping[str, Any], release: Mapping[str, Any], headers: Iterable[Mapping[str, Any]], *, archive_classes: Iterable[str]) -> None`；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:72`
- `__len__(self) -> int`；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:189`
- `__iter__(self) -> Iterator[Data]`；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:192`
- `__getitem__(self, index: Union[int, slice]) -> Union[Data, Dataset]`；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:195`
- `__getattr__(self, name: str) -> IdentifierIndex`；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:200`
- `subset(self, **criteria: Any) -> Dataset`；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:210`
- `__repr__(self) -> str`；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:248`
- `name(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:89`
- `dataset_id(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:93`
- `release_id(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:97`
- `version(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:101`
- `version_serial(self) -> int` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:105`
- `info(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:109`
- `handle(self) -> list[str]` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:131`
- `labels(self) -> Dict[str, int]` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:142`
- `core_labels(self) -> list[str]` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:149`
- `quality_levels(self) -> Dict[str, int]` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:153`
- `evidence_levels(self) -> Dict[str, int]` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:159`
- `identifier_namespaces(self) -> tuple[str, ...]` [读；只读]；owner `hotpot.ds.public.dataset.Dataset`；`hotpot/ds/public/dataset.py:170`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`collections.abc.Sequence`。

<a id="hotpot-ds-public-dataset-data"></a>
### `Data`

定义 owner `hotpot.ds.public.dataset.Data`；源码 `hotpot/ds/public/dataset.py:255`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.Data`, `hotpot.ds.public.dataset.Data`。

本类属性与方法：

- `__init__(self, dataset: Dataset, value: Mapping[str, Any]) -> None`；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:258`
- `__repr__(self) -> str`；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:387`
- `id(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:264`
- `record_id(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:271`
- `local_id(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:275`
- `metadata(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:279`
- `identifiers(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:283`
- `entities(self) -> list[Dict[str, Any]]` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:293`
- `label(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:297`
- `quality_level(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:320`
- `evidence_level(self) -> Optional[str]` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:324`
- `archive_class(self) -> str` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:328`
- `info(self) -> Dict[str, Any]` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:332`
- `structures(self) -> Structures` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:363`
- `molecule(self)` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:367`
- `crystal(self)` [读；只读]；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:374`

本类字段与枚举值：

- `dataset` = `dataset` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.public.dataset.Data`；`hotpot/ds/public/dataset.py:259`


<a id="hotpot-ds-public-search"></a>
## `hotpot.ds.public.search`

源码 `hotpot/ds/public/search.py`；SHA-256 `f736702211e5e9d15c1eec7e04a823cae2ddad12cf02c7e10c28529bbcac1099`。

显式 __all__：`('BaseSearcher', 'DataSearcher', 'DatasetSearcher', 'SmartsSearcher')`。

显式导出绑定：

- `BaseSearcher` → `hotpot.ds.public.search.BaseSearcher`
- `DataSearcher` → `hotpot.ds.public.search.DataSearcher`
- `DatasetSearcher` → `hotpot.ds.public.search.DatasetSearcher`
- `SmartsSearcher` → `hotpot.ds.public.search.SmartsSearcher`；构造声明 `__init__(self, smarts: str, *, semantics=None)`

<a id="hotpot-ds-public-search-basesearcher"></a>
### `BaseSearcher`

定义 owner `hotpot.ds.public.search.BaseSearcher`；源码 `hotpot/ds/public/search.py:123`。

基类原始声明：`ABC`；装饰器：无。

显式导出位置：`hotpot.ds.BaseSearcher`, `hotpot.ds.public.search.BaseSearcher`。

本类属性与方法：

- `match(self, item)`；装饰器 `abstractmethod`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:135`
- `__call__(self, item)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:139`
- `search(self, target)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:142`
- `__and__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:176`
- `__or__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:179`
- `__xor__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:182`
- `__invert__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:185`
- `not_(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:188`
- `__bool__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:192`

本类字段与枚举值：

- `scope` = `None` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:132`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`abc.ABC`。

<a id="hotpot-ds-public-search-datasearcher"></a>
### `DataSearcher`

定义 owner `hotpot.ds.public.search.DataSearcher`；源码 `hotpot/ds/public/search.py:235`。

基类原始声明：`_FieldSearcher`；装饰器：无。

显式导出位置：`hotpot.ds.DataSearcher`, `hotpot.ds.public.search.DataSearcher`。

本类字段与枚举值：

- `scope` = `'data'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.public.search.DataSearcher`；`hotpot/ds/public/search.py:238`

继承成员（逐项保留声明 owner）：

- `__init__(self, **criteria: Any)`；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:212`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `match(self, item)`；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:219`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__repr__(self)`；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:227`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `criteria(self)` [读；只读]；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:216`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__call__(self, item)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:139`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `search(self, target)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:142`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__and__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:176`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__or__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:179`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__xor__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:182`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__invert__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:185`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `not_(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:188`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__bool__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:192`；继承经 `hotpot.ds.public.search._FieldSearcher`

<a id="hotpot-ds-public-search-datasetsearcher"></a>
### `DatasetSearcher`

定义 owner `hotpot.ds.public.search.DatasetSearcher`；源码 `hotpot/ds/public/search.py:241`。

基类原始声明：`_FieldSearcher`；装饰器：无。

显式导出位置：`hotpot.ds.DatasetSearcher`, `hotpot.ds.public.search.DatasetSearcher`。

本类字段与枚举值：

- `scope` = `'dataset'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.public.search.DatasetSearcher`；`hotpot/ds/public/search.py:244`

继承成员（逐项保留声明 owner）：

- `__init__(self, **criteria: Any)`；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:212`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `match(self, item)`；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:219`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__repr__(self)`；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:227`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `criteria(self)` [读；只读]；owner `hotpot.ds.public.search._FieldSearcher`；`hotpot/ds/public/search.py:216`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__call__(self, item)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:139`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `search(self, target)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:142`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__and__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:176`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__or__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:179`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__xor__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:182`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__invert__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:185`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `not_(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:188`；继承经 `hotpot.ds.public.search._FieldSearcher`
- `__bool__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:192`；继承经 `hotpot.ds.public.search._FieldSearcher`

<a id="hotpot-ds-public-search-smartssearcher"></a>
### `SmartsSearcher`

定义 owner `hotpot.ds.public.search.SmartsSearcher`；源码 `hotpot/ds/public/search.py:248`。

基类原始声明：`BaseSearcher`；装饰器：无。

显式导出位置：`hotpot.ds.SmartsSearcher`, `hotpot.ds.public.search.SmartsSearcher`。

本类属性与方法：

- `__init__(self, smarts: str, *, semantics=None)`；owner `hotpot.ds.public.search.SmartsSearcher`；`hotpot/ds/public/search.py:253`
- `match(self, data)`；owner `hotpot.ds.public.search.SmartsSearcher`；`hotpot/ds/public/search.py:263`
- `__repr__(self)`；owner `hotpot.ds.public.search.SmartsSearcher`；`hotpot/ds/public/search.py:277`

本类字段与枚举值：

- `scope` = `'data'` [类声明值；Python 重绑定能力不构成事实写许可]；owner `hotpot.ds.public.search.SmartsSearcher`；`hotpot/ds/public/search.py:251`
- `smarts` = `smarts` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.public.search.SmartsSearcher`；`hotpot/ds/public/search.py:256`

继承成员（逐项保留声明 owner）：

- `__call__(self, item)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:139`；继承经 `hotpot.ds.public.search.BaseSearcher`
- `search(self, target)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:142`；继承经 `hotpot.ds.public.search.BaseSearcher`
- `__and__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:176`；继承经 `hotpot.ds.public.search.BaseSearcher`
- `__or__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:179`；继承经 `hotpot.ds.public.search.BaseSearcher`
- `__xor__(self, other)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:182`；继承经 `hotpot.ds.public.search.BaseSearcher`
- `__invert__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:185`；继承经 `hotpot.ds.public.search.BaseSearcher`
- `not_(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:188`；继承经 `hotpot.ds.public.search.BaseSearcher`
- `__bool__(self)`；owner `hotpot.ds.public.search.BaseSearcher`；`hotpot/ds/public/search.py:192`；继承经 `hotpot.ds.public.search.BaseSearcher`


<a id="hotpot-ds-public-structures"></a>
## `hotpot.ds.public.structures`

源码 `hotpot/ds/public/structures.py`；SHA-256 `5ce3e28fc140d89853497f1febcb235be6ac052a3875a8d668b099d40325d1aa`。

显式 __all__：`('AmbiguousStructureError', 'AssetIntegrityError', 'CrystalStructureUnavailableError', 'StructureAccessError', 'StructureUnavailableError', 'Structures')`。

显式导出绑定：

- `AmbiguousStructureError` → `hotpot.ds.public.structures.AmbiguousStructureError`
- `AssetIntegrityError` → `hotpot.ds.public.structures.AssetIntegrityError`
- `CrystalStructureUnavailableError` → `hotpot.ds.public.structures.CrystalStructureUnavailableError`
- `StructureAccessError` → `hotpot.ds.public.structures.StructureAccessError`
- `StructureUnavailableError` → `hotpot.ds.public.structures.StructureUnavailableError`
- `Structures` → `hotpot.ds.public.structures.Structures`；构造声明 `__init__(self, data: 'Data') -> None`

<a id="hotpot-ds-public-structures-structureaccesserror"></a>
### `StructureAccessError`

定义 owner `hotpot.ds.public.structures.StructureAccessError`；源码 `hotpot/ds/public/structures.py:13`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds.public.structures.StructureAccessError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.public.structures.RuntimeError`。

<a id="hotpot-ds-public-structures-ambiguousstructureerror"></a>
### `AmbiguousStructureError`

定义 owner `hotpot.ds.public.structures.AmbiguousStructureError`；源码 `hotpot/ds/public/structures.py:17`。

基类原始声明：`StructureAccessError`；装饰器：无。

显式导出位置：`hotpot.ds.public.structures.AmbiguousStructureError`。

<a id="hotpot-ds-public-structures-assetintegrityerror"></a>
### `AssetIntegrityError`

定义 owner `hotpot.ds.public.structures.AssetIntegrityError`；源码 `hotpot/ds/public/structures.py:21`。

基类原始声明：`StructureAccessError`；装饰器：无。

显式导出位置：`hotpot.ds.public.structures.AssetIntegrityError`。

<a id="hotpot-ds-public-structures-structureunavailableerror"></a>
### `StructureUnavailableError`

定义 owner `hotpot.ds.public.structures.StructureUnavailableError`；源码 `hotpot/ds/public/structures.py:25`。

基类原始声明：`StructureAccessError`；装饰器：无。

显式导出位置：`hotpot.ds.public.structures.StructureUnavailableError`。

<a id="hotpot-ds-public-structures-crystalstructureunavailableerror"></a>
### `CrystalStructureUnavailableError`

定义 owner `hotpot.ds.public.structures.CrystalStructureUnavailableError`；源码 `hotpot/ds/public/structures.py:29`。

基类原始声明：`StructureUnavailableError`；装饰器：无。

显式导出位置：`hotpot.ds.public.structures.CrystalStructureUnavailableError`。

<a id="hotpot-ds-public-structures-structures"></a>
### `Structures`

定义 owner `hotpot.ds.public.structures.Structures`；源码 `hotpot/ds/public/structures.py:37`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.public.structures.Structures`。

本类属性与方法：

- `__init__(self, data: 'Data') -> None`；owner `hotpot.ds.public.structures.Structures`；`hotpot/ds/public/structures.py:40`
- `__call__(self) -> list[str]`；owner `hotpot.ds.public.structures.Structures`；`hotpot/ds/public/structures.py:46`
- `content(self, fmt: str) -> Optional[str]`；owner `hotpot.ds.public.structures.Structures`；`hotpot/ds/public/structures.py:115`
- `path(self, fmt: str) -> Optional[Path]`；owner `hotpot.ds.public.structures.Structures`；`hotpot/ds/public/structures.py:121`
- `molecule(self)`；owner `hotpot.ds.public.structures.Structures`；`hotpot/ds/public/structures.py:125`
- `__getattr__(self, name: str) -> Optional[str]`；owner `hotpot.ds.public.structures.Structures`；`hotpot/ds/public/structures.py:142`
- `__repr__(self) -> str`；owner `hotpot.ds.public.structures.Structures`；`hotpot/ds/public/structures.py:147`


<a id="hotpot-ds-storage"></a>
## `hotpot.ds.storage`

源码 `hotpot/ds/storage.py`；SHA-256 `97a5d2114b8a35af39ec33d63a1b9f0146c892776ce8759f547118ec6eef6266`。

显式 __all__：`('TARGET_SCHEMA_ID', 'AssetIntegrityError', 'PolicyMismatchError', 'ReceiptConflictError', 'SnapshotError', 'TargetIngestResult', 'TargetSchemaError', 'TargetStore', 'TargetStoreError', 'asset_inventory_digest', 'case_content_digest', 'case_content_digests', 'case_set_digest', 'included_case_content_digest', 'persisted_formal_content_digest', 'persisted_included_case_content_digest')`。

显式导出绑定：

- `TARGET_SCHEMA_ID` → `hotpot.ds.storage.TARGET_SCHEMA_ID`
- `AssetIntegrityError` → `hotpot.ds.storage.AssetIntegrityError`
- `PolicyMismatchError` → `hotpot.ds.storage.PolicyMismatchError`
- `ReceiptConflictError` → `hotpot.ds.storage.ReceiptConflictError`
- `SnapshotError` → `hotpot.ds.storage.SnapshotError`
- `TargetIngestResult` → `hotpot.ds.storage.TargetIngestResult`；构造声明 `TargetIngestResult(receipt_id: str, store_uuid: str, dataset: str, release: str, release_serial: int, included_count: int, excluded_count: int, committed_at: str)`
- `TargetSchemaError` → `hotpot.ds.storage.TargetSchemaError`
- `TargetStore` → `hotpot.ds.storage.TargetStore`；构造声明 `__init__(self, path: Union[str, Path])`
- `TargetStoreError` → `hotpot.ds.storage.TargetStoreError`
- `asset_inventory_digest` → `hotpot.ds.storage.asset_inventory_digest`；`asset_inventory_digest(assets: Iterable[Mapping[str, Any]]) -> str`
- `case_content_digest` → `hotpot.ds.storage.case_content_digest`；`case_content_digest(cases: Iterable[Mapping[str, Any]]) -> str`
- `case_content_digests` → `hotpot.ds.storage.case_content_digests`；`case_content_digests(cases: Iterable[Mapping[str, Any]], *, include_incomplete: bool=False) -> Tuple[str, str]`
- `case_set_digest` → `hotpot.ds.storage.case_set_digest`；`case_set_digest(case_ids: Iterable[str]) -> str`
- `included_case_content_digest` → `hotpot.ds.storage.included_case_content_digest`；`included_case_content_digest(cases: Iterable[Mapping[str, Any]], *, include_incomplete: bool=False) -> str`
- `persisted_formal_content_digest` → `hotpot.ds.storage.persisted_formal_content_digest`；`persisted_formal_content_digest(connection: sqlite3.Connection, dataset: str, release: str) -> str`
- `persisted_included_case_content_digest` → `hotpot.ds.storage.persisted_included_case_content_digest`；`persisted_included_case_content_digest(connection: sqlite3.Connection, dataset: str, release: str) -> str`

<a id="hotpot-ds-storage-targetstoreerror"></a>
### `TargetStoreError`

定义 owner `hotpot.ds.storage.TargetStoreError`；源码 `hotpot/ds/storage.py:259`。

基类原始声明：`RuntimeError`；装饰器：无。

显式导出位置：`hotpot.ds.storage.TargetStoreError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.storage.RuntimeError`。

<a id="hotpot-ds-storage-targetschemaerror"></a>
### `TargetSchemaError`

定义 owner `hotpot.ds.storage.TargetSchemaError`；源码 `hotpot/ds/storage.py:263`。

基类原始声明：`TargetStoreError`；装饰器：无。

显式导出位置：`hotpot.ds.storage.TargetSchemaError`。

<a id="hotpot-ds-storage-snapshoterror"></a>
### `SnapshotError`

定义 owner `hotpot.ds.storage.SnapshotError`；源码 `hotpot/ds/storage.py:267`。

基类原始声明：`TargetStoreError`, `ValueError`；装饰器：无。

显式导出位置：`hotpot.ds.storage.SnapshotError`。

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.ds.storage.ValueError`。

<a id="hotpot-ds-storage-policymismatcherror"></a>
### `PolicyMismatchError`

定义 owner `hotpot.ds.storage.PolicyMismatchError`；源码 `hotpot/ds/storage.py:271`。

基类原始声明：`TargetStoreError`；装饰器：无。

显式导出位置：`hotpot.ds.storage.PolicyMismatchError`。

<a id="hotpot-ds-storage-receiptconflicterror"></a>
### `ReceiptConflictError`

定义 owner `hotpot.ds.storage.ReceiptConflictError`；源码 `hotpot/ds/storage.py:275`。

基类原始声明：`TargetStoreError`；装饰器：无。

显式导出位置：`hotpot.ds.storage.ReceiptConflictError`。

<a id="hotpot-ds-storage-assetintegrityerror"></a>
### `AssetIntegrityError`

定义 owner `hotpot.ds.storage.AssetIntegrityError`；源码 `hotpot/ds/storage.py:279`。

基类原始声明：`TargetStoreError`；装饰器：无。

显式导出位置：`hotpot.ds.storage.AssetIntegrityError`。

<a id="hotpot-ds-storage-targetingestresult"></a>
### `TargetIngestResult`

定义 owner `hotpot.ds.storage.TargetIngestResult`；源码 `hotpot/ds/storage.py:284`。

基类原始声明：无显式基类；装饰器：`dataclass(frozen=True)`。

显式导出位置：`hotpot.ds.storage.TargetIngestResult`。

dataclass 字段构造声明（field(...) 保留源码生成规则，非 inspect.signature）：

```python
TargetIngestResult(receipt_id: str, store_uuid: str, dataset: str, release: str, release_serial: int, included_count: int, excluded_count: int, committed_at: str)
```

本类属性与方法：

- `to_dict(self) -> Dict[str, Any]`；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:296`

本类字段与枚举值：

- `receipt_id: str` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:287`
- `store_uuid: str` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:288`
- `dataset: str` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:289`
- `release: str` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:290`
- `release_serial: int` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:291`
- `included_count: int` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:292`
- `excluded_count: int` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:293`
- `committed_at: str` [只读值字段]；owner `hotpot.ds.storage.TargetIngestResult`；`hotpot/ds/storage.py:294`

Dataclass generated __init__/__repr__/__eq__ and frozen behavior follow listed decorators and declared fields; explicit local methods take precedence. Signatures shown preserve source annotations, not evaluated runtime repr.

<a id="hotpot-ds-storage-case-set-digest"></a>
### `case_set_digest`

定义 owner `hotpot.ds.storage.case_set_digest`；源码 `hotpot/ds/storage.py:430`。

```python
case_set_digest(case_ids: Iterable[str]) -> str
```

<a id="hotpot-ds-storage-asset-inventory-digest"></a>
### `asset_inventory_digest`

定义 owner `hotpot.ds.storage.asset_inventory_digest`；源码 `hotpot/ds/storage.py:435`。

```python
asset_inventory_digest(assets: Iterable[Mapping[str, Any]]) -> str
```

<a id="hotpot-ds-storage-case-content-digest"></a>
### `case_content_digest`

定义 owner `hotpot.ds.storage.case_content_digest`；源码 `hotpot/ds/storage.py:445`。

```python
case_content_digest(cases: Iterable[Mapping[str, Any]]) -> str
```

<a id="hotpot-ds-storage-case-content-digests"></a>
### `case_content_digests`

定义 owner `hotpot.ds.storage.case_content_digests`；源码 `hotpot/ds/storage.py:454`。

```python
case_content_digests(cases: Iterable[Mapping[str, Any]], *, include_incomplete: bool=False) -> Tuple[str, str]
```

<a id="hotpot-ds-storage-included-case-content-digest"></a>
### `included_case_content_digest`

定义 owner `hotpot.ds.storage.included_case_content_digest`；源码 `hotpot/ds/storage.py:488`。

```python
included_case_content_digest(cases: Iterable[Mapping[str, Any]], *, include_incomplete: bool=False) -> str
```

<a id="hotpot-ds-storage-persisted-included-case-content-digest"></a>
### `persisted_included_case_content_digest`

定义 owner `hotpot.ds.storage.persisted_included_case_content_digest`；源码 `hotpot/ds/storage.py:500`。

```python
persisted_included_case_content_digest(connection: sqlite3.Connection, dataset: str, release: str) -> str
```

<a id="hotpot-ds-storage-persisted-formal-content-digest"></a>
### `persisted_formal_content_digest`

定义 owner `hotpot.ds.storage.persisted_formal_content_digest`；源码 `hotpot/ds/storage.py:654`。

```python
persisted_formal_content_digest(connection: sqlite3.Connection, dataset: str, release: str) -> str
```

<a id="hotpot-ds-storage-targetstore"></a>
### `TargetStore`

定义 owner `hotpot.ds.storage.TargetStore`；源码 `hotpot/ds/storage.py:778`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.ds.storage.TargetStore`。

本类属性与方法：

- `__init__(self, path: Union[str, Path])`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:781`
- `initialize(self) -> Dict[str, str]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:838`
- `metadata(self) -> Dict[str, str]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:871`
- `read_policy(self, dataset: str) -> Optional[Dict[str, Any]]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:875`
- `inspect_dataset(self, dataset: str) -> Optional[Dict[str, Any]]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:909`
- `load_release_metadata(self, dataset: str, release: str) -> Optional[Dict[str, Any]]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:966`
- `iter_cases(self, dataset: str, release: str, *, include_incomplete: bool=False, archive_class: Optional[str]=None) -> Iterator[Dict[str, Any]]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:1033`
- `iter_case_headers(self, dataset: str, release: str, *, archive_classes: Optional[Iterable[str]]=None) -> Iterator[Dict[str, Any]]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:1118`
- `identifier_namespaces(self, dataset: str, release: str, *, archive_classes: Iterable[str], visible_case_ids: Optional[Iterable[str]]=None) -> Tuple[str, ...]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:1161`
- `find_case_ids_by_identifier(self, dataset: str, release: str, namespace: str, value: str, *, archive_classes: Iterable[str]) -> Tuple[str, ...]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:1215`
- `get_case(self, dataset: str, release: str, case_id: str) -> Dict[str, Any]`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:1247`
- `ingest(self, snapshot: Mapping[str, Any], *, include_incomplete: bool=False, reason: Optional[str]=None, _precommit_check: Optional[Callable[[], None]]=None) -> TargetIngestResult`；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:1355`

本类字段与枚举值：

- `path` = `Path(path).expanduser().resolve()` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:782`
- `asset_root` = `Path(str(self.path) + '.assets')` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.ds.storage.TargetStore`；`hotpot/ds/storage.py:783`

模块值/类型别名（声明存在不构成事实写许可）：

- `TARGET_SCHEMA_ID` = `'hotpot-ds-store/compact-current'`；`hotpot/ds/storage.py:99`
- `SCHEMA` = <长声明表达式；完整值见库存及源码位置>；`hotpot/ds/storage.py:105`


<a id="hotpot-ds-tools"></a>
## `hotpot.ds.tools`

源码 `hotpot/ds/tools/__init__.py`；SHA-256 `0d4466fc9fc622987d058bcd0b104f428a5ecdcb5721b8f027e3bcbbe6eaa0ea`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-tools-analyze-structure-composition"></a>
## `hotpot.ds.tools.analyze_structure_composition`

源码 `hotpot/ds/tools/analyze_structure_composition.py`；SHA-256 `17ad056815806991760ce2dd80b18c200d516b3ca5ba105eea2e723991d98835`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `read_csv` → `hotpot.ds.tools.analyze_structure_composition.read_csv`；`read_csv(path: Path) -> list[dict[str, str]]`
- `write_csv` → `hotpot.ds.tools.analyze_structure_composition.write_csv`；`write_csv(path: Path, rows: list[dict[str, object]]) -> None`
- `target_metals` → `hotpot.ds.tools.analyze_structure_composition.target_metals`；`target_metals() -> dict[tuple[str, str, str], str]`
- `connected_components` → `hotpot.ds.tools.analyze_structure_composition.connected_components`；`connected_components(adjacency: list[set[int]], removed: int) -> list[set[int]]`
- `analyze` → `hotpot.ds.tools.analyze_structure_composition.analyze`；`analyze(path: Path, fmt: str, target_symbol: str) -> dict[str, object]`
- `distributions` → `hotpot.ds.tools.analyze_structure_composition.distributions`；`distributions(rows: list[dict[str, object]]) -> list[dict[str, object]]`
- `main` → `hotpot.ds.tools.analyze_structure_composition.main`；`main() -> None`

<a id="hotpot-ds-tools-analyze-structure-composition-read-csv"></a>
### `read_csv`

定义 owner `hotpot.ds.tools.analyze_structure_composition.read_csv`；源码 `hotpot/ds/tools/analyze_structure_composition.py:17`。

```python
read_csv(path: Path) -> list[dict[str, str]]
```

<a id="hotpot-ds-tools-analyze-structure-composition-write-csv"></a>
### `write_csv`

定义 owner `hotpot.ds.tools.analyze_structure_composition.write_csv`；源码 `hotpot/ds/tools/analyze_structure_composition.py:22`。

```python
write_csv(path: Path, rows: list[dict[str, object]]) -> None
```

<a id="hotpot-ds-tools-analyze-structure-composition-target-metals"></a>
### `target_metals`

定义 owner `hotpot.ds.tools.analyze_structure_composition.target_metals`；源码 `hotpot/ds/tools/analyze_structure_composition.py:30`。

```python
target_metals() -> dict[tuple[str, str, str], str]
```

<a id="hotpot-ds-tools-analyze-structure-composition-connected-components"></a>
### `connected_components`

定义 owner `hotpot.ds.tools.analyze_structure_composition.connected_components`；源码 `hotpot/ds/tools/analyze_structure_composition.py:46`。

```python
connected_components(adjacency: list[set[int]], removed: int) -> list[set[int]]
```

<a id="hotpot-ds-tools-analyze-structure-composition-analyze"></a>
### `analyze`

定义 owner `hotpot.ds.tools.analyze_structure_composition.analyze`；源码 `hotpot/ds/tools/analyze_structure_composition.py:62`。

```python
analyze(path: Path, fmt: str, target_symbol: str) -> dict[str, object]
```

<a id="hotpot-ds-tools-analyze-structure-composition-distributions"></a>
### `distributions`

定义 owner `hotpot.ds.tools.analyze_structure_composition.distributions`；源码 `hotpot/ds/tools/analyze_structure_composition.py:114`。

```python
distributions(rows: list[dict[str, object]]) -> list[dict[str, object]]
```

<a id="hotpot-ds-tools-analyze-structure-composition-main"></a>
### `main`

定义 owner `hotpot.ds.tools.analyze_structure_composition.main`；源码 `hotpot/ds/tools/analyze_structure_composition.py:131`。

```python
main() -> None
```


<a id="hotpot-ds-tools-electronic-state"></a>
## `hotpot.ds.tools.electronic_state`

源码 `hotpot/ds/tools/electronic_state/__init__.py`；SHA-256 `c04497646a6da2f2e495f34fcfcd4aa3895c75bf603d28b61020f647a638d809`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-tools-electronic-state-assess-quality"></a>
## `hotpot.ds.tools.electronic_state.assess_quality`

源码 `hotpot/ds/tools/electronic_state/assess_quality.py`；SHA-256 `fc5a2f5a22b9998f7b3d94b701da25aa467bf367f9c02773fab02a0354fa266b`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `assess` → `hotpot.ds.tools.electronic_state.assess_quality.assess`；`assess(database: Path) -> Dict[str, object]`
- `main` → `hotpot.ds.tools.electronic_state.assess_quality.main`；`main() -> int`
- `ELECTRONIC_STATE_ROOT` → `hotpot.ds.tools.electronic_state.assess_quality.ELECTRONIC_STATE_ROOT`
- `DEFAULT_DB` → `hotpot.ds.tools.electronic_state.assess_quality.DEFAULT_DB`
- `DEFAULT_OUTPUT` → `hotpot.ds.tools.electronic_state.assess_quality.DEFAULT_OUTPUT`
- `NUMERIC_SPIN_PATTERN` → `hotpot.ds.tools.electronic_state.assess_quality.NUMERIC_SPIN_PATTERN`

<a id="hotpot-ds-tools-electronic-state-assess-quality-assess"></a>
### `assess`

定义 owner `hotpot.ds.tools.electronic_state.assess_quality.assess`；源码 `hotpot/ds/tools/electronic_state/assess_quality.py:176`。

```python
assess(database: Path) -> Dict[str, object]
```

<a id="hotpot-ds-tools-electronic-state-assess-quality-main"></a>
### `main`

定义 owner `hotpot.ds.tools.electronic_state.assess_quality.main`；源码 `hotpot/ds/tools/electronic_state/assess_quality.py:352`。

```python
main() -> int
```

模块值/类型别名（声明存在不构成事实写许可）：

- `ELECTRONIC_STATE_ROOT` = `SOURCES_ROOT / 'csd-electronic-state'`；`hotpot/ds/tools/electronic_state/assess_quality.py:26`
- `DEFAULT_DB` = `ELECTRONIC_STATE_ROOT / 'electronic_states.sqlite3'`；`hotpot/ds/tools/electronic_state/assess_quality.py:27`
- `DEFAULT_OUTPUT` = `AUDITS_ROOT / 'csd-electronic-state-quality.json'`；`hotpot/ds/tools/electronic_state/assess_quality.py:28`
- `NUMERIC_SPIN_PATTERN` = `re.compile('(?<![A-Za-z])S\\s*=\\s*[-+]?\\d+(?:\\.\\d+)?(?:\\s*/\\s*\\d+)?|\\b(?:multiplicity|singlet|doublet|triplet|quartet|quintet|sextet|septet|octet)\\b', re.I)`；`hotpot/ds/tools/electronic_state/assess_quality.py:29`


<a id="hotpot-ds-tools-electronic-state-build-database"></a>
## `hotpot.ds.tools.electronic_state.build_database`

源码 `hotpot/ds/tools/electronic_state/build_database.py`；SHA-256 `9c699520a3ac2ba2fd9d0ea368fa260654cc85fc798f93991d6f6c46c0d3ed08`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `sha256_file` → `hotpot.ds.tools.electronic_state.build_database.sha256_file`；`sha256_file(path: Path) -> str`
- `scalar` → `hotpot.ds.tools.electronic_state.build_database.scalar`；`scalar(connection: sqlite3.Connection, sql: str, parameters=())`
- `parse_args` → `hotpot.ds.tools.electronic_state.build_database.parse_args`；`parse_args() -> argparse.Namespace`
- `main` → `hotpot.ds.tools.electronic_state.build_database.main`；`main() -> int`
- `OX_T1` → `hotpot.ds.tools.electronic_state.build_database.OX_T1`
- `SP_T1` → `hotpot.ds.tools.electronic_state.build_database.SP_T1`
- `ELECTRONIC_STATE_ROOT` → `hotpot.ds.tools.electronic_state.build_database.ELECTRONIC_STATE_ROOT`
- `DEFAULT_SCHEMA` → `hotpot.ds.tools.electronic_state.build_database.DEFAULT_SCHEMA`
- `DEFAULT_OUTPUT` → `hotpot.ds.tools.electronic_state.build_database.DEFAULT_OUTPUT`
- `DEFAULT_AUDIT_OUTPUT` → `hotpot.ds.tools.electronic_state.build_database.DEFAULT_AUDIT_OUTPUT`
- `DEFAULT_PROVENANCE_OUTPUT` → `hotpot.ds.tools.electronic_state.build_database.DEFAULT_PROVENANCE_OUTPUT`
- `SPIN_CONTEXT_PATTERNS` → `hotpot.ds.tools.electronic_state.build_database.SPIN_CONTEXT_PATTERNS`
- `NUMERIC_SPIN_PATTERN` → `hotpot.ds.tools.electronic_state.build_database.NUMERIC_SPIN_PATTERN`

<a id="hotpot-ds-tools-electronic-state-build-database-sha256-file"></a>
### `sha256_file`

定义 owner `hotpot.ds.tools.electronic_state.build_database.sha256_file`；源码 `hotpot/ds/tools/electronic_state/build_database.py:47`。

```python
sha256_file(path: Path) -> str
```

<a id="hotpot-ds-tools-electronic-state-build-database-scalar"></a>
### `scalar`

定义 owner `hotpot.ds.tools.electronic_state.build_database.scalar`；源码 `hotpot/ds/tools/electronic_state/build_database.py:55`。

```python
scalar(connection: sqlite3.Connection, sql: str, parameters=())
```

<a id="hotpot-ds-tools-electronic-state-build-database-parse-args"></a>
### `parse_args`

定义 owner `hotpot.ds.tools.electronic_state.build_database.parse_args`；源码 `hotpot/ds/tools/electronic_state/build_database.py:59`。

```python
parse_args() -> argparse.Namespace
```

<a id="hotpot-ds-tools-electronic-state-build-database-main"></a>
### `main`

定义 owner `hotpot.ds.tools.electronic_state.build_database.main`；源码 `hotpot/ds/tools/electronic_state/build_database.py:73`。

```python
main() -> int
```

模块值/类型别名（声明存在不构成事实写许可）：

- `OX_T1` = `'OX-T1_EXPLICIT_SINGLE_SITE_CANDIDATE'`；`hotpot/ds/tools/electronic_state/build_database.py:19`
- `SP_T1` = `'SP-T1_FIXED_SINGLE_METAL_CANDIDATE'`；`hotpot/ds/tools/electronic_state/build_database.py:20`
- `ELECTRONIC_STATE_ROOT` = `SOURCES_ROOT / 'csd-electronic-state'`；`hotpot/ds/tools/electronic_state/build_database.py:21`
- `DEFAULT_SCHEMA` = `DATASET_ROOT / 'schemas' / 'csd-electronic-state.sql'`；`hotpot/ds/tools/electronic_state/build_database.py:22`
- `DEFAULT_OUTPUT` = `ELECTRONIC_STATE_ROOT / 'electronic_states.sqlite3'`；`hotpot/ds/tools/electronic_state/build_database.py:23`
- `DEFAULT_AUDIT_OUTPUT` = `AUDITS_ROOT / 'csd-electronic-state-build-audit.json'`；`hotpot/ds/tools/electronic_state/build_database.py:24`
- `DEFAULT_PROVENANCE_OUTPUT` = `AUDITS_ROOT / 'csd-electronic-state-build-provenance.json'`；`hotpot/ds/tools/electronic_state/build_database.py:25`
- `SPIN_CONTEXT_PATTERNS` = <长声明表达式；完整值见库存及源码位置>；`hotpot/ds/tools/electronic_state/build_database.py:27`
- `NUMERIC_SPIN_PATTERN` = `re.compile('(?<![A-Za-z])S\\s*=\\s*[-+]?\\d+(?:\\.\\d+)?(?:\\s*/\\s*\\d+)?|\\b(?:multiplicity|singlet|doublet|triplet|quartet|quintet|sextet|septet|octet)\\b', re.I)`；`hotpot/ds/tools/electronic_state/build_database.py:40`


<a id="hotpot-ds-tools-oxidation"></a>
## `hotpot.ds.tools.oxidation`

源码 `hotpot/ds/tools/oxidation/__init__.py`；SHA-256 `14782b48492aac83aae9118cc52306aefb3261c637d32f1be07830273924bb4e`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-tools-oxidation-validate"></a>
## `hotpot.ds.tools.oxidation.validate`

源码 `hotpot/ds/tools/oxidation/validate.py`；SHA-256 `ea3bb9b98cccc262d695a04685dfd4fcb20fa9ceb386cd8fded98325dfef7be1`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `table` → `hotpot.ds.tools.oxidation.validate.table`；`table(path: Path) -> list[dict[str, str]]`
- `digest` → `hotpot.ds.tools.oxidation.validate.digest`；`digest(path: Path) -> str`
- `inputs` → `hotpot.ds.tools.oxidation.validate.inputs`；`inputs()`
- `validate` → `hotpot.ds.tools.oxidation.validate.validate`；`validate(item) -> dict[str, object]`
- `assert_independence` → `hotpot.ds.tools.oxidation.validate.assert_independence`；`assert_independence() -> None`
- `main` → `hotpot.ds.tools.oxidation.validate.main`；`main() -> None`
- `SOURCE_ROOT` → `hotpot.ds.tools.oxidation.validate.SOURCE_ROOT`

<a id="hotpot-ds-tools-oxidation-validate-table"></a>
### `table`

定义 owner `hotpot.ds.tools.oxidation.validate.table`；源码 `hotpot/ds/tools/oxidation/validate.py:20`。

```python
table(path: Path) -> list[dict[str, str]]
```

<a id="hotpot-ds-tools-oxidation-validate-digest"></a>
### `digest`

定义 owner `hotpot.ds.tools.oxidation.validate.digest`；源码 `hotpot/ds/tools/oxidation/validate.py:25`。

```python
digest(path: Path) -> str
```

<a id="hotpot-ds-tools-oxidation-validate-inputs"></a>
### `inputs`

定义 owner `hotpot.ds.tools.oxidation.validate.inputs`；源码 `hotpot/ds/tools/oxidation/validate.py:29`。

```python
inputs()
```

<a id="hotpot-ds-tools-oxidation-validate-validate"></a>
### `validate`

定义 owner `hotpot.ds.tools.oxidation.validate.validate`；源码 `hotpot/ds/tools/oxidation/validate.py:63`。

```python
validate(item) -> dict[str, object]
```

<a id="hotpot-ds-tools-oxidation-validate-assert-independence"></a>
### `assert_independence`

定义 owner `hotpot.ds.tools.oxidation.validate.assert_independence`；源码 `hotpot/ds/tools/oxidation/validate.py:91`。

```python
assert_independence() -> None
```

<a id="hotpot-ds-tools-oxidation-validate-main"></a>
### `main`

定义 owner `hotpot.ds.tools.oxidation.validate.main`；源码 `hotpot/ds/tools/oxidation/validate.py:105`。

```python
main() -> None
```

模块值/类型别名（声明存在不构成事实写许可）：

- `SOURCE_ROOT` = `SOURCES_ROOT / 'oxidation'`；`hotpot/ds/tools/oxidation/validate.py:17`


<a id="hotpot-ds-tools-spin"></a>
## `hotpot.ds.tools.spin`

源码 `hotpot/ds/tools/spin/__init__.py`；SHA-256 `24e24ff33f039681b30d5cbe30ae883b7aac9063a757f40fb7cf008fbf8439be`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-ds-tools-spin-validate"></a>
## `hotpot.ds.tools.spin.validate`

源码 `hotpot/ds/tools/spin/validate.py`；SHA-256 `31c887037378f397cbae2c165a417ca492799ea79969be14ba9b08378dcb158e`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `table` → `hotpot.ds.tools.spin.validate.table`；`table(path: Path) -> list[dict[str, str]]`
- `digest` → `hotpot.ds.tools.spin.validate.digest`；`digest(path: Path) -> str`
- `inputs` → `hotpot.ds.tools.spin.validate.inputs`；`inputs()`
- `validate` → `hotpot.ds.tools.spin.validate.validate`；`validate(item) -> dict[str, object]`
- `assert_independence` → `hotpot.ds.tools.spin.validate.assert_independence`；`assert_independence() -> None`
- `sco_pair_counts` → `hotpot.ds.tools.spin.validate.sco_pair_counts`；`sco_pair_counts() -> tuple[int, int]`
- `main` → `hotpot.ds.tools.spin.validate.main`；`main() -> None`
- `SOURCE_ROOT` → `hotpot.ds.tools.spin.validate.SOURCE_ROOT`

<a id="hotpot-ds-tools-spin-validate-table"></a>
### `table`

定义 owner `hotpot.ds.tools.spin.validate.table`；源码 `hotpot/ds/tools/spin/validate.py:20`。

```python
table(path: Path) -> list[dict[str, str]]
```

<a id="hotpot-ds-tools-spin-validate-digest"></a>
### `digest`

定义 owner `hotpot.ds.tools.spin.validate.digest`；源码 `hotpot/ds/tools/spin/validate.py:25`。

```python
digest(path: Path) -> str
```

<a id="hotpot-ds-tools-spin-validate-inputs"></a>
### `inputs`

定义 owner `hotpot.ds.tools.spin.validate.inputs`；源码 `hotpot/ds/tools/spin/validate.py:29`。

```python
inputs()
```

<a id="hotpot-ds-tools-spin-validate-validate"></a>
### `validate`

定义 owner `hotpot.ds.tools.spin.validate.validate`；源码 `hotpot/ds/tools/spin/validate.py:62`。

```python
validate(item) -> dict[str, object]
```

<a id="hotpot-ds-tools-spin-validate-assert-independence"></a>
### `assert_independence`

定义 owner `hotpot.ds.tools.spin.validate.assert_independence`；源码 `hotpot/ds/tools/spin/validate.py:90`。

```python
assert_independence() -> None
```

<a id="hotpot-ds-tools-spin-validate-sco-pair-counts"></a>
### `sco_pair_counts`

定义 owner `hotpot.ds.tools.spin.validate.sco_pair_counts`；源码 `hotpot/ds/tools/spin/validate.py:102`。

```python
sco_pair_counts() -> tuple[int, int]
```

<a id="hotpot-ds-tools-spin-validate-main"></a>
### `main`

定义 owner `hotpot.ds.tools.spin.validate.main`；源码 `hotpot/ds/tools/spin/validate.py:111`。

```python
main() -> None
```

模块值/类型别名（声明存在不构成事实写许可）：

- `SOURCE_ROOT` = `SOURCES_ROOT / 'spin'`；`hotpot/ds/tools/spin/validate.py:17`


<a id="hotpot-utils"></a>
## `hotpot.utils`

源码 `hotpot/utils/__init__.py`；SHA-256 `715b819f96fe1af99bae84686ed91f0335208ae8ac357e916c54c7eb6bea66e1`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-utils-chem"></a>
## `hotpot.utils.chem`

源码 `hotpot/utils/chem/__init__.py`；SHA-256 `e66478a15294268845047cb0157b7401b1e12fdc95bfb31ba6c7ea0496e8403d`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-utils-chem-atom"></a>
## `hotpot.utils.chem.atom`

源码 `hotpot/utils/chem/atom.py`；SHA-256 `2747de059e5a6d6b372d8f9307e7e76a9348aaf0a853161eca6cdd012e6f7302`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `calc_electron_config` → `hotpot.utils.chem.atom.calc_electron_config`；`calc_electron_config(_atomic_number)`

<a id="hotpot-utils-chem-atom-calc-electron-config"></a>
### `calc_electron_config`

定义 owner `hotpot.utils.chem.atom.calc_electron_config`；源码 `hotpot/utils/chem/atom.py:4`。

```python
calc_electron_config(_atomic_number)
```


<a id="hotpot-utils-configs"></a>
## `hotpot.utils.configs`

源码 `hotpot/utils/configs/__init__.py`；SHA-256 `4bc34480d0fb2c193053cbd7c0dd35bd48deb4e86a1ba293b556860e41f77caf`。

显式 __all__：未声明；以下为直接模块面。

无待展开的非私有本地定义或模块值；导入/导出绑定如上。


<a id="hotpot-utils-configs-logging-config"></a>
## `hotpot.utils.configs.logging_config`

源码 `hotpot/utils/configs/logging_config.py`；SHA-256 `e937263a2e25757d9ddf570cd51c8671d616bcb744fed4dd069090d75e37b6be`。

显式 __all__：`('setup_logging', 'RateLimitLogger', 'LoggerDict')`。

显式导出绑定：

- `setup_logging` → `hotpot.utils.configs.logging_config.setup_logging`；`setup_logging(debug=False, to_stdout=True)`
- `RateLimitLogger` → `hotpot.utils.configs.logging_config.RateLimitLogger`；构造声明 `__init__(self, interval_seconds=0, interval_count=0, just_once=False)`
- `LoggerDict` → `hotpot.utils.configs.logging_config.LoggerDict`；构造声明 `__init__(self, interval_seconds=0, interval_count=0, just_once=False)`

<a id="hotpot-utils-configs-logging-config-setup-logging"></a>
### `setup_logging`

定义 owner `hotpot.utils.configs.logging_config.setup_logging`；源码 `hotpot/utils/configs/logging_config.py:31`。

```python
setup_logging(debug=False, to_stdout=True)
```

<a id="hotpot-utils-configs-logging-config-ratelimitlogger"></a>
### `RateLimitLogger`

定义 owner `hotpot.utils.configs.logging_config.RateLimitLogger`；源码 `hotpot/utils/configs/logging_config.py:42`。

基类原始声明：无显式基类；装饰器：无。

显式导出位置：`hotpot.utils.configs.logging_config.RateLimitLogger`。

本类属性与方法：

- `__init__(self, interval_seconds=0, interval_count=0, just_once=False)`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:43`
- `allow(self) -> bool`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:57`
- `info(self, msg: str, *args, **kwargs)`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:82`
- `warning(self, msg: str, *args, **kwargs)`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:86`
- `error(self, msg: str, *args, **kwargs)`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:90`
- `debug(self, msg: str, *args, **kwargs)`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:94`
- `critical(self, msg: str, *args, **kwargs)`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:98`
- `exception(self, msg: str, *args, **kwargs)`；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:102`

本类字段与枚举值：

- `interval_seconds` = `interval_seconds` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:49`
- `interval_count` = `interval_count` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:50`
- `just_once` = `just_once` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.configs.logging_config.RateLimitLogger`；`hotpot/utils/configs/logging_config.py:51`

<a id="hotpot-utils-configs-logging-config-loggerdict"></a>
### `LoggerDict`

定义 owner `hotpot.utils.configs.logging_config.LoggerDict`；源码 `hotpot/utils/configs/logging_config.py:107`。

基类原始声明：`defaultdict`；装饰器：无。

显式导出位置：`hotpot.utils.configs.logging_config.LoggerDict`。

本类属性与方法：

- `__init__(self, interval_seconds=0, interval_count=0, just_once=False)`；owner `hotpot.utils.configs.logging_config.LoggerDict`；`hotpot/utils/configs/logging_config.py:108`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`collections.defaultdict`。


<a id="hotpot-utils-fmt-print"></a>
## `hotpot.utils.fmt_print`

源码 `hotpot/utils/fmt_print.py`；SHA-256 `1b9133b2d1838ec8e5ac19bf7c9b1f9e47d5ed9117d74fe5de7ab49fdfd2c477`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `hex_to_ansi` → `hotpot.utils.fmt_print.hex_to_ansi`；`hex_to_ansi(hex_color: str, style: PrintStyle='regular') -> str`
- `FmtString` → `hotpot.utils.fmt_print.FmtString`；构造声明 `__init__(self, fmt: str)`
- `FmtPrint` → `hotpot.utils.fmt_print.FmtPrint`；构造声明 `__init__(self, fmt: str)`
- `dict_to_df` → `hotpot.utils.fmt_print.dict_to_df`；`dict_to_df(metrics_dict: dict[str, Union[float, dict[str, float]]])`
- `dict_to_table` → `hotpot.utils.fmt_print.dict_to_table`；`dict_to_table(metrics_dict: dict[str, Union[float, dict[str, float]]], table_kw: dict=None, title: str=None)`
- `export_table` → `hotpot.utils.fmt_print.export_table`；`export_table(table: Table, save_path: str, width=200)`
- `rich_print` → `hotpot.utils.fmt_print.rich_print`；`rich_print(contents, **kwargs)`
- `PrintStyle` → `hotpot.utils.fmt_print.PrintStyle`
- `light_green` → `hotpot.utils.fmt_print.light_green`
- `dark_green` → `hotpot.utils.fmt_print.dark_green`
- `bold_dark_green` → `hotpot.utils.fmt_print.bold_dark_green`
- `bold_orange` → `hotpot.utils.fmt_print.bold_orange`
- `bold_magenta` → `hotpot.utils.fmt_print.bold_magenta`

<a id="hotpot-utils-fmt-print-hex-to-ansi"></a>
### `hex_to_ansi`

定义 owner `hotpot.utils.fmt_print.hex_to_ansi`；源码 `hotpot/utils/fmt_print.py:19`。

```python
hex_to_ansi(hex_color: str, style: PrintStyle='regular') -> str
```

<a id="hotpot-utils-fmt-print-fmtstring"></a>
### `FmtString`

定义 owner `hotpot.utils.fmt_print.FmtString`；源码 `hotpot/utils/fmt_print.py:36`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, fmt: str)`；owner `hotpot.utils.fmt_print.FmtString`；`hotpot/utils/fmt_print.py:37`
- `__call__(self, s: str) -> str`；owner `hotpot.utils.fmt_print.FmtString`；`hotpot/utils/fmt_print.py:40`

本类字段与枚举值：

- `fmt` = `fmt` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.fmt_print.FmtString`；`hotpot/utils/fmt_print.py:38`
- `reset` = `'\x1b[0m'` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.fmt_print.FmtString`；`hotpot/utils/fmt_print.py:39`

<a id="hotpot-utils-fmt-print-fmtprint"></a>
### `FmtPrint`

定义 owner `hotpot.utils.fmt_print.FmtPrint`；源码 `hotpot/utils/fmt_print.py:43`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, fmt: str)`；owner `hotpot.utils.fmt_print.FmtPrint`；`hotpot/utils/fmt_print.py:44`
- `__call__(self, contents)`；owner `hotpot.utils.fmt_print.FmtPrint`；`hotpot/utils/fmt_print.py:47`
- `from_hex(cls, hex_color: str, style: PrintStyle='regular') -> 'FmtPrint'` [classmethod]；装饰器 `classmethod`；owner `hotpot.utils.fmt_print.FmtPrint`；`hotpot/utils/fmt_print.py:51`

本类字段与枚举值：

- `fmt_string` = `FmtString(fmt)` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.fmt_print.FmtPrint`；`hotpot/utils/fmt_print.py:45`

<a id="hotpot-utils-fmt-print-dict-to-df"></a>
### `dict_to_df`

定义 owner `hotpot.utils.fmt_print.dict_to_df`；源码 `hotpot/utils/fmt_print.py:63`。

```python
dict_to_df(metrics_dict: dict[str, Union[float, dict[str, float]]])
```

<a id="hotpot-utils-fmt-print-dict-to-table"></a>
### `dict_to_table`

定义 owner `hotpot.utils.fmt_print.dict_to_table`；源码 `hotpot/utils/fmt_print.py:73`。

```python
dict_to_table(metrics_dict: dict[str, Union[float, dict[str, float]]], table_kw: dict=None, title: str=None)
```

<a id="hotpot-utils-fmt-print-export-table"></a>
### `export_table`

定义 owner `hotpot.utils.fmt_print.export_table`；源码 `hotpot/utils/fmt_print.py:118`。

```python
export_table(table: Table, save_path: str, width=200)
```

<a id="hotpot-utils-fmt-print-rich-print"></a>
### `rich_print`

定义 owner `hotpot.utils.fmt_print.rich_print`；源码 `hotpot/utils/fmt_print.py:128`。

```python
rich_print(contents, **kwargs)
```

模块值/类型别名（声明存在不构成事实写许可）：

- `PrintStyle` = `Literal['regular', 'bold', 'italic', 'underline', 'blink', 'reverse', 'hidden']`；`hotpot/utils/fmt_print.py:18`
- `light_green` = `FmtPrint('\x1b[92m')`；`hotpot/utils/fmt_print.py:54`
- `dark_green` = `FmtPrint('\x1b[32m')`；`hotpot/utils/fmt_print.py:56`
- `bold_dark_green` = `FmtPrint('\x1b[1m')`；`hotpot/utils/fmt_print.py:57`
- `bold_orange` = `FmtPrint('\x1b[38;5;208m')`；`hotpot/utils/fmt_print.py:59`
- `bold_magenta` = `FmtPrint('\x1b[1;35m')`；`hotpot/utils/fmt_print.py:60`


<a id="hotpot-utils-mp"></a>
## `hotpot.utils.mp`

源码 `hotpot/utils/mp.py`；SHA-256 `9596150b29580fd645f65615f96a266f2b1f2ccba3c77a75d4cf808b3acd2d86`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `decorator` → `hotpot.utils.mp.decorator`；`decorator(func: Callable) -> Callable`
- `mp_run` → `hotpot.utils.mp.mp_run`；`mp_run(func: Callable, args: Iterable[tuple], kwargs: Iterable[dict]=None, nproc: int=None, desc: str='', timeout: Optional[float]=None, error_to_None: bool=True, lazy_iter: bool=False, early_exit_condition: Optional[Callable[[dict, int], bool]]=None, branch_jobs: Iterable[Callable[[dict, dict, int], None]]=None)`
- `Pool` → `hotpot.utils.mp.Pool`；构造声明 `__init__(self, nproc: int=None, desc: str='', timeout: Optional[float]=None, error_to_None: bool=True)`

<a id="hotpot-utils-mp-decorator"></a>
### `decorator`

定义 owner `hotpot.utils.mp.decorator`；源码 `hotpot/utils/mp.py:18`。

```python
decorator(func: Callable) -> Callable
```

<a id="hotpot-utils-mp-mp-run"></a>
### `mp_run`

定义 owner `hotpot.utils.mp.mp_run`；源码 `hotpot/utils/mp.py:28`。

```python
mp_run(func: Callable, args: Iterable[tuple], kwargs: Iterable[dict]=None, nproc: int=None, desc: str='', timeout: Optional[float]=None, error_to_None: bool=True, lazy_iter: bool=False, early_exit_condition: Optional[Callable[[dict, int], bool]]=None, branch_jobs: Iterable[Callable[[dict, dict, int], None]]=None)
```

<a id="hotpot-utils-mp-pool"></a>
### `Pool`

定义 owner `hotpot.utils.mp.Pool`；源码 `hotpot/utils/mp.py:127`。

基类原始声明：无显式基类；装饰器：无。

本类属性与方法：

- `__init__(self, nproc: int=None, desc: str='', timeout: Optional[float]=None, error_to_None: bool=True)`；owner `hotpot.utils.mp.Pool`；`hotpot/utils/mp.py:128`
- `run(self, func: Callable, list_args: list[tuple], list_kwargs: list[dict]=None)`；owner `hotpot.utils.mp.Pool`；`hotpot/utils/mp.py:140`

本类字段与枚举值：

- `nproc` = `nproc` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.mp.Pool`；`hotpot/utils/mp.py:135`
- `desc` = `desc` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.mp.Pool`；`hotpot/utils/mp.py:136`
- `timeout` = `timeout` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.mp.Pool`；`hotpot/utils/mp.py:137`
- `error_to_None` = `error_to_None` [可写实例字段；非自动批准事实写许可]（self 赋值提取，非固定 schema）；owner `hotpot.utils.mp.Pool`；`hotpot/utils/mp.py:138`


<a id="hotpot-utils-tools"></a>
## `hotpot.utils.tools`

源码 `hotpot/utils/tools.py`；SHA-256 `be72fe4ea4611e2e1709aa467328cc8ac64068b641993edd3d00a1954bb81c1f`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `show_time` → `hotpot.utils.tools.show_time`；`show_time(func, *args, **kwargs)`
- `show_func_time` → `hotpot.utils.tools.show_func_time`；`show_func_time(func: Callable)`

<a id="hotpot-utils-tools-show-time"></a>
### `show_time`

定义 owner `hotpot.utils.tools.show_time`；源码 `hotpot/utils/tools.py:13`。

```python
show_time(func, *args, **kwargs)
```

<a id="hotpot-utils-tools-show-func-time"></a>
### `show_func_time`

定义 owner `hotpot.utils.tools.show_func_time`；源码 `hotpot/utils/tools.py:20`。

```python
show_func_time(func: Callable)
```


<a id="hotpot-utils-types"></a>
## `hotpot.utils.types`

源码 `hotpot/utils/types.py`；SHA-256 `1ad2b525541ec4f19ad12f3990ad2f67241f87c365f219e855d7ca1aa23f8957`。

显式 __all__：未声明；以下为直接模块面。

直接模块名称（非 __all__ 导出承诺）：

- `ModelLike` → `hotpot.utils.types.ModelLike`
- `ArrayLike` → `hotpot.utils.types.ArrayLike`
- `Vector` → `hotpot.utils.types.Vector`

<a id="hotpot-utils-types-modellike"></a>
### `ModelLike`

定义 owner `hotpot.utils.types.ModelLike`；源码 `hotpot/utils/types.py:23`。

基类原始声明：`Protocol`；装饰器：无。

本类属性与方法：

- `fit(self, *args, **kwargs) -> None`；owner `hotpot.utils.types.ModelLike`；`hotpot/utils/types.py:25`
- `predict(self, *args, **kwargs) -> Any`；owner `hotpot.utils.types.ModelLike`；`hotpot/utils/types.py:28`
- `score(self, *args, **kwargs) -> float`；owner `hotpot.utils.types.ModelLike`；`hotpot/utils/types.py:31`

未展开的外部/泛型基类（库存静态解析标签，非运行时身份确认）：`hotpot.utils.types.Protocol`。

模块值/类型别名（声明存在不构成事实写许可）：

- `ArrayLike` = `Union[Sequence, np.ndarray, 'torch.Tensor']`；`hotpot/utils/types.py:19`
- `Vector` = `Sequence[Union[int, float]]`；`hotpot/utils/types.py:20`
