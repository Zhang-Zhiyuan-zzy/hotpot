# 非本次主交付范围：辅助模块与导出索引

包括已有 dataset/ds、通用工具、绘图、机器学习/ComplexFormer、量化软件等插件。本页给当前模块、本地定义与显式导出；完整属性、字段、方法与继承成员另见 [辅助完整成员入口](auxiliary_index.md)；不声称已重新验收这些模块的全部行为或第三方动态 API。

完整参考：[dataset/ds 与工具](auxiliary_data_utils_api.md) · [ComplexFormer](auxiliary_complexformer_api.md) · [其他插件](auxiliary_plugins_api.md)。

## `hotpot.dataset`

`hotpot/dataset/__init__.py`


## `hotpot.ds`

`hotpot/ds/__init__.py`

显式 `__all__`: `('datasets_list', 'load', 'Dataset', 'Data', 'BaseSearcher', 'DataSearcher', 'DatasetSearcher', 'SmartsSearcher', 'load_bundle', 'Bundle', 'write_certificate')`。


导出别名：
- `datasets_list` → `hotpot.ds.public.catalog.datasets_list`
- `load` → `hotpot.ds.public.catalog.load`
- `Dataset` → `hotpot.ds.public.dataset.Dataset`
- `Data` → `hotpot.ds.public.dataset.Data`
- `BaseSearcher` → `hotpot.ds.public.search.BaseSearcher`
- `DataSearcher` → `hotpot.ds.public.search.DataSearcher`
- `DatasetSearcher` → `hotpot.ds.public.search.DatasetSearcher`
- `SmartsSearcher` → `hotpot.ds.public.search.SmartsSearcher`
- `load_bundle` → `hotpot.ds.public.bundle.load_bundle`
- `Bundle` → `hotpot.ds.public.bundle.Bundle`
- `write_certificate` → `hotpot.ds.public.bundle.write_certificate`

## `hotpot.ds._bundle_cleanup`

`hotpot/ds/_bundle_cleanup.py`

显式 `__all__`: `('BundleCleanupError', 'remove_bundle_intermediates')`。

- `BundleCleanupError`；基类 RuntimeError
- `remove_bundle_intermediates(root: Path, dataset: str, database: Optional[Path]=None) -> dict[str, Any]`

## `hotpot.ds._communication`

`hotpot/ds/_communication.py`

显式 `__all__`: `('CommunicationError', 'emit_envelope', 'extract_path', 'load_protocol', 'make_envelope', 'protocol_sha256', 'read_input')`。

- `CommunicationError`；基类 ValueError
- `load_protocol() -> dict[str, Any]`
- `protocol_sha256() -> str`
- `extract_path(document: Mapping[str, Any], path: str) -> Any`
- `make_envelope(key: str, payload: Any, *, channel: str='default') -> dict[str, Any]`
- `emit_envelope(key: str, payload: Any, *, channel: str='default', stream: TextIO | None=None) -> dict[str, Any]`
- `read_input(consumer: str, stream: TextIO | None=None) -> dict[str, Any]`

## `hotpot.ds._export`

`hotpot/ds/_export.py`

显式 `__all__`: `()`。

- `ExportError`；基类 ValueError
- `normalize_format(fmt: str) -> str`
- `render_structure_record(data: Any, fmt: str, *, structure_index: Optional[int]=None, structure_role: Optional[str]=None, structure_id: Optional[str]=None) -> dict[str, Any]`
- `export_records(dataset: Any, output: Path, *, fmt: Optional[str], selection: Mapping[str, Any]) -> dict[str, Any]`

## `hotpot.ds._paths`

`hotpot/ds/_paths.py`


## `hotpot.ds._store_copy`

`hotpot/ds/_store_copy.py`

显式 `__all__`: `('StoreCopyError', 'copy_store')`。

- `StoreCopyError`；基类 RuntimeError
- `copy_store(source_db: str | Path, target_package: str | Path) -> dict[str, Any]`

## `hotpot.ds.cli`

`hotpot/ds/cli.py`

- `load_cli_documentation(topic: Optional[str]=None) -> str`
- `add_arguments(parser: argparse.ArgumentParser) -> None`
- `run(args: argparse.Namespace) -> int`
- `build_parser() -> argparse.ArgumentParser`
- `main() -> int`

导出别名：
- `HELP_FORMATTER` → `argparse.ArgumentDefaultsHelpFormatter`

## `hotpot.ds.gates`

`hotpot/ds/gates/__init__.py`

显式 `__all__`: `('EvidenceGate', 'QualityGate', 'BenchmarkGate')`。


导出别名：
- `EvidenceGate` → `hotpot.ds.gates._gates.EvidenceGate`
- `QualityGate` → `hotpot.ds.gates._gates.QualityGate`
- `BenchmarkGate` → `hotpot.ds.gates._gates.BenchmarkGate`

## `hotpot.ds.gates._base`

`hotpot/ds/gates/_base.py`


## `hotpot.ds.gates._checker`

`hotpot/ds/gates/_checker.py`

- `CheckerIsolationUnavailableError`；基类 RuntimeError
- `CheckerExecutionError`；基类 RuntimeError
- `OSIsolatedCheckerRunner`；无显式基类
- `evaluate_configured_check(checker: _CheckerDefinition, case: Mapping[str, Any], bundle: Mapping[str, Any], *, builtin_result: Optional[Tuple[bool, str]]=None, isolated_runner: Optional[Any]=None) -> Tuple[bool, str]`

## `hotpot.ds.gates._gates`

`hotpot/ds/gates/_gates.py`

- `EvidenceGate`；基类 _ImmutableGate
- `QualityGate`；基类 _ImmutableGate
- `BenchmarkGate`；基类 _ImmutableGate

## `hotpot.ds.gates._persistence`

`hotpot/ds/gates/_persistence.py`

- `default_gates() -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]`
- `serialize_gate_configs(gates: Iterable[object]) -> list`
- `deserialize_gate_configs(configurations: Iterable[Mapping[str, object]]) -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]`
- `materialize_checker_sources(gates: Iterable[object], bundle_root: Path) -> Tuple[Path, ...]`

## `hotpot.ds.gates.benchmark`

`hotpot/ds/gates/benchmark.py`

显式 `__all__`: `('BenchmarkGate',)`。


导出别名：
- `BenchmarkGate` → `hotpot.ds.gates._gates.BenchmarkGate`

## `hotpot.ds.gates.evidence`

`hotpot/ds/gates/evidence.py`

显式 `__all__`: `('EvidenceGate',)`。


导出别名：
- `EvidenceGate` → `hotpot.ds.gates._gates.EvidenceGate`

## `hotpot.ds.gates.quality`

`hotpot/ds/gates/quality.py`

显式 `__all__`: `('QualityGate',)`。


导出别名：
- `QualityGate` → `hotpot.ds.gates._gates.QualityGate`

## `hotpot.ds.governance`

`hotpot/ds/governance.py`

显式 `__all__`: `('GovernanceAuthorizationError', 'GovernanceError', 'GovernanceVerificationError', 'GovernanceKey', 'PinnedRoot', 'VerifiedRegistry', 'load_pinned_root', 'verify_registry', 'verify_hotpot_attestation')`。

- `GovernanceError`；基类 ValueError
- `GovernanceVerificationError`；基类 GovernanceError
- `GovernanceAuthorizationError`；基类 GovernanceError
- `PinnedRoot`；无显式基类
- `GovernanceKey`；无显式基类
- `VerifiedRegistry`；无显式基类
- `load_pinned_root() -> PinnedRoot`
- `default_registry_snapshot() -> dict[str, Any]`
- `verify_registry(snapshot: Mapping[str, Any]) -> VerifiedRegistry`
- `verify_hotpot_attestation(attestation: Mapping[str, Any], dataset: str, release: str, case_id: str, registry: Mapping[str, Any] | VerifiedRegistry) -> tuple[bool, str]`

## `hotpot.ds.providers`

`hotpot/ds/providers/__init__.py`


## `hotpot.ds.providers._io`

`hotpot/ds/providers/_io.py`

- `download_file(fp, _url)`
- `uncompress_all_gz(file_path)`
- `uncompress_zip(zip_path, extract_dir)`

## `hotpot.ds.providers.lnqm`

`hotpot/ds/providers/lnqm/__init__.py`

- `LnQmDataset`；无显式基类

## `hotpot.ds.providers.lnqm.pyg_data`

`hotpot/ds/providers/lnqm/pyg_data.py`

- `LnQM_Dataset`；基类 InMemoryDataset

## `hotpot.ds.providers.tmqm`

`hotpot/ds/providers/tmqm/__init__.py`

- `DataGenerator`；无显式基类
- `TmQmDataset`；无显式基类

## `hotpot.ds.providers.tmqm.download`

`hotpot/ds/providers/tmqm/download.py`

显式 `__all__`: `('urls', 'download_file', 'download_files', 'uncompress_all_gz')`。

- `download_files()`

导出别名：
- `download_file` → `hotpot.ds.providers._io.download_file`
- `uncompress_all_gz` → `hotpot.ds.providers._io.uncompress_all_gz`

## `hotpot.ds.public`

`hotpot/ds/public/__init__.py`

显式 `__all__`: `()`。


## `hotpot.ds.public._assessment`

`hotpot/ds/public/_assessment.py`

显式 `__all__`: `('B_LEVELS', 'E_LEVELS', 'assess_case', 'assess_evidence', 'canonical_digest', 'canonical_json', 'compare_observations', 'find_conflicts', 'identity_key', 'observation_context', 'observation_property', 'summarize_assessments', 'validate_observation', 'validate_property_policy')`。

- `canonical_json(value: Any) -> str`
- `canonical_digest(value: Any) -> str`
- `validate_identity_schema(value: Mapping[str, Any]) -> Dict[str, Any]`
- `observation_property(observation: Mapping[str, Any]) -> Optional[str]`
- `validate_property_policy(definitions: Mapping[str, Any], comparators: Mapping[str, Any]) -> None`
- `validate_observation(observation: Mapping[str, Any], definitions: Mapping[str, Any], comparators: Mapping[str, Any]) -> None`
- `assess_evidence(case: Mapping[str, Any], required_targets: Optional[Iterable[str]]=None, *, asset_assessments: Sequence[Mapping[str, Any]]=(), policy: Optional[Mapping[str, Any]]=None, checker_context: Optional[Mapping[str, Any]]=None, isolated_runner: Optional[Any]=None) -> Dict[str, Any]`
- `identity_key(case: Mapping[str, Any], identity_schema: Mapping[str, Any]) -> Tuple[Optional[str], list[Dict[str, Any]]]`
- `sample_match(left: Mapping[str, Any], right: Mapping[str, Any], identity_schema: Mapping[str, Any]) -> Dict[str, Any]`
- `observation_context(observation: Mapping[str, Any], definition: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]`
- `observation_with_resolved_entities(case: Mapping[str, Any], observation: Mapping[str, Any]) -> Dict[str, Any]`
- `compare_observations(left: Mapping[str, Any], right: Mapping[str, Any], tolerances: Mapping[str, Mapping[str, float]], *, property_definitions: Optional[Mapping[str, Any]]=None, property_comparators: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]`
- `find_conflicts(cases: Iterable[Mapping[str, Any]], identity_schema: Mapping[str, Any], tolerances: Mapping[str, Mapping[str, float]], *, property_definitions: Optional[Mapping[str, Any]]=None, property_comparators: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]`
- `assess_case(case: Mapping[str, Any], *, core_labels: Sequence[str], core_info: Sequence[str], required_structure_roles: Sequence[str], asset_assessments: Sequence[Mapping[str, Any]], conflicting: bool, benchmark_policy: Mapping[str, Any], evidence_policy: Optional[Mapping[str, Any]]=None, quality_policy: Optional[Mapping[str, Any]]=None, conflict_assessments: Sequence[Mapping[str, Any]]=(), identity_reasons: Sequence[Mapping[str, Any]]=(), checker_context: Optional[Mapping[str, Any]]=None, isolated_runner: Optional[Any]=None) -> Dict[str, Any]`
- `summarize_assessments(assessments: Iterable[Mapping[str, Any]]) -> Dict[str, Any]`

## `hotpot.ds.public._bundle_state`

`hotpot/ds/public/_bundle_state.py`

显式 `__all__`: `('BUNDLE_SCHEMA', 'BundleAlreadyBuiltError', 'BundleNotBuiltError', 'BundleState', 'BundleStateError', 'ConcurrentStateError', 'DirtySourceStateError')`。

- `BundleStateError`；基类 ValueError
- `BundleAlreadyBuiltError`；基类 BundleStateError
- `BundleNotBuiltError`；基类 BundleStateError
- `ConcurrentStateError`；基类 BundleStateError
- `DirtySourceStateError`；基类 BundleStateError
- `BundleState`；无显式基类

## `hotpot.ds.public._generation`

`hotpot/ds/public/_generation.py`

显式 `__all__`: `('AssessmentGeneration', 'AssessmentGenerationError', 'AssetSourceChangedError', 'GenerationIntegrityError')`。

- `AssessmentGenerationError`；基类 ValueError
- `GenerationIntegrityError`；基类 AssessmentGenerationError
- `AssetSourceChangedError`；基类 AssessmentGenerationError
- `AssessmentGeneration`；无显式基类

## `hotpot.ds.public._implementation`

`hotpot/ds/public/_implementation.py`

显式 `__all__`: `('PolicyProtocolError', 'parser_binding_for_format', 'require_current_runtime_policy', 'runtime_policy', 'runtime_policy_digest', 'runtime_policy_is_current', 'validate_frozen_runtime_policy')`。

- `PolicyProtocolError`；基类 RuntimeError
- `runtime_policy() -> dict[str, Any]`
- `runtime_policy_digest() -> str`
- `validate_frozen_runtime_policy(value: Mapping[str, Any]) -> dict[str, Any]`
- `runtime_policy_is_current(value: Mapping[str, Any]) -> bool`
- `require_current_runtime_policy(value: Mapping[str, Any]) -> None`
- `parser_binding_for_format(format_name: str, policy: Mapping[str, Any] | None=None) -> dict[str, Any] | None`

## `hotpot.ds.public.bundle`

`hotpot/ds/public/bundle.py`

显式 `__all__`: `('Bundle', 'load_bundle', 'write_certificate')`。

- `ExistingDatasetPolicyError`；基类 RuntimeError
- `MissingCertificateError`；基类 RuntimeError
- `SealedReleaseError`；基类 RuntimeError
- `SealedIntegrityError`；基类 RuntimeError
- `Bundle`；无显式基类
- `load_bundle(root: Union[str, Path], ds_name: str, *, db: Optional[Union[str, Path]]=None) -> Bundle`
- `write_certificate(bundle: Bundle) -> Path`

## `hotpot.ds.public.case`

`hotpot/ds/public/case.py`

显式 `__all__`: `('Case', 'CaseCollection', 'CaseOwnershipError', 'CaseView', 'UnsynchronizedBundleError')`。

- `CaseOwnershipError`；基类 RuntimeError
- `UnsynchronizedBundleError`；基类 RuntimeError
- `Case`；无显式基类
- `CaseView`；基类 Sequence[Case]
- `CaseCollection`；基类 CaseView

## `hotpot.ds.public.catalog`

`hotpot/ds/public/catalog.py`

显式 `__all__`: `('DEFAULT_DATABASE', 'DatasetCatalog', 'datasets_list', 'load')`。

- `DatasetCatalog`；基类 Sequence[str]
- `load(ds_name: str, version: Optional[Union[str, int]]=None, *, db: Optional[Union[str, Path]]=None, archive_class: Optional[Union[str, Iterable[str]]]=None) -> Dataset`

导出别名：
- `DEFAULT_DATABASE` → `hotpot.ds._paths.DEFAULT_STORE_PATH`

## `hotpot.ds.public.certificate`

`hotpot/ds/public/certificate.py`

显式 `__all__`: `('CertificateError', 'CertificateFormatError', 'CertificateIntegrityError', 'CertificatePathError', 'certificate_digest', 'load_certificate_envelope', 'verify_bundle_certificate', 'write_bundle_certificate')`。

- `CertificateError`；基类 RuntimeError
- `CertificateFormatError`；基类 CertificateError
- `CertificateIntegrityError`；基类 CertificateError
- `CertificatePathError`；基类 CertificateIntegrityError
- `load_certificate_envelope(path: Union[str, Path]) -> Dict[str, Any]`
- `write_bundle_certificate(bundle: Any) -> Path`
- `verify_bundle_certificate(bundle: Any, path: Optional[Union[str, Path]]=None) -> Dict[str, Any]`
- `certificate_digest(path: Union[str, Path]) -> str`

## `hotpot.ds.public.dataset`

`hotpot/ds/public/dataset.py`

显式 `__all__`: `('Data', 'Dataset')`。

- `IdentifierIndex`；无显式基类
- `Dataset`；基类 Sequence['Data']
- `Data`；无显式基类

## `hotpot.ds.public.search`

`hotpot/ds/public/search.py`

显式 `__all__`: `('BaseSearcher', 'DataSearcher', 'DatasetSearcher', 'SmartsSearcher')`。

- `BaseSearcher`；基类 ABC
- `DataSearcher`；基类 _FieldSearcher
- `DatasetSearcher`；基类 _FieldSearcher
- `SmartsSearcher`；基类 BaseSearcher

## `hotpot.ds.public.structures`

`hotpot/ds/public/structures.py`

显式 `__all__`: `('AmbiguousStructureError', 'AssetIntegrityError', 'CrystalStructureUnavailableError', 'StructureAccessError', 'StructureUnavailableError', 'Structures')`。

- `StructureAccessError`；基类 RuntimeError
- `AmbiguousStructureError`；基类 StructureAccessError
- `AssetIntegrityError`；基类 StructureAccessError
- `StructureUnavailableError`；基类 StructureAccessError
- `CrystalStructureUnavailableError`；基类 StructureUnavailableError
- `Structures`；无显式基类

## `hotpot.ds.storage`

`hotpot/ds/storage.py`

显式 `__all__`: `('TARGET_SCHEMA_ID', 'AssetIntegrityError', 'PolicyMismatchError', 'ReceiptConflictError', 'SnapshotError', 'TargetIngestResult', 'TargetSchemaError', 'TargetStore', 'TargetStoreError', 'asset_inventory_digest', 'case_content_digest', 'case_content_digests', 'case_set_digest', 'included_case_content_digest', 'persisted_formal_content_digest', 'persisted_included_case_content_digest')`。

- `TargetStoreError`；基类 RuntimeError
- `TargetSchemaError`；基类 TargetStoreError
- `SnapshotError`；基类 TargetStoreError, ValueError
- `PolicyMismatchError`；基类 TargetStoreError
- `ReceiptConflictError`；基类 TargetStoreError
- `AssetIntegrityError`；基类 TargetStoreError
- `TargetIngestResult`；无显式基类
- `case_set_digest(case_ids: Iterable[str]) -> str`
- `asset_inventory_digest(assets: Iterable[Mapping[str, Any]]) -> str`
- `case_content_digest(cases: Iterable[Mapping[str, Any]]) -> str`
- `case_content_digests(cases: Iterable[Mapping[str, Any]], *, include_incomplete: bool=False) -> Tuple[str, str]`
- `included_case_content_digest(cases: Iterable[Mapping[str, Any]], *, include_incomplete: bool=False) -> str`
- `persisted_included_case_content_digest(connection: sqlite3.Connection, dataset: str, release: str) -> str`
- `persisted_formal_content_digest(connection: sqlite3.Connection, dataset: str, release: str) -> str`
- `TargetStore`；无显式基类

## `hotpot.ds.tools`

`hotpot/ds/tools/__init__.py`


## `hotpot.ds.tools.analyze_structure_composition`

`hotpot/ds/tools/analyze_structure_composition.py`

- `read_csv(path: Path) -> list[dict[str, str]]`
- `write_csv(path: Path, rows: list[dict[str, object]]) -> None`
- `target_metals() -> dict[tuple[str, str, str], str]`
- `connected_components(adjacency: list[set[int]], removed: int) -> list[set[int]]`
- `analyze(path: Path, fmt: str, target_symbol: str) -> dict[str, object]`
- `distributions(rows: list[dict[str, object]]) -> list[dict[str, object]]`
- `main() -> None`

## `hotpot.ds.tools.electronic_state`

`hotpot/ds/tools/electronic_state/__init__.py`


## `hotpot.ds.tools.electronic_state.assess_quality`

`hotpot/ds/tools/electronic_state/assess_quality.py`

- `assess(database: Path) -> Dict[str, object]`
- `main() -> int`

## `hotpot.ds.tools.electronic_state.build_database`

`hotpot/ds/tools/electronic_state/build_database.py`

- `sha256_file(path: Path) -> str`
- `scalar(connection: sqlite3.Connection, sql: str, parameters=())`
- `parse_args() -> argparse.Namespace`
- `main() -> int`

## `hotpot.ds.tools.oxidation`

`hotpot/ds/tools/oxidation/__init__.py`


## `hotpot.ds.tools.oxidation.validate`

`hotpot/ds/tools/oxidation/validate.py`

- `table(path: Path) -> list[dict[str, str]]`
- `digest(path: Path) -> str`
- `inputs()`
- `validate(item) -> dict[str, object]`
- `assert_independence() -> None`
- `main() -> None`

## `hotpot.ds.tools.spin`

`hotpot/ds/tools/spin/__init__.py`


## `hotpot.ds.tools.spin.validate`

`hotpot/ds/tools/spin/validate.py`

- `table(path: Path) -> list[dict[str, str]]`
- `digest(path: Path) -> str`
- `inputs()`
- `validate(item) -> dict[str, object]`
- `assert_independence() -> None`
- `sco_pair_counts() -> tuple[int, int]`
- `main() -> None`

## `hotpot.plugins.ComplexFormer`

`hotpot/plugins/ComplexFormer/__init__.py`


## `hotpot.plugins.ComplexFormer.callbacks`

`hotpot/plugins/ComplexFormer/callbacks.py`

- `get_inspect_table(inspect_dict, table_kw: dict=None, title: str=None)`
- `Pbar`；基类 ProgressBar
- `Debugger`；基类 Callback

## `hotpot.plugins.ComplexFormer.check`

`hotpot/plugins/ComplexFormer/check.py`

- `check_trainable_parameters(model: nn.Module)`
- `check_all_parameters_status(model: nn.Module)`
- `check_gradient_values(model: nn.Module)`

## `hotpot.plugins.ComplexFormer.config_task`

`hotpot/plugins/ComplexFormer/config_task.py`

- `config_task(cfg: ConfigArgs)`
- `wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable`
- `TaskComponents`；无显式基类
- `TaskBuilder`；无显式基类

## `hotpot.plugins.ComplexFormer.configs`

`hotpot/plugins/ComplexFormer/configs.py`

- `wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable`
- `config_tasks_from_multi_datasets(work_name: str, dataset_counts: int, inputs_getter: Callable, first_data: list[Data], core: M.CoreBase, predictor: M.Predictor, feature_extractor: list[dict[str, Callable]], target_getter: list[dict[str, tp.TargetGetter]], loss_fn: list[dict[str, tp.LossFn]], primary_metrics: list[dict[str, str]], other_metrics: Union[dict[str, Union[str, Iterable[str], tp.MetricFn]], list[tp.OtherMetricConfig]], hypers: Union[ParamSets, list[ParamSets]], task_names: Optional[list[Sequence[str]]]=None, batch_preprocessor: Optional[list[tp.BatchPreProcessor]]=None, inputs_preprocessor: Optional[list[Callable]]=None, xyz_index: Optional[list[Iterable[int]]]=None, with_sol: Optional[Union[bool, Iterable[bool]]]=None, with_med: Optional[Union[bool, Iterable[bool]]]=None, with_env: Optional[Union[bool, Iterable[bool]]]=None, xyz_perturb_sigma: Optional[list[float]]=None, xyz_perturb_mode: M.PerturbMode='uniform', extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]]=None, loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]]=None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='inverse-count', loss_fn_wrap_tasks: Optional[list[str, set[str]]]=None, onehot_types: Optional[list[dict[str, int]]]=None, x_masker: Optional[list[tp.XMasker]]=None, mask_need_task: Optional[list[list[str]]]=None, **kwargs)`
- `config(work_name: str, task_names: Union[list[list], list[str]], task_type: type, dataModule: DataModule, inputs_getter: Callable, core: M.CoreBase, predictor: M.Predictor, feature_extractor, target_getter, loss_fn, primary_metrics, other_metrics: Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]], hypers: ParamSets, batch_preprocessor: Optional[list[tp.BatchPreProcessor]]=None, inputs_preprocessor: Optional[list[Callable]]=None, with_xyz: Optional[Union[bool, Iterable[bool]]]=None, with_sol: Optional[Union[bool, Iterable[bool]]]=None, with_med: Optional[Union[bool, Iterable[bool]]]=None, with_env: Optional[Union[bool, Iterable[bool]]]=None, xyz_perturb_sigma: Optional[list[float]]=None, xyz_perturb_mode: M.PerturbMode='uniform', extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]]=None, loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]]=None, loss_fn_wrap_tasks: Optional[Union[bool, set[str], Iterable[set[str]]]]=None, loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='inverse-count', onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]]=None, x_masker: Optional[list[tp.XMasker]]=None, mask_need_task: Optional[list[list[str]]]=None, **kwargs)`

## `hotpot.plugins.ComplexFormer.data`

`hotpot/plugins/ComplexFormer/data/__init__.py`


## `hotpot.plugins.ComplexFormer.data.collate`

`hotpot/plugins/ComplexFormer/data/collate.py`

- `Collater`；基类 PyGCollater

## `hotpot.plugins.ComplexFormer.data.data`

`hotpot/plugins/ComplexFormer/data/data.py`

- `ExtractionData`；基类 Data

## `hotpot.plugins.ComplexFormer.data.data_module`

`hotpot/plugins/ComplexFormer/data/data_module.py`

- `get_first_data(dir_datasets: str) -> dict[str, Data]`
- `DataModule`；基类 L.LightningDataModule

## `hotpot.plugins.ComplexFormer.data.dataset`

`hotpot/plugins/ComplexFormer/data/dataset.py`

- `torch_load_data(p: str) -> Data`
- `load_data(dataset: Union[Iterable[Data], Sequence[Data]], sample_num: int=None)`
- `DataWrapper`；基类 Dataset
- `MConcatDataset`；基类 ConcatDataset
- `OnFlyLoadingDataset`；基类 Dataset
- `PathStoredDataset`；基类 Dataset

## `hotpot.plugins.ComplexFormer.data.loader`

`hotpot/plugins/ComplexFormer/data/loader.py`

- `DistConcatBatchSampler`；基类 Sampler
- `CDataLoader`；基类 DataLoader
- `DistConcatLoader`；基类 DataLoader

## `hotpot.plugins.ComplexFormer.data_process`

`hotpot/plugins/ComplexFormer/data_process/__init__.py`


## `hotpot.plugins.ComplexFormer.data_process.ccdc`

`hotpot/plugins/ComplexFormer/data_process/ccdc.py`

- `ccdc_struct_to_data(struct_dir: str, data_dir: str)`
- `convert_ml_pairs_to_cbond_broken_data(path_struct: str, data_dir: str, catoms_elements=None) -> Optional[set[str]]`

## `hotpot.plugins.ComplexFormer.data_process.normalize_data`

`hotpot/plugins/ComplexFormer/data_process/normalize_data.py`

- `Method`；基类 Enum
- `DataProfile`；无显式基类
- `DatasetAnalyzer`；无显式基类
- `TensorNormalizer`；无显式基类

## `hotpot.plugins.ComplexFormer.data_process.sc_logk`

`hotpot/plugins/ComplexFormer/data_process/sc_logk/__init__.py`


## `hotpot.plugins.ComplexFormer.data_process.sc_logk.core`

`hotpot/plugins/ComplexFormer/data_process/sc_logk/core.py`

显式 `__all__`: `('mp_process_SclogK',)`。

- `split_metal(metal_info: str)`
- `mp_process_SclogK(path_raw: str, data_dir: str, store_metal_cluster: bool=False)`

## `hotpot.plugins.ComplexFormer.data_process.sc_logk.data`

`hotpot/plugins/ComplexFormer/data_process/sc_logk/data.py`

- `ExtractionData`；基类 Data

## `hotpot.plugins.ComplexFormer.models`

`hotpot/plugins/ComplexFormer/models/__init__.py`


## `hotpot.plugins.ComplexFormer.models._core`

`hotpot/plugins/ComplexFormer/models/_core.py`

- `complete_graph_generator(ptr)`
- `CloudGraph`；基类 nn.Module
- `NodeProcessor`；基类 nn.Module
- `CoreBase`；基类 nn.Module
- `CompleteGraphExtractor`；无显式基类
- `CompleteGraphCore`；基类 CoreBase
- `AttnExtractor`；无显式基类
- `AttnCore`；基类 CoreBase

导出别名：
- `Core` → `hotpot.plugins.ComplexFormer.models._core.AttnCore`

## `hotpot.plugins.ComplexFormer.models.attn`

`hotpot/plugins/ComplexFormer/models/attn.py`

- `Encoder`；基类 nn.Module
- `EncoderLayer`；基类 nn.Module
- `MultiHeadAttention`；基类 nn.Module
- `scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor`

## `hotpot.plugins.ComplexFormer.models.cm`

`hotpot/plugins/ComplexFormer/models/cm.py`

- `Model`；基类 L.LightningModule

## `hotpot.plugins.ComplexFormer.models.core`

`hotpot/plugins/ComplexFormer/models/core/__init__.py`


导出别名：
- `Core` → `hotpot.plugins.ComplexFormer.models.core.attn_core.AttnCore`

## `hotpot.plugins.ComplexFormer.models.core._utils`

`hotpot/plugins/ComplexFormer/models/core/_utils.py`

显式 `__all__`: `('complete_graph_generator', 'split_padding', 'split_padding_deploy', 'seq_absmax_pooling')`。

- `complete_graph_generator(ptr)`
- `split_padding_deploy(x: torch.Tensor, nums: torch.Tensor)`
- `split_padding(x: torch.Tensor, nums: torch.Tensor)`
- `seq_absmax_pooling(seq)`

## `hotpot.plugins.ComplexFormer.models.core.attn_core`

`hotpot/plugins/ComplexFormer/models/core/attn_core.py`

显式 `__all__`: `('AttnExtractor', 'AttnCore')`。

- `AttnExtractor`；无显式基类
- `AttnCore`；基类 CoreBase

## `hotpot.plugins.ComplexFormer.models.core.base`

`hotpot/plugins/ComplexFormer/models/core/base.py`

- `CoreBase`；基类 nn.Module

## `hotpot.plugins.ComplexFormer.models.core.cloud`

`hotpot/plugins/ComplexFormer/models/core/cloud.py`

显式 `__all__`: `('CloudGraph',)`。

- `CloudGraph`；基类 nn.Module
- `SE3Cloud`；基类 nn.Module

## `hotpot.plugins.ComplexFormer.models.core.envs_encoder`

`hotpot/plugins/ComplexFormer/models/core/envs_encoder.py`

- `SolventNet`；基类 nn.Module

## `hotpot.plugins.ComplexFormer.models.core.graph`

`hotpot/plugins/ComplexFormer/models/core/graph.py`

显式 `__all__`: `('CompleteGraph',)`。

- `CompleteGraph`；基类 nn.Module

## `hotpot.plugins.ComplexFormer.models.core.graph_core`

`hotpot/plugins/ComplexFormer/models/core/graph_core.py`

显式 `__all__`: `('CompleteGraphExtractor', 'CompleteGraphCore')`。

- `CompleteGraphExtractor`；无显式基类
- `CompleteGraphCore`；基类 CoreBase

## `hotpot.plugins.ComplexFormer.models.core.node_processor`

`hotpot/plugins/ComplexFormer/models/core/node_processor.py`

显式 `__all__`: `('NodeProcessor',)`。

- `GraphCloud`；基类 nn.Module
- `NodeEmb`；基类 nn.Module
- `NodeProcessor`；基类 nn.Module

## `hotpot.plugins.ComplexFormer.models.core.se3net`

`hotpot/plugins/ComplexFormer/models/core/se3net.py`

- `auto_irreps(emb_dim: int, max_l: int, parity: Parity=-1) -> Irreps`
- `SE3EquivLayer`；基类 nn.Module
- `SE3Net`；基类 nn.Module

## `hotpot.plugins.ComplexFormer.models.graph`

`hotpot/plugins/ComplexFormer/models/graph.py`

- `CompleteGraph`；基类 nn.Module

## `hotpot.plugins.ComplexFormer.models.loss`

`hotpot/plugins/ComplexFormer/models/loss.py`

- `LossMethods`；无显式基类
- `label_weights_calculator(onehot_types: int, weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='cross-entropy')`
- `atl_calculator(metrics: dict[str, float], p: Union[int, float]=3, eps: float=0.0001)`

## `hotpot.plugins.ComplexFormer.models.metrics`

`hotpot/plugins/ComplexFormer/models/metrics.py`

- `Metrics`；无显式基类

## `hotpot.plugins.ComplexFormer.models.plot_makers`

`hotpot/plugins/ComplexFormer/models/plot_makers.py`

显式 `__all__`: `('plots_options',)`。

- `confusion_matrix(pred: np.ndarray, target: np.ndarray) -> plt.Figure`
- `binary_confusion_metrix(pred: np.ndarray, target: np.ndarray) -> plt.Figure`
- `r2_regression(pred: np.ndarray, target: np.ndarray) -> plt.Figure`
- `roc_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure`
- `det_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure`
- `precision_recall_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure`
- `multiclass_roc_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure`
- `hist(pred: np.ndarray, target: np.ndarray) -> plt.Figure`

## `hotpot.plugins.ComplexFormer.models.predictor`

`hotpot/plugins/ComplexFormer/models/predictor.py`

显式 `__all__`: `('TargetTypeName', 'Predictor', 'apply_lora_to_predictor')`。

- `replace_pyg_linear_in_mlp(mlp_module)`
- `Predictor`；基类 nn.Module
- `apply_lora_to_predictor(predictor_model, target_modules=None, rank=4, alpha=16, dropout=0.3, **kwargs)`

## `hotpot.plugins.ComplexFormer.models.utils`

`hotpot/plugins/ComplexFormer/models/utils.py`

- `norm_binary_to_zero_one(inp_vec: Union[torch.Tensor, np.ndarray]) -> Union[np.ndarray, torch.Tensor]`
- `oh2label(inp_vec: Union[torch.Tensor, np.ndarray])`
- `inverse_onehot(is_onehot, *onehot_vecs: Union[torch.Tensor, np.ndarray])`
- `where_metal(type_labels: Union[torch.Tensor, np.ndarray])`
- `weight_labels(labels: torch.Tensor, num_types: int=119, weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']='cross-entropy') -> torch.Tensor`
- `weight_binary(labels: torch.Tensor, eps: float=1e-07)`
- `get_x_input_attrs(inputs: Union[dict, list, tuple], input_x_index: Union[list, torch.Tensor], dtype: Optional[torch.dtype]=None)`
- `get_labeled_x_input_attrs(inputs, input_x_index: Union[list, torch.Tensor]=None)`
- `get_masked_input_and_labels(inp_vec: torch.Tensor, masked_idx: torch.Tensor, mask_vec: torch.Tensor, inp_atom_labels: torch.Tensor, label_mask: bool=False, to_mask_label: int=0)`
- `mask_atom_type(inputs: tuple, node_mask: Union[int, torch.Tensor])`
- `mask_metal_type(inputs: tuple, node_mask: Union[int, torch.Tensor])`
- `x_masker_func(inputs: tuple, masked_vec: torch.Tensor)`
- `metal_masker_func(inputs: tuple, masked_vec: torch.Tensor)`
- `torch_numpy_exchanger(nf: Callable, **kw)`
- `perturb_xyz(xyz: torch.Tensor, sigma: float=1.0, mode: PerturbMode='uniform')`

## `hotpot.plugins.ComplexFormer.optim_config`

`hotpot/plugins/ComplexFormer/optim_config.py`

- `OptimizerConfigure`；无显式基类

## `hotpot.plugins.ComplexFormer.run`

`hotpot/plugins/ComplexFormer/run/__init__.py`


## `hotpot.plugins.ComplexFormer.run.cv`

`hotpot/plugins/ComplexFormer/run/cv.py`

- `cross_validation(trainer: pl.Trainer, pl_module: LightPretrain, datamodule: DataModule, n_splits: int=5, lora_cfg: LoraConfig=None)`
- `run_cv(log_dir: os.PathLike, run_args: datacls.RunArgs, n_splits: int=5)`

## `hotpot.plugins.ComplexFormer.run.datacls`

`hotpot/plugins/ComplexFormer/run/datacls.py`

显式 `__all__`: `('merge_dataclass', 'RunArgs', 'DataModuleArgs', 'ConfigArgs')`。

- `merge_dataclass(target: T, *sources: Any, exclude_none: bool=False, **kwargs)`
- `RunArgs`；无显式基类
- `DataModuleArgs`；无显式基类
- `ConfigArgs`；无显式基类
- `CallbackConfig`；无显式基类

## `hotpot.plugins.ComplexFormer.run.run`

`hotpot/plugins/ComplexFormer/run/run.py`

- `run(run_args: datacls.RunArgs)`

## `hotpot.plugins.ComplexFormer.run.run_tools`

`hotpot/plugins/ComplexFormer/run/run_tools.py`

- `load_ckpt(work_dir, which: Optional[Union[int, str]]=-1)`
- `load_model_state_dict(model: nn.Module, ckpt: dict, extract_predictor: Optional[str]=None, strict_core_load: bool=True)`
- `init_model(core, task_kwargs: Union[dict, list[dict]], task: Union[tasks.SingleTask, tasks.MultiTask, tasks.MultiDataTask], optim_configure: optim_config.OptimizerConfigure)`
- `init_model_dir(work_dir, work_name, prefix: str='')`
- `prepare_trainer_pl_module(run_args: RunArgs, model_dir, task, core, logger, predictors: dict[str, M.Predictor])`

## `hotpot.plugins.ComplexFormer.run.train`

`hotpot/plugins/ComplexFormer/run/train.py`

- `LightPretrain`；基类 L.LightningModule

## `hotpot.plugins.ComplexFormer.tasks`

`hotpot/plugins/ComplexFormer/tasks/__init__.py`


## `hotpot.plugins.ComplexFormer.tasks.tasks`

`hotpot/plugins/ComplexFormer/tasks/tasks.py`

显式 `__all__`: `('specify_task_types', 'specify_single_dataset_task', 'NaNMetricError', 'BaseTask', 'Task', 'SingleTask', 'MultiTask', 'MultiDataTask')`。

- `add_mpl_font()`
- `specify_single_dataset_task(target_getter: Union[Callable, dict])`
- `specify_task_types(is_multi_datasets: bool, target_getter: Callable)`
- `NaNMetricError`；基类 Exception
- `BaseTask`；基类 ABC
- `Task`；基类 BaseTask, ABC
- `SingleTask`；基类 Task
- `MultiTask`；基类 Task
- `MultiDataTask`；基类 BaseTask

## `hotpot.plugins.ComplexFormer.tools`

`hotpot/plugins/ComplexFormer/tools.py`

- `specify_target_getter(first_data: Data, data_item: str, attrs: str=None, condition_fn: str=None)`
- `TargetGetter`；无显式基类
- `get_index(first_data, data_item: str, attrs: Union[str, Iterable[str]]=None) -> Union[int, list[int]]`

## `hotpot.plugins.ComplexFormer.types`

`hotpot/plugins/ComplexFormer/types.py`


## `hotpot.plugins.DataGen`

`hotpot/plugins/DataGen/__init__.py`


## `hotpot.plugins.DataGen.curve`

`hotpot/plugins/DataGen/curve.py`

- `curve_interpolate(X, Y, method=PchipInterpolator, num: int=10000)`
- `nice_extended_max(x: float) -> float`
- `plot_piecewise_colored_curve(X, Y, segment_bounds=None, color_cycle=None, show_points=False, xi=None, yi=None, point_kwargs=None, curve_kwargs=None, xlabel='x', ylabel='y', title=None, figsize=(10, 7), axis_label_fontsize=14, axis_label_weight='regular', xtick_fontsize=12, ytick_fontsize=16, save_path: str=None)`

## `hotpot.plugins.DataGen.samper_synthesizer`

`hotpot/plugins/DataGen/samper_synthesizer.py`

- `generate_predictions_with_r2(r2, sample_num: int=1000, low: float=0, high: float=1, mu: float=0, sigma: float=1, df: int=1.0, y_true=None, rand_generator='uniform', random_state=None)`

## `hotpot.plugins.PyG`

`hotpot/plugins/PyG/__init__.py`


## `hotpot.plugins.PyG.data`

`hotpot/plugins/PyG/data/__init__.py`


## `hotpot.plugins.PyG.data._base`

`hotpot/plugins/PyG/data/_base.py`

显式 `__all__`: `('BaseDataset',)`。

- `BaseDataset`；基类 ABC

## `hotpot.plugins.PyG.data.ccdc`

`hotpot/plugins/PyG/data/ccdc.py`

- `ComplexDataset`；基类 BaseDataset

## `hotpot.plugins.PyG.data.consts`

`hotpot/plugins/PyG/data/consts.py`

显式 `__all__`: `('edge_attr_names', 'rings_attr_names', 'pair_attr_names', 'num_atom_pair_attr')`。


## `hotpot.plugins.PyG.data.tmqm`

`hotpot/plugins/PyG/data/tmqm.py`

- `tmQmDataset`；基类 BaseDataset

## `hotpot.plugins.PyG.data.utils`

`hotpot/plugins/PyG/data/utils.py`

显式 `__all__`: `('direct_edge_to_indirect', 'extract_atom_attrs', 'extract_bond_attrs', 'extract_atom_pairs', 'extract_ring_attrs', 'merge_individual_data_to_block', 'make_empty_graph', 'mol_to_pyg_data', 'graph_extraction', 'extract_potentials_cbonds')`。

- `make_empty_graph(prefix: str='')`
- `mol_to_pyg_data(mol: Molecule=None, prefix: str='', with_batch: bool=True) -> Data`
- `graph_extraction(mol: Molecule=None, prefix: str='', with_batch: bool=False) -> dict`
- `direct_edge_to_indirect(attr_or_index: torch.Tensor, is_index=True) -> torch.Tensor`
- `extract_atom_attrs(mol: Molecule)`
- `extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str])`
- `extract_atom_pairs(mol: Molecule)`
- `extract_potentials_cbonds(mol: Molecule, catoms_symbols: Optional[Iterable[str]]=None)`
- `extract_ring_attrs(mol: Molecule, ring_attr_names: Iterable[str])`
- `merge_individual_data_to_block(indiv_data_dir, merged_data_dir, bundle_size: int=200000)`

## `hotpot.plugins`

`hotpot/plugins/__init__.py`


## `hotpot.plugins._harness`

`hotpot/plugins/_harness/__init__.py`

显式 `__all__`: `('ArtifactProvenance', 'ExecutableNotFoundError', 'NativeProcessResult', 'ProcessProvenance', 'ProcessRequest', 'ProcessTimeoutError', 'build_process_provenance', 'isolated_workspace', 'resolve_executable', 'run_process', 'sha256_file')`。


导出别名：
- `ArtifactProvenance` → `hotpot.plugins._harness.contracts.ArtifactProvenance`
- `ExecutableNotFoundError` → `hotpot.plugins._harness.executable.ExecutableNotFoundError`
- `NativeProcessResult` → `hotpot.plugins._harness.contracts.NativeProcessResult`
- `ProcessProvenance` → `hotpot.plugins._harness.contracts.ProcessProvenance`
- `ProcessRequest` → `hotpot.plugins._harness.contracts.ProcessRequest`
- `ProcessTimeoutError` → `hotpot.plugins._harness.process.ProcessTimeoutError`
- `build_process_provenance` → `hotpot.plugins._harness.provenance.build_process_provenance`
- `isolated_workspace` → `hotpot.plugins._harness.workspace.isolated_workspace`
- `resolve_executable` → `hotpot.plugins._harness.executable.resolve_executable`
- `run_process` → `hotpot.plugins._harness.process.run_process`
- `sha256_file` → `hotpot.plugins._harness.provenance.sha256_file`

## `hotpot.plugins._harness.contracts`

`hotpot/plugins/_harness/contracts.py`

显式 `__all__`: `('ArtifactProvenance', 'NativeProcessResult', 'ProcessProvenance', 'ProcessRequest')`。

- `ProcessRequest`；无显式基类
- `NativeProcessResult`；无显式基类
- `ArtifactProvenance`；无显式基类
- `ProcessProvenance`；无显式基类

## `hotpot.plugins._harness.executable`

`hotpot/plugins/_harness/executable.py`

显式 `__all__`: `('ExecutableNotFoundError', 'resolve_executable')`。

- `ExecutableNotFoundError`；基类 FileNotFoundError
- `resolve_executable(name: str, *, explicit: Optional[_ExecutablePath]=None, env_var: Optional[str]=None, env: Optional[Mapping[str, str]]=None) -> Path`

## `hotpot.plugins._harness.process`

`hotpot/plugins/_harness/process.py`

显式 `__all__`: `('ProcessTimeoutError', 'run_process')`。

- `ProcessTimeoutError`；基类 TimeoutError
- `run_process(request: ProcessRequest) -> NativeProcessResult`

## `hotpot.plugins._harness.provenance`

`hotpot/plugins/_harness/provenance.py`

显式 `__all__`: `('build_process_provenance', 'sha256_file')`。

- `sha256_file(path: Path) -> str`
- `build_process_provenance(*, executable: Path, result: NativeProcessResult, artifacts: Iterable[Path]) -> ProcessProvenance`

## `hotpot.plugins._harness.workspace`

`hotpot/plugins/_harness/workspace.py`

显式 `__all__`: `('isolated_workspace',)`。

- `isolated_workspace(parent: Optional[Path]=None, *, prefix: str='hotpot-', retain: bool=False) -> Iterator[Path]`

## `hotpot.plugins.ccdc_api`

`hotpot/plugins/ccdc_api/__init__.py`


## `hotpot.plugins.ccdc_api.convert`

`hotpot/plugins/ccdc_api/convert.py`

- `mol_from_ccdc_to_hp(c_mol: CMol)`

## `hotpot.plugins.ccdc_api.miner`

`hotpot/plugins/ccdc_api/miner.py`

- `EntryReader`；无显式基类
- `extract_metal_complexes(output_dir: str, fmt: str='mol2', nproc: int=None, timeout: float=100)`
- `extract_metal_complexes_(output_dir: str, fmt: str='mol2', nproc: int=None, timeout: float=20)`
- `extract_entry_info()`

## `hotpot.plugins.ccdc_api.solvents`

`hotpot/plugins/ccdc_api/solvents.py`

显式 `__all__`: `('solvents', 'solvents_smiles')`。


## `hotpot.plugins.ccdc_api.statistics`

`hotpot/plugins/ccdc_api/statistics.py`

- `get_stem(filename)`
- `get_metals(mol)`
- `ComplexStatistics`；基类 object

## `hotpot.plugins.dl`

`hotpot/plugins/dl/__init__.py`


## `hotpot.plugins.dl.function`

`hotpot/plugins/dl/function/__init__.py`


## `hotpot.plugins.dl.function.base`

`hotpot/plugins/dl/function/base.py`

- `eigen_abs(x: torch.Tensor) -> torch.Tensor`
- `is_symmetric(x, tol: float=5e-07) -> bool`
- `matmul_power(x: torch.Tensor, exp: Union[int, float, torch.Tensor], abs_val: bool=False) -> torch.Tensor`
- `kl_div_with_multivariate_normal(z: torch.Tensor, eps: float=0.01) -> torch.Tensor`
- `get_positional_encoding(max_seq_len: int, embed_dim: int, batch_size: int=None, base: int=None, device: torch.device=None)`

## `hotpot.plugins.dl.function.graph`

`hotpot/plugins/dl/function/graph.py`

- `spectral_similarity(spec1: Union[torch.Tensor, Sequence[torch.Tensor]], spec2: Union[torch.Tensor, Sequence[torch.Tensor]], metric: Union[Literal['min', 'mean'], Callable]=None)`
- `smiles_to_spectrum(smiles: str, length: int=4) -> torch.Tensor`
- `edge_index2adj(note_num: int, edge_index: torch.Tensor) -> torch.Tensor`
- `graph_spectrum(mat: torch.Tensor, descending: bool=False)`
- `adj2laplacian(adj: torch.Tensor, norm=True) -> torch.Tensor`
- `graph_data2spectrum(data: Data, confs_length: int=4, atomic_numbers_col=0) -> torch.Tensor`
- `smiles2spectrum_data(smiles: str, length: int=4, atomic_numbers_col: int=0) -> Data`
- `graph_spectrum_similarity(spec1: torch.Tensor, spec2: torch.Tensor, metric: Union[None, Literal['mean', 'mix'], Callable]='min') -> torch.Tensor`
- `batch_spectrum(batch_matrix: torch.Tensor, batch_atom_num: torch.Tensor)`
- `tensor_sort(inputs: torch.Tensor, key: Callable=None, dim: int=None, descending: bool=True)`

## `hotpot.plugins.dl.generate`

`hotpot/plugins/dl/generate/__init__.py`


## `hotpot.plugins.dl.generate.vae`

`hotpot/plugins/dl/generate/vae.py`

- `VAE`；基类 nn.Module
- `GraphEncoder`；基类 nn.Module
- `NodeNumDecoder`；基类 nn.Module
- `MatrixDecoder`；基类 nn.Module
- `GraphDecoder`；基类 nn.Module
- `KernelLinear`；基类 nn.Module
- `MultiKernelMLP`；基类 nn.Module
- `SpectrumLoss`；基类 nn.Module
- `spectrum_loss(pred_spectrum: torch.Tensor, target_spectrum: torch.Tensor, metric: Union[Literal['min', 'mean'], Callable]='min', to_distance: bool=False, eps: float=1e-06) -> torch.Tensor`
- `graph_vae_loss(true_atom_num: torch.Tensor, pred_atom_num: torch.Tensor, true_spectrum: torch.Tensor, pred_spectrum: torch.Tensor, latent_vector: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor)`

## `hotpot.plugins.dl.graph`

`hotpot/plugins/dl/graph/__init__.py`


## `hotpot.plugins.dl.graph.gen_decoder`

`hotpot/plugins/dl/graph/gen_decoder.py`

- `NodeEdgeDecoder`；基类 nn.Module

## `hotpot.plugins.dl.pytorch_func`

`hotpot/plugins/dl/pytorch_func.py`

- `TensorContainer`；无显式基类
- `train(model, dataloader, loss_fn, opti_cls, learning_rate: float, epoch_num: int, device=torch.device('cpu'), data_organizer: Callable=None, loss_organizer: Callable=None, valid_func: Callable=None, valid_per_batch: int=100)`
- `validate(item_num: int, model, batch, valid_func: Callable, device: torch, data_organizer: Callable=None)`
- `show_gpu_memory_utils(device=torch.device('cuda'))`
- `get_model_size(module)`
- `loss_organizer(target, out)`
- `to_valid(target, out)`

## `hotpot.plugins.ml`

`hotpot/plugins/ml/__init__.py`


## `hotpot.plugins.ml.features`

`hotpot/plugins/ml/features/__init__.py`


## `hotpot.plugins.ml.features._features`

`hotpot/plugins/ml/features/_features.py`

- `get_rdkit_mol_descriptor(list_smi, mol_names=None) -> pd.DataFrame`

## `hotpot.plugins.ml.features.elements`

`hotpot/plugins/ml/features/elements/__init__.py`

显式 `__all__`: `('element_features',)`。

- `ElementFeature`；无显式基类
- `determine_elements_block(atom_numbers: pd.Series)`

## `hotpot.plugins.ml.features.elements.elements`

`hotpot/plugins/ml/features/elements/elements.py`

显式 `__all__`: `('element_features',)`。

- `ElementFeature`；无显式基类
- `determine_elements_block(atom_numbers: pd.Series)`

## `hotpot.plugins.ml.features.sol_media`

`hotpot/plugins/ml/features/sol_media/__init__.py`

显式 `__all__`: `('get_all_media_names', 'to_media_features')`。

- `get_all_media_names() -> list[str]`
- `to_media_features(lst_media, fill_miss: Union[list[str], bool]=True) -> pd.DataFrame`

## `hotpot.plugins.ml.features.solvents`

`hotpot/plugins/ml/features/solvents/__init__.py`

显式 `__all__`: `('get_all_solvents_names', 'to_solvent_features')`。

- `get_all_solvents_names() -> list[str]`
- `to_solvent_features(lst_sols: list, fill_miss: Union[list, bool]=False) -> pd.DataFrame`

## `hotpot.plugins.ml.utils`

`hotpot/plugins/ml/utils.py`

- `scatter_discriminant_analysis(X: np.ndarray, y: np.ndarray, split_points: Union[np.ndarray, Sequence]=None)`
- `plot_scatter_discriminant_lines(X, y, lines, dim_method=TSNE(n_components=2))`

## `hotpot.plugins.ml.wf`

`hotpot/plugins/ml/wf.py`

- `expectation_improvement(y, mu, sigma)`
- `get_rdkit_mol_descriptor(list_smi) -> pd.DataFrame`
- `LinearAddLightGBM`；基类 BaseEstimator
- `CrossValidation`；无显式基类
- `MachineLearning`；无显式基类
- `MachineLearning_`；无显式基类
- `linear_leave_one_out_analysis(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray)`

## `hotpot.plugins.opti`

`hotpot/plugins/opti/__init__.py`


## `hotpot.plugins.opti.beyes`

`hotpot/plugins/opti/beyes/__init__.py`


## `hotpot.plugins.opti.beyes.opti`

`hotpot/plugins/opti/beyes/opti.py`

显式 `__all__`: `('next_params', 'draw_comics_map')`。

- `generate_power_ladder(index: int, batch_size: int=DEFAULT_BATCH_SIZE, max_epsilon: float=DEFAULT_ACQ_EPS_MAX, power: float=DEFAULT_ACQ_POWER) -> float`
- `AcquisitionFunction`；无显式基类
- `GaussianProcess`；基类 gpytorch.models.ExactGP
- `BayesianOptimizer`；无显式基类
- `ParamPreprocessor`；无显式基类
- `preprocess_inputs(X: ArrayLike, y: ArrayLike, param_range: np.ndarray, param_names: List[str], log_indices: Optional[Union[int, List[int]]], mesh_counts: int) -> Tuple[ParamPreprocessor, np.ndarray, np.ndarray, torch.Tensor]`
- `run_bayesian_optimization(X: ArrayLike, y: ArrayLike, design_points: ArrayLike, batch_size: int=DEFAULT_BATCH_SIZE) -> Tuple[BayesianOptimizer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
- `next_params(X: ArrayLike, y: ArrayLike, param_range: np.ndarray, param_names: List[str], next_param_path: Union[str, os.PathLike], mesh_counts: int=DEFAULT_MESH_COUNTS, figpath: Optional[Union[str, os.PathLike]]=None, log_indices: Optional[Union[int, List[int]]]=None, to_coutourf: bool=True, cmap: str='Greys') -> None`
- `draw_comics_map(X: ArrayLike, y: ArrayLike, init_index: int, batch_size: int, param_range: Optional[np.ndarray]=None, param_names: Optional[Sequence[str]]=None, mesh_counts: int=DEFAULT_MESH_COUNTS, log_indices: Optional[Union[int, List[int]]]=None, figpath_dir: Optional[Union[str, os.PathLike]]=None, emb_method: Union[TSNE, MDS]=TSNE(), to_coutourf: bool=True, cmap: str='Greys') -> None`

## `hotpot.plugins.opti.ev`

`hotpot/plugins/opti/ev/__init__.py`


## `hotpot.plugins.opti.ev.core`

`hotpot/plugins/opti/ev/core.py`

- `get_params(param_name: str, *objs)`
- `UniversalParams`；无显式基类
- `GeneLibrary`；无显式基类
- `Individual`；无显式基类
- `Gene`；无显式基类
- `GeneSeries`；无显式基类
- `GeneSpace`；无显式基类
- `DiscreteGenSpace`；基类 GeneSpace
- `ContinuousGenSpace`；基类 GeneSpace
- `formalize_params_space(name, space: Sequence)`
- `Species`；无显式基类
- `Earth`；无显式基类
- `EvoluteOptimizer`；无显式基类
- `ml_optimizer(models: dict[str, type(ModelLike)], hypers: dict[str, Sequence], X: np.ndarray, y: np.ndarray, X_test: np.ndarray=None, y_test: np.ndarray=None, features: list[str]=None, fitness_function: Callable=None, initial_size: Union[int, list[int]]=3, *, feature_engineering: bool=False, fe_kwargs: dict[str, Any]=None, which_return: Union[Literal['best', 'top'], int]='best', **kwargs)`
- `MachineLearningOptimizer`；无显式基类
- `MLOptimizer`；无显式基类

## `hotpot.plugins.opti.ev.func`

`hotpot/plugins/opti/ev/func.py`

- `LivingFunc`；无显式基类
- `MachineLearningR2Fitness`；无显式基类

## `hotpot.plugins.opti.exp_design`

`hotpot/plugins/opti/exp_design/__init__.py`


## `hotpot.plugins.opti.exp_design.core`

`hotpot/plugins/opti/exp_design/core.py`

- `orthogonal_design(seq_of_params: Sequence[Sequence], num_exper: int)`

## `hotpot.plugins.opti.params_space`

`hotpot/plugins/opti/params_space.py`

显式 `__all__`: `('ParamSpace', 'optuna_optimize', 'ParamSets')`。

- `CategoricalParams`；无显式基类
- `DiscreteUniformParams`；无显式基类
- `FloatParams`；无显式基类
- `IntParams`；无显式基类
- `ParamSpace`；无显式基类
- `optuna_optimize(response: Callable[[dict], float], space: ParamSpace, n_trials: int=100, sampler: optuna.samplers.BaseSampler=optuna.samplers.GPSampler, return_study: bool=False)`
- `ParamSets`；无显式基类

## `hotpot.plugins.plots`

`hotpot/plugins/plots/__init__.py`

- `refresh_cache()`
- `init_fonts()`

## `hotpot.plugins.plots.base`

`hotpot/plugins/plots/base.py`

- `ImmutableDict`；基类 dict
- `Plotter`；无显式基类
- `Plot`；基类 ABC
- `FigAxesModifier`；基类 ABC

## `hotpot.plugins.plots.colors`

`hotpot/plugins/plots/colors.py`

显式 `__all__`: `('create_saturation_colormap', 'register_cmap_to_hotpot', 'load_defined_cmap')`。

- `create_saturation_colormap(name: str, color_rgb_names: Union[str, tuple[int, int, int]])`
- `register_cmap_to_hotpot(cmap: mcolors.Colormap, name: Optional[str]=True)`
- `get_defined_cmap_name()`
- `load_defined_cmap(cmap_name: str, cmap_type: Literal['linear', 'list']='linear')`
- `load_cmap(cmap_name: str, cmap_type: Literal['linear', 'list']='linear')`

## `hotpot.plugins.plots.defaults`

`hotpot/plugins/plots/defaults.py`

- `Settings`；无显式基类

## `hotpot.plugins.plots.plots`

`hotpot/plugins/plots/plots/__init__.py`


## `hotpot.plugins.plots.plots.base`

`hotpot/plugins/plots/plots/base.py`

显式 `__all__`: `('Plot',)`。

- `Plot`；基类 ABC

## `hotpot.plugins.plots.plots.beyes`

`hotpot/plugins/plots/plots/beyes.py`

显式 `__all__`: `('BayesDesignSpaceMap',)`。

- `BayesDesignSpaceMap`；基类 SciPlotter

## `hotpot.plugins.plots.plots.general`

`hotpot/plugins/plots/plots/general.py`

显式 `__all__`: `('Hist',)`。

- `Hist`；基类 Plot

## `hotpot.plugins.plots.plots.ml`

`hotpot/plugins/plots/plots/ml.py`

显式 `__all__`: `('R2Regression', 'FeatureImportance', 'HierarchicalTree', 'PearsonMatrix', 'Pearson', 'SciPlotter', 'EmbeddingDataTo2dMap', 'ConfusionMatrix', 'ROCCurve', 'DETCurve', 'PrecisionRecallCurve', 'MultiClassROCCurve', 'SHAPlot')`。

- `ConfusionMatrix`；基类 Plot
- `ROCCurve`；基类 Plot
- `DETCurve`；基类 Plot
- `PrecisionRecallCurve`；基类 Plot
- `MultiClassROCCurve`；基类 Plot
- `R2Regression`；基类 Plot
- `FeatureImportance`；无显式基类
- `HierarchicalTree`；基类 Plot
- `Pearson`；基类 Plot
- `PearsonMatrix`；基类 Plot
- `SHAPairPlot(exp: shap.Explanation, *args, **kwargs)`
- `SHAPlot`；基类 Plot
- `EmbeddingDataTo2dMap`；基类 Plot

导出别名：
- `SciPlotter` → `hotpot.plugins.plots.plotter.SciPlotter`

## `hotpot.plugins.plots.plots.perodotable`

`hotpot/plugins/plots/plots/perodotable.py`

- `PeriodicTable`；无显式基类

## `hotpot.plugins.plots.plotter`

`hotpot/plugins/plots/plotter.py`

- `axes_set_method(method: Callable)`
- `SciPlotter`；基类 Plotter
- `LegendOrganizer`；无显式基类

## `hotpot.plugins.plots.utils`

`hotpot/plugins/plots/utils.py`

- `calc_active_span(nrows, ncols, i, j)`
- `abs2rela_pos(ax: plt.Axes, left: float, bottom: float, width: float, height: float, rows_interval: float=None, cols_interval: float=None)`
- `insert_axes(main_ax, ins_ax, relative_pos: Sequence[float]=None)`
- `specify_colorbar_pos(main_ax: plt.Axes, orientation: Literal['horizontal', 'vertical']='vertical')`
- `add_colorbar(ax: plt.Axes, mappable=None, value_lim: Sequence=None, colorbar_label: str=None, colorbar_pos: Sequence[float]=None, orientation: Literal['horizontal', 'vertical']='vertical', norm: mpl.colors.Normalize=None, cmap: Union[str, mpl.colors.Colormap]=None, **kwargs)`
- `add_text(ax, xfrac: float, yfrac: float, s: str, fontdict: dict=None)`
- `sample_density(a, b, ratio=0.4)`
- `later_call(func: Callable)`
- `calculate_scatter_density(xy: np.ndarray)`
- `scatter_to_coutourf(x, y, c, mesh_num=50)`

## `hotpot.plugins.pubchem`

`hotpot/plugins/pubchem/__init__.py`


## `hotpot.plugins.pubchem.get_data`

`hotpot/plugins/pubchem/get_data.py`

- `download_pubchem_sdf_files(output_dir, base_url='https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/')`
- `download_file(url, dest_path)`

## `hotpot.plugins.pubchem.utils`

`hotpot/plugins/pubchem/utils.py`

- `get_compounds_cid(identifier: str, id_type: str='name')`
- `async_get_compounds_cid(identifier: str, id_type: str='name')`

## `hotpot.plugins.qm`

`hotpot/plugins/qm/__init__.py`


## `hotpot.plugins.qm.gaussian`

`hotpot/plugins/qm/gaussian/__init__.py`


## `hotpot.plugins.qm.gaussian._works`

`hotpot/plugins/qm/gaussian/_works.py`

- `run_gaussian(mol: Molecule, link0: str=None, route: str=None, g16root: Union[str, PathLike]=None, gjf_save_path: Union[str, PathLike]=None, log_save_path: Union[str, PathLike]=None, err_save_path: Union[str, PathLike]=None, report_set_resource_error: bool=False, options: Options=None, test: bool=True, **kwargs)`
- `export_results(*log_file_path: Union[str, PathLike], skip_errors: bool=True, retrieve_mol: bool=True, nproc: int=None, timeout: float=None)`
- `parse_gjf(path_gjf: Union[str, Path])`
- `reorganize_gjf(parsed_gjf: dict)`
- `ladder_opti(mol: Molecule, ladder: list[str], *args, **kwargs)`
- `update_gjf_coordinates(old_gjf_file: Union[str, PathLike], log_file: Union[str, PathLike])`
- `update_gjf(old_gjf_file: Union[str, PathLike], update_dict: dict)`
- `parse_route(route: str) -> dict`
- `ResultsExtract`；无显式基类

## `hotpot.plugins.qm.gaussian.gauss`

`hotpot/plugins/qm/gaussian/gauss.py`

- `extract_times_from_gaussian_log(log_content)`
- `OptionPath`；无显式基类
- `Options`；无显式基类
- `Gaussian`；无显式基类
- `GaussOut`；无显式基类

## `hotpot.plugins.zeoxx`

`hotpot/plugins/zeoxx/__init__.py`


## `hotpot.plugins.zeoxx.core`

`hotpot/plugins/zeoxx/core.py`

- `ZeoppRunner`；无显式基类
- `concat_results(results: Dict[str, pd.DataFrame], tasks: Optional[Sequence[str]]=None, keep_units: bool=True, chan_mode: str='expand', chan_agg: str='max', include_context: bool=False, add_task_suffix_on_collision: bool=True, sort_rows: bool=True) -> pd.DataFrame`

## `hotpot.utils`

`hotpot/utils/__init__.py`


## `hotpot.utils.chem`

`hotpot/utils/chem/__init__.py`


## `hotpot.utils.chem.atom`

`hotpot/utils/chem/atom.py`

- `calc_electron_config(_atomic_number)`

## `hotpot.utils.configs`

`hotpot/utils/configs/__init__.py`


## `hotpot.utils.configs.logging_config`

`hotpot/utils/configs/logging_config.py`

显式 `__all__`: `('setup_logging', 'RateLimitLogger', 'LoggerDict')`。

- `setup_logging(debug=False, to_stdout=True)`
- `RateLimitLogger`；无显式基类
- `LoggerDict`；基类 defaultdict

## `hotpot.utils.fmt_print`

`hotpot/utils/fmt_print.py`

- `hex_to_ansi(hex_color: str, style: PrintStyle='regular') -> str`
- `FmtString`；无显式基类
- `FmtPrint`；无显式基类
- `dict_to_df(metrics_dict: dict[str, Union[float, dict[str, float]]])`
- `dict_to_table(metrics_dict: dict[str, Union[float, dict[str, float]]], table_kw: dict=None, title: str=None)`
- `export_table(table: Table, save_path: str, width=200)`
- `rich_print(contents, **kwargs)`

## `hotpot.utils.mp`

`hotpot/utils/mp.py`

- `decorator(func: Callable) -> Callable`
- `mp_run(func: Callable, args: Iterable[tuple], kwargs: Iterable[dict]=None, nproc: int=None, desc: str='', timeout: Optional[float]=None, error_to_None: bool=True, lazy_iter: bool=False, early_exit_condition: Optional[Callable[[dict, int], bool]]=None, branch_jobs: Iterable[Callable[[dict, dict, int], None]]=None)`
- `Pool`；无显式基类

## `hotpot.utils.tools`

`hotpot/utils/tools.py`

- `show_time(func, *args, **kwargs)`
- `show_func_time(func: Callable)`

## `hotpot.utils.types`

`hotpot/utils/types.py`

- `ModelLike`；基类 Protocol
