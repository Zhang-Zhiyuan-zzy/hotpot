# 当前 CLI 静态入口与参数声明

静态读取 `hotpot/__main__.py` 与命令模块中的 `add_argument/add_parser` 调用；没有执行 `--help`、模型或引擎。动态循环展开/条件默认值以原始调用式保留，故本表不替代运行环境中的 argparse 帮助。

顶级入口 `hotpot` / `python -m hotpot`。
命令：`convert`, `optimize`, `ml_train`, `mca`, `cbond`, `models`, `ff`, `xtb`, `run`, `ds`。
内置 pipeline stage 为 `cbond`, `ff`, `xtb`；MCA 为独立 inference/CLI 入口，并未注册为当前内置 stage。

## `hotpot.__main__`

```python
parser.add_argument('-d', '--debug', action='store_true', help='debug mode')  # hotpot/__main__.py:98
parser.add_argument('-b', '--background', action='store_true', help="run command in background; use together with '&' or nohup")  # hotpot/__main__.py:99
parser.add_argument('-v', '--version', action='store_true', dest='show_version')  # hotpot/__main__.py:105
parser.add_subparsers(title='works', dest='works')  # hotpot/__main__.py:107
works.add_parser(command, help=spec.help)  # hotpot/__main__.py:109
```

## `hotpot.cheminfo.operations.forcefields.stage`

```python
geometry_mode.add_argument('--rebuild', action='store_true')  # hotpot/cheminfo/operations/forcefields/stage.py:324
geometry_mode.add_argument('--optimize-only', action='store_true')  # hotpot/cheminfo/operations/forcefields/stage.py:325
parser.add_argument('--route', choices=('auto', 'organic', 'complex'), default='auto')  # hotpot/cheminfo/operations/forcefields/stage.py:326
parser.add_argument('--forcefield', choices=tuple(_FORCEFIELDS), default='auto')  # hotpot/cheminfo/operations/forcefields/stage.py:331
parser.add_argument('--algorithm', choices=('conjugate', 'steepest'), default='conjugate')  # hotpot/cheminfo/operations/forcefields/stage.py:336
parser.add_argument('--epochs', type=_positive_int, default=100)  # hotpot/cheminfo/operations/forcefields/stage.py:341
parser.add_argument('--steps-per-epoch', type=_positive_int, default=100)  # hotpot/cheminfo/operations/forcefields/stage.py:342
parser.add_argument('--convergence-level', choices=tuple(_CONVERGENCE_LEVELS), default='fast')  # hotpot/cheminfo/operations/forcefields/stage.py:343
parser.add_argument('--no-add-hydrogens', dest='add_hydrogens', action='store_false')  # hotpot/cheminfo/operations/forcefields/stage.py:348
parser.add_argument('--quality', choices=('off', 'basic', 'standard', 'strict'), default='standard')  # hotpot/cheminfo/operations/forcefields/stage.py:354
parser.add_argument('--seed', type=int)  # hotpot/cheminfo/operations/forcefields/stage.py:359
parser.add_argument('--timeout', type=_positive_float, default=1000.0)  # hotpot/cheminfo/operations/forcefields/stage.py:360
parser.add_argument('--trajectory', action='store_true')  # hotpot/cheminfo/operations/forcefields/stage.py:361
parser.add_argument('--trajectory-start', choices=tuple(_TRAJECTORY_STARTS))  # hotpot/cheminfo/operations/forcefields/stage.py:362
parser.add_argument('--sample-break-on', action='append', choices=_FF_CODES, default=[])  # hotpot/cheminfo/operations/forcefields/stage.py:366
parser.add_argument('--batch-break-on', action='append', choices=_FF_CODES, default=[])  # hotpot/cheminfo/operations/forcefields/stage.py:367
```

## `hotpot.cheminfo.operations.inference.cbond.stage`

```python
parser.add_argument('metal')  # hotpot/cheminfo/operations/inference/cbond/stage.py:139
parser.add_argument('ligand')  # hotpot/cheminfo/operations/inference/cbond/stage.py:140
parser.add_argument('--input-format')  # hotpot/cheminfo/operations/inference/cbond/stage.py:141
parser.add_argument('--threshold', type=float)  # hotpot/cheminfo/operations/inference/cbond/stage.py:142
parser.add_argument('--first-threshold', type=float)  # hotpot/cheminfo/operations/inference/cbond/stage.py:143
parser.add_argument('--no-greedy', dest='greedy', action='store_false')  # hotpot/cheminfo/operations/inference/cbond/stage.py:144
parser.add_argument('--all-structures', action='store_true')  # hotpot/cheminfo/operations/inference/cbond/stage.py:146
parser.add_argument('--max-states', type=_positive_int, default=DEFAULT_MAX_STATES)  # hotpot/cheminfo/operations/inference/cbond/stage.py:147
parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')  # hotpot/cheminfo/operations/inference/cbond/stage.py:148
parser.add_argument('--model-dir')  # hotpot/cheminfo/operations/inference/cbond/stage.py:153
parser.add_argument('--model-source', choices=('auto', 'local', 'hub'))  # hotpot/cheminfo/operations/inference/cbond/stage.py:154
```

## `hotpot.commands.cbond`

```python
parser.add_argument('--doc', action=MarkdownDocumentationAction, nargs=0, document_loader=load_cli_documentation, help='show detailed Markdown usage documentation and exit')  # hotpot/commands/cbond.py:27
parser.add_argument('metal', help='metal element symbol or atomic number, such as Eu or 63')  # hotpot/commands/cbond.py:34
parser.add_argument('ligand', metavar='LIGAND_SMILES/FILE', help='ligand SMILES string or molecule file')  # hotpot/commands/cbond.py:38
parser.add_argument('-o', '--output', help='write the result to this file instead of standard output')  # hotpot/commands/cbond.py:43
parser.add_argument('--input-format', help='input format override, such as smi, sdf, mol2, or cif')  # hotpot/commands/cbond.py:48
parser.add_argument('--threshold', type=float, help=f'minimum raw CBond score for later bonds; when explicitly supplied without --first-threshold, also applies to the first bond (later-bond default: {DEFAULT_SUBSEQUENT_CBOND_THRESHOLD})')  # hotpot/commands/cbond.py:52
parser.add_argument('--first-threshold', type=float, help=f'minimum raw CBond score for the first bond (default: {DEFAULT_FIRST_CBOND_THRESHOLD})')  # hotpot/commands/cbond.py:61
parser.add_argument('--no-greedy', dest='greedy', action='store_false', help='stop when the highest-scoring candidate is already connected')  # hotpot/commands/cbond.py:69
parser.add_argument('--all-structures', action='store_true', help='enumerate and rank every terminal coordination structure')  # hotpot/commands/cbond.py:75
parser.add_argument('--bond-detail', action='store_true', help='show atom indices, elements, and raw scores for selected bonds')  # hotpot/commands/cbond.py:80
parser.add_argument('--max-states', type=int, default=DEFAULT_MAX_STATES, help=f'maximum unique states explored by --all-structures (default: {DEFAULT_MAX_STATES})')  # hotpot/commands/cbond.py:85
parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='inference device (default: auto)')  # hotpot/commands/cbond.py:94
parser.add_argument('--model-dir', help='directory containing the CBond ONNX release')  # hotpot/commands/cbond.py:100
parser.add_argument('--model-source', choices=('auto', 'local', 'hub'), help='model source policy (default: HOTPOT_MODEL_SOURCE or auto)')  # hotpot/commands/cbond.py:104
```

## `hotpot.commands.convert`

```python
parser.add_argument('infile', type=str, help='input file')  # hotpot/commands/convert.py:28
parser.add_argument('-f', '--output_file', type=str, help='output file or directory')  # hotpot/commands/convert.py:29
parser.add_argument('-i', '--inputs-format', type=str, help='input format')  # hotpot/commands/convert.py:35
parser.add_argument('-o', '--output-format', type=str, help='output format')  # hotpot/commands/convert.py:36
```

## `hotpot.commands.ff`

```python
parser.add_argument('--doc', action=MarkdownDocumentationAction, nargs=0, document_loader=load_cli_documentation, help='show detailed Markdown usage documentation and exit')  # hotpot/commands/ff.py:144
parser.add_argument('inputs', nargs='+', metavar='SMILES/FILE', help="one or more SMILES strings, molecule files, or '-' for stdin")  # hotpot/commands/ff.py:151
parser.add_argument('-o', '--output', help="write the molecular payload to FILE instead of stdout; '-' means stdout")  # hotpot/commands/ff.py:157
parser.add_argument('--input-format', help='input format override, such as smi, sdf, or mol2')  # hotpot/commands/ff.py:162
parser.add_argument('--output-format', help='output molecule format (default: output suffix, otherwise mol2)')  # hotpot/commands/ff.py:166
geometry_mode.add_argument('--rebuild', action='store_true', help='discard existing coordinates and build a new 3D starting geometry')  # hotpot/commands/ff.py:171
geometry_mode.add_argument('--optimize-only', action='store_true', help='optimize existing 3D coordinates and fail when they are absent')  # hotpot/commands/ff.py:176
parser.add_argument('--route', choices=('auto', 'organic', 'complex'), default='auto', help='force-field workflow route (default: auto from molecular topology)')  # hotpot/commands/ff.py:181
parser.add_argument('--forcefield', choices=tuple(_FORCEFIELDS), default='auto', help='force-field backend (default: auto; complexes currently require UFF)')  # hotpot/commands/ff.py:187
parser.add_argument('--algorithm', choices=('conjugate', 'steepest'), default='conjugate', help='Open Babel optimization algorithm (default: conjugate)')  # hotpot/commands/ff.py:193
parser.add_argument('--epochs', type=_positive_int, default=100, help='optimization epochs (default: 100)')  # hotpot/commands/ff.py:199
parser.add_argument('--steps-per-epoch', type=_positive_int, default=100, help='Open Babel steps submitted per epoch (default: 100)')  # hotpot/commands/ff.py:205
parser.add_argument('--convergence-level', choices=tuple(_CONVERGENCE_LEVELS), default='fast', help='optimizer convergence evidence level (default: fast)')  # hotpot/commands/ff.py:211
parser.add_argument('--no-add-hydrogens', dest='add_hydrogens', action='store_false', help='do not add missing hydrogens before building or optimization')  # hotpot/commands/ff.py:217
parser.add_argument('--quality', choices=('off', 'basic', 'standard', 'strict'), default='standard', help='terminal structure quality gate (default: standard)')  # hotpot/commands/ff.py:224
parser.add_argument('--seed', type=int, help='seed Open Babel building and Hotpot perturbations')  # hotpot/commands/ff.py:230
parser.add_argument('--timeout', type=_positive_float, default=1000.0, help='coordinate-build timeout in seconds (default: 1000)')  # hotpot/commands/ff.py:235
parser.add_argument('--trajectory', metavar='DIRECTORY', help='persist the complete topology-aware force-field trajectory')  # hotpot/commands/ff.py:241
parser.add_argument('--trajectory-start', choices=tuple(_TRAJECTORY_STARTS), help='earliest trajectory stage to retain (default: workflow-specific)')  # hotpot/commands/ff.py:246
parser.add_argument('--report', metavar='FILE', help='write a JSON execution and scientific-quality report')  # hotpot/commands/ff.py:251
parser.add_argument('--jobs', type=_positive_int, default=1, help='independent molecule worker processes (default: 1)')  # hotpot/commands/ff.py:256
parser.add_argument('--overwrite', action='store_true', help='replace existing output, report, and trajectory paths')  # hotpot/commands/ff.py:262
```

## `hotpot.commands.mca`

```python
parser.add_argument('--doc', action=MarkdownDocumentationAction, nargs=0, document_loader=load_cli_documentation, help='show detailed Markdown usage documentation and exit')  # hotpot/commands/mca.py:34
parser.add_argument('inputs', nargs='+', metavar='SMILES/FILE', help='one or more SMILES strings or molecule files')  # hotpot/commands/mca.py:41
parser.add_argument('-o', '--output', help='write the text table to this file instead of standard output')  # hotpot/commands/mca.py:47
parser.add_argument('--plot', metavar='IMAGE', help='draw MCA-coloured atoms to an image file')  # hotpot/commands/mca.py:52
parser.add_argument('--all-site', action='store_true', help='colour every atom in the plot (default: detected nucleophilic sites only)')  # hotpot/commands/mca.py:57
parser.add_argument('--input-format', help='input format override, such as smi, sdf, mol2, or cif')  # hotpot/commands/mca.py:62
parser.add_argument('--model-dir', help='directory containing the MCA ONNX release')  # hotpot/commands/mca.py:66
parser.add_argument('--model-source', choices=('auto', 'local', 'hub'), help='model source policy (default: HOTPOT_MODEL_SOURCE or auto)')  # hotpot/commands/mca.py:67
parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu', help='inference backend (default: cpu)')  # hotpot/commands/mca.py:72
parser.add_argument('--variant', choices=('fp32', 'fp16'))  # hotpot/commands/mca.py:78
parser.add_argument('--batch-size', type=int, help='CPU atom-row batch size (default: 64)')  # hotpot/commands/mca.py:79
parser.add_argument('--cpu-count', type=int, help='total CPUs shared by CPU preprocessing and ONNX (default: up to 8)')  # hotpot/commands/mca.py:84
gpu_selection.add_argument('--gpu-count', type=int, help='use logical GPUs 0 through N-1')  # hotpot/commands/mca.py:90
gpu_selection.add_argument('--gpu-indices', type=lambda value: tuple((int(index) for index in value.split(','))), help='comma-separated logical GPU indices, such as 0,2')  # hotpot/commands/mca.py:95
parser.add_argument('--site-cpu-per-gpu', type=int)  # hotpot/commands/mca.py:100
parser.add_argument('--infer-cpu-per-gpu', type=int)  # hotpot/commands/mca.py:101
parser.add_argument('--gpu-batch-size', type=int, choices=SUPPORTED_GPU_BATCH_SIZES, help='override the total-VRAM-derived GPU atom-row batch')  # hotpot/commands/mca.py:102
parser.add_argument('--conformer-seed', type=int, default=DEFAULT_CONFORMER_SEED)  # hotpot/commands/mca.py:108
parser.add_argument('--allow-charged', action='store_true', help='allow out-of-domain predictions for charged molecules')  # hotpot/commands/mca.py:113
```

## `hotpot.commands.ml_train`

```python
parser.add_argument('data_file', type=str, help='The ChemData to train the ML model, given by excel file')  # hotpot/commands/ml_train.py:48
parser.add_argument('-f', '--features', type=str, help='A list of features name or feature indices (with -i flag)the features are separated by comma ",", such as feature1,feature2,feature3 By default, all columns excluding the last would be treat as the features.')  # hotpot/commands/ml_train.py:49
parser.add_argument('-ef', '--exclude-features', type=str, help='A list of feature names or feature indices, excluding from selected list the features are separated by comma ",", such as feature1,feature2,feature3 ')  # hotpot/commands/ml_train.py:57
parser.add_argument('-t', '--target', type=str, help='The target column name or feature indices in the ChemData excel file')  # hotpot/commands/ml_train.py:64
parser.add_argument('-i', '--by-index', help='Specify the params by their index in the input excel', action='store_true')  # hotpot/commands/ml_train.py:70
parser.add_argument('-a', '--algorithm', type=str, help='Which algorithm to implement to build the model, defaults to XGBoost', choices=['SVM', 'RF', 'Ridge', 'GBDT', 'XGBoost', 'DecisionTree', 'LeastSquares', 'KNN', 'AdaBoost'], default='XGBoost')  # hotpot/commands/ml_train.py:76
parser.add_argument('-d', '--work-dir', type=str, help='The working directory to save the results, defaults to CURRENT_DIR/work_dir')  # hotpot/commands/ml_train.py:83
parser.add_argument('-m', '--method', type=str, help='Which method to train the model, "Regression(R)", "Classification(R)", "Cluster", or ""', default='Regression')  # hotpot/commands/ml_train.py:88
parser.add_argument('--hyper', type=str, help='Path to a json file with hyperparameters')  # hotpot/commands/ml_train.py:94
parser.add_argument('--examples', type=str, help='Run the examples', choices=['logK_LnAn', 'SF_LnAn'])  # hotpot/commands/ml_train.py:99
```

## `hotpot.commands.models`

```python
parser.add_subparsers(dest='models_action', required=True)  # hotpot/commands/models.py:14
actions.add_parser('list', help='list pinned model artifacts')  # hotpot/commands/models.py:15
actions.add_parser(action, help=help_text)  # hotpot/commands/models.py:22
action_parser.add_argument('models', nargs='*', metavar='MODEL', help='model name; all models are selected when omitted')  # hotpot/commands/models.py:23
action_parser.add_argument('--all', action='store_true', help='select every registered model')  # hotpot/commands/models.py:29
```

## `hotpot.commands.optimize`

```python
optimize_parser.add_argument('excel_file', type=str, help='the path of excel file')  # hotpot/commands/optimize.py:193
optimize_parser.add_argument('result_dir', type=str, help='the directory to save the result files')  # hotpot/commands/optimize.py:194
opti_base.add_argument('-m', '--method', action='store', default='Bayesian', help="The method to perform parameters optimizer, choose from ['Bayesian(B)', 'Evolution(E)']")  # hotpot/commands/optimize.py:201
opti_base.add_argument('-p', '--params', help='Which items is the parameters you want to optimize, the default is all columns in the input sheet,excluding of the last column. The items can be specified by a range of item using a connection "-", or individual items separated by a comma ",", for example:\n    python hotpot optimize excel_file result_dir -p item1-item2\n    python hotpot optimize excel_file result_dir -p item0,item1,item2,item3\n    python hotpot optimize excel_file result_dir -p item0,item1-item2,item3')  # hotpot/commands/optimize.py:207
opti_base.add_argument('-ep', '--exclude-params', help='Exclude params in the input sheet columns from the optimization')  # hotpot/commands/optimize.py:216
opti_base.add_argument('-i', '--by-index', help='Specify the params by their index in the input excel', action='store_true')  # hotpot/commands/optimize.py:220
opti_base.add_argument('-t', '--target', help='Which are your optimizing target or indicator in the input sheet, the default is the last column.')  # hotpot/commands/optimize.py:226
opti_base.add_argument('--plot', action='store_true', help='Plot the 2D manifold comics to show the mu and sigma charges')  # hotpot/commands/optimize.py:231
opti_base.add_argument('--mesh', default=10, help='The design parameter space will be weaved, This argument specify the mesh number for each parameter, the default is the 20', type=int)  # hotpot/commands/optimize.py:237
opti_base.add_argument('-b', '--batch-size', default=5, help='The batch size in the each iteration', type=int)  # hotpot/commands/optimize.py:245
opti_bayes.add_argument('-s', '--params-space', help="Specify the low:high bounds of the each parameter, separated by comma ','\nFollowing are example:\n    python hotpot optimize excel_file result_dir -s [name0|index0]:low0:high0,[name1|index1]:low1:high1,[name2|index2]:low2:high2 ...\nThe counts of specified space must be equal to the number of specified parameters")  # hotpot/commands/optimize.py:252
opti_bayes.add_argument('--log-params', help="Which parameters are optimized in log scale, specified parameters by their names or indices (with -i flog) and each parameters is separated by comma ','")  # hotpot/commands/optimize.py:261
opti_plot.add_argument('--init-index', help="This arguments is only used when '--plot' flag is set to true.\nThe start sample index to reproduce the optimizing procedure\nThe specified value is range [-sample_num, sample_num]", type=int)  # hotpot/commands/optimize.py:267
opti_plot.add_argument('-e', '--emb-method', default='TSNE', help='The embedding method to project parameters to 2D manifold map')  # hotpot/commands/optimize.py:275
opti_plot.add_argument('--scatter-map', action='store_true', help='Whether to use a scatter plot to visualize the parameters space')  # hotpot/commands/optimize.py:281
opti_plot.add_argument('--cmap', default='Greys', help='The ColorMap to make the picture')  # hotpot/commands/optimize.py:287
opti_plot.add_argument('--examples', choices=['COF'], help='Reproduce the optimization procedure of COF experiment')  # hotpot/commands/optimize.py:293
```

## `hotpot.ds.cli`

```python
parser.add_argument('--doc', action=MarkdownDocumentationAction, nargs=0, document_loader=partial(load_cli_documentation, topic), help='show detailed Markdown usage documentation and exit')  # hotpot/ds/cli.py:124
subparsers.add_parser(name, help=help, description=description, formatter_class=HELP_FORMATTER)  # hotpot/ds/cli.py:141
parser.add_argument('--db', required=required, default=argparse.SUPPRESS, metavar='PATH', help='current Hotpot dataset store')  # hotpot/ds/cli.py:152
parser.add_argument('dataset', nargs=None if required else '?', metavar='DATASET', help='public dataset name')  # hotpot/ds/cli.py:162
parser.add_argument('--version', default=argparse.SUPPRESS, metavar='VERSION', help='release ID, semantic version, or three-digit release serial')  # hotpot/ds/cli.py:168
parser.add_argument('--archive-class', action='append', choices=('ordinary', 'benchmark', 'incomplete'), default=argparse.SUPPRESS, help='opt into explicit persisted archive classes; repeat to combine')  # hotpot/ds/cli.py:174
parser.add_argument('--root', dest='bundle_root_option', default=argparse.SUPPRESS, metavar='BUNDLE_ROOT', help='built Bundle root; defaults to the current directory')  # hotpot/ds/cli.py:185
parser.add_argument('--dataset', dest='bundle_dataset', default=argparse.SUPPRESS, metavar='NAME', help='assert the authoritative bundle.json dataset name')  # hotpot/ds/cli.py:192
parser.add_argument('--db', dest='bundle_db', default=argparse.SUPPRESS, metavar='PATH', help='current Hotpot dataset store')  # hotpot/ds/cli.py:199
parser.add_argument('--archive-class', action='append', choices=('incomplete', 'ordinary', 'benchmark'), help='select archive classes; repeat for IN semantics')  # hotpot/ds/cli.py:209
parser.add_argument('--quality', action='append', choices=_B_LEVELS, help='select exact B levels')  # hotpot/ds/cli.py:215
parser.add_argument('--min-quality', choices=_B_LEVELS, help='select a minimum B level')  # hotpot/ds/cli.py:216
parser.add_argument('--evidence', action='append', choices=_E_LEVELS, help='select exact E levels')  # hotpot/ds/cli.py:217
parser.add_argument('--min-evidence', choices=_E_LEVELS, help='select a minimum E level')  # hotpot/ds/cli.py:218
parser.add_argument('--benchmark-status', action='append', help='select benchmark statuses')  # hotpot/ds/cli.py:219
parser.add_argument('--reason', action='append', help='select assessment reason codes')  # hotpot/ds/cli.py:220
parser.add_argument('--verification', action='append', choices=('AI', 'Human'), help='select evidence verification methods')  # hotpot/ds/cli.py:221
parser.add_argument('--attestation', action='append', choices=('Hotpot', 'User', 'Other'), help='select attestation authorities')  # hotpot/ds/cli.py:222
parser.add_subparsers(dest='ds_action', metavar='COMMAND', required=True)  # hotpot/ds/cli.py:231
get.add_argument('selector', metavar='SELECTOR', help='index, identifier, Hotpot ID, or - for protocol stdin')  # hotpot/ds/cli.py:244
modes.add_argument('--index', action='store_true', help='interpret SELECTOR as a sequence index (default)')  # hotpot/ds/cli.py:246
modes.add_argument('--id', action='store_true', help='interpret SELECTOR as a Hotpot data ID')  # hotpot/ds/cli.py:247
modes.add_argument('--identifier', nargs='?', const=_AUTO_IDENTIFIER, metavar='NAMESPACE', help='interpret SELECTOR as an external identifier')  # hotpot/ds/cli.py:248
presentation.add_argument('--smi', action='store_true', help='render SMILES')  # hotpot/ds/cli.py:250
presentation.add_argument('--xyz', action='store_true', help='render XYZ')  # hotpot/ds/cli.py:251
structures.add_argument('--structure-index', type=int, metavar='N', help='select a zero-based attached structure')  # hotpot/ds/cli.py:253
structures.add_argument('--structure-role', metavar='ROLE', help='select the unique structure with ROLE')  # hotpot/ds/cli.py:254
structures.add_argument('--structure-id', metavar='ID', help='select an exact attached structure ID')  # hotpot/ds/cli.py:255
subset.add_argument('--where', action='append', default=[], metavar='FIELD=JSON', help='field equality or IN filter; repeat to combine')  # hotpot/ds/cli.py:259
subset.add_argument('-q', '--quality', type=int, choices=range(5), default=0, metavar='LEVEL', help='minimum B level; zero disables the threshold')  # hotpot/ds/cli.py:260
subset.add_argument('-e', '--evidence', type=int, choices=range(5), default=0, metavar='LEVEL', help='minimum E level; zero preserves not-applicable records')  # hotpot/ds/cli.py:261
subset.add_argument('--limit', type=int, metavar='N', help='emit at most N records')  # hotpot/ds/cli.py:262
search.add_argument('--scope', choices=('data', 'dataset', 'smarts'), default='data', help='object class to search')  # hotpot/ds/cli.py:266
search.add_argument('--where', action='append', default=[], metavar='FIELD=JSON', help='field equality or IN filter')  # hotpot/ds/cli.py:267
search.add_argument('--smarts', metavar='PATTERN', help='SMARTS query for smarts scope')  # hotpot/ds/cli.py:268
search.add_argument('--limit', type=int, metavar='N', help='emit at most N matches')  # hotpot/ds/cli.py:269
export.add_argument('output', metavar='OUTPUT', help='new export directory')  # hotpot/ds/cli.py:273
scope.add_argument('item', nargs='?', metavar='ITEM', help='one index/identifier or - for protocol stdin')  # hotpot/ds/cli.py:275
scope.add_argument('--range', nargs=2, type=int, dest='index_range', metavar=('START', 'STOP'), help='half-open sequence range')  # hotpot/ds/cli.py:276
scope.add_argument('--where', action='append', metavar='FIELD=JSON', help='field subset filter')  # hotpot/ds/cli.py:277
modes.add_argument('--index', action='store_true', help='interpret ITEM as an index (default)')  # hotpot/ds/cli.py:279
modes.add_argument('--id', action='store_true', help='interpret ITEM as a Hotpot ID')  # hotpot/ds/cli.py:280
modes.add_argument('--identifier', nargs='?', const=_AUTO_IDENTIFIER, metavar='NAMESPACE', help='interpret ITEM as an external identifier')  # hotpot/ds/cli.py:281
export.add_argument('--format', metavar='FORMAT', help='convert structures to FORMAT')  # hotpot/ds/cli.py:282
copy.add_argument('target_package', metavar='TARGET_HOTPOT_PACKAGE', help='target hotpot package directory containing __init__.py and dataset/')  # hotpot/ds/cli.py:291
copy.add_argument('--db', default=argparse.SUPPRESS, metavar='PATH', help='source ingested store; defaults to the imported package store')  # hotpot/ds/cli.py:296
detail.add_subparsers(dest='detail_action', metavar='COMMAND')  # hotpot/ds/cli.py:304
query.add_subparsers(dest='detail_query_target', metavar='TARGET', required=True)  # hotpot/ds/cli.py:309
claims.add_argument('--dataset', metavar='NAME', help='restrict claims to one dataset')  # hotpot/ds/cli.py:312
claims.add_argument('--evidence-level', metavar='LEVEL', help='restrict claims to one Case E level')  # hotpot/ds/cli.py:313
claims.add_argument('--limit', type=int, metavar='N', help='return at most N claims')  # hotpot/ds/cli.py:314
bundle.add_subparsers(dest='bundle_action', metavar='COMMAND', required=True)  # hotpot/ds/cli.py:318
build.add_argument('bundle_root', nargs='?', default='.', metavar='BUNDLE_ROOT', help='unbuilt current-format intake root; defaults to the current directory')  # hotpot/ds/cli.py:320
build.add_argument('--dataset', default=argparse.SUPPRESS, metavar='NAME', help='assert the authoritative bundle.json dataset name')  # hotpot/ds/cli.py:327
bundle_rm.add_argument('--all', action='store_true', required=True, help='remove every allowlisted reproducible intermediate')  # hotpot/ds/cli.py:344
bundle_get.add_argument('view', choices=('info', 'summary', 'cases', 'case.info', 'case.assessment'), help='allowlisted Bundle or Case view')  # hotpot/ds/cli.py:352
bundle_get.add_argument('case_id', nargs='?', metavar='CASE_ID', help='Case ID for a case.* view')  # hotpot/ds/cli.py:357
bundle_set.add_argument('method', choices=('case.set_core_info', 'case.replace_structure', 'case.add_observation', 'case.add_evidence', 'case.verify_evidence', 'case.attest_benchmark'), help='allowlisted Case method')  # hotpot/ds/cli.py:361
bundle_set.add_argument('input', metavar='JSON_FILE', help='closed method-argument JSON object')  # hotpot/ds/cli.py:373
gate.add_subparsers(dest='gate_name', metavar='GATE', required=True)  # hotpot/ds/cli.py:376
child.add_argument('--show', action='store_true', help='show the current immutable Gate')  # hotpot/ds/cli.py:379
child.add_argument('--add-check', action='append', default=[], metavar='NAME[=BOOL]', help='add or replace an ordinary check')  # hotpot/ds/cli.py:380
child.add_argument('--rm-check', action='append', default=[], metavar='NAME', help='remove an ordinary check')  # hotpot/ds/cli.py:381
child.add_argument('--checker', action='append', default=[], metavar='NAME=FILE:ENTRYPOINT', help='bind source for a same-named custom check')  # hotpot/ds/cli.py:382
child.add_argument('--set-level', action='append', default=[], metavar='LEVEL=NAMES', help='set one cumulative B/E check list')  # hotpot/ds/cli.py:384
child.add_argument('--rm-level', action='append', default=[], metavar='LEVEL', help='remove a level before replacing it in the same edit')  # hotpot/ds/cli.py:385
child.add_argument('--add-structure-check', action='append', default=[], metavar='NAME[=BOOL]', help='add or replace a structure check')  # hotpot/ds/cli.py:387
child.add_argument('--rm-structure-check', action='append', default=[], metavar='NAME', help='remove a structure check')  # hotpot/ds/cli.py:388
child.add_argument('--add-direct', action='append', default=[], metavar='NAME[=BOOL]', help='add or replace a direct-entry check')  # hotpot/ds/cli.py:390
child.add_argument('--rm-direct', action='append', default=[], metavar='NAME', help='remove a direct-entry check')  # hotpot/ds/cli.py:391
child.add_argument('--add-veto', action='append', default=[], metavar='NAME', help='add a benchmark veto')  # hotpot/ds/cli.py:392
child.add_argument('--rm-veto', action='append', default=[], metavar='NAME', help='remove a benchmark veto')  # hotpot/ds/cli.py:393
child.add_argument('--set-tolerance', action='append', default=[], metavar='PROPERTY=ABS[,REL]', help='set numeric conflict tolerances')  # hotpot/ds/cli.py:394
child.add_argument('--rm-tolerance', action='append', default=[], metavar='PROPERTY', help='remove numeric conflict tolerances')  # hotpot/ds/cli.py:395
certify.add_argument('bundle_root', metavar='BUNDLE_ROOT', help='built synchronized Bundle root')  # hotpot/ds/cli.py:398
certify.add_argument('--dataset', default=argparse.SUPPRESS, metavar='NAME', help='assert bundle.json.ds_name')  # hotpot/ds/cli.py:399
ingest.add_argument('bundle_root', metavar='BUNDLE_ROOT', help='built synchronized certified Bundle root')  # hotpot/ds/cli.py:403
ingest.add_argument('--dataset', default=argparse.SUPPRESS, metavar='NAME', help='assert bundle.json.ds_name')  # hotpot/ds/cli.py:404
ingest.add_argument('--include-incomplete', action='store_true', help='force incomplete archival into its separate partition')  # hotpot/ds/cli.py:406
ingest.add_argument('--reason', help='required justification for --include-incomplete')  # hotpot/ds/cli.py:407
```

## `hotpot.ds.tools.electronic_state.assess_quality`

```python
parser.add_argument('--db', type=Path, default=DEFAULT_DB)  # hotpot/ds/tools/electronic_state/assess_quality.py:354
parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)  # hotpot/ds/tools/electronic_state/assess_quality.py:355
```

## `hotpot.ds.tools.electronic_state.build_database`

```python
parser.add_argument('--entry-jsonl', type=Path, required=True)  # hotpot/ds/tools/electronic_state/build_database.py:61
parser.add_argument('--site-jsonl', type=Path, required=True)  # hotpot/ds/tools/electronic_state/build_database.py:62
parser.add_argument('--canonical-db', type=Path, required=True)  # hotpot/ds/tools/electronic_state/build_database.py:63
parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)  # hotpot/ds/tools/electronic_state/build_database.py:64
parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)  # hotpot/ds/tools/electronic_state/build_database.py:65
parser.add_argument('--audit-output', type=Path, default=DEFAULT_AUDIT_OUTPUT)  # hotpot/ds/tools/electronic_state/build_database.py:66
parser.add_argument('--provenance-output', type=Path, default=DEFAULT_PROVENANCE_OUTPUT)  # hotpot/ds/tools/electronic_state/build_database.py:67
parser.add_argument('--limit', type=int)  # hotpot/ds/tools/electronic_state/build_database.py:68
parser.add_argument('--replace', action='store_true')  # hotpot/ds/tools/electronic_state/build_database.py:69
```

## `hotpot.pipeline.cli`

```python
parser.add_argument('--doc', action=MarkdownDocumentationAction, nargs=0, document_loader=load_cli_documentation, help='show detailed Markdown usage documentation and exit')  # hotpot/pipeline/cli.py:131
parser.add_argument('--results-dir', required=True, metavar='DIRECTORY', help='create this exact directory for ordered pipeline artifacts')  # hotpot/pipeline/cli.py:138
parser.add_argument('pipeline_tokens', nargs='*', metavar='WORKFLOW.json/-- STAGE ...', help="JSON workflow file or inline stages separated by exact '::' tokens")  # hotpot/pipeline/cli.py:144
```

## `hotpot.plugins.xtb.cli`

```python
parser.add_argument('--doc', action=MarkdownDocumentationAction, nargs=0, document_loader=load_cli_documentation, help='show detailed Markdown usage documentation and exit')  # hotpot/plugins/xtb/cli.py:142
parser.add_argument('input', nargs='?', metavar='3D-STRUCTURE-FILE/-', help="3D molecular file or '-' for standard input")  # hotpot/plugins/xtb/cli.py:149
parser.add_argument('--method', choices=tuple(_METHODS), default='gfn2', help='numerical method (default: gfn2)')  # hotpot/plugins/xtb/cli.py:155
parser.add_argument('--task', choices=tuple(_TASKS), default='optimize', help='calculation task (default: optimize)')  # hotpot/plugins/xtb/cli.py:161
parser.add_argument('--charge', type=int, help='authoritative total charge')  # hotpot/plugins/xtb/cli.py:167
parser.add_argument('--unpaired-electrons', type=_nonnegative_int, help='authoritative xTB UHF/unpaired-electron count')  # hotpot/plugins/xtb/cli.py:168
parser.add_argument('--state-policy', choices=('strict', 'coordination-screening-v1'), default='strict', help='scientific-state decision policy (default: strict)')  # hotpot/plugins/xtb/cli.py:173
parser.add_argument('--input-format', help='input format override; required for standard input')  # hotpot/plugins/xtb/cli.py:179
parser.add_argument('--output-format', default='sdf', help='molecular output format (default: sdf)')  # hotpot/plugins/xtb/cli.py:183
parser.add_argument('-o', '--output', help="write molecular payload to FILE; '-' means standard output")  # hotpot/plugins/xtb/cli.py:188
parser.add_argument('--report', help='write a structured JSON report to FILE')  # hotpot/plugins/xtb/cli.py:193
parser.add_argument('--native-log', help='write captured native xTB stdout and stderr to FILE')  # hotpot/plugins/xtb/cli.py:194
parser.add_argument('--xtb-executable', help='path to the official xTB executable')  # hotpot/plugins/xtb/cli.py:198
parser.add_argument('--threads', type=_positive_int, help='threads assigned to each xTB process')  # hotpot/plugins/xtb/cli.py:202
parser.add_argument('--jobs', type=_positive_int, default=1, help='independent xTB molecule processes (default: 1)')  # hotpot/plugins/xtb/cli.py:207
parser.add_argument('--work-directory', help='parent directory for isolated native workspaces')  # hotpot/plugins/xtb/cli.py:213
parser.add_argument('--keep-work-directory', action='store_true', help='retain native workspaces and artifacts')  # hotpot/plugins/xtb/cli.py:217
parser.add_argument('--post-check', choices=_POST_CHECKS, default='off', help='optional Hotpot geometry gate after xTB (default: off)')  # hotpot/plugins/xtb/cli.py:222
parser.add_argument('--overwrite', action='store_true', help='replace existing output, report, and native-log files')  # hotpot/plugins/xtb/cli.py:228
```

## `hotpot.plugins.xtb.stage`

```python
parser.add_argument('--method', choices=tuple(_METHODS), default='gfn2')  # hotpot/plugins/xtb/stage.py:246
parser.add_argument('--task', choices=tuple(_TASKS), default='optimize')  # hotpot/plugins/xtb/stage.py:251
parser.add_argument('--charge', type=int)  # hotpot/plugins/xtb/stage.py:256
parser.add_argument('--unpaired-electrons', type=_nonnegative_int)  # hotpot/plugins/xtb/stage.py:257
parser.add_argument('--state-policy', choices=('strict', 'coordination-screening-v1'), default='strict')  # hotpot/plugins/xtb/stage.py:258
parser.add_argument('--xtb-executable')  # hotpot/plugins/xtb/stage.py:263
parser.add_argument('--threads', type=_positive_int)  # hotpot/plugins/xtb/stage.py:264
parser.add_argument('--post-check', choices=('off', 'basic', 'standard', 'strict'), default='off')  # hotpot/plugins/xtb/stage.py:265
parser.add_argument('--sample-break-on', action='append', choices=_XTB_CODES, default=[])  # hotpot/plugins/xtb/stage.py:270
parser.add_argument('--batch-break-on', action='append', choices=_XTB_CODES, default=[])  # hotpot/plugins/xtb/stage.py:271
```
