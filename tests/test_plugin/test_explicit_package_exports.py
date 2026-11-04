"""Actual optional facades preserve leaf identities and last-writer exports."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


CASES = (
    pytest.param(
        "hotpot.plugins.ComplexFormer.data_process.sc_logk", ("core", "data"),
        marks=pytest.mark.optional_dependency,
    ),
    pytest.param(
        "hotpot.plugins.ComplexFormer.models",
        ("core", "loss", "metrics", "predictor", "plot_makers", "utils"),
        marks=pytest.mark.optional_dependency,
    ),
    pytest.param(
        "hotpot.plugins.ComplexFormer.models.core", ("_utils",),
        marks=pytest.mark.optional_dependency,
    ),
    pytest.param(
        "hotpot.plugins.ComplexFormer.run", ("datacls",),
        marks=pytest.mark.optional_dependency,
    ),
    pytest.param(
        "hotpot.plugins.ComplexFormer.tasks", ("tasks",),
        marks=pytest.mark.optional_dependency,
    ),
    pytest.param(
        "hotpot.plugins.plots", ("plots", "colors"),
        marks=pytest.mark.optional_dependency,
    ),
    pytest.param(
        "hotpot.plugins.plots.plots", ("general", "beyes", "ml", "base"),
        marks=pytest.mark.optional_dependency,
    ),
    ("hotpot.plugins.zeoxx", ("core",)),
    ("hotpot.utils.chem", ("atom",)),
    ("hotpot.utils.configs", ("logging_config",)),
)


@pytest.mark.parametrize("package,leaves", CASES)
def test_facade_exports_are_actual_leaf_values_in_original_order(package, leaves, tmp_path):
    source = textwrap.dedent('''
        import importlib
        import json
        from pathlib import Path
        import sys

        package, leaves = json.loads(sys.argv[1])
        import hotpot
        product = Path(hotpot.__file__).resolve().parent
        assert not any(name.split('.')[0] in {
            'torch', 'rdkit', 'openbabel', 'onnxruntime', 'matplotlib',
        } for name in sys.modules)
        assert not any(name.startswith((
            'hotpot.plugins.ComplexFormer', 'hotpot.plugins.plots',
        )) for name in sys.modules)

        facade = importlib.import_module(package)
        winners = {}
        leaf_exports = {}
        for suffix in leaves:
            leaf = importlib.import_module(package + '.' + suffix)
            names = getattr(leaf, '__all__', tuple(
                name for name in vars(leaf) if not name.startswith('_')
            ))
            leaf_exports[leaf.__name__] = list(names)
            for name in names:
                winners[name] = (leaf, getattr(leaf, name))
        assert all(getattr(facade, name) is value
                   for name, (_, value) in winners.items())

        # Keep implicit helper/module exports and normal from-package consumers.
        assert '__all__' not in vars(facade)
        namespace = {}
        exec('from ' + package + ' import *', namespace)
        public = {name: value for name, value in vars(facade).items()
                  if not name.startswith('_')}
        assert set(namespace) - {'__builtins__'} == set(public)
        assert all(namespace[name] is value for name, value in public.items())
        for name, (_, value) in winners.items():
            imported = {}
            exec('from ' + package + ' import ' + name, imported)
            assert imported[name] is value

        if package in {'hotpot.plugins.ComplexFormer.models',
                       'hotpot.plugins.ComplexFormer.models.core'}:
            core = importlib.import_module('hotpot.plugins.ComplexFormer.models.core')
            assert facade.Core is core.AttnCore
            for child in ('se3net', 'node_processor', 'envs_encoder', 'base', 'attn_core'):
                assert getattr(facade, child) is importlib.import_module(core.__name__ + '.' + child)
        if package == 'hotpot.plugins.ComplexFormer.models':
            cm = importlib.import_module(package + '.cm')
            utils = importlib.import_module(package + '.utils')
            assert facade.Model is cm.Model
            for name in ('weight_labels', 'weight_binary', 'torch_numpy_exchanger',
                         'Union', 'Literal', 'np', 'torch', 'F'):
                assert getattr(facade, name) is getattr(utils, name)
        if package == 'hotpot.plugins.ComplexFormer.run':
            for child, name in (('train', 'LightPretrain'), ('run', 'run'), ('cv', 'run_cv')):
                assert getattr(facade, name) is getattr(importlib.import_module(package + '.' + child), name)
        if package in {'hotpot.plugins.plots', 'hotpot.plugins.plots.plots'}:
            plotter = importlib.import_module('hotpot.plugins.plots.plotter')
            plots = importlib.import_module('hotpot.plugins.plots.plots')
            assert facade.SciPlotter is plotter.SciPlotter
            for child in ('general', 'beyes', 'ml', 'base'):
                assert getattr(facade, child) is importlib.import_module(plots.__name__ + '.' + child)
            assert facade.Hist is plots.general.Hist
            assert facade.Plot is plots.base.Plot

        origins = {name: str(Path(module.__file__).resolve())
                   for name, module in sys.modules.items()
                   if (name == 'hotpot' or name.startswith('hotpot.'))
                   and getattr(module, '__file__', None)}
        assert all(Path(path).is_relative_to(product) for path in origins.values())
        print(json.dumps({'package': package, 'leaf_exports': leaf_exports,
                          'winning_owners': {name: leaf.__name__
                                             for name, (leaf, _) in winners.items()},
                          'public_namespace': list(public), 'origins': origins}))
    ''')
    argv = [sys.executable, "-s", "-B", "-c", source, json.dumps([package, leaves])]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    observation = {
        "argv": argv, "cwd": str(Path.cwd()), "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    }
    (tmp_path / "actual-package-exports.json").write_text(json.dumps(observation, indent=2) + "\n")
    assert result.returncode == 0, observation
    assert json.loads(result.stdout.splitlines()[-1])["package"] == package
