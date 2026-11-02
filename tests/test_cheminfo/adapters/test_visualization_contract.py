"""Drawing input sequencing, live font resources and real format consumers."""
import hashlib
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import rdkit
from packaging.version import InvalidVersion
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

import hotpot
from hotpot.cheminfo.operations import visualization
from hotpot.cheminfo.adapters.visualization import draw

FONT_HASHES = {'Arial.ttf': '35c0f3559d8db569e36c31095b8a60d441643d95f59139de40e23fada819b833', 'Arial_Black.ttf': 'dad7c04acb26e23dfe4780e79375ca193ddaf68409317e81577a30674668830e', 'Arial_Bold.ttf': '4044aa6b5bebbc36980206b45b0aaaaa5681552a48bcadb41746d5d1d71fd7b4', 'Arial_Bold_Italic.ttf': '2f371cd9d96b3ac544519d85c16dc43ceacdfcea35090ee8ddf3ec5857c50328', 'Arial_Italic.ttf': '70ade233175a6a6675e4501461af9326e6f78b1ffdf787ca0da5ab0fc8c9cfd6'}
FORMATS = ('svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp')


def _svg(text):
    assert ET.fromstring(text).tag.endswith('svg')


def test_drawing_and_utility_definitions_have_their_approved_owners():
    assert visualization.__all__ == ['draw_single_mol', 'draw_grid']
    assert draw.__all__ == ['compute_atom_bond_colors', 'create_svg_colorbar', 'merge_colorbar_to_svg']
    for name in visualization.__all__:
        assert getattr(visualization, name).__module__ == 'hotpot.cheminfo.operations.visualization'
        assert not hasattr(hotpot, name)
    for name in draw.__all__:
        assert getattr(draw, name).__module__ == 'hotpot.cheminfo.adapters.visualization.draw'


def test_rd_input_is_selected_first_and_real_compute_2d_mutates_caller_value():
    class WithSmiles(Chem.Mol):
        @property
        def smiles(self):
            raise AssertionError('RD input must precede carrier getter')
    mol = WithSmiles(Chem.MolFromSmiles('CCO'))
    conformer = Chem.Conformer(3)
    for row in range(3):
        conformer.SetAtomPosition(row, (row, row + 2., row + 4.))
    mol.AddConformer(conformer)
    _svg(visualization.draw_single_mol(mol))
    assert mol.GetNumConformers() == 1
    assert all(mol.GetConformer().GetAtomPosition(i).z == 0. for i in range(3))


def test_grid_fully_prepares_each_yield_before_advancing_and_retains_two_smiles_reads(monkeypatch):
    events = []
    class Carrier:
        def __init__(self, value):
            self.value = value
        @property
        def smiles(self):
            events.append(('smiles', self.value))
            return self.value
        def to_rdmol(self):
            raise AssertionError('drawing uses its public smiles branch')
    parse, compute = draw._parse_smiles, draw._compute_2d_coords
    def parsed(value, **kwargs):
        events.append(('parse', value))
        return parse(value, **kwargs)
    def computed(mol):
        value = Chem.MolToSmiles(mol)
        result = compute(mol)
        events.append(('computed', value))
        return result
    monkeypatch.setattr(draw, '_parse_smiles', parsed)
    monkeypatch.setattr(draw, '_compute_2d_coords', computed)
    def inputs():
        events.append(('yield', 'CCO'))
        yield Carrier('CCO')
        assert events[-1] == ('computed', 'CCO')
        events.append(('yield', 'CCN'))
        yield Carrier('CCN')
    _svg(visualization.draw_grid(inputs(), n_cols=2))
    assert events == [('yield', 'CCO'), ('smiles', 'CCO'), ('smiles', 'CCO'),
                      ('parse', 'CCO'), ('computed', 'CCO'),
                      ('yield', 'CCN'), ('smiles', 'CCN'), ('smiles', 'CCN'),
                      ('parse', 'CCN'), ('computed', 'CCN')]


def test_real_string_single_grid_success_and_distinct_invalid_errors():
    _svg(visualization.draw_single_mol('CCO'))
    _svg(visualization.draw_grid(['CCO', 'CCN'], n_cols=2))
    with pytest.raises(ValueError, match='^Invalid molecule input: invalid-smiles$'):
        visualization.draw_single_mol('invalid-smiles')
    with pytest.raises(ValueError, match='^Could not parse SMILES: invalid-smiles$'):
        visualization._load_rdmol('invalid-smiles')
    with pytest.raises(ValueError, match='^Could not parse SMILES: invalid-smiles$'):
        visualization.draw_grid(['invalid-smiles'])
    with pytest.raises(ValueError, match='^No valid molecules provided for grid drawing[.]$'):
        visualization.draw_grid([])


@pytest.mark.parametrize('operation', [visualization.draw_single_mol, lambda value: visualization.draw_grid([value])])
def test_version_check_precedes_actual_rd_input_test_and_carrier_getter(monkeypatch, operation):
    class Carrier:
        @property
        def smiles(self):
            raise AssertionError('version check must precede carrier access')
    monkeypatch.setattr(rdkit, '__version__', '0.0.0')
    with pytest.raises(InvalidVersion, match='Expected version of rdkit should later than 2025.9, got 0.0.0'):
        operation(Carrier())


def test_all_five_packaged_font_bytes_and_live_render_resource(monkeypatch):
    root = resources.files('hotpot.cheminfo.adapters.visualization').joinpath('fonts', 'Arial')
    assert {entry.name: hashlib.sha256(entry.read_bytes()).hexdigest() for entry in root.iterdir()} == FONT_HASHES
    original = draw._draw_configuration
    observed = []
    def configure(*args, **kwargs):
        options = original(*args, **kwargs)
        path = Path(options.fontFile)
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == FONT_HASHES['Arial_Bold.ttf']
        observed.append(path.name)
        return options
    monkeypatch.setattr(draw, '_draw_configuration', configure)
    _svg(visualization.draw_single_mol('CCO', font='Arial', fontweight='bold'))
    _svg(visualization.draw_grid(['CCO', 'CCN'], font='Arial', fontweight='bold'))
    assert observed == ['Arial_Bold.ttf', 'Arial_Bold.ttf']


def test_extracted_font_lifetime_contains_the_actual_single_and_grid_render(monkeypatch, tmp_path):
    paths = []
    @contextmanager
    def extracted(resource):
        path = tmp_path / (str(len(paths)) + '-' + resource.name)
        path.write_bytes(resource.read_bytes())
        paths.append(path)
        try:
            yield path
        finally:
            path.unlink()
    monkeypatch.setattr(draw.resources, 'as_file', extracted)
    single, grid = rdMolDraw2D.PrepareAndDrawMolecule, Draw.MolsToGridImage
    calls = []
    def draw_single(*args, **kwargs):
        assert paths[-1].is_file()
        assert hashlib.sha256(paths[-1].read_bytes()).hexdigest() == FONT_HASHES['Arial_Bold.ttf']
        calls.append('single')
        return single(*args, **kwargs)
    def draw_grid(*args, **kwargs):
        assert paths[-1].is_file()
        assert hashlib.sha256(paths[-1].read_bytes()).hexdigest() == FONT_HASHES['Arial_Bold.ttf']
        calls.append('grid')
        return grid(*args, **kwargs)
    monkeypatch.setattr(rdMolDraw2D, 'PrepareAndDrawMolecule', draw_single)
    monkeypatch.setattr(Draw, 'MolsToGridImage', draw_grid)
    _svg(visualization.draw_single_mol('CCO'))
    assert not paths[-1].exists()
    _svg(visualization.draw_grid(['CCO', 'CCN']))
    assert not paths[-1].exists()
    assert calls == ['single', 'grid']


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('grid', [False, True])
def test_real_formats_write_files_and_retain_svg_return(fmt, grid, tmp_path):
    path = tmp_path / ('grid.' + fmt if grid else 'single.' + fmt)
    if grid:
        value = visualization.draw_grid(['CCO', 'CCN'], save_path=str(path), output_format=fmt, n_cols=2)
    else:
        value = visualization.draw_single_mol('CCO', save_path=str(path), output_format=fmt)
    _svg(value)
    data = path.read_bytes()
    assert data
    if fmt == 'svg':
        assert data.decode() == value
    elif fmt == 'pdf':
        assert data.startswith(b'%PDF-')
    elif fmt in ('ps', 'eps'):
        assert data.startswith(b'%!PS-Adobe-')
    else:
        expected = {'png': 'PNG', 'jpeg': 'JPEG', 'jpg': 'JPEG', 'tiff': 'TIFF', 'bmp': 'BMP', 'webp': 'WEBP'}
        with Image.open(path) as image:
            assert image.format == expected[fmt]
            assert image.width > 0 and image.height > 0
            image.load()


def test_real_color_utility_and_colorbar_rendering_consumers():
    mol = Chem.MolFromSmiles('CCO')
    atoms, bonds, norm, scalar = draw.compute_atom_bond_colors(mol, np.array([-1., 0., 1.]), threshold=0.)
    assert set(atoms) == {0, 2} and set(bonds) == {0, 1}
    assert norm.vmin == -1. and norm.vmax == 1.
    assert scalar.norm is norm
    bar = draw.create_svg_colorbar(norm=norm)
    _svg(bar)
    _svg(draw.merge_colorbar_to_svg(visualization.draw_single_mol('CCO'), bar, (600, 600), 'vertical'))
    _svg(visualization.draw_single_mol('CCO', atom_hl_values=np.array([-1., 0., 1.]), colorbar=True))
