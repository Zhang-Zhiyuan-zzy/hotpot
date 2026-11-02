"""Caller-owned thermo proxies, ordered properties and exact fallback behavior."""
import builtins

import pytest
from rdkit import Chem

from hotpot.cheminfo.operations import thermo
from hotpot.cheminfo.adapters.thermo import chemical

# Frozen independently from C61, never derived from the implementation at runtime.
EXPECTED_PROPERTIES = ('T', 'P', 'phase', 'ID', 'CAS', 'PubChem', 'MW', 'formula', 'atoms', 'similarity_variable', 'smiles', 'InChI', 'InChI_Key', 'IUPAC_name', 'synonyms', 'Tm', 'Tb', 'Tc', 'Pc', 'Vc', 'Zc', 'rhoc', 'rhocm', 'omega', 'StielPolar', 'Tt', 'Pt', 'Hfus', 'Hfusm', 'Hsub', 'Hsubm', 'Hfm', 'Hf', 'Hfgm', 'Hfg', 'Hcm', 'Hc', 'Hcm_lower', 'Hc_lower', 'S0m', 'S0', 'S0gm', 'S0g', 'Gfm', 'Gf', 'Gfgm', 'Gfg', 'Sfm', 'Sf', 'Sfgm', 'Sfg', 'Hcgm', 'Hcg', 'Hcgm_lower', 'Hcg_lower', 'Tflash', 'Tautoignition', 'LFL', 'UFL', 'TWA', 'STEL', 'Ceiling', 'Skin', 'Carcinogen', 'dipole', 'Stockmayer', 'molecular_diameter', 'GWP', 'ODP', 'logP', 'legal_status', 'economic_status', 'RI', 'RIT', 'conductivity', 'conductivityT', 'VaporPressure', 'EnthalpyVaporization', 'VolumeSolid', 'VolumeLiquid', 'VolumeGas', 'HeatCapacitySolid', 'HeatCapacityLiquid', 'HeatCapacityGas', 'ViscosityLiquid', 'ViscosityGas', 'ThermalConductivityLiquid', 'ThermalConductivityGas', 'SurfaceTension', 'Permittivity', 'Psat_298', 'phase_STP', 'Vml_Tb', 'Vml_Tm', 'Vml_STP', 'rhoml_STP', 'Vmg_STP', 'Vms_Tm', 'rhos_Tm', 'Hvap_Tbm', 'Hvap_Tb', 'Hvapm_298', 'Hvap_298', 'alpha', 'alphag', 'alphal', 'API', 'aromatic_rings', 'atom_fractions', 'Bvirial', 'charge', 'Cp', 'Cpg', 'Cpgm', 'Cpl', 'Cplm', 'Cpm', 'Cps', 'Cpsm', 'Cvg', 'Cvgm', 'eos', 'Hill', 'Hvap', 'Hvapm', 'isentropic_exponent', 'isobaric_expansion', 'isobaric_expansion_g', 'isobaric_expansion_l', 'JT', 'JTg', 'JTl', 'k', 'kg', 'kl', 'mass_fractions', 'mu', 'mug', 'mul', 'nu', 'nug', 'nul', 'Parachor', 'permittivity', 'Poynting', 'Pr', 'Prg', 'Prl', 'Psat', 'PSRK_groups', 'rdkitmol', 'rdkitmol_Hs', 'rho', 'rhog', 'rhogm', 'rhol', 'rholm', 'rhom', 'rhos', 'rhosm', 'rings', 'SG', 'SGg', 'SGl', 'SGs', 'sigma', 'solubility_parameter', 'UNIFAC_Dortmund_groups', 'UNIFAC_groups', 'UNIFAC_R', 'UNIFAC_Q', 'Van_der_Waals_area', 'Van_der_Waals_volume', 'Vm', 'Vmg', 'Vml', 'Vms', 'Z', 'Zg', 'Zl', 'Zs')


class Carrier:
    def __init__(self):
        self.events = []
        self.rdmol = Chem.MolFromSmiles('CCO')

    def to_rdmol(self):
        self.events.append('rdmol')
        return Chem.Mol(self.rdmol)

    @property
    def cas(self):
        self.events.append('cas')
        return '64-17-5'

    @property
    def cid(self):
        self.events.append('cid')
        return 702


def _chemical(monkeypatch, constructor):
    monkeypatch.setattr(chemical, '_chemical_type', lambda: constructor)


def test_thermo_defaults_full_ordered_properties_and_unrestricted_forwarding(monkeypatch):
    token = object()
    executable = lambda value: value + 1
    class Chemical:
        def __init__(self, identifier, **conditions):
            self.identifier = identifier
            self.conditions = conditions
            self.unlisted_value = token
            self.executable = executable
    _chemical(monkeypatch, Chemical)
    source = Carrier()
    first, second = thermo._get_thermo(source), thermo._get_thermo(source)
    assert type(first) is thermo.Thermo
    assert type(first).__module__ == 'hotpot.cheminfo.operations.thermo'
    assert first.mol is source and second.mol is source
    assert first is not second and first._chem is not second._chem
    assert first.temp == 298.15 and first.pressure == 101325
    assert first._chem.conditions == {'P': 101325, 'T': 298.15}
    assert thermo.Thermo.properties == EXPECTED_PROPERTIES
    assert len(EXPECTED_PROPERTIES) == 181
    assert set(EXPECTED_PROPERTIES) <= set(dir(first))
    assert first.unlisted_value is token and first.executable is executable
    assert first.executable(4) == 5
    assert not any(isinstance(value, thermo.Thermo) for value in source.__dict__.values())
    with pytest.raises(AttributeError):
        first.nonexistent_chemical_property


@pytest.mark.parametrize('reject_count', range(5))
def test_fallback_identifiers_conditions_and_lazy_getter_order(monkeypatch, reject_count):
    source = Carrier()
    calls = []
    result = object()
    def construct(identifier, **conditions):
        calls.append((identifier, conditions))
        if len(calls) <= reject_count:
            raise ValueError('rejected identifier')
        return result
    _chemical(monkeypatch, construct)
    if reject_count == 4:
        with pytest.raises(ValueError, match='^Unknown thermo properties fro CCO$'):
            thermo.Thermo(source, temp=310., pressure=202650.)
    else:
        assert thermo.Thermo(source, temp=310., pressure=202650.)._chem is result
    identifiers = ['SMILES=CCO', Chem.MolToInchi(source.rdmol), '64-17-5', 'PubChem=702']
    count = min(reject_count + 1, 4)
    assert calls == [(identifier, {'P': 202650., 'T': 202650. if index == 3 else 310.})
                     for index, identifier in enumerate(identifiers[:count])]
    expected_events = [
        ['rdmol'],
        ['rdmol', 'rdmol'],
        ['rdmol', 'rdmol', 'cas'],
        ['rdmol', 'rdmol', 'cas', 'cid'],
        ['rdmol', 'rdmol', 'cas', 'cid', 'rdmol'],
    ]
    assert source.events == expected_events[reject_count]


@pytest.mark.parametrize('error', [RuntimeError('constructor runtime'), TypeError('constructor type')])
def test_fallback_catches_only_value_error(monkeypatch, error):
    source = Carrier()
    def construct(identifier, **conditions):
        raise error
    _chemical(monkeypatch, construct)
    with pytest.raises(type(error), match=str(error)):
        thermo.Thermo(source)
    assert source.events == ['rdmol']


def test_get_chem_preserves_order_duplicates_all_columns_and_per_row_failures(monkeypatch):
    calls = []
    class Chemical:
        def __init__(self, identifier, **conditions):
            calls.append((identifier, conditions))
            if identifier == 'bad':
                raise RuntimeError('unavailable ID')
            self.identifier = identifier
            self.T, self.P = conditions['T'], conditions['P']
            self.ID = identifier
        def __getattr__(self, name):
            if name == 'MW':
                raise RuntimeError('one property unavailable')
            return f'{self.identifier}:{name}'
    _chemical(monkeypatch, Chemical)
    proxy = thermo.Thermo(Carrier(), temp=310., pressure=202650.)
    calls.clear()
    raw_rows = []
    dataframe = chemical._dataframe
    def capture_rows(rows, *, index):
        raw_rows.append(rows)
        return dataframe(rows, index=index)
    monkeypatch.setattr(chemical, '_dataframe', capture_rows)
    frame = proxy.get_chem(['first', 'bad', 'first', 'last'])
    assert list(frame.index) == ['first', 'bad', 'first', 'last']
    assert tuple(frame.columns) == EXPECTED_PROPERTIES
    assert calls == [(name, {'T': 310., 'P': 202650.}) for name in frame.index]
    assert frame.iloc[0]['CAS'] == frame.iloc[2]['CAS'] == 'first:CAS'
    assert frame.iloc[3]['CAS'] == 'last:CAS'
    # Raw output rows preserve None exactly. pandas may normalize missing
    # values in numeric or inferred string columns on supported versions.
    assert frame.iloc[0]['MW'] is None
    failed = frame.iloc[1]
    assert failed['ID'] == 'bad' and failed['T'] == 310. and failed['P'] == 202650.
    assert all(raw_rows[0][1][name] is None for name in EXPECTED_PROPERTIES
               if name not in ('ID', 'T', 'P'))
    assert failed.drop(['ID', 'T', 'P']).isna().all()
    single = proxy.get_chem('single')
    assert list(single.index) == ['single'] and tuple(single.columns) == EXPECTED_PROPERTIES


def test_missing_chemical_lookup_escapes_before_any_carrier_getter_or_row_catch(monkeypatch):
    source = Carrier()
    proxy = object.__new__(thermo.Thermo)
    proxy.temp, proxy.pressure = 298.15, 101325.
    def missing():
        raise ModuleNotFoundError('missing thermo dependency', name='thermo')
    monkeypatch.setattr(chemical, '_chemical_type', missing)
    with pytest.raises(ModuleNotFoundError, match='missing thermo dependency'):
        thermo.Thermo(source)
    with pytest.raises(ModuleNotFoundError, match='missing thermo dependency'):
        proxy.get_chem(['first', 'last'])
    assert source.events == []


def test_missing_pandas_at_final_dataframe_boundary_escapes(monkeypatch):
    _chemical(monkeypatch, lambda identifier, **kwargs: object())
    proxy = thermo.Thermo(Carrier())
    original_import = builtins.__import__
    def missing_pandas(name, *args, **kwargs):
        if name == 'pandas':
            raise ModuleNotFoundError('missing pandas output dependency', name='pandas')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing_pandas)
    with pytest.raises(ModuleNotFoundError, match='missing pandas output dependency'):
        proxy.get_chem(['first', 'last'])
