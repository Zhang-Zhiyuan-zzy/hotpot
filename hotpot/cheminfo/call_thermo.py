"""
@File Name:        thermo
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/17 11:58
@Project:          Hotpot

Ref Doc: https://thermo.readthedocs.io/thermo.chemical.html#thermo.chemical.Chemical.A
"""
from typing import Union, Iterable
import pandas as pd
from thermo.chemical import Chemical
from rdkit import Chem


__all__ = [
    'properties',
    'get_chem',
    'mol_to_thermo'
]

properties = (
    'T',                    # Temperature of the chemical [K]
    'P',                    # Pressure of the chemical [Pa]
    'phase',                # Phase at current state; 's', 'l', 'g', or 'l/g'
    'ID',                   # User-specified identifier for lookup
    'CAS',                  # CAS registry number of the chemical
    'PubChem',              # PubChem compound identifier (CID)
    'MW',                   # Molecular weight [g/mol]
    'formula',              # Molecular formula
    'atoms',                # Dict: element symbol -> atom count
    'similarity_variable',  # Similarity variable [mol/g]
    'smiles',               # SMILES representation
    'InChI',                # IUPAC InChI string
    'InChI_Key',            # 25-character InChI key (hashed InChI)
    'IUPAC_name',           # Preferred IUPAC name
    'synonyms',             # List of synonyms (PubChem, sorted by popularity)

    'Tm',                   # Melting temperature [K]
    'Tb',                   # Normal boiling temperature [K]
    'Tc',                   # Critical temperature [K]
    'Pc',                   # Critical pressure [Pa]
    'Vc',                   # Critical molar volume [m^3/mol]
    'Zc',                   # Critical compressibility factor [-]
    'rhoc',                 # Critical mass density [kg/m^3]
    'rhocm',                # Critical molar density [mol/m^3]
    'omega',                # Acentric factor [-]
    'StielPolar',           # Stiel polar factor [-]
    'Tt',                   # Triple-point temperature [K]
    'Pt',                   # Triple-point pressure [Pa]

    'Hfus',                 # Enthalpy of fusion (s→l), mass basis [J/kg]
    'Hfusm',                # Enthalpy of fusion (s→l), molar basis [J/mol]
    'Hsub',                 # Enthalpy of sublimation (s→g), mass basis [J/kg]
    'Hsubm',                # Enthalpy of sublimation (s→g), molar basis [J/mol]

    'Hfm',                  # Std-state molar enthalpy of formation [J/mol]
    'Hf',                   # Std-state mass enthalpy of formation [J/kg]
    'Hfgm',                 # Ideal-gas molar enthalpy of formation [J/mol]
    'Hfg',                  # Ideal-gas mass enthalpy of formation [J/kg]

    'Hcm',                  # Higher (gross) molar heat of combustion [J/mol]
    'Hc',                   # Higher (gross) mass heat of combustion [J/kg]
    'Hcm_lower',            # Lower (net) molar heat of combustion [J/mol]
    'Hc_lower',             # Lower (net) mass heat of combustion [J/kg]

    'S0m',                  # Std-state absolute molar entropy [J/mol/K]
    'S0',                   # Std-state absolute mass entropy [J/kg/K]
    'S0gm',                 # Ideal-gas absolute molar entropy [J/mol/K]
    'S0g',                  # Ideal-gas absolute mass entropy [J/kg/K]

    'Gfm',                  # Std-state molar Gibbs energy of formation [J/mol]
    'Gf',                   # Std-state mass Gibbs energy of formation [J/kg]
    'Gfgm',                 # Ideal-gas molar Gibbs energy of formation [J/mol]
    'Gfg',                  # Ideal-gas mass Gibbs energy of formation [J/kg]

    'Sfm',                  # Std-state molar entropy change of formation [J/mol/K]
    'Sf',                   # Std-state mass entropy change of formation [J/kg/K]
    'Sfgm',                 # Ideal-gas molar entropy change of formation [J/mol/K]
    'Sfg',                  # Ideal-gas mass entropy change of formation [J/kg/K]

    'Hcgm',                 # Higher molar heat of combustion (ideal gas) [J/mol]
    'Hcg',                  # Higher mass heat of combustion (ideal gas) [J/kg]
    'Hcgm_lower',           # Lower molar heat of combustion (ideal gas) [J/mol]
    'Hcg_lower',            # Lower mass heat of combustion (ideal gas) [J/kg]

    'Tflash',               # Flash point [K]
    'Tautoignition',        # Autoignition temperature [K]
    'LFL',                  # Lower flammability limit at STP (mole fraction)
    'UFL',                  # Upper flammability limit at STP (mole fraction)
    'TWA',                  # Time-weighted average exposure limit (quantity, unit)
    'STEL',                 # Short-term exposure limit (quantity, unit)
    'Ceiling',              # Ceiling exposure limit (quantity, unit)
    'Skin',                 # True if significant skin absorption occurs
    'Carcinogen',           # Carcinogenic status (str or dict)

    'dipole',               # Dipole moment [debye]
    'Stockmayer',           # Lennard-Jones epsilon/k [K]
    'molecular_diameter',   # Lennard-Jones molecular diameter [angstrom]
    'GWP',                  # Global warming potential (relative to CO2) [-]
    'ODP',                  # Ozone depletion potential (relative to CFC-11) [-]
    'logP',                 # Octanol–water partition coefficient (log10 P) [-]
    'legal_status',         # Legal status information (str or dict)
    'economic_status',      # Economic status indicators (list/dict)

    'RI',                   # Refractive index (Na D line) [-]
    'RIT',                  # Temperature of RI measurement [K]
    'conductivity',         # Electrical conductivity [S/m]
    'conductivityT',        # Temperature of conductivity measurement [K]

    'VaporPressure',        # Vapor pressure correlation object
    'EnthalpyVaporization', # Enthalpy of vaporization correlation object
    'VolumeSolid',          # Solid molar volume correlation object
    'VolumeLiquid',         # Liquid molar volume correlation object
    'VolumeGas',            # Gas molar volume correlation object
    'HeatCapacitySolid',    # Solid heat capacity correlation object
    'HeatCapacityLiquid',   # Liquid heat capacity correlation object
    'HeatCapacityGas',      # Gas heat capacity correlation object
    'ViscosityLiquid',      # Liquid viscosity correlation object
    'ViscosityGas',         # Gas viscosity correlation object
    'ThermalConductivityLiquid',  # Liquid thermal conductivity correlation object
    'ThermalConductivityGas',     # Gas thermal conductivity correlation object
    'SurfaceTension',       # Surface tension correlation object
    'Permittivity',         # Liquid permittivity correlation object

    'Psat_298',             # Vapor pressure at 298.15 K [Pa]
    'phase_STP',            # Phase at 298.15 K and 101325 Pa ('s','l','g','l/g')
    'Vml_Tb',               # Liquid molar volume at normal boiling point [m^3/mol]
    'Vml_Tm',               # Liquid molar volume at melting point [m^3/mol]
    'Vml_STP',              # Liquid molar volume at 298.15 K, 101325 Pa [m^3/mol]
    'rhoml_STP',            # Liquid molar density at 298.15 K, 101325 Pa [mol/m^3]
    'Vmg_STP',              # Gas molar volume at 298.15 K, 101325 Pa (ideal gas) [m^3/mol]
    'Vms_Tm',               # Solid molar volume at melting point [m^3/mol]
    'rhos_Tm',              # Solid mass density at melting point [kg/m^3]
    'Hvap_Tbm',             # Molar enthalpy of vaporization at Tb [J/mol]
    'Hvap_Tb',              # Mass enthalpy of vaporization at Tb [J/kg]
    'Hvapm_298',            # Molar enthalpy of vaporization at 298.15 K [J/mol]
    'Hvap_298',             # Mass enthalpy of vaporization at 298.15 K [J/kg]

    'alpha',                # Thermal diffusivity at current state [m^2/s]
    'alphag',               # Gas-phase thermal diffusivity at current T,P [m^2/s]
    'alphal',               # Liquid-phase thermal diffusivity at current T,P [m^2/s]

    'API',                  # API gravity of the liquid [°API]
    'aromatic_rings',       # Number of aromatic rings (RDKit from SMILES)
    'atom_fractions',       # Dict: element -> atomic fraction
    'Bvirial',              # Second virial coefficient at current state [mol/m^3]
    'charge',               # Net molecular charge (RDKit from SMILES)

    'Cp',                   # Mass-basis heat capacity at current phase,T [J/kg/K]
    'Cpg',                  # Gas-phase mass heat capacity at current T [J/kg/K]
    'Cpgm',                 # Gas-phase ideal-gas molar Cp at current T [J/mol/K]
    'Cpl',                  # Liquid-phase mass heat capacity at current T [J/kg/K]
    'Cplm',                 # Liquid-phase molar heat capacity at current T [J/mol/K]
    'Cpm',                  # Molar heat capacity at current phase,T [J/mol/K]
    'Cps',                  # Solid-phase mass heat capacity at current T [J/kg/K]
    'Cpsm',                 # Solid-phase molar heat capacity at current T [J/mol/K]
    'Cvg',                  # Gas-phase ideal-gas Cv, mass basis [J/kg/K]
    'Cvgm',                 # Gas-phase ideal-gas Cv, molar basis [J/mol/K]

    'eos',                  # Equation-of-state object for thermodynamic props

    'Hill',                 # Hill formula (element-sorted formula)
    'Hvap',                 # Current mass enthalpy of vaporization [J/kg]
    'Hvapm',                # Current molar enthalpy of vaporization [J/mol]

    'isentropic_exponent',  # Gas-phase isentropic exponent k = Cp/Cv [-]
    'isobaric_expansion',   # Isobaric expansion coeff. (current phase) [1/K]
    'isobaric_expansion_g', # Isobaric expansion coeff. (gas phase) [1/K]
    'isobaric_expansion_l', # Isobaric expansion coeff. (liquid phase) [1/K]
    'JT',                   # Joule–Thomson coeff. (current phase) [K/Pa]
    'JTg',                  # Joule–Thomson coeff. (gas phase) [K/Pa]
    'JTl',                  # Joule–Thomson coeff. (liquid phase) [K/Pa]

    'k',                    # Thermal conductivity (current phase) [W/m/K]
    'kg',                   # Thermal conductivity (gas phase) [W/m/K]
    'kl',                   # Thermal conductivity (liquid phase) [W/m/K]

    'mass_fractions',       # Dict: component -> mass fraction
    'mu',                   # Dynamic viscosity (current phase) [Pa·s]
    'mug',                  # Dynamic viscosity (gas phase) [Pa·s]
    'mul',                  # Dynamic viscosity (liquid phase) [Pa·s]
    'nu',                   # Kinematic viscosity (current phase) [m^2/s]
    'nug',                  # Kinematic viscosity (gas phase) [m^2/s]
    'nul',                  # Kinematic viscosity (liquid phase) [m^2/s]

    'Parachor',             # Parachor at current T,P [N^0.25*m^2.75/mol]
    'permittivity',         # Relative permittivity (dielectric constant) [-]
    'Poynting',             # Poynting correction factor for phase equilibria [-]
    'Pr',                   # Prandtl number (current phase) [-]
    'Prg',                  # Prandtl number (gas phase) [-]
    'Prl',                  # Prandtl number (liquid phase) [-]

    'Psat',                 # Vapor pressure at current T [Pa]
    'PSRK_groups',          # PSRK subgroup counts (dict: group -> count)
    'rdkitmol',             # RDKit molecule object (no explicit H)
    'rdkitmol_Hs',          # RDKit molecule object (with explicit H)

    'rho',                  # Mass density (current phase) [kg/m^3]
    'rhog',                 # Mass density (gas phase) [kg/m^3]
    'rhogm',                # Molar density (gas phase) [mol/m^3]
    'rhol',                 # Mass density (liquid phase) [kg/m^3]
    'rholm',                # Molar density (liquid phase) [mol/m^3]
    'rhom',                 # Molar density (current phase) [mol/m^3]
    'rhos',                 # Mass density (solid phase) [kg/m^3]
    'rhosm',                # Molar density (solid phase) [mol/m^3]

    'rings',                # Total number of rings (RDKit from SMILES)
    'SG',                   # Specific gravity of the chemical [-]
    'SGg',                  # Specific gravity of gas phase [-]
    'SGl',                  # Specific gravity of liquid phase [-]
    'SGs',                  # Specific gravity of solid phase [-]

    'sigma',                # Surface tension at current T [N/m]
    'solubility_parameter', # Solubility parameter at current T,P [Pa^0.5]

    'UNIFAC_Dortmund_groups', # Dortmund-UNIFAC subgroup counts (dict)
    'UNIFAC_groups',          # UNIFAC subgroup counts (dict)
    'UNIFAC_R',               # UNIFAC R (normalized VdW volume) [-]
    'UNIFAC_Q',               # UNIFAC Q (normalized VdW area) [-]

    'Van_der_Waals_area',   # Unnormalized van der Waals area [m^2/mol]
    'Van_der_Waals_volume', # Unnormalized van der Waals volume [m^3/mol]

    'Vm',                   # Molar volume (current phase) [m^3/mol]
    'Vmg',                  # Molar volume (gas phase) [m^3/mol]
    'Vml',                  # Molar volume (liquid phase) [m^3/mol]
    'Vms',                  # Molar volume (solid phase) [m^3/mol]

    'Z',                    # Compressibility factor (current phase) [-]
    'Zg',                   # Compressibility factor (gas phase) [-]
    'Zl',                   # Compressibility factor (liquid phase) [-]
    'Zs',                   # Compressibility factor (solid phase) [-]
)

def get_chem(
    ids: Union[str, Iterable[str]],
    T: float = 298.15,
    P: float = 101325.0,
) -> pd.DataFrame:
    """
    Query thermo.chemical.Chemical for a list of IDs at a given T, P
    and return their properties in a pandas DataFrame.

    Compared to the old implementation based on ChemicalConstantsPackage:
    - This uses Chemical(ID, T, P) per species
    - It supports temperature- and pressure-dependent properties
    - It still returns a wide DataFrame with one row per input ID and
      one column per property listed in the global `properties` tuple

    Parameters
    ----------
    ids : str or Iterable[str]
        Chemical identifiers accepted by thermo.chemical.Chemical:
        - Name (IUPAC/common/synonym)
        - InChI ("InChI=1S/..." or "InChI=1/...")
        - InChI key ("InChIKey=...")
        - PubChem CID ("PubChem=...")
        - SMILES ("SMILES=...")
        - CAS number
    T : float, optional
        Temperature at which to instantiate Chemical objects, [K].
        Default is 298.15 K.
    P : float, optional
        Pressure at which to instantiate Chemical objects, [Pa].
        Default is 101325 Pa.

    Returns
    -------
    pandas.DataFrame
        - Index: input IDs, in the same order as provided.
        - Columns: all properties listed in the global `properties`
          (after applying `_replace_heads`), or a subset depending on
          `species`.
        - Any property which is missing or raises during access is
          set to None for that chemical.
    """
    # Normalize ids to a list
    if isinstance(ids, str):
        ids_list = [ids]
    else:
        ids_list = list(ids)

    rows: list[dict[str, object]] = []

    for identifier in ids_list:
        # Construct a Chemical object at the requested T and P.
        # Any failure here should not stop the whole batch; we just
        # fill the row with None.
        try:
            chem = Chemical(identifier, T=T, P=P)
        except Exception:
            # If construction fails, keep a row of None values.
            row = {prop: None for prop in properties}
            # But still record minimal ID/T/P information if present in our property list
            if 'ID' in properties:
                row['ID'] = identifier
            if 'T' in properties:
                row['T'] = T
            if 'P' in properties:
                row['P'] = P
            rows.append(row)
            continue

        row: dict[str, object] = {}
        for prop in properties:
            # For each requested property name, try to read it
            # from the Chemical instance. Any problem -> None.
            try:
                value = getattr(chem, prop)
            except Exception:
                value = None
            row[prop] = value

        rows.append(row)

    # Build a DataFrame from all rows
    return pd.DataFrame(rows, index=ids_list)


def mol_to_thermo(
        mol,
        T: float = 298.15,
        P: float = 101325.0,
):
    try:
        return Chemical(f'SMILES={Chem.MolToSmiles(mol.to_rdmol(), kekuleSmiles=True)}', P=P, T=T)
    except ValueError:
        pass

    try:
        return Chemical(Chem.MolToInchi(mol.to_rdmol()), P=P, T=T)
    except ValueError:
        pass

    try:  # CAS
        return Chemical(mol.cas, P=P, T=T)
    except ValueError:
        pass

    try:  # PubChem CID
        return Chemical(f'PubChem={mol.cid}', P=P, T=T)
    except ValueError:
        pass

    raise ValueError(f'Unknown thermo properties fro {Chem.MolToSmiles(mol.to_rdmol(), kekuleSmiles=True)}')

if __name__ == '__main__':
    # col = [
    #     'CAS', 'IUPAC_name', 'formula', 'smiles', 'PubChem', 'InChI_Key', 'InChI', 'MW', 'similarity_variable', 'Vmg_STP',
    #     'rhogm', 'rhog', 'similarity_variable', 'Cpg', 'Cpgm', 'Cpl', 'Cplm', 'Cps', 'Cpsm', 'Cvg', 'Cvgm',
    #     "isentropic_exponent", 'JTg', 'kl', 'rhog', 'SGg', 'rhos',  'MW', "Tm"
    # ]
    #
    # df = get_chem(['7647-01-0'])  # '64-17-5'是乙醇，'7732-18-5'是水
    #
    # df = df[col]

    import hotpot as hp
    mol_to_thermo(hp.read_mol('CCO'))
