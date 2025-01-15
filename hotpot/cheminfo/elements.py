import numpy as np
import periodictable
import openbabel.openbabel as ob

class Element:
    """ Represents library to query elements information """

    _symbols = (
        "0",
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "", ""
    )

    _atomic_orbital = [       # Periodic
        [2],                  # 1: 1s
        [2, 6],               # 2: 2s, 2p
        [2, 6],               # 3: 3s, 3p
        [2, 10, 6],           # 4: 4s, 3d, 4p
        [2, 10, 6],           # 5: 5s, 4d, 5p
        [2, 14, 10, 6],       # 6: 6s, 4f, 5d, 6p
        [2, 14, 10, 6],       # 7: 7s, 5f, 6d, 7p
        [2, 18, 14, 10, 6]    # 8: 8s, 5g, 6f, 7d, 8p
    ]

    _default_valence = {
        1: 1,    # Hydrogen (H)
        2: 0,    # Helium (He) - inert
        3: 1,    # Lithium (Li)
        4: 2,    # Beryllium (Be)
        5: 3,    # Boron (B)
        6: 4,    # Carbon (C)
        7: 3,    # Nitrogen (N)
        8: 2,    # Oxygen (O)
        9: 1,    # Fluorine (F)
        10: 0,   # Neon (Ne) - inert
        11: 1,   # Sodium (Na)
        12: 2,   # Magnesium (Mg)
        13: 3,   # Aluminium (Al)
        14: 4,   # Silicon (Si)
        15: 3,   # Phosphorus (P)
        16: 2,   # Sulfur (S)
        17: 1,   # Chlorine (Cl)
        18: 0,   # Argon (Ar) - inert
        19: 1,   # Potassium (K)
        20: 2,   # Calcium (Ca)
        21: 3,   # Scandium (Sc)
        22: 4,   # Titanium (Ti)
        23: 5,   # Vanadium (V)
        24: 3,   # Chromium (Cr)
        25: 2,   # Manganese (Mn)
        26: 2,   # Iron (Fe)
        27: 3,   # Cobalt (Co)
        28: 2,   # Nickel (Ni)
        29: 2,   # Copper (Cu)
        30: 2,   # Zinc (Zn)
        31: 3,   # Gallium (Ga)
        32: 4,   # Germanium (Ge)
        33: 3,   # Arsenic (As)
        34: 2,   # Selenium (Se)
        35: 1,   # Bromine (Br)
        36: 0,   # Krypton (Kr) - inert
        37: 1,   # Rubidium (Rb)
        38: 2,   # Strontium (Sr)
        39: 3,   # Yttrium (Y)
        40: 4,   # Zirconium (Zr)
        41: 5,   # Niobium (Nb)
        42: 6,   # Molybdenum (Mo)
        43: 7,   # Technetium (Tc)
        44: 4,   # Ruthenium (Ru)
        45: 3,   # Rhodium (Rh)
        46: 2,   # Palladium (Pd)
        47: 1,   # Silver (Ag)
        48: 2,   # Cadmium (Cd)
        49: 3,   # Indium (In)
        50: 4,   # Tin (Sn)
        51: 3,   # Antimony (Sb)
        52: 2,   # Tellurium (Te)
        53: 1,   # Iodine (I)
        54: 0,   # Xenon (Xe) - inert
        55: 1,   # Cesium (Cs)
        56: 2,   # Barium (Ba)
        57: 3,   # Lanthanum (La)
        58: 3,   # Cerium (Ce)
        59: 3,   # Praseodymium (Pr)
        60: 3,   # Neodymium (Nd)
        61: 3,   # Promethium (Pm)
        62: 3,   # Samarium (Sm)
        63: 3,   # Europium (Eu)
        64: 3,   # Gadolinium (Gd)
        65: 3,   # Terbium (Tb)
        66: 3,   # Dysprosium (Dy)
        67: 3,   # Holmium (Ho)
        68: 3,   # Erbium (Er)
        69: 3,   # Thulium (Tm)
        70: 3,   # Ytterbium (Yb)
        71: 3,   # Lutetium (Lu)
        72: 4,   # Hafnium (Hf)
        73: 5,   # Tantalum (Ta)
        74: 6,   # Tungsten (W)
        75: 5,   # Rhenium (Re)
        76: 4,   # Osmium (Os)
        77: 3,   # Iridium (Ir)
        78: 2,   # Platinum (Pt)
        79: 1,   # Gold (Au)
        80: 2,   # Mercury (Hg)
        81: 3,   # Thallium (Tl)
        82: 4,   # Lead (Pb)
        83: 3,   # Bismuth (Bi)
        84: 2,   # Polonium (Po)
        85: 1,   # Astatine (At)
        86: 0,   # Radon (Rn) - inert
        87: 1,   # Francium (Fr)
        88: 2,   # Radium (Ra)
        89: 3,   # Actinium (Ac)
        90: 4,   # Thorium (Th)
        91: 5,   # Protactinium (Pa)
        92: 6,   # Uranium (U)
        93: 5,   # Neptunium (Np)
        94: 6,   # Plutonium (Pu)
        95: 3,   # Americium (Am)
        96: 3,   # Curium (Cm)
        97: 3,   # Berkelium (Bk)
        98: 3,   # Californium (Cf)
        99: 3,   # Einsteinium (Es)
        100: 3,  # Fermium (Fm)
        101: 3,  # Mendelevium (Md)
        102: 3,  # Nobelium (No)
        103: 3,  # Lawrencium (Lr)
        104: 4,  # Rutherfordium (Rf)
        105: 5,  # Dubnium (Db)
        106: 6,  # Seaborgium (Sg)
        107: 7,  # Bohrium (Bh)
        108: 4,  # Hassium (Hs)
        109: 3,  # Meitnerium (Mt)
        110: 4,  # Darmstadtium (Ds)
        111: 1,  # Roentgenium (Rg)
        112: 2,  # Copernicium (Cn)
        113: 3,  # Nihonium (Nh)
        114: 4,  # Flerovium (Fl)
        115: 3,  # Moscovium (Mc)
        116: 2,  # Livermorium (Lv)
        117: 1,  # Tennessine (Ts)
        118: 0,  # Oganesson (Og) - inert
    }

    _valence_dict = {
        1: {"stable": [1], "unstable": [-1]},  # Hydrogen
        2: {"stable": [0], "unstable": []},  # Helium
        3: {"stable": [1], "unstable": []},  # Lithium
        4: {"stable": [2], "unstable": []},  # Beryllium
        5: {"stable": [3], "unstable": [-3]},  # Boron
        6: {"stable": [4], "unstable": [2]},  # Carbon
        7: {"stable": [-3, -2, -1, 3, 4, 5], "unstable": [1, 2]},  # Nitrogen
        8: {"stable": [2], "unstable": [-2]},  # Oxygen
        9: {"stable": [1], "unstable": [-1]},  # Fluorine
        10: {"stable": [0], "unstable": []},  # Neon
        11: {"stable": [1], "unstable": []},  # Sodium
        12: {"stable": [2], "unstable": []},  # Magnesium
        13: {"stable": [3], "unstable": []},  # Aluminum
        14: {"stable": [-4, 4], "unstable": [2]},  # Silicon
        15: {"stable": [-3, 1, 3, 5], "unstable": []},  # Phosphorus
        16: {"stable": [-2, 2, 4, 6], "unstable": []},  # Sulfur
        17: {"stable": [-1, 1, 3, 5, 7], "unstable": [2, 4]},  # Chlorine
        18: {"stable": [0], "unstable": []},  # Argon
        19: {"stable": [1], "unstable": []},  # Potassium
        20: {"stable": [2], "unstable": []},  # Calcium
        21: {"stable": [3], "unstable": []},  # Scandium
        22: {"stable": [2, 3, 4], "unstable": []},  # Titanium
        23: {"stable": [2, 3, 4, 5], "unstable": []},  # Vanadium
        24: {"stable": [2, 3, 6], "unstable": []},  # Chromium
        25: {"stable": [2, 4, 7], "unstable": [3, 6]},  # Manganese
        26: {"stable": [2, 3], "unstable": [4, 6]},  # Iron
        27: {"stable": [2, 3], "unstable": [4]},  # Cobalt
        28: {"stable": [2], "unstable": [1, 3, 4]},  # Nickel
        29: {"stable": [1, 2], "unstable": [3]},  # Copper
        30: {"stable": [2], "unstable": []},  # Zinc
        31: {"stable": [3], "unstable": [2]},  # Gallium
        32: {"stable": [-4, 2, 4], "unstable": []},  # Germanium
        33: {"stable": [-3, 3, 5], "unstable": [2]},  # Arsenic
        34: {"stable": [-2, 4, 6], "unstable": [2]},  # Selenium
        35: {"stable": [-1, 1, 5], "unstable": [3, 4]},  # Bromine
        36: {"stable": [0], "unstable": []},  # Krypton
        37: {"stable": [1], "unstable": []},  # Rubidium
        38: {"stable": [2], "unstable": []},  # Strontium
        39: {"stable": [3], "unstable": []},  # Yttrium
        40: {"stable": [4], "unstable": [2, 3]},  # Zirconium
        41: {"stable": [3, 5], "unstable": [2, 4]},  # Niobium
        42: {"stable": [3, 6], "unstable": [2, 4, 5]},  # Molybdenum
        43: {"stable": [6], "unstable": []},  # Technetium
        44: {"stable": [3, 4, 8], "unstable": [2, 6, 7]},  # Ruthenium
        45: {"stable": [4], "unstable": [2, 3, 6]},  # Rhodium
        46: {"stable": [2, 4], "unstable": [6]},  # Palladium
        47: {"stable": [1], "unstable": [2, 3]},  # Silver
        48: {"stable": [2], "unstable": [1]},  # Cadmium
        49: {"stable": [3], "unstable": [1, 2]},  # Indium
        50: {"stable": [2, 4], "unstable": []},  # Tin
        51: {"stable": [-3, 3, 5], "unstable": [4]},  # Antimony
        52: {"stable": [-2, 4, 6], "unstable": [2]},  # Tellurium
        53: {"stable": [-1, 1, 5, 7], "unstable": [3, 4]},  # Iodine
        54: {"stable": [0], "unstable": []},  # Xenon
        55: {"stable": [1], "unstable": []},  # Cesium
        56: {"stable": [2], "unstable": []},  # Barium
        57: {"stable": [3], "unstable": []},  # Lanthanum
        58: {"stable": [3, 4], "unstable": []},  # Cerium
        59: {"stable": [3], "unstable": []},  # Praseodymium
        60: {"stable": [3, 4], "unstable": []},  # Neodymium
        61: {"stable": [3], "unstable": []},  # Promethium
        62: {"stable": [3], "unstable": [2]},  # Samarium
        63: {"stable": [3], "unstable": [2]},  # Europium
        64: {"stable": [3], "unstable": []},  # Gadolinium
        65: {"stable": [3, 4], "unstable": []},  # Terbium
        66: {"stable": [3], "unstable": []},  # Dysprosium
        67: {"stable": [3], "unstable": []},  # Holmium
        68: {"stable": [3], "unstable": []},  # Erbium
        69: {"stable": [3], "unstable": [2]},  # Thulium
        70: {"stable": [3], "unstable": [2]},  # Ytterbium
        71: {"stable": [3], "unstable": []},  # Lutetium
        72: {"stable": [4], "unstable": []},  # Hafnium
        73: {"stable": [5], "unstable": [3, 4]},  # Tantalum
        74: {"stable": [6], "unstable": [2, 3, 4, 5]},  # Tungsten
        75: {"stable": [2, 4, 6, 7], "unstable": [-1, 1, 3, 5]},  # Rhenium
        76: {"stable": [3, 4, 6, 8], "unstable": [2]},  # Osmium
        77: {"stable": [3, 4, 6], "unstable": [1, 2]},  # Iridium
        78: {"stable": [2, 4, 6], "unstable": [1, 3]},  # Platinum
        79: {"stable": [1, 3], "unstable": [2]},  # Gold
        80: {"stable": [1, 2], "unstable": []},  # Mercury
        81: {"stable": [1, 3], "unstable": [2]},  # Thallium
        82: {"stable": [2, 4], "unstable": []},  # Lead
        83: {"stable": [3], "unstable": [-3, 2, 4, 5]},  # Bismuth
        84: {"stable": [2, 4], "unstable": [-2, 6]},  # Polonium
        85: {"stable": [-1], "unstable": []},  # Astatine
        86: {"stable": [0], "unstable": []},  # Radon
        87: {"stable": [1], "unstable": []},  # Francium
        88: {"stable": [2], "unstable": []},  # Radium
        89: {"stable": [3], "unstable": []},  # Actinium
        90: {"stable": [4], "unstable": []},  # Thorium
        91: {"stable": [5], "unstable": []},  # Protactinium
        92: {"stable": [3, 4, 6], "unstable": [2, 5]}  # Uranium
    }

    _electronegativity = {
        1: 2.20,  # Hydrogen (H)
        2: None,  # Helium (He)
        3: 0.98,  # Lithium (Li)
        4: 1.57,  # Beryllium (Be)
        5: 2.04,  # Boron (B)
        6: 2.55,  # Carbon (C)
        7: 3.04,  # Nitrogen (N)
        8: 3.44,  # Oxygen (O)
        9: 3.98,  # Fluorine (F)
        10: None,  # Neon (Ne)
        11: 0.93,  # Sodium (Na)
        12: 1.31,  # Magnesium (Mg)
        13: 1.61,  # Aluminum (Al)
        14: 1.90,  # Silicon (Si)
        15: 2.19,  # Phosphorus (P)
        16: 2.58,  # Sulfur (S)
        17: 3.16,  # Chlorine (Cl)
        18: None,  # Argon (Ar)
        19: 0.82,  # Potassium (K)
        20: 1.00,  # Calcium (Ca)
        21: 1.36,  # Scandium (Sc)
        22: 1.54,  # Titanium (Ti)
        23: 1.63,  # Vanadium (V)
        24: 1.66,  # Chromium (Cr)
        25: 1.55,  # Manganese (Mn)
        26: 1.83,  # Iron (Fe)
        27: 1.88,  # Cobalt (Co)
        28: 1.91,  # Nickel (Ni)
        29: 1.90,  # Copper (Cu)
        30: 1.65,  # Zinc (Zn)
        31: 1.81,  # Gallium (Ga)
        32: 2.01,  # Germanium (Ge)
        33: 2.18,  # Arsenic (As)
        34: 2.55,  # Selenium (Se)
        35: 2.96,  # Bromine (Br)
        36: 3.00,  # Krypton (Kr)
        37: 0.82,  # Rubidium (Rb)
        38: 0.95,  # Strontium (Sr)
        39: 1.22,  # Yttrium (Y)
        40: 1.33,  # Zirconium (Zr)
        41: 1.60,  # Niobium (Nb)
        42: 2.16,  # Molybdenum (Mo)
        43: 1.90,  # Technetium (Tc)
        44: 2.20,  # Ruthenium (Ru)
        45: 2.28,  # Rhodium (Rh)
        46: 2.20,  # Palladium (Pd)
        47: 1.93,  # Silver (Ag)
        48: 1.69,  # Cadmium (Cd)
        49: 1.78,  # Indium (In)
        50: 1.96,  # Tin (Sn)
        51: 2.05,  # Antimony (Sb)
        52: 2.10,  # Tellurium (Te)
        53: 2.66,  # Iodine (I)
        54: 2.60,  # Xenon (Xe)
        55: 0.79,  # Cesium (Cs)
        56: 0.89,  # Barium (Ba)
        57: 1.10,  # Lanthanum (La)
        58: 1.12,  # Cerium (Ce)
        59: 1.13,  # Praseodymium (Pr)
        60: 1.14,  # Neodymium (Nd)
        61: 1.13,  # Promethium (Pm)
        62: 1.17,  # Samarium (Sm)
        63: 1.20,  # Europium (Eu)
        64: 1.20,  # Gadolinium (Gd)
        65: 1.22,  # Terbium (Tb)
        66: 1.23,  # Dysprosium (Dy)
        67: 1.24,  # Holmium (Ho)
        68: 1.24,  # Erbium (Er)
        69: 1.25,  # Thulium (Tm)
        70: 1.10,  # Ytterbium (Yb)
        71: 1.27,  # Lutetium (Lu)
        72: 1.30,  # Hafnium (Hf)
        73: 1.50,  # Tantalum (Ta)
        74: 2.36,  # Tungsten (W)
        75: 1.90,  # Rhenium (Re)
        76: 2.20,  # Osmium (Os)
        77: 2.20,  # Iridium (Ir)
        78: 2.28,  # Platinum (Pt)
        79: 2.54,  # Gold (Au)
        80: 2.00,  # Mercury (Hg)
        81: 1.62,  # Thallium (Tl)
        82: 2.33,  # Lead (Pb)
        83: 2.02,  # Bismuth (Bi)
        84: 2.00,  # Polonium (Po)
        85: 2.20,  # Astatine (At)
        86: None,  # Radon (Rn)
        87: 0.70,  # Francium (Fr)
        88: 0.89,  # Radium (Ra)
        89: 1.10,  # Actinium (Ac)
        90: 1.30,  # Thorium (Th)
        91: 1.50,  # Protactinium (Pa)
        92: 1.38,  # Uranium (U)
        93: 1.36,  # Neptunium (Np)
        94: 1.28,  # Plutonium (Pu)
        95: 1.30,  # Americium (Am)
        96: 1.30,  # Curium (Cm)
        97: 1.30,  # Berkelium (Bk)
        98: 1.30,  # Californium (Cf)
        99: 1.30,  # Einsteinium (Es)
        100: 1.30,  # Fermium (Fm)
        101: 1.30,  # Mendelevium (Md)
        102: 1.30,  # Nobelium (No)
        103: None,  # Lawrencium (Lr)
        104: None,  # Rutherfordium (Rf)
        105: None,  # Dubnium (Db)
        106: None,  # Seaborgium (Sg)
        107: None,  # Bohrium (Bh)
        108: None,  # Hassium (Hs)
        109: None,  # Meitnerium (Mt)
        110: None,  # Darmstadtium (Ds)
        111: None,  # Roentgenium (Rg)
        112: None,  # Copernicium (Cn)
        113: None,  # Nihonium (Nh)
        114: None,  # Flerovium (Fl)
        115: None,  # Moscovium (Mc)
        116: None,  # Livermorium (Lv)
        117: None,  # Tennessine (Ts)
        118: None  # Oganesson (Og)
    }

    # Element categorize in periodic tabel
    _alkali_metals = {3, 11, 19, 37, 55, 87}  # Group 1
    _alkaline_earth_metals = {4, 12, 20, 38, 56, 88}  # Group 2
    _transition_metals = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))
    _post_transition_metals = {13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116}
    _lanthanides = set(range(57, 72))
    _actinides = set(range(89, 104))
    metal_ = _alkali_metals|_alkaline_earth_metals|_transition_metals|_post_transition_metals|_lanthanides|_actinides

    _nonmetals = [1, 6, 7, 8, 15, 16, 34]
    _metalloids = [5, 14, 32, 33, 51, 52, 84]
    _noble_gases = [2, 10, 18, 36, 54, 86, 118]
    _halogens = [9, 17, 35, 53, 85, 117]

    covalent_radii = np.array([0.] + [getattr(periodictable, ob.GetSymbol(i)).covalent_radius or 0. for i in range(1, 119)])
