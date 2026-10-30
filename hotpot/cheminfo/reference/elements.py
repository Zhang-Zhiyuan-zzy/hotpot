"""Immutable element facts; public mutable snapshots live in elements.py."""

from .properties import _ELEMENT_SYMBOLS, _freeze


__all__ = ()


_ATOMIC_ORBITAL = _freeze([       # Periodic
        [2],                  # 1: 1s
        [2, 6],               # 2: 2s, 2p
        [2, 6],               # 3: 3s, 3p
        [2, 10, 6],           # 4: 4s, 3d, 4p
        [2, 10, 6],           # 5: 5s, 4d, 5p
        [2, 14, 10, 6],       # 6: 6s, 4f, 5d, 6p
        [2, 14, 10, 6],       # 7: 7s, 5f, 6d, 7p
        [2, 18, 14, 10, 6]    # 8: 8s, 5g, 6f, 7d, 8p
    ])


_DEFAULT_VALENCE = _freeze({
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
    })


_VALENCE_DICT = _freeze({
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
    })


_ATOMIC_RADII = _freeze([
    # Unit: pm
        120,  # 1, Hydrogen, H
        140,  # 2, Helium, He
        182,  # 3, Lithium, Li
        153,  # 4, Beryllium, Be
        192,  # 5, Boron, B
        170,  # 6, Carbon, C
        155,  # 7, Nitrogen, N
        152,  # 8, Oxygen, O
        147,  # 9, Fluorine, F
        154,  # 10, Neon, Ne
        227,  # 11, Sodium, Na
        173,  # 12, Magnesium, Mg
        184,  # 13, Aluminum, Al
        210,  # 14, Silicon, Si
        180,  # 15, Phosphorus, P
        180,  # 16, Sulfur, S
        175,  # 17, Chlorine, Cl
        188,  # 18, Argon, Ar
        275,  # 19, Potassium, K
        231,  # 20, Calcium, Ca
        211,  # 21, Scandium, Sc
        187,  # 22, Titanium, Ti
        179,  # 23, Vanadium, V
        189,  # 24, Chromium, Cr
        197,  # 25, Manganese, Mn
        194,  # 26, Iron, Fe
        192,  # 27, Cobalt, Co
        163,  # 28, Nickel, Ni
        140,  # 29, Copper, Cu
        139,  # 30, Zinc, Zn
        187,  # 31, Gallium, Ga
        211,  # 32, Germanium, Ge
        185,  # 33, Arsenic, As
        190,  # 34, Selenium, Se
        185,  # 35, Bromine, Br
        202,  # 36, Krypton, Kr
        303,  # 37, Rubidium, Rb
        249,  # 38, Strontium, Sr
        212,  # 39, Yttrium, Y
        206,  # 40, Zirconium, Zr
        198,  # 41, Niobium, Nb
        190,  # 42, Molybdenum, Mo
        183,  # 43, Technetium, Tc
        178,  # 44, Ruthenium, Ru
        173,  # 45, Rhodium, Rh
        169,  # 46, Palladium, Pd
        165,  # 47, Silver, Ag
        158,  # 48, Cadmium, Cd
        193,  # 49, Indium, In
        217,  # 50, Tin, Sn
        206,  # 51, Antimony, Sb
        206,  # 52, Tellurium, Te
        198,  # 53, Iodine, I
        216,  # 54, Xenon, Xe
        343,  # 55, Cesium, Cs
        268,  # 56, Barium, Ba
        240,  # 57, Lanthanum, La
        235,  # 58, Cerium, Ce
        239,  # 59, Praseodymium, Pr
        229,  # 60, Neodymium, Nd
        236,  # 61, Promethium, Pm
        229,  # 62, Samarium, Sm
        233,  # 63, Europium, Eu
        237,  # 64, Gadolinium, Gd
        221,  # 65, Terbium, Tb
        229,  # 66, Dysprosium, Dy
        216,  # 67, Holmium, Ho
        214,  # 68, Erbium, Er
        213,  # 69, Thulium, Tm
        242,  # 70, Ytterbium, Yb
        221,  # 71, Lutetium, Lu
        212,  # 72, Hafnium, Hf
        217,  # 73, Tantalum, Ta
        210,  # 74, Tungsten, W
        217,  # 75, Rhenium, Re
        216,  # 76, Osmium, Os
        202,  # 77, Iridium, Ir
        209,  # 78, Platinum, Pt
        166,  # 79, Gold, Au
        209,  # 80, Mercury, Hg
        196,  # 81, Thallium, Tl
        202,  # 82, Lead, Pb
        207,  # 83, Bismuth, Bi
        197,  # 84, Polonium, Po
        None,  # 85, Astatine, At
        220,  # 86, Radon, Rn
        None,  # 87, Francium, Fr
        283,  # 88, Radium, Ra
        260,  # 89, Actinium, Ac
        237,  # 90, Thorium, Th
        243,  # 91, Protactinium, Pa
        246,  # 92, Uranium, U
        241,  # 93, Neptunium, Np
        243,  # 94, Plutonium, Pu
        244,  # 95, Americium, Am
        None,  # 96, Curium, Cm
        None,  # 97, Berkelium, Bk
        None,  # 98, Californium, Cf
        None,  # 99, Einsteinium, Es
        None,  # 100, Fermium, Fm
        None,  # 101, Mendelevium, Md
        None,  # 102, Nobelium, No
        None,  # 103, Lawrencium, Lr
        None,  # 104, Rutherfordium, Rf
        None,  # 105, Dubnium, Db
        None,  # 106, Seaborgium, Sg
        None,  # 107, Bohrium, Bh
        None,  # 108, Hassium, Hs
        None,  # 109, Meitnerium, Mt
        None,  # 110, Darmstadtium, Ds
        None,  # 111, Roentgenium, Rg
        None,  # 112, Copernicium, Cn
        None,  # 113, Nihonium, Nh
        None,  # 114, Flerovium, Fl
        None,  # 115, Moscovium, Mc
        None,  # 116, Livermorium, Lv
        None,  # 117, Tennessine, Ts
        None,  # 118, Oganesson, Og
    ])


_IONIC_RADII = _freeze({
        # Period 1
        1: {  # Hydrogen (H)
            1: {  # H⁺
                2: {"spin": "norm", "crystal_radius": 0.004, "ionic_radius": 0.018},  # II
                1: {"spin": "norm", "crystal_radius": 0.024, "ionic_radius": 0.038},  # I
            },
            -1: {  # H⁻
                6: {"spin": "norm", "crystal_radius": 0.154, "ionic_radius": 0.154},  # VI
            }
        },
        2: {  # Helium (He)
            # 无常见离子
        },

        # Period 2
        3: {  # Lithium (Li)
            1: {
                4: {"spin": "norm", "crystal_radius": 0.059, "ionic_radius": 0.076},
                6: {"spin": "norm", "crystal_radius": 0.076, "ionic_radius": 0.090},
            }
        },
        4: {  # Beryllium (Be)
            2: {
                4: {"spin": "norm", "crystal_radius": 0.027, "ionic_radius": 0.045},
                6: {"spin": "norm", "crystal_radius": 0.045, "ionic_radius": 0.059},
            }
        },
        5: {  # Boron (B)
            3: {
                4: {"spin": "norm", "crystal_radius": 0.023, "ionic_radius": 0.027},
                6: {"spin": "norm", "crystal_radius": 0.027, "ionic_radius": 0.041},
            }
        },
        6: {  # Carbon (C)
            4: {
                4: {"spin": "norm", "crystal_radius": 0.016, "ionic_radius": 0.029},
                6: {"spin": "norm", "crystal_radius": 0.029, "ionic_radius": 0.036},
            },
            -4: {
                6: {"spin": "norm", "crystal_radius": 0.260, "ionic_radius": 0.260},
            },
        },
        7: {  # Nitrogen (N)
            -3: {
                4: {"spin": "norm", "crystal_radius": 0.132, "ionic_radius": 0.146},
                6: {"spin": "norm", "crystal_radius": 0.171, "ionic_radius": 0.171},
            },
            3: {
                6: {"spin": "norm", "crystal_radius": 0.030, "ionic_radius": 0.016},
            },
            5: {
                3: {"spin": "norm", "crystal_radius": 0.044, "ionic_radius": -0.0104},
                6: {"spin": "norm", "crystal_radius": 0.027, "ionic_radius": 0.013},
            }
        },
        8: {  # Oxygen (O)
            -2: {
                4: {"spin": "norm", "crystal_radius": 0.140, "ionic_radius": 0.140},
                6: {"spin": "norm", "crystal_radius": 0.140, "ionic_radius": 0.140}
            },
            2: {
                4: {"spin": "norm", "crystal_radius": 0.014, "ionic_radius": 0.014},
                6: {"spin": "norm", "crystal_radius": 0.026, "ionic_radius": 0.026}
            }
        },
        9: {  # Fluorine (F)
            -1: {
                4: {"spin": "norm", "crystal_radius": 0.119, "ionic_radius": 0.119},
                6: {"spin": "norm", "crystal_radius": 0.133, "ionic_radius": 0.133},
            }
        },
        10: {  # Neon (Ne)
            # 无常见离子
        },

        # Period 3
        11: {  # Sodium (Na)
            1: {
                4: {"spin": "norm", "crystal_radius": 0.099, "ionic_radius": 0.099},  # IV
                6: {"spin": "norm", "crystal_radius": 0.102, "ionic_radius": 0.102},  # VI
                8: {"spin": "norm", "crystal_radius": 0.116, "ionic_radius": 0.118},  # VIII
            }
        },
        12: {  # Magnesium (Mg)
            2: {
                4: {"spin": "norm", "crystal_radius": 0.057, "ionic_radius": 0.057},  # IV
                6: {"spin": "norm", "crystal_radius": 0.072, "ionic_radius": 0.072},  # VI
            }
        },
        13: {  # Aluminum (Al)
            3: {
                4: {"spin": "norm", "crystal_radius": 0.039, "ionic_radius": 0.039},  # IV
                6: {"spin": "norm", "crystal_radius": 0.054, "ionic_radius": 0.054},  # VI
            }
        },
        14: {  # Silicon (Si)
            4: {
                4: {"spin": "norm", "crystal_radius": 0.026, "ionic_radius": 0.026},  # IV
                6: {"spin": "norm", "crystal_radius": 0.040, "ionic_radius": 0.040},  # VI
            }
        },
        15: {  # Phosphorus (P)
            3: {
                4: {"spin": "norm", "crystal_radius": 0.029, "ionic_radius": 0.029},  # IV
                6: {"spin": "norm", "crystal_radius": 0.044, "ionic_radius": 0.044},  # VI
            },
            5: {
                6: {"spin": "norm", "crystal_radius": 0.038, "ionic_radius": 0.038},  # VI
            }
        },
        16: {  # Sulfur (S)
            -2: {
                6: {"spin": "norm", "crystal_radius": 0.184, "ionic_radius": 0.184},  # VI
            },
            4: {
                4: {"spin": "norm", "crystal_radius": 0.037, "ionic_radius": 0.037},  # IV
            },
            6: {
                6: {"spin": "norm", "crystal_radius": 0.030, "ionic_radius": 0.030},  # VI
            }
        },
        17: {  # Chlorine (Cl)
            -1: {
                6: {"spin": "norm", "crystal_radius": 0.181, "ionic_radius": 0.181},  # VI
            }
        },
        18: {  # Argon (Ar)
            # 稳定稀有气体，无常见离子
        },

        # Period 4
        19: {  # Potassium (K)
            1: {
                6: {"spin": "norm", "crystal_radius": 0.138, "ionic_radius": 0.138},  # VI
                8: {"spin": "norm", "crystal_radius": 0.151, "ionic_radius": 0.151},  # VIII
                12: {"spin": "norm", "crystal_radius": 0.169, "ionic_radius": 0.169}, # XII
            }
        },
        20: {  # Calcium (Ca)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.100, "ionic_radius": 0.100},  # VI
                8: {"spin": "norm", "crystal_radius": 0.112, "ionic_radius": 0.112},  # VIII
            }
        },
        21: {  # Scandium (Sc)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.075, "ionic_radius": 0.075},  # VI
                8: {"spin": "norm", "crystal_radius": 0.087, "ionic_radius": 0.087},  # VIII
            }
        },
        22: {  # Titanium (Ti)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.086, "ionic_radius": 0.086},  # VI
            },
            3: {
                6: {"spin": "norm", "crystal_radius": 0.067, "ionic_radius": 0.067},  # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.060, "ionic_radius": 0.060},  # VI
                8: {"spin": "norm", "crystal_radius": 0.074, "ionic_radius": 0.074},  # VIII
            }
        },
        23: {  # Vanadium (V)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.079, "ionic_radius": 0.079},  # VI
            },
            3: {
                6: {"spin": "norm", "crystal_radius": 0.064, "ionic_radius": 0.064},  # VI
                8: {"spin": "norm", "crystal_radius": 0.079, "ionic_radius": 0.079},  # VIII
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.058, "ionic_radius": 0.058},  # VI
            },
            5: {
                6: {"spin": "norm", "crystal_radius": 0.054, "ionic_radius": 0.054},  # VI
            }
        },
        24: {  # Chromium (Cr)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.073, "ionic_radius": 0.073},  # VI
            },
            3: {
                6: {"spin": "high-spin", "crystal_radius": 0.062, "ionic_radius": 0.062},  # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.055, "ionic_radius": 0.055},   # VI 低自旋
            },
            6: {
                4: {"spin": "norm", "crystal_radius": 0.026, "ionic_radius": 0.026},  # IV
            }
        },
        25: {  # Manganese (Mn)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.083, "ionic_radius": 0.083},  # VI
            },
            3: {
                6: {"spin": "high-spin", "crystal_radius": 0.064, "ionic_radius": 0.064},   # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.058, "ionic_radius": 0.058},    # VI 低自旋
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.053, "ionic_radius": 0.053},  # VI
            },
            7: {
                4: {"spin": "norm", "crystal_radius": 0.046, "ionic_radius": 0.046},  # IV
            }
        },
        26: {  # Iron (Fe)
            2: {
                6: {"spin": "high-spin", "crystal_radius": 0.092, "ionic_radius": 0.092},  # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.078, "ionic_radius": 0.078},  # VI 低自旋
            },
            3: {
                6: {"spin": "high-spin", "crystal_radius": 0.064, "ionic_radius": 0.064},  # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.055, "ionic_radius": 0.055},   # VI 低自旋
            }
        },
        27: {  # Cobalt (Co)
            2: {
                6: {"spin": "high-spin", "crystal_radius": 0.088, "ionic_radius": 0.088},  # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.065, "ionic_radius": 0.065},   # VI 低自旋
            },
            3: {
                6: {"spin": "high-spin", "crystal_radius": 0.061, "ionic_radius": 0.061},   # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.054, "ionic_radius": 0.054},    # VI 低自旋
            }
        },
        28: {  # Nickel (Ni)
            2: {
                6: {"spin": "high-spin", "crystal_radius": 0.083, "ionic_radius": 0.083},  # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.069, "ionic_radius": 0.069},   # VI 低自旋
            },
            3: {
                6: {"spin": "high-spin", "crystal_radius": 0.060, "ionic_radius": 0.060},   # VI 高自旋
                # 6: {"spin": "low-spin", "crystal_radius": 0.056, "ionic_radius": 0.056},    # VI 低自旋
            }
        },
        29: {  # Copper (Cu)
            1: {
                6: {"spin": "norm", "crystal_radius": 0.096, "ionic_radius": 0.096},  # VI
            },
            2: {
                4: {"spin": "norm", "crystal_radius": 0.071, "ionic_radius": 0.071},  # IV
                6: {"spin": "norm", "crystal_radius": 0.073, "ionic_radius": 0.073},  # VI
            }
        },
        30: {  # Zinc (Zn)
            2: {
                4: {"spin": "norm", "crystal_radius": 0.074, "ionic_radius": 0.074},  # IV
                6: {"spin": "norm", "crystal_radius": 0.074, "ionic_radius": 0.074},  # VI
            }
        },
        31: {  # Gallium (Ga)
            3: {
                4: {"spin": "norm", "crystal_radius": 0.047, "ionic_radius": 0.047},  # IV
                6: {"spin": "norm", "crystal_radius": 0.062, "ionic_radius": 0.062},  # VI
            }
        },
        32: {  # Germanium (Ge)
            4: {
                4: {"spin": "norm", "crystal_radius": 0.039, "ionic_radius": 0.039},  # IV
                6: {"spin": "norm", "crystal_radius": 0.053, "ionic_radius": 0.053},  # VI
            }
        },
        33: {  # Arsenic (As)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.058, "ionic_radius": 0.058},  # VI
            },
            5: {
                6: {"spin": "norm", "crystal_radius": 0.046, "ionic_radius": 0.046},  # VI
            }
        },
        34: {  # Selenium (Se)
            -2: {
                6: {"spin": "norm", "crystal_radius": 0.198, "ionic_radius": 0.198},  # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.050, "ionic_radius": 0.050},  # VI
            },
            6: {
                6: {"spin": "norm", "crystal_radius": 0.042, "ionic_radius": 0.042},  # VI
            }
        },
        35: {  # Bromine (Br)
            -1: {
                6: {"spin": "norm", "crystal_radius": 0.196, "ionic_radius": 0.196},  # VI
            }
        },
        36: {  # Krypton (Kr)
            # 稳定稀有气体，无常见离子
        },

        # Period 5
        37: {  # Rubidium (Rb)
            1: {
                6: {"spin": "norm", "crystal_radius": 0.152, "ionic_radius": 0.152},   # VI
                8: {"spin": "norm", "crystal_radius": 0.166, "ionic_radius": 0.166},   # VIII
                12: {"spin": "norm", "crystal_radius": 0.183, "ionic_radius": 0.183},  # XII
            }
        },
        38: {  # Strontium (Sr)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.118, "ionic_radius": 0.118},   # VI
                8: {"spin": "norm", "crystal_radius": 0.132, "ionic_radius": 0.132},   # VIII
            }
        },
        39: {  # Yttrium (Y)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.090, "ionic_radius": 0.090},   # VI
                8: {"spin": "norm", "crystal_radius": 0.104, "ionic_radius": 0.104},   # VIII
            }
        },
        40: {  # Zirconium (Zr)
            4: {
                6: {"spin": "norm", "crystal_radius": 0.072, "ionic_radius": 0.072},   # VI
                8: {"spin": "norm", "crystal_radius": 0.084, "ionic_radius": 0.084},   # VIII
            }
        },
        41: {  # Niobium (Nb)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.086, "ionic_radius": 0.086},   # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.078, "ionic_radius": 0.078},   # VI
            },
            5: {
                6: {"spin": "norm", "crystal_radius": 0.069, "ionic_radius": 0.069},   # VI
                8: {"spin": "norm", "crystal_radius": 0.080, "ionic_radius": 0.080},   # VIII
            }
        },
        42: {  # Molybdenum (Mo)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.069, "ionic_radius": 0.069},   # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.065, "ionic_radius": 0.065},   # VI
            },
            5: {
                6: {"spin": "norm", "crystal_radius": 0.061, "ionic_radius": 0.061},   # VI
            },
            6: {
                4: {"spin": "norm", "crystal_radius": 0.041, "ionic_radius": 0.041},   # IV
            }
        },
        43: {  # Technetium (Tc)
            4: {
                6: {"spin": "norm", "crystal_radius": 0.064, "ionic_radius": 0.064},   # VI
            },
            7: {
                4: {"spin": "norm", "crystal_radius": 0.045, "ionic_radius": 0.045},   # IV
            }
        },
        44: {  # Ruthenium (Ru)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.068, "ionic_radius": 0.068},   # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.062, "ionic_radius": 0.062},   # VI
            },
            8: {
                4: {"spin": "norm", "crystal_radius": 0.040, "ionic_radius": 0.040},   # IV （Ru8+）
            }
        },
        45: {  # Rhodium (Rh)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.067, "ionic_radius": 0.067},   # VI
            }
        },
        46: {  # Palladium (Pd)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.086, "ionic_radius": 0.086},   # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.065, "ionic_radius": 0.065},   # VI
            }
        },
        47: {  # Silver (Ag)
            1: {
                6: {"spin": "norm", "crystal_radius": 0.129, "ionic_radius": 0.129},   # VI
                8: {"spin": "norm", "crystal_radius": 0.142, "ionic_radius": 0.142},   # VIII
            },
            2: {
                6: {"spin": "norm", "crystal_radius": 0.089, "ionic_radius": 0.089},   # VI
            },
            3: {
                6: {"spin": "norm", "crystal_radius": 0.069, "ionic_radius": 0.069},   # VI
            }
        },
        48: {  # Cadmium (Cd)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.097, "ionic_radius": 0.097},   # VI
                8: {"spin": "norm", "crystal_radius": 0.109, "ionic_radius": 0.109},   # VIII
            }
        },
        49: {  # Indium (In)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.094, "ionic_radius": 0.094},   # VI
                8: {"spin": "norm", "crystal_radius": 0.107, "ionic_radius": 0.107},   # VIII
            }
        },
        50: {  # Tin (Sn)
            2: {
                6: {"spin": "norm", "crystal_radius": 0.112, "ionic_radius": 0.112},   # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.083, "ionic_radius": 0.083},   # VI
                8: {"spin": "norm", "crystal_radius": 0.095, "ionic_radius": 0.095},   # VIII
            }
        },
        51: {  # Antimony (Sb)
            3: {
                6: {"spin": "norm", "crystal_radius": 0.090, "ionic_radius": 0.090},   # VI
            },
            5: {
                6: {"spin": "norm", "crystal_radius": 0.076, "ionic_radius": 0.076},   # VI
            }
        },
        52: {  # Tellurium (Te)
            -2: {
                6: {"spin": "norm", "crystal_radius": 0.221, "ionic_radius": 0.221},   # VI
            },
            4: {
                6: {"spin": "norm", "crystal_radius": 0.097, "ionic_radius": 0.097},   # VI
            },
            6: {
                6: {"spin": "norm", "crystal_radius": 0.089, "ionic_radius": 0.089},   # VI
            }
        },
        53: {  # Iodine (I)
            -1: {
                6: {"spin": "norm", "crystal_radius": 0.220, "ionic_radius": 0.220},   # VI
            }
        },
        54: {  # Xenon (Xe)
            8: {
                4: {"spin": "norm", "crystal_radius": 0.074, "ionic_radius": 0.074},   # IV
            }
        },

        # Period 6
        55: {  # Cesium (Cs)
            1: {
                6:  {"spin": "norm", "crystal_radius": 0.167, "ionic_radius": 0.167},   # VI
                8:  {"spin": "norm", "crystal_radius": 0.181, "ionic_radius": 0.181},   # VIII
                12: {"spin": "norm", "crystal_radius": 0.202, "ionic_radius": 0.202},   # XII
            }
        },
        56: {  # Barium (Ba)
            2: {
                6:  {"spin": "norm", "crystal_radius": 0.135, "ionic_radius": 0.135},   # VI
                8:  {"spin": "norm", "crystal_radius": 0.149, "ionic_radius": 0.149},   # VIII
            }
        },
        # 镧系（57–71）
        57: {  # Lanthanum (La)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.103, "ionic_radius": 0.103},   # VI
                7:  {"spin": "norm", "crystal_radius": 0.116, "ionic_radius": 0.116},   # VII
                8:  {"spin": "norm", "crystal_radius": 0.122, "ionic_radius": 0.122},   # VIII
                9:  {"spin": "norm", "crystal_radius": 0.132, "ionic_radius": 0.132},   # IX
            }
        },
        58: {  # Cerium (Ce)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.101, "ionic_radius": 0.101},   # VI
                7:  {"spin": "norm", "crystal_radius": 0.114, "ionic_radius": 0.114},   # VII
                8:  {"spin": "norm", "crystal_radius": 0.119, "ionic_radius": 0.119},   # VIII
            },
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.087, "ionic_radius": 0.087},   # VI
                8:  {"spin": "norm", "crystal_radius": 0.097, "ionic_radius": 0.097},   # VIII
            }
        },
        59: {  # Praseodymium (Pr)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.099, "ionic_radius": 0.099},
                8:  {"spin": "norm", "crystal_radius": 0.118, "ionic_radius": 0.118},
            }
        },
        60: {  # Neodymium (Nd)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.098, "ionic_radius": 0.098},
                7:  {"spin": "norm", "crystal_radius": 0.110, "ionic_radius": 0.110},
                8:  {"spin": "norm", "crystal_radius": 0.112, "ionic_radius": 0.112},
            }
        },
        61: {  # Promethium (Pm)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.097, "ionic_radius": 0.097},
                8:  {"spin": "norm", "crystal_radius": 0.110, "ionic_radius": 0.110},
            }
        },
        62: {  # Samarium (Sm)
            2: {
                7:  {"spin": "norm", "crystal_radius": 0.125, "ionic_radius": 0.125},
                8:  {"spin": "norm", "crystal_radius": 0.132, "ionic_radius": 0.132},
            },
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.096, "ionic_radius": 0.096},
                7:  {"spin": "norm", "crystal_radius": 0.107, "ionic_radius": 0.107},
                8:  {"spin": "norm", "crystal_radius": 0.122, "ionic_radius": 0.122},
            }
        },
        63: {  # Europium (Eu)
            2: {
                7:  {"spin": "norm", "crystal_radius": 0.131, "ionic_radius": 0.131},
                8:  {"spin": "norm", "crystal_radius": 0.135, "ionic_radius": 0.135},
            },
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.095, "ionic_radius": 0.095},
                7:  {"spin": "norm", "crystal_radius": 0.106, "ionic_radius": 0.106},
                8:  {"spin": "norm", "crystal_radius": 0.120, "ionic_radius": 0.120},
            }
        },
        64: {  # Gadolinium (Gd)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.093, "ionic_radius": 0.093},
                7:  {"spin": "norm", "crystal_radius": 0.105, "ionic_radius": 0.105},
                8:  {"spin": "norm", "crystal_radius": 0.118, "ionic_radius": 0.118},
            }
        },
        65: {  # Terbium (Tb)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.092, "ionic_radius": 0.092},
                7:  {"spin": "norm", "crystal_radius": 0.104, "ionic_radius": 0.104},
                8:  {"spin": "norm", "crystal_radius": 0.117, "ionic_radius": 0.117},
            }
        },
        66: {  # Dysprosium (Dy)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.091, "ionic_radius": 0.091},
                7:  {"spin": "norm", "crystal_radius": 0.103, "ionic_radius": 0.103},
                8:  {"spin": "norm", "crystal_radius": 0.116, "ionic_radius": 0.116},
            }
        },
        67: {  # Holmium (Ho)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.090, "ionic_radius": 0.090},
                8:  {"spin": "norm", "crystal_radius": 0.115, "ionic_radius": 0.115},
            }
        },
        68: {  # Erbium (Er)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.089, "ionic_radius": 0.089},
                7:  {"spin": "norm", "crystal_radius": 0.102, "ionic_radius": 0.102},
                8:  {"spin": "norm", "crystal_radius": 0.114, "ionic_radius": 0.114},
            }
        },
        69: {  # Thulium (Tm)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.088, "ionic_radius": 0.088},
                8:  {"spin": "norm", "crystal_radius": 0.113, "ionic_radius": 0.113},
            }
        },
        70: {  # Ytterbium (Yb)
            2: {
                8:  {"spin": "norm", "crystal_radius": 0.125, "ionic_radius": 0.125},
            },
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.087, "ionic_radius": 0.087},
                8:  {"spin": "norm", "crystal_radius": 0.112, "ionic_radius": 0.112},
            }
        },
        71: {  # Lutetium (Lu)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.086, "ionic_radius": 0.086},
                7:  {"spin": "norm", "crystal_radius": 0.097, "ionic_radius": 0.097},
                8:  {"spin": "norm", "crystal_radius": 0.111, "ionic_radius": 0.111},
            }
        },
        72: {  # Hafnium (Hf)
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.071, "ionic_radius": 0.071},
                8:  {"spin": "norm", "crystal_radius": 0.083, "ionic_radius": 0.083},
            }
        },
        73: {  # Tantalum (Ta)
            5: {
                6:  {"spin": "norm", "crystal_radius": 0.064, "ionic_radius": 0.064},
            }
        },
        74: {  # Tungsten (W)
            6: {
                4:  {"spin": "norm", "crystal_radius": 0.042, "ionic_radius": 0.042},
                6:  {"spin": "norm", "crystal_radius": 0.060, "ionic_radius": 0.060},
            }
        },
        75: {  # Rhenium (Re)
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.063, "ionic_radius": 0.063},
            },
            7: {
                4:  {"spin": "norm", "crystal_radius": 0.044, "ionic_radius": 0.044},
            }
        },
        76: {  # Osmium (Os)
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.062, "ionic_radius": 0.062},
            },
            8: {
                4:  {"spin": "norm", "crystal_radius": 0.037, "ionic_radius": 0.037},
            }
        },
        77: {  # Iridium (Ir)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.068, "ionic_radius": 0.068},
            },
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.062, "ionic_radius": 0.062},
            }
        },
        78: {  # Platinum (Pt)
            2: {
                6:  {"spin": "norm", "crystal_radius": 0.094, "ionic_radius": 0.094},
            },
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.064, "ionic_radius": 0.064},
            }
        },
        79: {  # Gold (Au)
            1: {
                6:  {"spin": "norm", "crystal_radius": 0.137, "ionic_radius": 0.137},
            },
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.085, "ionic_radius": 0.085},
            }
        },
        80: {  # Mercury (Hg)
            1: {
                6:  {"spin": "norm", "crystal_radius": 0.119, "ionic_radius": 0.119},
            },
            2: {
                6:  {"spin": "norm", "crystal_radius": 0.116, "ionic_radius": 0.116},
                8:  {"spin": "norm", "crystal_radius": 0.128, "ionic_radius": 0.128},
            }
        },
        81: {  # Thallium (Tl)
            1: {
                6:  {"spin": "norm", "crystal_radius": 0.159, "ionic_radius": 0.159},
            },
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.103, "ionic_radius": 0.103},
            }
        },
        82: {  # Lead (Pb)
            2: {
                6:  {"spin": "norm", "crystal_radius": 0.119, "ionic_radius": 0.119},
                8:  {"spin": "norm", "crystal_radius": 0.133, "ionic_radius": 0.133},
            },
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.094, "ionic_radius": 0.094},
            }
        },
        83: {  # Bismuth (Bi)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.107, "ionic_radius": 0.107},
            },
            5: {
                6:  {"spin": "norm", "crystal_radius": 0.076, "ionic_radius": 0.076},
            }
        },
        84: {  # Polonium (Po)
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.094, "ionic_radius": 0.094},
            },
            6: {
                6:  {"spin": "norm", "crystal_radius": 0.089, "ionic_radius": 0.089},
            }
        },
        85: {  # Astatine (At)
            -1: {
                6:  {"spin": "norm", "crystal_radius": 0.230, "ionic_radius": 0.230},   # VI
            }
        },
        86: {  # Radon (Rn)
            # 无常见离子
        },
        # 锕系（90–103），谨摘录表格可查部分
        89: {  # Actinium (Ac)
            3: {
                6:  {"spin": "norm", "crystal_radius": 0.112, "ionic_radius": 0.112},
            }
        },
        90: {  # Thorium (Th)
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.094, "ionic_radius": 0.094},
                8:  {"spin": "norm", "crystal_radius": 0.104, "ionic_radius": 0.104},
            }
        },
        91: {  # Protactinium (Pa)
            5: {
                6:  {"spin": "norm", "crystal_radius": 0.090, "ionic_radius": 0.090},
            }
        },
        92: {  # Uranium (U)
            3: {
                8:  {"spin": "norm", "crystal_radius": 0.117, "ionic_radius": 0.117},
            },
            4: {
                6:  {"spin": "norm", "crystal_radius": 0.089, "ionic_radius": 0.089},
                8:  {"spin": "norm", "crystal_radius": 0.103, "ionic_radius": 0.103},
            },
            5: {
                6:  {"spin": "norm", "crystal_radius": 0.076, "ionic_radius": 0.076},
                8:  {"spin": "norm", "crystal_radius": 0.090, "ionic_radius": 0.090},
            },
            6: {
                6:  {"spin": "norm", "crystal_radius": 0.073, "ionic_radius": 0.073},
            }
        },
        93: {  # Neptunium (Np)
            3: {
                8:  {"spin": "norm", "crystal_radius": 0.115, "ionic_radius": 0.115},
            },
            4: {
                8:  {"spin": "norm", "crystal_radius": 0.101, "ionic_radius": 0.101},
            },
            5: {
                8:  {"spin": "norm", "crystal_radius": 0.089, "ionic_radius": 0.089},
            },
            6: {
                8:  {"spin": "norm", "crystal_radius": 0.087, "ionic_radius": 0.087},
            },
            7: {
                8:  {"spin": "norm", "crystal_radius": 0.086, "ionic_radius": 0.086},
            }
        },
        94: {  # Plutonium (Pu)
            3: {
                8:  {"spin": "norm", "crystal_radius": 0.114, "ionic_radius": 0.114},
            },
            4: {
                8:  {"spin": "norm", "crystal_radius": 0.100, "ionic_radius": 0.100},
            },
            5: {
                8:  {"spin": "norm", "crystal_radius": 0.088, "ionic_radius": 0.088},
            },
            6: {
                8:  {"spin": "norm", "crystal_radius": 0.086, "ionic_radius": 0.086},
            }
        },
        95: {  # Americium (Am)
            3: {
                8:  {"spin": "norm", "crystal_radius": 0.111, "ionic_radius": 0.111},
            }
        },
        96: {  # Curium (Cm)
            3: {
                8:  {"spin": "norm", "crystal_radius": 0.109, "ionic_radius": 0.109},
            }
        },
        # 97–103无表内可靠数据，仅保留空白结构或略去
        104: { # Rutherfordium (Rf)
            4: {
                8:  {"spin": "norm", "crystal_radius": 0.100, "ionic_radius": 0.100},  # 少见超重元素估算
            }
        },
        # 其余105-118, 若查无则不加

    })


_IONIZATION_ENERGIES = _freeze([
        [13.59, None, None],  # Hydrogen, H
        [24.58, 54.41, None],  # Helium, He
        [5.39, 75.64, 122.45],  # Lithium, Li
        [9.32, 18.21, 153.89],  # Beryllium, Be
        [8.29, 25.15, 37.93],  # Boron, B
        [11.26, 24.38, 47.88],  # Carbon, C
        [14.53, 29.60, 47.44],  # Nitrogen, N
        [13.61, 35.11, 54.93],  # Oxygen, O
        [17.42, 34.97, 62.70],  # Fluorine, F
        [21.56, 40.96, 63.45],  # Neon, Ne
        [5.13, 47.28, 71.62],  # Sodium, Na
        [7.64, 15.03, 80.14],  # Magnesium, Mg
        [5.98, 18.82, 28.44],  # Aluminum, Al
        [8.15, 16.34, 33.49],  # Silicon, Si
        [10.48, 19.76, 30.20],  # Phosphorus, P
        [10.36, 23.33, 34.79],  # Sulfur, S
        [12.96, 23.81, 39.61],  # Chlorine, Cl
        [15.75, 27.62, 40.74],  # Argon, Ar
        [4.34, 31.63, 45.80],  # Potassium, K
        [6.11, 11.87, 50.91],  # Calcium, Ca
        [6.56, 12.79, 24.75],  # Scandium, Sc
        [6.82, 13.57, 27.49],  # Titanium, Ti
        [6.74, 14.66, 29.31],  # Vanadium, V
        [6.76, 16.48, 30.96],  # Chromium, Cr
        [7.43, 15.63, 33.66],  # Manganese, Mn
        [7.90, 16.18, 30.65],  # Iron, Fe
        [7.88, 17.08, 33.50],  # Cobalt, Co
        [7.63, 18.16, 35.19],  # Nickel, Ni
        [7.72, 20.29, 36.84],  # Copper, Cu
        [9.39, 17.96, 39.72],  # Zinc, Zn
        [5.99, 20.51, 30.71],  # Gallium, Ga
        [7.89, 15.93, 34.22],  # Germanium, Ge
        [9.78, 18.63, 28.35],  # Arsenic, As
        [9.75, 21.19, 30.82],  # Selenium, Se
        [11.81, 21.8, 36.00],  # Bromine, Br
        [13.99, 24.35, 36.95],  # Krypton, Kr
        [4.17, 27.28, 40.00],  # Rubidium, Rb
        [5.69, 11.03, 42.89],  # Strontium, Sr
        [6.21, 12.24, 20.52],  # Yttrium, Y
        [6.63, 13.13, 22.99],  # Zirconium, Zr
        [6.75, 14.32, 25.04],  # Niobium, Nb
        [7.09, 16.16, 27.13],  # Molybdenum, Mo
        [7.28, 15.26, 29.54],  # Technetium, Tc
        [7.36, 16.76, 28.47],  # Ruthenium, Ru
        [7.45, 18.08, 31.06],  # Rhodium, Rh
        [8.33, 19.43, 32.93],  # Palladium, Pd
        [7.57, 21.49, 34.83],  # Silver, Ag
        [8.99, 16.90, 37.48],  # Cadmium, Cd
        [5.78, 18.86, 28.03],  # Indium, In
        [7.34, 14.63, 30.50],  # Tin, Sn
        [8.60, 16.53, 25.30],  # Antimony, Sb
        [9.00, 18.60, 27.96],  # Tellurium, Te
        [10.45, 19.13, 33.00],  # Iodine, I
        [12.12, 21.20, 32.12],  # Xenon, Xe
        [3.89, 23.15, None],  # Cesium, Cs
        [5.21, 10.00, None],  # Barium, Ba
        [5.57, 11.06, 19.17],  # Lanthanum, La
        [5.53, 10.85, 20.19],  # Cerium, Ce
        [5.47, 10.55, 21.62],  # Praseodymium, Pr
        [5.52, 10.73, 22.21],  # Neodymium, Nd
        [5.58, 10.90, 22.30],  # Promethium, Pm
        [5.64, 11.07, 23.40],  # Samarium, Sm
        [5.67, 11.24, 24.92],  # Europium, Eu
        [6.15, 12.09, 20.63],  # Gadolinium, Gd
        [5.86, 11.52, 21.91],  # Terbium, Tb
        [5.93, 11.67, 22.80],  # Dysprosium, Dy
        [6.02, 11.80, 22.84],  # Holmium, Ho
        [6.10, 11.93, 22.74],  # Erbium, Er
        [6.18, 12.05, 23.63],  # Thulium, Tm
        [6.25, 12.17, 25.05],  # Ytterbium, Yb
        [5.42, 13.90, 20.95],  # Lutetium, Lu
        [6.82, 14.90, 23.30],  # Hafnium, Hf
        [7.54, None, None],  # Tantalum, Ta
        [7.86, None, None],  # Tungsten, W
        [7.83, None, None],  # Rhenium, Re
        [8.43, None, None],  # Osmium, Os
        [8.96, None, None],  # Iridium, Ir
        [8.95, 18.56, 28.00],  # Platinum, Pt
        [9.22, 20.50, 30.00],  # Gold, Au
        [10.43, 18.75, 34.20],  # Mercury, Hg
        [6.10, 20.42, 29.83],  # Thallium, Tl
        [7.41, 15.03, 31.93],  # Lead, Pb
        [7.28, 16.69, 25.56],  # Bismuth, Bi
        [8.41, None, None],  # Polonium, Po
        [9.31, None, None],  # Astatine, At
        [10.74, None, None],  # Radon, Rn
        [4.07, None, None],  # Francium, Fr
        [5.27, 10.14, None],  # Radium, Ra
        [5.17, 12.10, None],  # Actinium, Ac
        [6.30, 11.50, 20.00],  # Thorium, Th
        [5.89, None, None],  # Protactinium, Pa
        [6.19, None, None],  # Uranium, U
        [6.26, None, None],  # Neptunium, Np
        [6.02, None, None],  # Plutonium, Pu
        [5.97, None, None],  # Americium, Am
        [5.99, None, None],  # Curium, Cm
        [6.19, None, None],  # Berkelium, Bk
        [6.28, None, None],  # Californium, Cf
        [6.42, None, None],  # Einsteinium, Es
        [6.50, None, None],  # Fermium, Fm
        [6.58, None, None],  # Mendelevium, Md
        [6.65, None, None],  # Nobelium, No
        [4.90, None, None],  # Lawrencium, Lr
        [6.00, None, None],  # Rutherfordium, Rf
        [None, None, None],  # Dubnium, Db
        [None, None, None],  # Seaborgium, Sg
        [None, None, None],  # Bohrium, Bh
        [None, None, None],  # Hassium, Hs
        [None, None, None],  # Meitnerium, Mt
        [None, None, None],  # Darmstadtium, Ds
        [None, None, None],  # Roentgenium, Rg
        [None, None, None],  # Copernicium, Cn
        [None, None, None],  # Nihonium, Nh
        [None, None, None],  # Flerovium, Fl
        [None, None, None],  # Moscovium, Mc
        [None, None, None],  # Livermorium, Lv
        [None, None, None],  # Tennessine, Ts
        [None, None, None],  # Oganesson, Og
    ])


_ELEMENT_DENSITIES = _freeze([
        0.0899,  # Hydrogen, H (g/L, 0°C, 101.325 kPa)
        0.179,  # Helium, He (g/L, 0°C, 101.325 kPa)
        0.535,  # Lithium, Li (g/cm³)
        1.85,  # Beryllium, Be (g/cm³)
        2.47,  # Boron, B (g/cm³)
        2.26,  # Carbon, C (g/cm³, graphite; diamond: 3.51)
        1.25,  # Nitrogen, N (g/L, 0°C, 101.325 kPa)
        1.43,  # Oxygen, O (g/L, 0°C, 101.325 kPa)
        1.67,  # Fluorine, F (g/L, 0°C, 101.325 kPa)
        0.9,  # Neon, Ne (g/L, 0°C, 101.325 kPa)
        0.968,  # Sodium, Na (g/cm³)
        1.74,  # Magnesium, Mg (g/cm³)
        2.7,  # Aluminum, Al (g/cm³)
        2.33,  # Silicon, Si (g/cm³)
        1.823,  # Phosphorus, P (g/cm³)
        1.96,  # Sulfur, S (g/cm³)
        3.214,  # Chlorine, Cl (g/L, 0°C, 101.325 kPa)
        1.79,  # Argon, Ar (g/L, 0°C, 101.325 kPa)
        0.856,  # Potassium, K (g/cm³)
        1.55,  # Calcium, Ca (g/cm³)
        2.985,  # Scandium, Sc (g/cm³)
        4.507,  # Titanium, Ti (g/cm³)
        6.11,  # Vanadium, V (g/cm³)
        7.19,  # Chromium, Cr (g/cm³)
        7.47,  # Manganese, Mn (g/cm³)
        7.875,  # Iron, Fe (g/cm³)
        8.9,  # Cobalt, Co (g/cm³)
        8.908,  # Nickel, Ni (g/cm³)
        8.96,  # Copper, Cu (g/cm³)
        7.14,  # Zinc, Zn (g/cm³)
        5.904,  # Gallium, Ga (g/cm³)
        5.323,  # Germanium, Ge (g/cm³)
        5.73,  # Arsenic, As (g/cm³)
        4.82,  # Selenium, Se (g/cm³)
        3.12,  # Bromine, Br (g/cm³)
        3.75,  # Krypton, Kr (g/L, 0°C, 101.325 kPa)
        1.53,  # Rubidium, Rb (g/cm³)
        2.63,  # Strontium, Sr (g/cm³)
        4.472,  # Yttrium, Y (g/cm³)
        6.511,  # Zirconium, Zr (g/cm³)
        8.57,  # Niobium, Nb (g/cm³)
        10.28,  # Molybdenum, Mo (g/cm³)
        11.5,  # Technetium, Tc (g/cm³)
        12.37,  # Ruthenium, Ru (g/cm³)
        12.45,  # Rhodium, Rh (g/cm³)
        12.023,  # Palladium, Pd (g/cm³)
        10.5,  # Silver, Ag (g/cm³)
        8.65,  # Cadmium, Cd (g/cm³)
        7.31,  # Indium, In (g/cm³)
        7.31,  # Tin, Sn (g/cm³)
        6.7,  # Antimony, Sb (g/cm³)
        6.24,  # Tellurium, Te (g/cm³)
        4.94,  # Iodine, I (g/cm³)
        5.9,  # Xenon, Xe (g/L, 0°C, 101.325 kPa)
        1.88,  # Cesium, Cs (g/cm³)
        3.51,  # Barium, Ba (g/cm³)
        6.15,  # Lanthanum, La (g/cm³)
        6.69,  # Cerium, Ce (g/cm³)
        6.64,  # Praseodymium, Pr (g/cm³)
        7.01,  # Neodymium, Nd (g/cm³)
        7.26,  # Promethium, Pm (g/cm³)
        7.35,  # Samarium, Sm (g/cm³)
        5.25,  # Europium, Eu (g/cm³)
        7.9,  # Gadolinium, Gd (g/cm³)
        8.22,  # Terbium, Tb (g/cm³)
        8.55,  # Dysprosium, Dy (g/cm³)
        8.8,  # Holmium, Ho (g/cm³)
        9.06,  # Erbium, Er (g/cm³)
        9.32,  # Thulium, Tm (g/cm³)
        6.57,  # Ytterbium, Yb (g/cm³)
        9.84,  # Lutetium, Lu (g/cm³)
        13.31,  # Hafnium, Hf (g/cm³)
        16.65,  # Tantalum, Ta (g/cm³)
        19.25,  # Tungsten, W (g/cm³)
        21.02,  # Rhenium, Re (g/cm³)
        22.6,  # Osmium, Os (g/cm³)
        22.56,  # Iridium, Ir (g/cm³)
        21.45,  # Platinum, Pt (g/cm³)
        19.3,  # Gold, Au (g/cm³)
        13.535,  # Mercury, Hg (g/cm³)
        11.85,  # Thallium, Tl (g/cm³)
        11.34,  # Lead, Pb (g/cm³)
        9.78,  # Bismuth, Bi (g/cm³)
        9.16,  # Polonium, Po (g/cm³)
        None,  # Astatine, At (no data)
        9.73,  # Radon, Rn (g/L, 0°C, 101.325 kPa)
        None,  # Francium, Fr (no data)
        5.0,  # Radium, Ra (g/cm³)
        10.07,  # Actinium, Ac (g/cm³)
        11.725,  # Thorium, Th (g/cm³)
        15.37,  # Protactinium, Pa (g/cm³)
        19.05,  # Uranium, U (g/cm³)
        20.45,  # Neptunium, Np (g/cm³)
        19.82,  # Plutonium, Pu (g/cm³)
        13.67,  # Americium, Am (g/cm³)
        13.51,  # Curium, Cm (g/cm³)
        14.79,  # Berkelium, Bk (g/cm³)
        15.1,  # Californium, Cf (g/cm³)
        None,  # Einsteinium, Es (no data)
        None,  # Fermium, Fm (no data)
        None,  # Mendelevium, Md (no data)
        None,  # Nobelium, No (no data)
        None,  # Lawrencium, Lr (no data)
        None,  # Rutherfordium, Rf (no data)
        None,  # Dubnium, Db (no data)
        None,  # Seaborgium, Sg (no data)
        None,  # Bohrium, Bh (no data)
        None,  # Hassium, Hs (no data)
        None,  # Meitnerium, Mt (no data)
        None,  # Darmstadtium, Ds (no data)
        None,  # Roentgenium, Rg (no data)
        None,  # Copernicium, Cn (no data)
        None,  # Nihonium, Nh (no data)
        None,  # Flerovium, Fl (no data)
        None,  # Moscovium, Mc (no data)
        None,  # Livermorium, Lv (no data)
        None,  # Tennessine, Ts (no data)
        None,  # Oganesson, Og (no data)
    ])


_ELECTRONEGATIVITY = _freeze({
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
    })


_ELECTRON_CONFIGS = (
    #   (n, s, p, d, f, g)
        (0, 0, 0, 0, 0, 0),  # Unknown
        (1, 1, 0, 0, 0, 0),  # 1, H
        (1, 2, 0, 0, 0, 0),  # 2, He

        # Z=3~10 (Period 2)
        (2, 1, 0, 0, 0, 0),  # 3, Li
        (2, 2, 0, 0, 0, 0),  # 4, Be
        (2, 2, 1, 0, 0, 0),  # 5, B
        (2, 2, 2, 0, 0, 0),  # 6, C
        (2, 2, 3, 0, 0, 0),  # 7, N
        (2, 2, 4, 0, 0, 0),  # 8, O
        (2, 2, 5, 0, 0, 0),  # 9, F
        (2, 2, 6, 0, 0, 0),  # 10, Ne

        # Z=11~18 (Period 3)
        (3, 1, 0, 0, 0, 0),  # 11, Na
        (3, 2, 0, 0, 0, 0),  # 12, Mg
        (3, 2, 1, 0, 0, 0),  # 13, Al
        (3, 2, 2, 0, 0, 0),  # 14, Si
        (3, 2, 3, 0, 0, 0),  # 15, P
        (3, 2, 4, 0, 0, 0),  # 16, S
        (3, 2, 5, 0, 0, 0),  # 17, Cl
        (3, 2, 6, 0, 0, 0),  # 18, Ar

        # Z=19~36 (Period 4)
        (4, 1, 0, 0, 0, 0),  # 19, K
        (4, 2, 0, 0, 0, 0),  # 20, Ca
        (4, 2, 0, 1, 0, 0),  # 21, Sc  (4s2 3d1)
        (4, 2, 0, 2, 0, 0),  # 22, Ti
        (4, 2, 0, 3, 0, 0),  # 23, V
        (4, 1, 0, 5, 0, 0),  # 24, Cr  (4s1 3d5)
        (4, 2, 0, 5, 0, 0),  # 25, Mn
        (4, 2, 0, 6, 0, 0),  # 26, Fe
        (4, 2, 0, 7, 0, 0),  # 27, Co
        (4, 2, 0, 8, 0, 0),  # 28, Ni
        (4, 1, 0, 10, 0, 0),  # 29, Cu (4s1 3d10)
        (4, 2, 0, 10, 0, 0),  # 30, Zn
        (4, 2, 1, 0, 0, 0),  # 31, Ga
        (4, 2, 2, 0, 0, 0),  # 32, Ge
        (4, 2, 3, 0, 0, 0),  # 33, As
        (4, 2, 4, 0, 0, 0),  # 34, Se
        (4, 2, 5, 0, 0, 0),  # 35, Br
        (4, 2, 6, 0, 0, 0),  # 36, Kr

        # Z=37~54 (Period 5)
        (5, 1, 0, 0, 0, 0),  # 37, Rb
        (5, 2, 0, 0, 0, 0),  # 38, Sr
        (5, 2, 0, 1, 0, 0),  # 39, Y
        (5, 2, 0, 2, 0, 0),  # 40, Zr
        (5, 1, 0, 4, 0, 0),  # 41, Nb (5s1 4d4)
        (5, 1, 0, 5, 0, 0),  # 42, Mo
        (5, 2, 0, 5, 0, 0),  # 43, Tc
        (5, 1, 0, 7, 0, 0),  # 44, Ru
        (5, 1, 0, 8, 0, 0),  # 45, Rh
        (5, 0, 0, 10, 0, 0),  # 46, Pd (5s0 4d10)
        (5, 1, 0, 10, 0, 0),  # 47, Ag
        (5, 2, 0, 10, 0, 0),  # 48, Cd
        (5, 2, 1, 0, 0, 0),  # 49, In
        (5, 2, 2, 0, 0, 0),  # 50, Sn
        (5, 2, 3, 0, 0, 0),  # 51, Sb
        (5, 2, 4, 0, 0, 0),  # 52, Te
        (5, 2, 5, 0, 0, 0),  # 53, I
        (5, 2, 6, 0, 0, 0),  # 54, Xe

        # Z=55~86 (Period 6)
        (6, 1, 0, 0, 0, 0),  # 55, Cs
        (6, 2, 0, 0, 0, 0),  # 56, Ba
        (6, 2, 0, 1, 0, 0),  # 57, La (5d1 6s2)
        (6, 2, 0, 1, 1, 0),  # 58, Ce (4f1 5d1 6s2)
        (6, 2, 0, 0, 3, 0),  # 59, Pr (4f3 6s2)
        (6, 2, 0, 0, 4, 0),  # 60, Nd
        (6, 2, 0, 0, 5, 0),  # 61, Pm
        (6, 2, 0, 0, 6, 0),  # 62, Sm
        (6, 2, 0, 0, 7, 0),  # 63, Eu
        (6, 2, 0, 1, 7, 0),  # 64, Gd (4f7 5d1)
        (6, 2, 0, 0, 9, 0),  # 65, Tb (4f9)
        (6, 2, 0, 0, 10, 0),  # 66, Dy
        (6, 2, 0, 0, 11, 0),  # 67, Ho
        (6, 2, 0, 0, 12, 0),  # 68, Er
        (6, 2, 0, 0, 13, 0),  # 69, Tm
        (6, 2, 0, 0, 14, 0),  # 70, Yb
        (6, 2, 0, 1, 14, 0),  # 71, Lu (4f14 5d1)
        (6, 2, 0, 2, 14, 0),  # 72, Hf
        (6, 2, 0, 3, 14, 0),  # 73, Ta
        (6, 2, 0, 4, 14, 0),  # 74, W
        (6, 2, 0, 5, 14, 0),  # 75, Re
        (6, 2, 0, 6, 14, 0),  # 76, Os
        (6, 2, 0, 7, 14, 0),  # 77, Ir
        (6, 1, 0, 9, 14, 0),  # 78, Pt (5d9 6s1)
        (6, 1, 0, 10, 14, 0),  # 79, Au (5d10, 6s1)
        (6, 2, 0, 10, 14, 0),  # 80, Hg
        (6, 2, 1, 10, 14, 0),  # 81, Tl
        (6, 2, 2, 10, 14, 0),  # 82, Pb
        (6, 2, 3, 10, 14, 0),  # 83, Bi
        (6, 2, 4, 10, 14, 0),  # 84, Po
        (6, 2, 5, 10, 14, 0),  # 85, At
        (6, 2, 6, 10, 14, 0),  # 86, Rn

        # Z=87~118 (Period 7)
        (7, 1, 0, 0, 0, 0),  # 87, Fr
        (7, 2, 0, 0, 0, 0),  # 88, Ra
        (7, 2, 0, 1, 0, 0),  # 89, Ac
        (7, 2, 0, 2, 0, 0),  # 90, Th
        (7, 2, 0, 1, 2, 0),  # 91, Pa (5f2 6d1 7s2)
        (7, 2, 0, 1, 3, 0),  # 92, U

        (7, 2, 0, 1, 4, 0),  # 93, Np (5f4 6d1 7s2)
        (7, 2, 0, 0, 6, 0),  # 94, Pu (5f6 7s2)
        (7, 2, 0, 0, 7, 0),  # 95, Am
        (7, 2, 0, 1, 7, 0),  # 96, Cm (5f7 6d1 7s2)
        (7, 2, 0, 0, 9, 0),  # 97, Bk
        (7, 2, 0, 0, 10, 0),  # 98, Cf
        (7, 2, 0, 0, 11, 0),  # 99, Es
        (7, 2, 0, 0, 12, 0),  # 100, Fm
        (7, 2, 0, 0, 13, 0),  # 101, Md
        (7, 2, 0, 0, 14, 0),  # 102, No
        (7, 2, 1, 0, 14, 0),  # 103, Lr (5f14 7s2 7p1)
        (7, 2, 0, 2, 14, 0),  # 104, Rf (5f14 6d2 7s2)
        (7, 2, 0, 3, 14, 0),  # 105, Db
        (7, 2, 0, 4, 14, 0),  # 106, Sg
        (7, 2, 0, 5, 14, 0),  # 107, Bh
        (7, 2, 0, 6, 14, 0),  # 108, Hs
        (7, 2, 0, 7, 14, 0),  # 109, Mt
        (7, 2, 0, 8, 14, 0),  # 110, Ds
        (7, 2, 0, 9, 14, 0),  # 111, Rg
        (7, 2, 0, 10, 14, 0),  # 112, Cn
        (7, 2, 1, 10, 14, 0),  # 113, Nh (7p1)
        (7, 2, 2, 10, 14, 0),  # 114, Fl
        (7, 2, 3, 10, 14, 0),  # 115, Mc
        (7, 2, 4, 10, 14, 0),  # 116, Lv
        (7, 2, 5, 10, 14, 0),  # 117, Ts
        (7, 2, 6, 10, 14, 0),  # 118, Og

        # Z=119 (第8周期碱金属，理论预测)
        (8, 1, 0, 0, 0, 0)  # 119, Uue
    )


_Z_TO_GROUP = _freeze({
        # group 1
        1: 1, 3: 1, 11: 1, 19: 1, 37: 1, 55: 1, 87: 1, 119: 1,
        # group 2
        4: 2, 12: 2, 20: 2, 38: 2, 56: 2, 88: 2, 120: 2,
        # group 3 (含镧系、锕系)
        21: 3, 39: 3, 57: 3, 58: 3, 59: 3, 60: 3, 61: 3, 62: 3,
        63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 3,
        89: 3, 90: 3, 91: 3, 92: 3, 93: 3, 94: 3, 95: 3, 96: 3,
        97: 3, 98: 3, 99: 3, 100: 3, 101: 3, 102: 3, 103: 3,
        # group 4
        22: 4, 40: 4, 72: 4, 104: 4,
        # group 5
        23: 5, 41: 5, 73: 5, 105: 5,
        # group 6
        24: 6, 42: 6, 74: 6, 106: 6,
        # group 7
        25: 7, 43: 7, 75: 7, 107: 7,
        # group 8
        26: 8, 44: 8, 76: 8, 108: 8,
        # group 9
        27: 9, 45: 9, 77: 9, 109: 9,
        # group 10
        28: 10, 46: 10, 78: 10, 110: 10,
        # group 11
        29: 11, 47: 11, 79: 11, 111: 11,
        # group 12
        30: 12, 48: 12, 80: 12, 112: 12,
        # group 13
        5: 13, 13: 13, 31: 13, 49: 13, 81: 13, 113: 13,
        # group 14
        6: 14, 14: 14, 32: 14, 50: 14, 82: 14, 114: 14,
        # group 15
        7: 15, 15: 15, 33: 15, 51: 15, 83: 15, 115: 15,
        # group 16
        8: 16, 16: 16, 34: 16, 52: 16, 84: 16, 116: 16,
        # group 17
        9: 17, 17: 17, 35: 17, 53: 17, 85: 17, 117: 17,
        # group 18
        2: 18, 10: 18, 18: 18, 36: 18, 54: 18, 86: 18, 118: 18,
    })


_ALKALI_METALS = frozenset({3, 11, 19, 37, 55, 87})


_ALKALINE_EARTH_METALS = frozenset({4, 12, 20, 38, 56, 88})


_TRANSITION_METALS = frozenset(set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113)))


_POST_TRANSITION_METALS = frozenset({13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116})


_LANTHANIDES = frozenset(set(range(57, 72)))


_ACTINIDES = frozenset(set(range(89, 104)))


_METAL = frozenset(_ALKALI_METALS|_ALKALINE_EARTH_METALS|_TRANSITION_METALS|_POST_TRANSITION_METALS|_LANTHANIDES|_ACTINIDES)


_METALLOID_1ST = frozenset({5, 14, 33, 52})


_METALLOID_2ND = frozenset({32, 51, 84})


_METALLOID = frozenset(_METALLOID_1ST|_METALLOID_2ND)


_METAL = frozenset(_METAL|_METALLOID_2ND)


_NONMETALS = _freeze([1, 6, 7, 8, 15, 16, 34])


_METALLOIDS = _freeze([5, 14, 32, 33, 51, 52, 84])


_NOBLE_GASES = _freeze([2, 10, 18, 36, 54, 86, 118])


_HALOGENS = _freeze([9, 17, 35, 53, 85, 117])


_COVALENT_RADII = _freeze([0.0, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.2, 1.19, 1.2, 1.2, 1.16, 2.2, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.4, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.9, 1.87, 1.87, 1.75, 1.7, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.4, 1.5, 1.5, 2.6, 2.21, 2.15, 2.06, 2.0, 1.96, 1.9, 1.87, 1.8, 1.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


_DENSITY = _freeze([0.0, 0.0708, 0.122, 0.534, 1.848, 2.34, 2.2, 0.808, 1.14, 1.5, 1.207, 0.971, 1.738, 2.6989, 2.33, 1.82, 2.07, 1.56, 1.4, 0.862, 1.55, 2.989, 4.54, 6.11, 7.19, 7.33, 7.874, 8.9, 8.902, 8.96, 7.133, 5.904, 5.323, 5.73, 4.79, 3.12, 2.16, 1.532, 2.54, 4.469, 6.506, 8.57, 10.22, 11.5, 12.41, 12.41, 12.02, 10.5, 8.65, 7.31, 7.31, 6.691, 6.24, 4.93, 3.52, 1.873, 3.5, 6.145, 6.77, 6.773, 7.008, 7.264, 7.52, 5.244, 7.901, 8.23, 8.551, 8.795, 9.066, 9.321, 6.966, 9.841, 13.31, 16.654, 19.3, 21.02, 22.57, 22.42, 21.45, 19.3, 13.546, 11.85, 11.35, 9.747, 9.32, 0.0, 0.0, 0.0, 0.0, 0.0, 11.72, 15.37, 18.95, 20.25, 19.84, 13.67, 13.51, 14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# Open Babel 3.2.1 Z=0..118 values captured before extraction.
_MASS = (0.0, 1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994, 18.9984032, 20.1797, 22.98977, 24.305, 26.981538, 28.0855, 30.973761, 32.065, 35.453, 39.948, 39.0983, 40.078, 44.95591, 47.867, 50.9415, 51.9961, 54.938049, 55.845, 58.9332, 58.6934, 63.546, 65.38, 69.723, 72.64, 74.9216, 78.96, 79.904, 83.798, 85.4678, 87.62, 88.90585, 91.224, 92.90638, 95.96, 98.0, 101.07, 102.9055, 106.42, 107.8682, 112.411, 114.818, 118.701, 121.76, 127.6, 126.90447, 131.293, 132.90545, 137.327, 138.9055, 140.116, 140.90765, 144.24, 145.0, 150.36, 151.964, 157.25, 158.92534, 162.5, 164.93032, 167.259, 168.93421, 173.054, 174.9668, 178.49, 180.9479, 183.84, 186.207, 190.23, 192.217, 195.078, 196.96655, 200.59, 204.3833, 207.2, 208.9804, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0381, 231.03588, 238.02891, 237.05, 244.06, 243.06, 247.07, 247.07, 251.08, 252.08, 257.1, 258.1, 259.1, 262.11, 265.12, 268.13, 271.13, 270.0, 277.15, 276.15, 281.16, 280.16, 285.17, 284.18, 289.19, 288.19, 293.0, 294.0, 294.0)
_EXACT_MASS = (0.0, 1.007825032, 4.002603254, 7.01600455, 9.0121822, 11.0093054, 12.0, 14.003074005, 15.99491462, 18.99840322, 19.992440175, 22.989769281, 23.9850417, 26.98153863, 27.976926532, 30.97376163, 31.972071, 34.96885268, 39.962383123, 38.96370668, 39.96259098, 44.9559119, 47.9479463, 50.9439595, 51.9405075, 54.9380451, 55.9349375, 58.933195, 57.9353429, 62.9295975, 63.929142, 68.925573, 73.921177, 74.921596, 79.916521, 78.918337, 83.911507, 84.911789, 87.905612, 88.905848, 89.904704, 92.906378, 97.905408, 97.907216, 101.904349, 102.905504, 105.903486, 106.905097, 113.903358, 114.903878, 119.902194, 120.903815, 129.906224, 126.904473, 131.904153, 132.905451, 137.905247, 138.906353, 139.905438, 140.907652, 141.907723, 144.912749, 151.919732, 152.92123, 157.924103, 158.925346, 163.929174, 164.930322, 165.930293, 168.934213, 173.938862, 174.940771, 179.94655, 180.947995, 183.950931, 186.955753, 191.96148, 192.962926, 194.964791, 196.966568, 201.970643, 204.974427, 207.976652, 208.980398, 208.98243, 209.987148, 222.017577, 223.019735, 226.025409, 227.027752, 232.038055, 231.035884, 238.050788, 237.048173, 244.064204, 243.061381, 247.070354, 247.070307, 251.079587, 252.08298, 257.095105, 258.098431, 259.10103, 262.10963, 261.10877, 262.11408, 263.11832, 264.1246, 265.13009, 268.13873, 281.162061, 280.164473, 285.174105, 284.17808, 289.187279, 288.192492, 292.199786, 292.20755, 293.21467)
