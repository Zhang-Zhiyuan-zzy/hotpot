/* File: atom.c */
#include <stdbool.h>
#include <stdio.h>

static int default_valence[119] = {
    /* Atomic number 0 is unused or reserved. Just set to 0 or a sentinel value. */
    0,
    1, /* 1: Hydrogen (H)              */
    0, /* 2: Helium (He) - inert       */
    1, /* 3: Lithium (Li)             */
    2, /* 4: Beryllium (Be)           */
    3, /* 5: Boron (B)                */
    4, /* 6: Carbon (C)               */
    3, /* 7: Nitrogen (N)             */
    2, /* 8: Oxygen (O)               */
    1, /* 9: Fluorine (F)             */
    0, /* 10: Neon (Ne) - inert       */
    1, /* 11: Sodium (Na)             */
    2, /* 12: Magnesium (Mg)          */
    3, /* 13: Aluminium (Al)          */
    4, /* 14: Silicon (Si)            */
    3, /* 15: Phosphorus (P)          */
    2, /* 16: Sulfur (S)              */
    1, /* 17: Chlorine (Cl)           */
    0, /* 18: Argon (Ar) - inert      */
    1, /* 19: Potassium (K)           */
    2, /* 20: Calcium (Ca)            */
    3, /* 21: Scandium (Sc)           */
    4, /* 22: Titanium (Ti)           */
    5, /* 23: Vanadium (V)            */
    3, /* 24: Chromium (Cr)           */
    2, /* 25: Manganese (Mn)          */
    2, /* 26: Iron (Fe)               */
    3, /* 27: Cobalt (Co)             */
    2, /* 28: Nickel (Ni)             */
    2, /* 29: Copper (Cu)             */
    2, /* 30: Zinc (Zn)               */
    3, /* 31: Gallium (Ga)            */
    4, /* 32: Germanium (Ge)          */
    3, /* 33: Arsenic (As)            */
    2, /* 34: Selenium (Se)           */
    1, /* 35: Bromine (Br)            */
    0, /* 36: Krypton (Kr) - inert    */
    1, /* 37: Rubidium (Rb)           */
    2, /* 38: Strontium (Sr)          */
    3, /* 39: Yttrium (Y)             */
    4, /* 40: Zirconium (Zr)          */
    5, /* 41: Niobium (Nb)            */
    6, /* 42: Molybdenum (Mo)         */
    7, /* 43: Technetium (Tc)         */
    4, /* 44: Ruthenium (Ru)          */
    3, /* 45: Rhodium (Rh)           */
    2, /* 46: Palladium (Pd)          */
    1, /* 47: Silver (Ag)             */
    2, /* 48: Cadmium (Cd)            */
    3, /* 49: Indium (In)             */
    4, /* 50: Tin (Sn)                */
    3, /* 51: Antimony (Sb)           */
    2, /* 52: Tellurium (Te)          */
    1, /* 53: Iodine (I)              */
    0, /* 54: Xenon (Xe) - inert      */
    1, /* 55: Cesium (Cs)             */
    2, /* 56: Barium (Ba)             */
    3, /* 57: Lanthanum (La)          */
    3, /* 58: Cerium (Ce)             */
    3, /* 59: Praseodymium (Pr)       */
    3, /* 60: Neodymium (Nd)          */
    3, /* 61: Promethium (Pm)         */
    3, /* 62: Samarium (Sm)           */
    3, /* 63: Europium (Eu)           */
    3, /* 64: Gadolinium (Gd)         */
    3, /* 65: Terbium (Tb)            */
    3, /* 66: Dysprosium (Dy)         */
    3, /* 67: Holmium (Ho)            */
    3, /* 68: Erbium (Er)             */
    3, /* 69: Thulium (Tm)            */
    3, /* 70: Ytterbium (Yb)          */
    3, /* 71: Lutetium (Lu)           */
    4, /* 72: Hafnium (Hf)            */
    5, /* 73: Tantalum (Ta)           */
    6, /* 74: Tungsten (W)            */
    5, /* 75: Rhenium (Re)            */
    4, /* 76: Osmium (Os)             */
    3, /* 77: Iridium (Ir)            */
    2, /* 78: Platinum (Pt)           */
    1, /* 79: Gold (Au)               */
    2, /* 80: Mercury (Hg)            */
    3, /* 81: Thallium (Tl)           */
    4, /* 82: Lead (Pb)               */
    3, /* 83: Bismuth (Bi)            */
    2, /* 84: Polonium (Po)           */
    1, /* 85: Astatine (At)           */
    0, /* 86: Radon (Rn) - inert      */
    1, /* 87: Francium (Fr)           */
    2, /* 88: Radium (Ra)             */
    3, /* 89: Actinium (Ac)           */
    4, /* 90: Thorium (Th)            */
    5, /* 91: Protactinium (Pa)       */
    6, /* 92: Uranium (U)             */
    5, /* 93: Neptunium (Np)          */
    6, /* 94: Plutonium (Pu)          */
    3, /* 95: Americium (Am)          */
    3, /* 96: Curium (Cm)             */
    3, /* 97: Berkelium (Bk)          */
    3, /* 98: Californium (Cf)        */
    3, /* 99: Einsteinium (Es)        */
    3, /* 100: Fermium (Fm)           */
    3, /* 101: Mendelevium (Md)       */
    3, /* 102: Nobelium (No)          */
    3, /* 103: Lawrencium (Lr)        */
    4, /* 104: Rutherfordium (Rf)     */
    5, /* 105: Dubnium (Db)           */
    6, /* 106: Seaborgium (Sg)        */
    7, /* 107: Bohrium (Bh)           */
    4, /* 108: Hassium (Hs)           */
    3, /* 109: Meitnerium (Mt)        */
    4, /* 110: Darmstadtium (Ds)      */
    1, /* 111: Roentgenium (Rg)       */
    2, /* 112: Copernicium (Cn)       */
    3, /* 113: Nihonium (Nh)          */
    4, /* 114: Flerovium (Fl)         */
    3, /* 115: Moscovium (Mc)         */
    2, /* 116: Livermorium (Lv)       */
    1, /* 117: Tennessine (Ts)        */
    0  /* 118: Oganesson (Og) - inert */
};


/* Define sets for metals. */
const int alkali_metals[]          = {3, 11, 19, 37, 55, 87};
const int alkaline_earth_metals[]  = {4, 12, 20, 38, 56, 88};
const int transition_metals[]      = {
    21,22,23,24,25,26,27,28,29,30,
    39,40,41,42,43,44,45,46,47,48,
    72,73,74,75,76,77,78,79,80,
    104,105,106,107,108,109,110,111,112
};
const int post_transition_metals[] = {13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116};
const int lanthanides[]            = {57,58,59,60,61,62,63,64,65,66,67,68,69,70,71};
const int actinides[]              = {89,90,91,92,93,94,95,96,97,98,99,100,101,102,103};


bool is_metal(int atomic_number)
{
    // Helper macro to check membership
    #define IN_ARRAY(elem, arr)                                \
        for (size_t i = 0; i < (sizeof(arr)/sizeof(arr[0])); i++) { \
            if ((elem) == arr[i]) return true;                      \
        }

    IN_ARRAY(atomic_number, alkali_metals);
    IN_ARRAY(atomic_number, alkaline_earth_metals);
    IN_ARRAY(atomic_number, transition_metals);
    IN_ARRAY(atomic_number, post_transition_metals);
    IN_ARRAY(atomic_number, lanthanides);
    IN_ARRAY(atomic_number, actinides);

    return false;
}


extern int sum_list(int *neigh_atoms, int neigh_length)
{
    int sum = 0;
    for (int i = 0; i < neigh_length; i++) {
        sum = sum + neigh_atoms[i];
    }
    return sum;
}



int get_valence(int atomic_number, int *neigh_atoms, int neigh_length)
{
    /* Count how many oxygen neighbors exist. */
    int oxygen_count = 0;
    for (int i = 0; i < neigh_length; i++) {
        if (neigh_atoms[i] == 8) {
            oxygen_count++;
        }
    }

    switch (atomic_number) {
        case 6:  /* C */
        case 14: /* Si */
            return 4;

        case 8:  /* O */
            return 2;

        case 7:  /* N */
        case 15: /* P */
        {
            /* If there's no oxygen neighbor => valence = 3
               else => valence = max(5, 2x(oxygen_count)+1). */
            if (oxygen_count == 0) {
                return 3;
            } else {
                int val = (2 * oxygen_count) + 1;
                return (val > 5) ? val : 5;
            }
        }

        case 16: /* S */
        {
            /* Without sum_covalent_orders, use a simplified approach:
               If no oxygen neighbors => 2, else => 6. */
            if (oxygen_count == 0) {
                return 2;
            } else {
                return 6;
            }
        }

        case 5:  /* B */
            return 3;

        case 1:  /* H */
            return 1;

        default:
            /* For all other elements, fall back to a default lookup. */
            /* Ensure atomic_number is within valid range (1..118). */
            if (atomic_number >= 0 && atomic_number < 119) {
                return default_valence[atomic_number];
            } else {
                /* Handle invalid atomic numbers however you prefer. */
                return -1; /* Or assert/error out. */
            }
    }
}


int calc_implicit_hydrogens(int atomic_number, bool is_aromatic, int *neigh_atoms, int neigh_length)
{
    /* 1) Emulate: if self.is_metal => return 0. */
    if (is_metal(atomic_number))
        return 0;

    /* 2) Check hydrogen special case: if atomic_number == 1 => if neighbor is also H => return 1; else 0. */
    if (atomic_number == 1) {
        /* In Python, it checks self.neighbours[0].atomic_number == 1
           so we do the same if we have at least one neighbor. */
        if (neigh_length > 0 && neigh_atoms[0] == 1) {
            return 1;
        } else {
            return 0;
        }
    }

    /* 3) If aromatic => replicate that logic: */
    if (is_aromatic) {
        /* Count how many neighbors are neither H nor metal. */
        int num_non_H_non_metal = 0;
        for (int i = 0; i < neigh_length; i++) {
            if ((neigh_atoms[i] != 1) && (!is_metal(neigh_atoms[i]))) {
                num_non_H_non_metal++;
            }
        }

        /* a) C(6) or Si(14) */
        if (atomic_number == 6 || atomic_number == 14) {
            if (num_non_H_non_metal == 3) {
                return 0;
            } else {
                return 1;
            }
        }
        /* b) N(7), P(15), As(33) */
        else if (atomic_number == 7  || atomic_number == 15 || atomic_number == 33) {
            /* In Python: if num==3 OR sum_heavy_cov_orders>2 => 0, else => 1
               We do not have sum_heavy_cov_orders param, so we only check num. */
            if (num_non_H_non_metal == 3) {
                return 0;
            } else {
                return 1;
            }
        }
        /* c) O(8), S(16), Se(34) => return 0 */
        else if (atomic_number == 8  || atomic_number == 16 || atomic_number == 34) {
            return 0;
        }
        /* d) B(5) => return 1 */
        else if (atomic_number == 5) {
            return 1;
        }
        /* e) Ge(32) => return 0 */
        else if (atomic_number == 32) {
            return 0;
        }
        /* f) Otherwise => error in Python. For C, we can either return -1 or assert. */
        else {
            fprintf(stderr, "Unexpected atomic_number %d in aromatic context!\n", atomic_number);
            return -1;
        }
    }

    /* 4) “Else” block in Python => max(self.valence - self.sum_heavy_cov_orders, 0).
       We do not have valence or sum_heavy_cov_orders in this signature.
       Return 0 or adapt as needed. */
    return 0;
}