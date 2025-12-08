import unittest as ut
import hotpot as hp
from hotpot.cheminfo import search

class TestSearch(ut.TestCase):

    def test_base(self):
        mol = next(hp.MolReader('c1cnncc1c2cccnc2C(=O)O'))

        sub = search.Substructure()
        for _ in range(6):
            qa = search.QueryAtom(atomic_number=[6, 7])
            sub.add_atom(qa)

        for i in range(5):
            sub.add_bond(i, i+1)
        sub.add_bond(0, 5)

        searcher = search.Searcher(sub)
        hits = searcher.search(mol)

        for hit in hits:
            print(hit.atoms)
            print(hit.bonds)


class TestSmartsSearch(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        # ~20 test molecules from SMILES, including a simple metal complex
        cls.mols = {
            # 1. Ethanol
            "ethanol": hp.read_mol("CCO"),
            # 2. Acetic acid
            "acetic_acid": hp.read_mol("CC(=O)O"),
            # 3. Benzene
            "benzene": hp.read_mol("c1ccccc1"),
            # 4. Toluene
            "toluene": hp.read_mol("Cc1ccccc1"),
            # 5. Benzyl alcohol
            "benzyl_alcohol": hp.read_mol("OCc1ccccc1"),
            # 6. Benzoic acid
            "benzoic_acid": hp.read_mol("OC(=O)c1ccccc1"),
            # 7. Benzamide
            "benzamide": hp.read_mol("NC(=O)c1ccccc1"),
            # 8. Acetonitrile
            "acetonitrile": hp.read_mol("CC#N"),
            # 9. Ethyl acetate
            "ethyl_acetate": hp.read_mol("CC(=O)OCC"),
            # 10. p‑Nitro methyl benzoate
            "p_nitro_methyl_benzoate": hp.read_mol(
                "COC(=O)c1ccc(cc1)[N+](=O)[O-]"
            ),
            # 11. Chlorobenzene
            "chlorobenzene": hp.read_mol("Clc1ccccc1"),
            # 12. Bromobenzene
            "bromobenzene": hp.read_mol("Brc1ccccc1"),
            # 13. Fluoroethane
            "fluoroethane": hp.read_mol("CCF"),
            # 14. Simple thioether
            "thioether": hp.read_mol("CCSCC"),
            # 15. Ethylamine
            "ethylamine": hp.read_mol("CCN"),
            # 16. Guanidine‑like fragment
            "guanidine_like": hp.read_mol("N=C(N)N"),
            # 17. Pyridine
            "pyridine": hp.read_mol("n1ccccc1"),
            # 18. Pyrrole
            "pyrrole": hp.read_mol("c1cc[nH]c1"),
            # 19. Simple Fe(II) complex with two pyridines
            "fe_pyridine2": hp.read_mol("[Fe](n1ccccc1)(n1ccccc1)"),
            # 20. Sodium acetate (metal–carboxylate salt)
            "na_acetate": hp.read_mol("[Na+].[O-]C(=O)C", "smi"),

            # 21. Eu_pair
            "Eu_pair": hp.read_mol('[Eu]O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C(C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3'),
            # 22. Am_pair
            "Am_pair": hp.read_mol('[Am]O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C(C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3')
        }

        # ~20 substructure SMARTS patterns
        cls.smarts_patterns = {
            # 1. Alcohol O (very simple: X2 O bearing an H)
            "alcohol_O": "[OX2H]",
            # 2. Carboxylic acid fragment C(=O)O or its deprotonated form
            "carboxylic_acid": "C(=O)[O,H-]",
            # 3. Benzene ring
            "benzene_ring": "c1ccccc1",
            # 4. Pyridine ring
            "pyridine_ring": "n1ccccc1",
            # 5. Pyrrolic N–H
            "pyrrolic_NH": "[nH]",
            # 6. Thioether C–S–C
            "thioether": "C-S-C",
            # 7. Primary amine C–NH2 (very simple)
            "primary_amine": "CN",
            # 8. Amide C(=O)N
            "amide": "C(=O)N",
            # 9. Nitrile C#N
            "nitrile": "C#N",
            # 10. Ester C(=O)O–C
            "ester": "C(=O)O[C;X4]",
            # 11. Nitro group
            "nitro": "[N+](=O)[O-]",
            # 12. Aryl chloride
            "aryl_cl": "Clc",
            # 13. Aryl bromide
            "aryl_br": "Brc",
            # 14. Terminal fluoroalkane motif C–C–F
            "alkyl_f": "CCF",
            # 15. Benzylic CH2 (very rough pattern)
            "benzyl_CH2": "c-CH2-",
            # 16. Guanidinium‑like core
            "guanidinium_core": "NC(=NH)N",
            # 17. Aryl ester fragment Ar‑C(=O)O–
            "aryl_ester": "cC(=O)O",
            # 18. Aryl amide fragment Ar‑C(=O)N–
            "aryl_amide": "cC(=O)N",
            # 19. Metal–N coordination (very rough Fe–N pattern)
            "metal_N_coord": "[Fe]n",
            # 20. Metal–carboxylate (Na–O− plus C(=O))
            "metal_carboxylate": "[Na+].[O-]C(=O)",

            "Ln-O": "[Ln]O",
            "An-O": "[An]O",
        }

        # Expected presence/absence table:
        # (molecule_name, smarts_name) -> bool (True = should find match)
        cls.expectations = {
            # Alcohol O
            ("ethanol", "alcohol_O"): True,
            ("benzyl_alcohol", "alcohol_O"): True,
            ("acetic_acid", "alcohol_O"): False,

            # Carboxylic acid
            ("acetic_acid", "carboxylic_acid"): True,
            ("benzoic_acid", "carboxylic_acid"): True,
            ("na_acetate", "carboxylic_acid"): True,
            ("ethyl_acetate", "carboxylic_acid"): False,

            # Benzene ring
            ("benzene", "benzene_ring"): True,
            ("toluene", "benzene_ring"): True,
            ("ethylamine", "benzene_ring"): False,

            # Pyridine ring
            ("pyridine", "pyridine_ring"): True,
            ("fe_pyridine2", "pyridine_ring"): True,
            ("benzene", "pyridine_ring"): False,

            # Pyrrolic N–H
            ("pyrrole", "pyrrolic_NH"): True,
            ("pyridine", "pyrrolic_NH"): False,

            # Thioether
            ("thioether", "thioether"): True,
            ("ethanol", "thioether"): False,

            # Primary amine
            ("ethylamine", "primary_amine"): True,
            ("benzamide", "primary_amine"): False,

            # Amide
            ("benzamide", "amide"): True,
            ("ethyl_acetate", "amide"): False,

            # Nitrile
            ("acetonitrile", "nitrile"): True,
            ("ethanol", "nitrile"): False,

            # Ester
            ("ethyl_acetate", "ester"): True,
            ("p_nitro_methyl_benzoate", "ester"): True,
            ("acetic_acid", "ester"): False,

            # Nitro
            ("p_nitro_methyl_benzoate", "nitro"): True,
            ("benzene", "nitro"): False,

            # Aryl halides
            ("chlorobenzene", "aryl_cl"): True,
            ("bromobenzene", "aryl_cl"): False,

            ("bromobenzene", "aryl_br"): True,
            ("chlorobenzene", "aryl_br"): False,

            # Alkyl fluoride
            ("fluoroethane", "alkyl_f"): True,
            ("ethanol", "alkyl_f"): False,

            # Benzylic CH2
            ("toluene", "benzyl_CH2"): True,
            ("benzyl_alcohol", "benzyl_CH2"): True,

            # Guanidinium‑like core
            ("guanidine_like", "guanidinium_core"): True,

            # Aryl ester
            ("p_nitro_methyl_benzoate", "aryl_ester"): True,
            ("ethyl_acetate", "aryl_ester"): False,

            # Aryl amide
            ("benzamide", "aryl_amide"): True,
            ("ethylamine", "aryl_amide"): False,

            # Metal–pyridine coordination (very rough)
            ("fe_pyridine2", "metal_N_coord"): True,
            ("pyridine", "metal_N_coord"): False,

            # Metal–carboxylate
            ("na_acetate", "metal_carboxylate"): True,
            ("acetic_acid", "metal_carboxylate"): False,

            # 21.
            ('Eu_pair', 'Ln-O'): True,
            ('Eu_pair', 'An-O'): False,
            ('Am_pair', 'Ln-O'): False,
            ('Am_pair', 'An-O'): True,
        }

    @staticmethod
    def _has_match(mol, sub: "hp.Substructure") -> bool:
        """
        Wrap the actual substructure search.

        Replace this with your real search API if needed.
        """
        searcher = hp.Searcher(sub)
        hits = searcher.search(mol)
        return bool(hits)

    def _run_single_case(self, mol_name: str, smarts_name: str, expected: bool):
        mol = self.mols[mol_name]
        smarts = self.smarts_patterns[smarts_name]

        sub = hp.Substructure.from_smarts(smarts)
        has = self._has_match(mol, sub)

        if has != expected:
            # Collect rich debug information on failure
            smiles = getattr(mol, "smiles", None)
            # fall back to hp API if you have one, e.g. hp.to_smiles(mol)
            if smiles is None and hasattr(hp, "to_smiles"):
                smiles = hp.to_smiles(mol)

            debug_lines = [
                "Substructure search mismatch:",
                f"  molecule name : {mol_name}",
                f"  molecule SMILES: {smiles}",
                f"  SMARTS name   : {smarts_name}",
                f"  SMARTS pattern: {smarts}",
                "",
                f"  Substructure  : {sub!r}",
                f"  query_atoms   : {getattr(sub, 'query_atoms', None)!r}",
                f"  query_bonds   : {getattr(sub, 'query_bonds', None)!r}",
                "",
                f"  expected match: {expected}",
                f"  actual match  : {has}",
            ]
            msg = "\n".join(debug_lines)
        else:
            msg = (
                f"SMARTS '{smarts_name}' = {smarts} on molecule '{mol_name}' "
                f"expected {expected} and got {has}"
            )

        self.assertEqual(has, expected, msg=msg)

    def test_smarts_search(self):
        """
        Main test: iterate over (molecule, SMARTS) expectation table and
        assert whether a match exists or not.

        On failure, prints:
          - molecule SMILES
          - SMARTS pattern
          - Substructure object
          - sub.query_atoms
          - sub.query_bonds
        """
        for (mol_name, smarts_name), expected in self.expectations.items():
            with self.subTest(molecule=mol_name, smarts=smarts_name):
                self._run_single_case(mol_name, smarts_name, expected)


if __name__ == "__main__":
    ut.main()
