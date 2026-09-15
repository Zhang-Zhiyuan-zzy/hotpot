import math
import unittest

import hotpot as hp
from hotpot.calculator import formal_charge
from hotpot.cheminfo.core import OPENBABEL_PARTIAL_CHARGE_MODELS


class TestPartialChargeModels(unittest.TestCase):
    def test_all_openbabel_charge_models_are_exposed(self):
        self.assertEqual(
            {
                "eem", "eem2015ha", "eem2015hm", "eem2015hn",
                "eem2015ba", "eem2015bm", "eem2015bn",
                "gasteiger", "mmff94", "qeq", "qtpie", "eqeq",
                "fromfile", "none",
            },
            set(OPENBABEL_PARTIAL_CHARGE_MODELS),
        )

    def test_assign_uses_requested_model(self):
        mol = hp.read_mol("CCO", "smi")

        charges = mol.assign_partial_charge("gasteiger")

        self.assertEqual(charges, tuple(atom.partial_charge for atom in mol.atoms))
        self.assertTrue(all(math.isfinite(charge) for charge in charges))
        self.assertAlmostEqual(0.0, sum(charges), places=12)

        cleared = mol.assign_partial_charge("none")
        self.assertEqual((0.0, 0.0, 0.0), cleared)
        self.assertEqual(cleared, tuple(atom.partial_charge for atom in mol.atoms))

    def test_unavailable_model_fails_explicitly(self):
        mol = hp.read_mol("CCO", "smi")

        with self.assertRaisesRegex(ValueError, "unavailable"):
            mol.get_partial_charge("not-a-charge-model")


class TestFormalChargeCalculator(unittest.TestCase):
    @staticmethod
    def _recalculate(smiles, model="valence"):
        mol = hp.read_mol(smiles, "smi")
        target_charge = mol.charge
        for atom in mol.atoms:
            atom.formal_charge = 0
        mol.charge = target_charge
        charges = formal_charge(mol, model=model)
        return mol, charges

    def test_neutral_organic_molecule(self):
        mol, charges = self._recalculate("CCO")

        self.assertEqual((0, 0, 0), charges)
        self.assertEqual(0, mol.charge)
        self.assertEqual(mol.charge, mol.sum_atoms_charge)

    def test_common_main_group_ions(self):
        ammonium, ammonium_charges = self._recalculate("[NH4+]")
        acetate, acetate_charges = self._recalculate("CC(=O)[O-]")
        nitromethane, nitromethane_charges = self._recalculate("C[N+](=O)[O-]")
        borohydride, borohydride_charges = self._recalculate("[BH4-]")

        self.assertEqual((1,), ammonium_charges)
        self.assertEqual(1, ammonium.charge)
        self.assertEqual((0, 0, 0, -1), acetate_charges)
        self.assertEqual(-1, acetate.charge)
        self.assertEqual((0, 1, 0, -1), nitromethane_charges)
        self.assertEqual(0, nitromethane.charge)
        self.assertEqual((-1,), borohydride_charges)
        self.assertEqual(-1, borohydride.charge)

    def test_total_charge_constraint_resolves_carbocation(self):
        mol, charges = self._recalculate("[CH3+]", "valence-constrained")

        self.assertEqual((1,), charges)
        self.assertEqual(1, mol.charge)
        self.assertEqual(mol.charge, mol.sum_atoms_charge)

    def test_metal_and_ligands_are_resolved_separately(self):
        mol = hp.read_mol("[Zn](Cl)Cl", "smi")
        bond_count = len(mol.bonds)

        charges = formal_charge(mol)

        self.assertEqual((2, -1, -1), charges)
        self.assertEqual(0, mol.charge)
        self.assertEqual(mol.charge, mol.sum_atoms_charge)
        self.assertEqual(bond_count, len(mol.bonds))

    def test_custom_metal_resolver_is_an_extension_point(self):
        mol = hp.read_mol("[Zn](Cl)Cl", "smi")

        def trivalent_zinc(metal, parent):
            self.assertIs(metal, parent.metals[0])
            self.assertEqual(2, len(metal.neighbours))
            return 3

        charges = formal_charge(
            mol,
            metal_model=trivalent_zinc,
        )

        self.assertEqual((3, -1, -1), charges)
        self.assertEqual(1, mol.charge)

    def test_preserve_supports_authoritative_charges(self):
        mol = hp.read_mol("CC", "smi")
        mol.atoms[0].formal_charge = 1
        mol.atoms[1].formal_charge = -1
        mol.charge = 7

        charges = formal_charge(mol, model="preserve")

        self.assertEqual((1, -1), charges)
        self.assertEqual(0, mol.charge)
        self.assertEqual(mol.charge, mol.sum_atoms_charge)

    def test_invalid_models_fail_explicitly(self):
        mol = hp.read_mol("CCO", "smi")

        with self.assertRaisesRegex(ValueError, "Unknown formal-charge model"):
            formal_charge(mol, model="unknown")
        with self.assertRaisesRegex(ValueError, "Unknown metal formal-charge model"):
            formal_charge(mol, metal_model="unknown")


if __name__ == "__main__":
    unittest.main()
