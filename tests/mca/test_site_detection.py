from rdkit import Chem

from mca.site_detection import find_nucleophilic_sites


def test_piperidine_has_one_symmetry_unique_nitrogen_site():
    sites = find_nucleophilic_sites(Chem.MolFromSmiles("C1CCCCN1"))
    nitrogen_sites = [site for site in sites if site.site_type == "Amine"]
    assert len(nitrogen_sites) == 1
    assert nitrogen_sites[0].atom_index == 5
