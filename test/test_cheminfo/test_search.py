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
