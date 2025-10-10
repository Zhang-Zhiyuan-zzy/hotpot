import os.path as osp
import glob
import torch
import hotpot as hp

from hotpot.cheminfo.AImodels.cbond.apply import cbond_prediction, auto_build_cbond


if __name__=='__main__':
    # cbond_data_dir = osp.join(osp.dirname(__file__), 'example_data', 'cbond')
    # cbond_data_files = glob.glob(osp.join(cbond_data_dir, '*.pt'))

    # list_data = [torch.load(f, weights_only=False) for f in cbond_data_files]

    mol = hp.read_mol('NC(=O)c1ccc2c(n1)c1nc(ccc1c(c2)C[C](C)(C)(C)C)C1=NC=CC1')
    # Am = mol.add_atom(hp.Atom(symbol='Am'))
    # mol.add_bond(8, -1)
    #
    # pred_cb, cb_index, is_cb = cbond_prediction(mol)
    new_mol = auto_build_cbond(mol, 'Am')