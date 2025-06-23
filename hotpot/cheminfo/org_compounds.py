# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : org_compounds
 Created   : 2025/6/15 11:38
 Author    : zhang
 Python    :
-----------------------------------------------------------
 Description
 ----------------------------------------------------------

===========================================================
"""
import os.path as osp
import pandas as pd



_dir_data = osp.join(osp.dirname(osp.realpath(__file__)), 'ChemData', 'OrganicCompounds.xlsx')

data = pd.read_excel(_dir_data)


# To add cid to the sheet
if __name__ == '__main__':
    from tqdm import tqdm

    from hotpot.cheminfo.pubchem import get_compounds_cid

    names = data.values.tolist()
    list_cid = [get_compounds_cid(name) for name in tqdm(names)]
