"""
@File Name:        cook_raw_data
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/14 17:52
@Project:          Hotpot
"""
import pandas as pd


_cols = ['SMILES', 'Metal', 'Sol.SMI', 'Sol.CAS', 'Med.SMI', 'Med.CAS', 'Med.Conc.(M)']


def cook_beta_g(raw_data: str, cooked_data: str):
    df_raw = pd.read_excel(raw_data)[_cols]


