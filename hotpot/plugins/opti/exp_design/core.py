"""
python v3.9.0
@Project: hotpot
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2024/9/26
@Time   : 18:17
Notes:
    For orthogonal experiment design
"""
from typing import Sequence

import numpy as np
import pyDOE2

def orthogonal_design(seq_of_params: Sequence[Sequence], num_exper: int):
    levels = [len(s) for s in seq_of_params]

    indices = pyDOE2.gsd(levels, num_exper)

    num_indices = len(indices)
    interval = num_indices // num_exper

    indices = indices[::interval]

    design_params = []
    for index in indices:
        design_params.append([s[i] for s, i in zip(seq_of_params, index)])

    return np.array(design_params)


if __name__ == '__main__':
    import pandas as pd
    params_space = [
        [10*i for i in range(1, 7)], # H4
        [20*i for i in range(1, 7)], # Mg
        [2*i for i in range(1, 6)], # DMF
        [2 * i for i in range(1, 6)],  # CH3OH
    ]

    params = orthogonal_design(params_space, 10)
    params = pd.DataFrame(
        params,
        columns=['H4', 'Mg', 'DMF', 'CH4OH']
    )

    params.to_csv('/home/zz1/params.csv')
