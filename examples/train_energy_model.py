"""
python v3.9.0
@Project: hotpot
@File   : train_energy_model
@Auther : Zhiyuan Zhang
@Data   : 2023/8/22
@Time   : 20:48
"""
from hotpot.tasks.ml.graph.module import get_atom_energy_tensor


if __name__ == "__main__":
    e = get_atom_energy_tensor("M062X", "Def2SVP", "water")

