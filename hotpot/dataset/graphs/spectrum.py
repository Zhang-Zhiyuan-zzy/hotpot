"""
python v3.9.0
@Project: hotpot
@File   : spectrum
@Auther : Zhiyuan Zhang
@Data   : 2024/8/24
@Time   : 10:30
"""
from typing import Any

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles

from hotpot.plugins.dl.function.graph import graph_data2spectrum
from hotpot.dataset import load_dataset


class SpectralData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'spectrum':
            return self.x.shape[0]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'spectrum':
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)

    @classmethod
    def from_data(cls, data: Data, length: int = 4, atomic_numbers_col: int = 0) -> 'SpectralData':
        data = cls(**data.to_dict())
        data['spectrum'] = graph_data2spectrum(data, length, atomic_numbers_col)

        return data

    @classmethod
    def from_smiles(cls, smiles: str, length: int = 4, atomic_numbers_col: int = 0) -> 'SpectralData':
        data = from_smiles(smiles, with_hydrogen=True)
        data = cls.from_data(data, length, atomic_numbers_col)
        data['smiles'] = smiles

        return data


class SpectrumDataset:
    """"""
    def __init__(self):
        self.smi_loader = load_dataset('SMILES')

    def __getitem__(self, index):
        return next(self)

    def __len__(self):
        return 999999999999999

    def __next__(self):
        return self.get()

    def get(self):
        try:
            smi = next(self.smi_loader)
        except StopIteration:
            self.smi_loader = load_dataset('SMILES')
            smi = next(self.smi_loader)

        return SpectralData.from_smiles(smi)


if __name__ == '__main__':
    loader = DataLoader(SpectrumDataset(), batch_size=4096)
    for batch in iter(loader):
        print(batch.shape)
