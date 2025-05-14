from typing import Union, Sequence, Optional, List, Any
from typing_extensions import override

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader.dataloader import Collater as PyGCollater


class Collater(PyGCollater):
    @override
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        if exclude_keys is None:
            exclude_keys = []
        else:
            exclude_keys = list(exclude_keys)

        first_data = dataset[0]
        if isinstance(first_data, BaseData):
            attr_names = set(first_data.keys())
            for attr in attr_names:
                if attr[-6:] == '_names' and attr[:-6] in attr_names:
                    exclude_keys.append(attr)

        super(Collater, self).__init__(dataset, follow_batch, exclude_keys)

    @override
    def __call__(self, batch: List[Any]) -> Any:
        batch = super(Collater, self).__call__(batch)
        data = self.dataset[0]
        if isinstance(batch, Batch):
            batch_stores = batch.stores[0]
            for key in self.exclude_keys:
                batch_stores[key] = getattr(data, key)

        return batch
