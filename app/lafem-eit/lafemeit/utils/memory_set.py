
from typing import (
    Optional, Union, Tuple, List, Sequence,
    Any, Callable, Iterator,
    TypeVar
)

from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import Dataset, BatchSampler, RandomSampler, DataLoader


_KT = TypeVar('_KT')
_device = torch.device
ArrayFunction = Callable[[NDArray[Any]], NDArray[Any]]


class MemoryDataset(Dataset):
    def __init__(
            self,
            sample_keys: Sequence[_KT],
            reader: Callable[[_KT], Tuple[Tensor, ...]], *,
            num_workers: int = 0,
            device: Union[_device, str, None] = None,
            pin_memory: Optional[bool] = False,
            tqdm: bool = False
        ) -> None:
        """Preload data from disk to memory, as a Dataset object."""
        super().__init__()
        kwargs = {'device': device, 'pin_memory': pin_memory}
        NUM = len(sample_keys)
        _preloader = DataLoader(dataset=sample_keys,
                                batch_size=None,
                                num_workers=num_workers,
                                collate_fn=reader)
        self._header_data_read = False
        self.data: List[Tensor] = []

        if tqdm:
            from tqdm import tqdm as _tqdm
            _preloader = _tqdm(_preloader, desc=f"Loading", unit=f'sample')

        for sample_id, pair in enumerate(_preloader):
            if not isinstance(pair, tuple):
                pair = (pair,)

            if not self._header_data_read: # the first data
                for col in pair: # check the shape of the first data
                    self.data.append(torch.empty((NUM, *col.shape), dtype=col.dtype, **kwargs))
                self._header_data_read = True

            for col_id, col in enumerate(pair):
                self.data[col_id][sample_id].copy_(col, non_blocking=True)

    def __len__(self) -> int:
        return len(self.data[0])

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if len(self.data) == 1:
            return self.data[0][index]
        return tuple(col[index] for col in self.data)

    def __getitems__(self, indices) -> Tuple[Tensor, Tensor]:
        if len(self.data) == 1:
            return self.data[0][indices]
        return tuple(col[indices] for col in self.data)

    def loader(self, batch_size: int, drop_last=False):
        return _Loader(dataset=self, batch_size=batch_size, drop_last=drop_last)


class _Loader():
    dataset: MemoryDataset
    batch_size: int
    sampler: RandomSampler
    batch_sampler: BatchSampler
    _iterator: Iterator
    __initialized: bool = False

    def __init__(self, dataset: MemoryDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = RandomSampler(self.dataset, num_samples=len(self.dataset))
        self.batch_sampler = BatchSampler(self.sampler, batch_size=batch_size, drop_last=drop_last)
        self.reset_iterator()

    def reset_iterator(self):
        self.__initialized = False
        self._iterator = iter(self.batch_sampler)
        self.__initialized = True

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self._iterator)
        if hasattr(self.dataset, '__getitems__'):
            return self.dataset.__getitems__(indices)
        else:
            raise NotImplementedError

    def __iter__(self):
        self.reset_iterator()
        return self

    def __setattr__(self, attr: str, val: Any) -> None:
        if self.__initialized and attr in {
            'dataset', 'batch_size', 'sampler', 'batch_sampler', '_iterator',
        }:
            raise RuntimeError(f"Cannot set attribute {attr} after {self.__class__.__name__} is initialized.")

        super().__setattr__(attr, val)
