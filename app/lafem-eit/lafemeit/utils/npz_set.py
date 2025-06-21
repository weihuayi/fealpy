
import os
from typing import (
    Optional, Union,
    List, Tuple, Dict, Sequence,
    Any, Callable
)

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import Dataset


ArrayFunction = Callable[[NDArray[Any]], NDArray[Any]]


class NPYDataset(Dataset):
    def __init__(self, folder: str, names: Sequence[str]) -> None:
        super().__init__()
        self.folder = folder
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def read_data(self, file_name: str):
        data = np.load(os.path.join(self.folder, file_name + ".npy"))
        return torch.from_numpy(data)

    def __getitem__(self, index: int):
        return self.read_data(self.names[index])

    def read_batch(self, names: Sequence[str]):
        return [self.read_data(name) for name in names]

    def __getitems__(self, indices: Sequence[int]):
        samples = []
        for index in indices:
            sample = self.read_data(self.names[index])
            samples.append(sample)
        return torch.stack(samples, dim=0)


class NPZDataset(Dataset):
    path: str
    names_seq: Sequence
    channel_keys: List[str]
    label_key: str
    keep_dim: bool
    _cache: Optional[List[Optional[Tuple[Tensor, Tensor]]]]

    def __init__(
            self,
            folder: str,
            names: Union[Sequence[Any], int] = -1, *,
            channel_keys: Optional[Sequence[str]] = None,
            label_key = 'label',
            keep_dim = False,
            use_cache = False
        ) -> None:
        """Initialize a dataset from `.npz` files.
        A single file is a sample, containing multiple channels and the label.
        (The array named `lebel_key` is the label, and others are channels.)

        Args:
            folder (str): The path to the folder containing `.npz` files.
            num (int): The number of samples whose indices range from 0 to num-1.
            label_key (str): The name of the label data in a `.npz` file.
        """
        super().__init__()
        assert isinstance(folder, str)

        self.path = os.path.join(folder, '')

        if isinstance(names, int):
            if names == -1:
                self.names_seq = [f[:-4] for f in os.listdir(folder) if f.endswith('.npz')]
            else:
                self.names_seq = range(names)
        else:
            # NOTE: The number of files must be limited, so the type of `names`
            # provided here must be Sequence instead of Iterable.
            self.names_seq = names

        NUM = len(self.names_seq)
        self.channel_keys = list(channel_keys) if (channel_keys is not None) else []
        self.label_key = label_key
        self.keep_dim = keep_dim
        self._cache = [None, ] * NUM if use_cache else None

    def __len__(self) -> int:
        return len(self.names_seq)

    def has_cache(self, index: int):
        if self._cache is None:
            return False
        return self._cache[index] is not None

    def _read_data(self, fname: Any):
        file_name = os.path.join(self.path, str(fname) + ".npz")

        with np.load(file_name) as f:
            datadict = dict(f)

        label = datadict[self.label_key]
        del datadict[self.label_key]

        if len(self.channel_keys) != 0:
            channels = [datadict[key] for key in self.channel_keys]
            if not self.keep_dim and len(channels) == 1:
                data = channels[0]
            else:
                data = np.stack(channels, axis=0)
            pair = torch.from_numpy(data), torch.from_numpy(label)
        else:
            pair = (torch.from_numpy(label), )

        return pair

    def _read_batch(self, names: Sequence[Any]):
        return [self._read_data(name) for name in names]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.has_cache(index):
            return self._cache[index]
        else:
            pair = self._read_data(self.names_seq[index])

            if self._cache is not None:
                self._cache[index] = pair
            return pair

    def transform_to(self, func: Callable[[Dict[str, NDArray]], None],
                     destination_folder: str, *, tqdm=False):
        """Transform the data and save to a new folder."""
        os.path.join(destination_folder, '')
        os.makedirs(destination_folder, exist_ok=True)

        if tqdm:
            from tqdm import tqdm
            iterator = tqdm(self.names_seq, desc=f"Transform", unit='sample')
        else:
            iterator = self.names_seq

        for fname in iterator:
            pathname_from = os.path.join(self.path, f"{fname}.npz")
            pathname_to = os.path.join(destination_folder, f"{fname}.npz")
            datadict = dict(np.load(pathname_from))
            func(datadict)
            np.savez(pathname_to, **datadict)
