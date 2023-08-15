
from typing import Tuple, List

import torch
from torch import device, Tensor

from .sampler import Sampler


class Collocator(Sampler):
    """
    Generate collocation points.
    """
    def __init__(self, *start_stop_step_tuples: Tuple[float, float, int],
                 dtype=torch.float64,
                 device: device=None, requires_grad: bool=False) -> None:
        super().__init__(0, dtype, device, requires_grad)
        self.settings = start_stop_step_tuples
        example = self.run()
        self.m = example.shape[-1]

    def run(self):
        ND = len(self.settings)
        lins: List[Tensor] = []
        for i in range(ND):
            conf = self.settings[i]
            lins.append(torch.linspace(*conf, dtype=self.dtype, device=self.device,
                                       requires_grad=self.requires_grad))
        return torch.stack(torch.meshgrid(*lins, indexing='ij'), dim=-1).reshape(-1, ND)
