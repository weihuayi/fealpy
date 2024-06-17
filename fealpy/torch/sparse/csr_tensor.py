
from typing import Optional, Union, overload, List

import torch


_Size = torch.Size
_device = torch.device
Tensor = torch.Tensor
Number = Union[int, float]


class CSRTensor():
    def __init__(self, crow: Tensor, col: Tensor, values: Optional[Tensor],
                 spshape: _Size) -> None:
        """Initializes CSR format sparse tensor.

        Parameters:
            crow (Tensor): _description_
            col (Tensor): _description_
            values (Optional[Tensor]): _description_
            spshape (_Size): _description_
        """
        self.crow = crow
        self.col = col
        self.values = values

        if spshape is None:
            nrow = crow.size(0) - 1
            ncol = col.max().item() + 1
            self._spshape = torch.Size((nrow, ncol))
        else:
            self._spshape = torch.Size(spshape)
