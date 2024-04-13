
import torch
from torch import Tensor, Size


class LazyCOOMatrix():
    def __init__(self, data: Tensor, row: Tensor, col: Tensor):
        self.data = data
        self.row = row
        self.col = col
