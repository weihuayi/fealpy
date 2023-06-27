
import numpy as np
import torch
from torch import Tensor

from .mesh_ds import MeshDataStructure, Redirector


class Mesh1dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of 1-d mesh.\
           This is an abstract class and can not be used directly.
    """
    TD = 1
    edge: Redirector[Tensor] = Redirector('cell')

    @property
    def face(self):
        return torch.arange(0, self.NN, device=self.device).unsqueeze(-1)
