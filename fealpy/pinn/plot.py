
import numpy as np
from numpy.typing import NDArray
import torch
from torch.nn import Module
from torch.autograd import Variable


def meshgrid_mapping(nn: Module, *xi: NDArray):
    """
    Parameters
    ---
    nn: torch.nn.Module.
        Neural network.
    *xi: ArrayLike.
        See `numpy.meshgrid`.

    Return
    ---
    outputs, (X1, X2, ..., Xn)
    """
    mesh = np.meshgrid(*xi)
    flat_mesh = [np.ravel(x).reshape(-1, 1) for x in mesh]
    mesh_pt = [Variable(torch.from_numpy(x).float(), requires_grad=True) for x in flat_mesh]
    pt_u: torch.Tensor = nn(torch.cat(mesh_pt, 1))
    u_plot: NDArray = pt_u.data.cpu().numpy()
    return u_plot.reshape(mesh[0].shape), mesh
