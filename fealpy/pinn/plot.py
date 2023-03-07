
import numpy as np
from numpy.typing import NDArray
import torch
from torch.autograd import Variable

from .nntyping import TensorFunction


def meshgrid_mapping(func: TensorFunction, *xi: NDArray):
    """
    Parameters
    ---
    func: TensorFunction.
    *xi: ArrayLike.
        See `numpy.meshgrid`.

    Return
    ---
    outputs, (X1, X2, ..., Xn)
    """
    mesh = np.meshgrid(*xi)
    flat_mesh = [np.ravel(x).reshape(-1, 1) for x in mesh]
    mesh_pt = [Variable(torch.from_numpy(x).float(), requires_grad=True) for x in flat_mesh]
    pt_u = func(torch.cat(mesh_pt, 1))
    u_plot: NDArray = pt_u.data.cpu().numpy()
    return u_plot.reshape(mesh[0].shape), mesh
