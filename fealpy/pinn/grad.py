
import torch
from torch import Tensor
from torch.autograd import grad


def gradient(output: Tensor, input: Tensor,
             create_graph=False, allow_unused=False, split: bool=False):
    g = grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        allow_unused=allow_unused
    )[0]
    if split:
        return torch.split(g, 1, dim=-1)
    return g


def grad_by_fts(scaler_out: Tensor, vector_in: Tensor,
                create_graph=False, allow_unused=False, split: bool=False):
    if len(scaler_out.shape) != 2:
        raise Exception("Arg 'scaler_out' must has samples in dim-0 and only 1 feature in dim-1.")
    if scaler_out.shape[1] != 1:
        raise ValueError("For multiple features, gradients of them will be added.")
    g = grad(
        outputs=scaler_out,
        inputs=vector_in,
        grad_outputs=torch.ones_like(scaler_out),
        create_graph=create_graph,
        allow_unused=allow_unused
    )[0]
    if split:
        return torch.split(g, 1, dim=-1)
    return g


def grad_of_fts(vector_out: Tensor, vector_in: Tensor, ft_idx: int,
                create_graph=False, allow_unused=False, split: bool=False):
    if len(vector_out.shape) != 2:
        raise Exception("Arg 'scaler_out' must has samples in dim-0 and features in dim-1.")
    ret = torch.zeros_like(vector_out)
    for i, ft_out in enumerate(torch.split(vector_out, 1, dim=1)):
        g = grad_by_fts(
            scaler_out=ft_out,
            vector_in=vector_in,
            create_graph=create_graph,
            allow_unused=allow_unused,
            split=False
        )
        ret[:, i] = g[:, ft_idx]
    if split:
        return torch.split(ret, 1, dim=-1)
    return ret
