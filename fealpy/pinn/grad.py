
import torch
from torch.autograd import grad


def gradient(output: torch.Tensor, input: torch.Tensor,
             create_graph=False, split: bool=False):
    g = grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph
    )[0]
    if split:
        return torch.split(g, 1, dim=-1)
    return g
