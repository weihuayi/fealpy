
import torch
from torch import Tensor
from torch.autograd import grad


def gradient(output: Tensor, input: Tensor,
             create_graph=False, allow_unused=False, split: bool=False):
    """
        Compute the gradient of the output with respect to the input tensor.
        
        This function calculates the gradient using automatic differentiation and provides
        options for further computation and handling of special cases.

        Parameters:
            output: The output tensor for which to compute gradients.
            input: The input tensor with respect to which gradients are computed.
            create_graph: If True, graph of the derivative will be constructed, allowing
                        computation of higher order derivatives. Default is False.
            allow_unused: If True, allows unused parameters in the computation graph.
                        Default is False.
            split: If True, splits the gradient tensor along the last dimension.
                   Default is False.

        Returns:
            Tensor or tuple[Tensor]: The computed gradient(s). Returns a single tensor unless
                                    split=True, in which case returns a tuple of tensors.
    """
    g = grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        allow_unused=allow_unused
    )[0]
    if g is None:
        raise ValueError(f"Gradient for input '{input}' with respect to output '{output}' is None.")

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
