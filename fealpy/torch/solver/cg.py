
from typing import Optional

import torch
from torch import Tensor, sum
from torch.sparse import mm
from torch.autograd import Function

from .. import logger


def sparse_cg(A: Tensor, b: Tensor, x0: Optional[Tensor]=None, *,
              batch_first: bool=False,
              atol: float=1e-12, rtol: float=1e-8, maxiter: Optional[int]=10000) -> Tensor:
    """Solve a linear system Ax = b using the Conjugate Gradient (CG) method.

    Args:
        A (Tensor): The coefficient matrix of the linear system, must be a 2D sparse CSR or COO tensor.
        b (Tensor): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (Tensor): Initial guess for the solution, a 1D or 2D tensor.\
        Must have the same shape as b when reshaped appropriately.
        batch_first (bool, optional): Whether the batch dimension of `b` and `x0`\
        is the first dimension. Default is False.
        atol (float, optional): Absolute tolerance for convergence. Default is 1e-12.
        rtol (float, optional): Relative tolerance for convergence. Default is 1e-8.
        maxiter (int, optional): Maximum number of iterations allowed. Default is 10000.\
        If not provided, the method will continue until convergence based on the given tolerances.

    Returns:
        Tensor: The approximate solution to the system Ax = b.

    Raises:
        ValueError: If inputs do not meet the specified conditions (e.g., A is not sparse, dimensions mismatch).

    Note:
        This implementation assumes that A is a symmetric positive-definite matrix,
        which is a common requirement for the Conjugate Gradient method to work correctly.
    """
    assert isinstance(A, Tensor), "A must be a torch.Tensor"
    assert isinstance(b, Tensor), "b must be a torch.Tensor"
    if x0 is not None:
        assert isinstance(x0, Tensor), "x0 must be a torch.Tensor if not None"
    unsqueezed = False

    if not (A.is_sparse_csr or A.is_sparse):
        raise ValueError("A must be a sparse CSR or COO matrix")

    if (A.ndim != 2) or (A.shape[1] != A.shape[0]):
        raise ValueError("A must be a square matrix (2D tensor)")

    if b.ndim != 2:
        if b.ndim == 1:
            b = b.unsqueeze(1)
            unsqueezed = True
        else:
            raise ValueError("b must be a 2D dense tensor")

    if x0 is None:
        x0 = torch.zeros_like(b, requires_grad=False)
    else:
        if x0.ndim == 1:
            x0 = x0.unsqueeze(1)
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")

    if batch_first:
        b = b.transpose(0, 1)
        x0 = x0.transpose(0, 1)

    if A.shape[1] != b.shape[0]:
        raise ValueError("b and A must have the same number of rows")

    sol = SparseCG.apply(A, b, x0, atol, rtol, maxiter)

    if unsqueezed:
        sol = sol.squeeze(1)
    if batch_first:
        sol = sol.transpose(0, 1)
    return sol


class SparseCG(Function):
    @staticmethod
    def forward(A, b, x0, atol, rtol, maxiter):
        # initialize
        A = A.detach()
        b = b.detach()
        x = x0.detach()        # (dof, batch)
        r = b - mm(A, x)       # (dof, batch)
        p = r                  # (dof, batch)
        n_iter = 0
        b_norm = b.norm()

        # iterate
        while True:
            Ap = mm(A, p)      # (dof, batch)
            rTr = sum(r**2, dim=0)
            alpha = rTr / sum(p*Ap, dim=0)  # r @ r / (p @ Ap) # (batch,)
            x = x + alpha.unsqueeze(0) * p  # (dof, batch)
            r_new = r - alpha.unsqueeze(0) * Ap
            rTr_new = sum(r_new**2, dim=0)  # (batch,)
            r_norm_new = rTr_new.sum().sqrt()

            n_iter += 1

            if r_norm_new < atol:
                logger.info(f"SparseCG: converged in {n_iter} iterations, "
                            "stopped by absolute tolerance.")
                break

            if r_norm_new / b_norm < rtol:
                logger.info(f"SparseCG: converged in {n_iter} iterations, "
                            "stopped by relative tolerance.")
                break

            if (maxiter is not None) and (n_iter >= maxiter):
                logger.info(f"SparseCG: stopped by maxiter ({maxiter}).")
                break

            beta = rTr_new / rTr # (batch,)
            p = r_new + beta.unsqueeze(0) * p
            r = r_new

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, b, x0, atol, rtol, maxiter = inputs
        x = output
        ctx.save_for_backward(A, x)

    # NOTE: This backward function is not implemented yet.
    # Now only the shape is correct to test the autograd, while values do not make sense.
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        A, x = ctx.saved_tensors
        grad_A = grad_b = None
        # NOTE: A[ij]  x[jb]  grad_output[jb]

        # 如果第零个或者第一个需要求导
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # 区分 A 的不同稀疏格式，用不同的方法求 A 的逐个元素倒数
            if A.is_sparse_csr:
                inv_A = torch.sparse_csr_tensor(A.crow_indices(), A.col_indices(), 1/A.values(), A.size())
            elif A.is_sparse: # COO
                inv_A = torch.sparse_coo_tensor(A.indices(), 1/A.values(), A.size())
            else:
                raise RuntimeError("SparseCGBackward: A must be a sparse CSR, CSC or COO matrix.")

        if ctx.needs_input_grad[0]:
            weights = -sum(grad_output*x, dim=-1).unsqueeze(0)
            weights = torch.broadcast_to(weights, A.shape)
            grad_A = inv_A * weights

        if ctx.needs_input_grad[1]:
            weights = grad_output.mean(dim=-1, keepdim=True)
            grad_b = mm(inv_A, weights)
            grad_b = torch.broadcast_to(grad_b, x.shape)

        return grad_A, grad_b, None, None, None, None
