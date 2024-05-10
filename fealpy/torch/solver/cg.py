
import torch
from torch import Tensor, sum
from torch.sparse import mm
from torch.autograd import Function

from .. import logger


def sparse_cg(A: Tensor, b: Tensor, x0: Tensor,
              atol: float=1e-12, rtol: float=1e-8, maxiter=1000) -> Tensor:
    unsqueezed = False

    if b.ndim == 1:
        b = b.unsqueeze(1)
        unsqueezed = True
    if x0.ndim == 1:
        x0 = x0.unsqueeze(1)

    sol = SparseCG.apply(A, b, x0, atol, rtol, maxiter)

    if unsqueezed:
        sol = sol.squeeze(1)
    return sol


class SparseCG(Function):
    @staticmethod
    def forward(A, b, x0, atol, rtol, maxiter):
        # check inputs
        assert isinstance(A, Tensor), "A must be a torch.Tensor"
        assert isinstance(b, Tensor), "b must be a torch.Tensor"
        assert isinstance(x0, Tensor), "x0 must be a torch.Tensor"
        if not (A.is_sparse_csr or A.is_sparse):
            raise ValueError("A must be a sparse CSR or COO matrix")
        if A.ndim != 2:
            raise ValueError("A must be a 2-dimensional sparse tensor")
        if b.ndim != 2:
            raise ValueError("b must be a 2D dense tensor")
        if A.shape[0] != b.shape[0]:
            raise ValueError("A and b must have the same number of rows")
        if A.shape[1] != A.shape[0]:
            raise ValueError("A must be a square matrix")
        if x0.ndim != 2:
            raise ValueError("x0 must be a 2D dense tensor")
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")

        # initialize
        x = x0                 # (dof, batch)
        r = b - mm(A, x0)      # (dof, batch)
        p = r                  # (dof, batch)
        n_iter = 0

        #iterate
        while True:
            Ap = mm(A, p)      # (dof, batch)
            alpha = sum(r**2, dim=0) / sum(p*Ap, dim=0)  # r @ r / (p @ Ap) # (batch,)
            x = x + alpha.unsqueeze(0) * p # (dof, batch)
            r_new = r - alpha.unsqueeze(0) * Ap

            n_iter += 1

            if r_new.norm() < atol:
                logger.info(f"SparseCG: converged in {n_iter} iterations, "
                            "stopped by absolute tolerance.")
                break

            if r_new.norm() / b.norm() < rtol:
                logger.info(f"SparseCG: converged in {n_iter} iterations, "
                            "stopped by relative tolerance.")
                break

            if n_iter >= maxiter:
                logger.info(f"SparseCG: stopped by maxiter ({maxiter}).")
                break

            beta = sum(r_new**2, dim=0) / sum(r**2, dim=0) # (batch,)
            p = r_new + beta.unsqueeze(0) * p
            r = r_new

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, b, x0, atol, rtol, maxiter = inputs
        x = output
        ctx.save_for_backward(A, x)

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

        return grad_A, grad_b, None, None, None, None
