
from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike

from .. import logger


class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...


def cg(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None, M: Optional[SupportsMatmul] = None, *,
       batch_first: bool=False,
       atol: float=1e-12, rtol: float=1e-8,
       maxit: Optional[int]=10000,returninfo: bool=False) -> TensorLike:
    """Solve a linear system Ax = b using the Conjugate Gradient (CG) method.

    Parameters:
        A (SupportsMatmul): The coefficient matrix of the linear system.
        b (TensorLike): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (TensorLike): Initial guess for the solution, a 1D or 2D tensor.\
        Must have the same shape as b when reshaped appropriately.
        batch_first (bool, optional): Whether the batch dimension of `b` and `x0`\
        is the first dimension. Ignored if `b` is an 1-d tensor. Default is False.
        atol (float, optional): Absolute tolerance for convergence. Default is 1e-12.
        rtol (float, optional): Relative tolerance for convergence. Default is 1e-8.
        maxit (int, optional): Maximum number of iterations allowed. Default is 10000.\
        If not provided, the method will continue until convergence based on the given tolerances.
        returninfo(bool):if or not return info{['residual],['niter]}

    Returns:
        Tensor: The approximate solution to the system Ax = b.

    Raises:
        ValueError: If inputs do not meet the specified conditions (e.g., A is not sparse, dimensions mismatch).

    Note:
        This implementation assumes that A is a symmetric positive-definite matrix,
        which is a common requirement for the Conjugate Gradient method to work correctly.
    """
 

    assert isinstance(b, TensorLike), "b must be a Tensor"
    if x0 is not None:
        assert isinstance(x0, TensorLike), "x0 must be a Tensor if not None"
    single_vector = b.ndim == 1

    if b.ndim not in {1, 2}:
        raise ValueError("b must be a 1D or 2D dense tensor")

    if x0 is None:
        x0 = bm.zeros_like(b)
    else:
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")
    
    # if M is not None  and M.shape != A.shape:
    #     raise ValueError("A and M must have the same shape")


    if (not single_vector) and batch_first:
        b = bm.swapaxes(b, 0, 1)
        x0 = bm.swapaxes(x0, 0, 1)

    sol,info = _cg_impl(A, b, x0,M,atol, rtol, maxit)

    if (not single_vector) and batch_first:
        sol = bm.swapaxes(sol, 0, 1)
    if returninfo is True:
        return sol,info
    else:
        return sol


def _cg_impl(A: SupportsMatmul, b: TensorLike, x0: TensorLike, M: SupportsMatmul, atol, rtol, maxit):
    # initialize
    info = {}
    x = x0              # (dof, batch)
    r = b - A @ x       # (dof, batch)
    z = M @ r if M is not None else r
    p = z               # (dof, batch)
    n_iter = 0
    b_norm = bm.linalg.norm(b)
    sum_func = bm.sum
    sqrt_func = bm.sqrt
    rTr = sum_func(r * z, axis=0)
    Ap = A @ p
    # iterate
    while True:
        Ap = A @ p      # (dof, batch)
        alpha = rTr / sum_func(p*Ap, axis=0)  # r @ r / (p @ Ap) # (batch,)
        x = x + alpha[None, ...] * p  # (dof, batch)
        r_new = r - alpha[None, ...] * Ap
        z_new = M @ r_new if M is not None  else r_new 
        rTr_new = sum_func(r_new*z_new, axis=0)  # (batch,)
        r_norm_new = sqrt_func(sum_func(rTr_new))

        n_iter += 1
        info['residual'] = r_norm_new
        info['niter'] = n_iter
        if r_norm_new < atol:
            logger.info(f"CG: converged in {n_iter} iterations, "
                        "stopped by absolute tolerance.")
            break

        if r_norm_new < rtol * b_norm:
            logger.info(f"CG: converged in {n_iter} iterations, "
                        "stopped by relative tolerance.")
            break

        if (maxit is not None) and (n_iter >= maxit):
            logger.info(f"CG: failed, stopped by maxit ({maxit}).")
            break

        beta = rTr_new / rTr # (batch,)
        p = z_new + beta[None, ...] * p
        r, z, rTr = r_new, z_new, rTr_new

    return x,info

    # @staticmethod
    # def setup_context(ctx, inputs, output):
    #     A, b, x0, atol, rtol, maxit = inputs
    #     x = output
    #     ctx.save_for_backward(A, x)

    # # NOTE: This backward function is not implemented yet.
    # # Now only the shape is correct to test the autograd, while values do not make sense.
    # @staticmethod
    # def backward(ctx, grad_output: Tensor):
    #     A, x = ctx.saved_tensors
    #     grad_A = grad_b = None
    #     # NOTE: A[ij]  x[jb]  grad_output[jb]

    #     # 如果第零个或者第一个需要求导
    #     if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
    #         # 区分 A 的不同稀疏格式，用不同的方法求 A 的逐个元素倒数
    #         if A.is_sparse_csr:
    #             inv_A = torch.sparse_csr_tensor(A.crow_indices(), A.col_indices(), 1/A.values(), A.size())
    #         elif A.is_sparse: # COO
    #             inv_A = torch.sparse_coo_tensor(A.indices(), 1/A.values(), A.size())
    #         else:
    #             raise RuntimeError("SparseCGBackward: A must be a sparse CSR, CSC or COO matrix.")

    #     if ctx.needs_input_grad[0]:
    #         weights = -sum(grad_output*x, dim=-1).unsqueeze(0)
    #         weights = torch.broadcast_to(weights, A.shape)
    #         grad_A = inv_A * weights

    #     if ctx.needs_input_grad[1]:
    #         weights = grad_output.mean(dim=-1, keepdim=True)
    #         grad_b = mm(inv_A, weights)
    #         grad_b = torch.broadcast_to(grad_b, x.shape)

    #     return grad_A, grad_b, None, None, None, None
