import torch
from torch import Tensor
from torch.autograd.functional import jacobian

def minimize_levmarq(
    xs,
    ys,
    get_y_hat,
    eps_metric=1e-4,
    lam=1e-4,
    eps_grad=None,
    eps_x=None,
    eps_reduced_chi2=None,
    lam_decrease_factor=9.0,
    lam_increase_factor=20.0,
    max_iters=10000,
    small_number=0.0,
    lam_min=1e-7,
    lam_max=1e7,
) -> Tensor:
    
    """
    Levenberg-Marquardt minimizer, based on implementation 1 from `Gavin 2022 <https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf>`_.

    This minimizes :math:`|\\mathbf{y} - \\mathbf{\\hat{y}}(\\mathbf{x})|^2`, and is
    batched over parameters `x` and datapoints `y`.

    Args:
        x: batch of initial parameters.
        y: corresponding batch of target data.
        get_y_hat: function taking components of `x` as arguments and returning a
            prediction of `y`. Must be `vmap`able.
        eps_metric: tolerance for deciding when to decrease :math:`\\lambda`.
        lam: damping parameter :math:`\\lambda`.
        eps_grad: gradient convergence threshold. If `None`, doesn't check this
            convergence metric.
        eps_x: parameter convergence threshold. If `None`, doesn't check this
            convergence metric.
        eps_reduced_chi2: reduced :math:`\\chi^2` convergence threshold. If `None`,
            doesn't check this convergence metric.
        lam_decrease_factor: factor by which to decrease :math:`\\lambda` after accepting
            an update.
        lam_increase_factor: factor by which to increase :math:`\\lambda` after rejecting
            an update.
        max_iters: maximum number of iterations.
        small_number: this is added to the diagonal of the approximate Hessian and the
            parameter values to avoid inverting a singular matrix or dividing by zero.
        lam_min: minimum value permitted for :math:`\\lambda`.
        lam_max: maximum value permitted for :math:`\\lambda`.

    Notes:
        - Switch to jacfwd
        - Should be able to get value and Jacobian simultaneously
    """
    
    if len(xs) != len(ys):
        raise ValueError("x and y must having matching batch dimension")

    n = xs.shape[0]
    m = ys.shape[0]
    
    if m - n + 1 <= 0:
        raise ValueError(
            "number of data points per batch must be at least the number of parameters "
            "minus 1"
        )
        
    J = jacobian(get_y_hat, xs).view(xs.shape[0], -1)
    chi2_p = ((ys - get_y_hat(xs)) ** 2).sum(-1)
    
    for _ in range(max_iters):

        # Reduced chi^2 convergence check
        if (
            eps_reduced_chi2 is not None
            and chi2_p.max() / (m - n + 1) < eps_reduced_chi2
        ):
            break

        d_y_yhat = ys - get_y_hat(xs)
        JT_d_y_yhat = J.T @ d_y_yhat
        
        # Gradient convergence check
        if eps_grad is not None and JT_d_y_yhat.max() < eps_grad:
            break

        # Propose an update
        JTJ = J.T @ J
        H_approxs = (lam * torch.diag_embed(torch.diagonal(JTJ)))
        dxs = torch.linalg.inv(JTJ + H_approxs) @ JT_d_y_yhat
        
        if eps_x is not None and (dxs / (xs + small_number)).abs().max() < eps_x:
            break

        x_new = xs + dxs
        chi2 = ((ys - get_y_hat(x_new)) ** 2).sum(-1)
        
        # Decide whether to accept update
        metrics = (chi2_p- chi2) / (
            (dxs.T @ JT_d_y_yhat) + (dxs.T @(H_approxs @ dxs))
        ).abs()
    
        if metrics.max() > eps_metric:
            xs = x_new
            chi2_p = chi2
            lam = max(lam / lam_decrease_factor, lam_min)

        else:
            lam = min(lam * lam_increase_factor, lam_max)

    return xs