
import torch
from torch.autograd import Function


class SparseCG(Function):
    @staticmethod
    def forward(ctx, A, b, x0, atol, rtol, maxiter):
        ...

    @staticmethod
    def backward(ctx, grad_output):
        ...
