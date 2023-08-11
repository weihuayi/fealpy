from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor
from .modules import TensorMapping

from .nntyping import Operator, GeneralSampler


class LearningMachine():
    """Neural network trainer."""
    def __init__(self, s: TensorMapping, cost_function: Optional[Module]=None) -> None:
        self.__module = s

        if cost_function:
            self.cost = cost_function
        else:
            self.cost = torch.nn.MSELoss(reduction='mean')

    @property
    def solution(self):
        return self.__module


    def loss(self, sampler: GeneralSampler, func: Operator,
             target: Optional[Tensor]=None, output_samples: bool=False):
        """
        @brief Calculate loss value.

        @param sampler: Sampler.
        @param func: Operator.\
                     A function get x and u(x) as args. (e.g A pde or boundary condition.)
        @param target: Tensor or None.\
                       If `None`, the output will be compared with zero tensor.
        @param output_samples: bool. Defaults to `False`.\
                               Return samples from the `sampler` if `True`.

        @note Arg `func` should be defined like:
        ```
            def equation(p: Tensor, u: TensorFunction) -> Tensor:
                ...
        ```
        Here `u` may be a function of `p`.
        """
        inputs = sampler.run()
        param = next(self.solution.parameters())

        if param is not None:
            inputs = inputs.to(param.device)

        outputs = func(inputs, self.solution.forward)

        if not target:
            target = torch.zeros_like(outputs)

        ret: Tensor = self.cost(outputs, target)

        if output_samples:
            return ret, inputs
        return ret
