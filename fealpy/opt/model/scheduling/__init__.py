from typing import TypeVar, Protocol
from fealpy.backend import TensorLike

class SchedulingProtocol(Protocol):
    """
    Protocol defining the interface of scheduling optimization problems.

    Any scheduling problem compatible with the optimization framework
    should implement this protocol.

    Required methods:
        evaluate(x):
            Compute the objective value (fitness) of one or more candidate
            scheduling solutions.

    Notes:
        - The protocol is backend-agnostic.
        - Both single-solution and batched evaluation are supported,
          provided the returned tensor matches the input shape.
    """
    def evaluate(self, x: TensorLike) -> TensorLike: ...

SchedulingT = TypeVar('SchedulingT', bound=SchedulingProtocol)

DATA_TABLE = {
    1: ('permutation_flow_shopscheduling_prob', 'PermutationFlowShopschedulingProb')
}