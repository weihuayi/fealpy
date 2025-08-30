from typing import NamedTuple
from fealpy.backend import TensorLike
from .model_manager import PathPlanningModelManager

class OptResult(NamedTuple):
    solution: TensorLike
    cost: float