
from .module import TensorMapping, Solution, ZeroMapping, Fixed, Extracted, Projected
from .function_space import TensorSpace, Function
from .linear import Standardize, Distance, MultiLinear
from .boundary import BoxDBCSolution, BoxDBCSolution1d, BoxDBCSolution2d, BoxNBCSolution
from .attention import GradAttention
from .rfm import RandomFeaturePoUSpace, LocalRandomFeatureSpace, RandomFeatureSpace
from .activate import Sin, Cos, Tanh
from .pou import PoUA, PoUSin
from .loss import ScaledMSELoss
