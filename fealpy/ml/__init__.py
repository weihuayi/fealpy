from .grad import gradient
from .sampler.sampler import Sampler
from .modules.module import Solution
from .tools import proj, mkfs, use_mkfs, as_tensor_func
from .torch_mapping import optimizers, activations
from .poisson_penn_model import PoissonPENNModel
from .poisson_pinn_model import PoissonPINNModel
from .helmholtz_pinn_model import HelmholtzPINNModel

