from typing import Any, Optional, Union

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
from fealpy.model.stokes import StokesPDEDataT

from fealpy.functionspace import functionspace 
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import DirichletBC

class StokesLFEMModel(ComputationalModel):

    def __init__(self, options: dict = None):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )

