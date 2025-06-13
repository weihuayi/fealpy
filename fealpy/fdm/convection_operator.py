import math
from typing import Optional, Callable, Union
import inspect
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, spdiags, SparseTensor
from ..mesh import UniformMesh

from .operator_base import OpteratorBase, assemblymethod

class ConvectionOperator(OpteratorBase):
    """
    Discrete approximation of the first-order differential operator:

        b(x) · ∇u(x)

    on structured (uniform Cartesian) meshes using finite difference methods.

    This operator appears in many classes of PDEs, including but not limited to:
        - pure transport (hyperbolic) equations
        - convection-diffusion-reaction problems
        - advection-dominated elliptic equations
        - adjoint equations in PDE-constrained optimization
    -----------------------------------
    Parameters:
        mesh           : The structured mesh over which the operator is defined.
        convection_coef: Can be  a tensor(shape:(GD,)) 
                         or a function that returns a tensor, 
                         or a int and float,
                         or a tuple or list of length GD.

        method         : Optional string: 'central_const_2' or 'upwind_const_1'.

    -------------------------------------------------------------------------------
    Method extensibility:
        Different finite difference schemes are supported via the `assemblymethod`
        decorator mechanism. Users may switch between schemes by specifying the
        `method` parameter in the constructor.

    -------------------------------------------------------------------------------

    Examples:
        - 'upwind_const_1'   : first-order upwind, constant velocity
        - 'central_func_2'   : second-order central, function-valued b(x)
        - 'nonupwind_func_3' : third-order non-upwind scheme, function-valued b(x)

    Default method is 'upwind_const_1' if not specified.

    -------------------------------------------------------------------------------
    Attributes:
        mesh            : UniformMesh object
        convection_coef : Callable returning velocity b (constant or function-valued)
    """

    def __init__(self,
                 mesh: UniformMesh,
                 convection_coef: Union[Callable, TensorLike, int, float],
                 method: Optional[str] = None):

        method = 'assembly' if method is None else method
        super().__init__(method=method)

        self.mesh = mesh
        self.convection_coef = convection_coef

    def assembly(self) -> SparseTensor:
        """
        Assemble the convection matrix using first‐order upwind differences
        for constant velocity vector b (scheme: 'upwind_const_1').

        This is the default implementation if no method is specified.

        Returns:
            SparseTensor: the discrete convection operator matrix.
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        context = bm.context(node)
        
        if callable(self.convection_coef):
        # 处理函数情况
            sig = inspect.signature(self.convection_coef)
            l = len(sig.parameters)
            if l == 2:
                b = self.convection_coef(node)
            else:
                b = self.convection_coef()
        elif isinstance(self.convection_coef, (int, float)):
            # 处理单个数值情况
            data = [float(self.convection_coef)] * GD
            b = bm.array(data)
        elif isinstance(self.convection_coef, (list, tuple)):
            # 处理列表或元组情况
            b = bm.array(self.convection_coef, dtype=bm.float64)
        elif isinstance(self.convection_coef, TensorLike):
            b = self.convection_coef
        else:
            raise ValueError(f"Invalid data type: convection_coef must be an int, float, list, tuple, tensor, or callable(e.g. function). \
                             Now is {type(self.convection_coef)}.")
            
        h = mesh.h                                # uniform spacing in each dimension
        NN = mesh.number_of_nodes()
        K = mesh.linear_index_map('node')         # multi-index map
        shape = K.shape
        full = (slice(None),) * GD                # full slicing tuple

        c = bm.abs(b / h)   # component-wise |b_i|/h_i
        diag = bm.full((NN,), bm.sum(c), **context)  # diagonal term: sum_i |b_i|/h_i
        A = spdiags(diag, 0, NN, NN, format='csr')

        for i in range(GD):
            n_shift = math.prod(cnt for idx, cnt in enumerate(shape) if idx != i)
            off = bm.full((NN - n_shift,), -c[i], **context)

            s1 = full[:i] + (slice(1, None),) + full[i+1:]
            s2 = full[:i] + (slice(None, -1),) + full[i+1:]
            I = K[s1].ravel()
            J = K[s2].ravel()

            if b[i] > 0:
                A += csr_matrix((off, (I, J)), shape=(NN, NN))  # backward difference
            else:
                A += csr_matrix((off, (J, I)), shape=(NN, NN))  # forward difference

        return A

    @assemblymethod('central_const_2')
    def assembly_central_const(self) -> SparseTensor:
        """
        Assemble the convection matrix using second-order central differences
        for constant velocity vector b (scheme: 'central_const_2').

        Stencil: (u_{j+1} - u_{j-1}) / (2 h)

        Returns:
            SparseTensor: the discrete convection operator matrix.
        """
        mesh = self.mesh
        node = mesh.entity('node')
        context = bm.context(node)
        GD = mesh.geo_dimension()
        NN = mesh.number_of_nodes()
        K = mesh.linear_index_map('node')
        shape = K.shape
        full = (slice(None),) * GD

         
        if callable(self.convection_coef):
        # 处理函数情况
            sig = inspect.signature(self.convection_coef)
            l = len(sig.parameters)
            if l == 2:
                b = self.convection_coef(node)
            else:
                b = self.convection_coef()
        elif isinstance(self.convection_coef, (int, float)):
            # 处理单个数值情况
            data = [float(self.convection_coef)] * GD
            b = bm.array(data)
        elif isinstance(self.convection_coef, (list, tuple)):
            # 处理列表或元组情况
            b = bm.array(self.convection_coef, dtype=bm.float64)
        elif isinstance(self.convection_coef, TensorLike):
            b = self.convection_coef
        else:
            raise ValueError(f"Invalid data type: convection_coef must be an int, float, list, tuple, tensor, or callable(e.g. function). \
                             Now is {type(self.convection_coef)}.")
        

        c = b / mesh.h / 2.0                            # central difference coefficient

        zero_diag = bm.zeros(NN, dtype=bm.float64) 
        A = spdiags(zero_diag, 0, NN, NN, format='csr')
        for i in range(GD):
            n_shift = math.prod(cnt for idx, cnt in enumerate(shape) if idx != i)
            off = bm.full((NN - n_shift,), c[i], **context)

            s1 = full[:i] + (slice(1, None),) + full[i+1:]   # 1:
            s0 = full[:i] + (slice(None, -1),) + full[i+1:]  # 0:-1
            I = K[s1].ravel()
            J = K[s0].ravel()

            A += csr_matrix((-off, (I, J)), shape=(NN, NN))
            A += csr_matrix(( off, (J, I)), shape=(NN, NN))

        return A

