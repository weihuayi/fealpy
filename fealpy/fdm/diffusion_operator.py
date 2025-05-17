import math
from typing import Optional
import inspect
from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, spdiags, SparseTensor
from ..mesh import UniformMesh
from .operator_base import OpteratorBase, assemblymethod


class DiffusionOperator(OpteratorBase):
    """
    Discrete approximation of the second‐order diffusion operator:

        −∇·(D(x) ∇u(x))

    on structured (uniform Cartesian) meshes using finite difference methods.

    This operator appears in many classes of PDEs, including but not limited to:
        - heat equation
        - reaction–diffusion systems
        - diffusion‐dominated convection–diffusion equations

    -------------------------------------------------------------------------------
    Method extensibility:
        Different coefficient types or optimizations can be supported via the
        `assemblymethod` decorator mechanism. Users may switch between
        implementations by specifying the `method` parameter in the constructor.

    -------------------------------------------------------------------------------
    Attributes:
        mesh           : UniformMesh object
        diffusion_coef : Function or constant tensor D(x) of shape (GD, GD)
    """
    def __init__(self, mesh: UniformMesh, diffusion_coef,
                 method: Optional[str]=None):
        """
        Initialize the diffusion operator.

        Parameters:
            mesh            : The structured mesh over which the operator is defined.
            diffusion_coef  : Either a constant tensor D of shape (GD,GD),
                              or a function taking node coordinates p and returning D(p).
            method          : Optional string key identifying which assembly
                              implementation to use. Defaults to 'assembly'.
        """
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.mesh = mesh  # Store the mesh for later assembly
        self.diffusion_coef = diffusion_coef 
        
    def assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the diffusion operator
        using second‐order central differences.

        The discrete stencil in each dimension i contributes:
            −(1/h_i^2) * [ u_{i+1} − 2 u_i + u_{i−1} ]

        Returns:
            SparseTensor: CSR-format matrix of size (NN, NN), where NN is total nodes.
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        # Evaluate diffusion tensor at all nodes (if function) or take constant
        # D = self.diffusion_coef(node)    # shape == (GD, GD) or broadcastable
        n = len(inspect.signature(self.diffusion_coef).parameters) 
        if n == 0:
            D = self.diffusion_coef()       
        else:
            D =self.diffusion_coef(node) # shape == (GD, GD) or broadcastable
        
        # Mesh spacing and coefficient vector per dimension
        h = mesh.h                       # tuple of length GD
        c = 1.0 / (h ** 2) # 1/h_i^2
        # If D is nontrivial tensor, contract: c_i ← Σ_j D_{ij} c_j
        c = D * c                        # shape (GD,)
        NN = mesh.number_of_nodes()      # total nodes
        K  = mesh.linear_index_map('node')  
        shape = K.shape                  # multi-index grid shape

        # Main diagonal: sum over dims of 2c_i
        diag_val = bm.full(NN, 2 * bm.sum(c.diagonal()), dtype=mesh.ftype)
        I = K.flat
        J = K.flat
        A = csr_matrix((diag_val, (I, J)), shape=(NN, NN))

        # Prepare slicing for neighbor access
        full = (slice(None),) * GD
        shifts = [math.prod(shape[j] for j in range(GD) if j != i) for i in range(GD)]

        # Off-diagonal contributions in each dimension
        for i in range(GD):
            # shift between nodes in dim i
            off_val = bm.full(NN - shifts[i], -c[i, i], dtype=mesh.ftype)

            # build slices for forward/backward neighbors
            s_plus  = full[:i] + (slice(1, None),) + full[i+1:]
            s_minus = full[:i] + (slice(None, -1),) + full[i+1:]
            I_idx = K[s_plus].flat
            J_idx = K[s_minus].flat

            # add coupling to both directions
            A += csr_matrix((off_val, (I_idx, J_idx)), shape=(NN, NN))
            A += csr_matrix((off_val, (J_idx, I_idx)), shape=(NN, NN))
            
            for j in range(i+1, GD):
                if c[i, j] != 0:
                    # +i +j
                    s_ip_jp = full[:i] + (slice(1,None),) + full[i+1:j] \
                            + (slice(1,None),) + full[j+1:]
                    # -i +j
                    s_im_jp = full[:i] + (slice(None,-1),) + full[i+1:j] \
                            + (slice(1,None),) + full[j+1:]
                    # +i -j
                    s_ip_jm = full[:i] + (slice(1,None),) + full[i+1:j] \
                            + (slice(None,-1),) + full[j+1:]
                    # -i -j
                    s_im_jm = full[:i] + (slice(None,-1),) + full[i+1:j] \
                            + (slice(None,-1),) + full[j+1:]
              
                    s_c_ip_jp = full[:i] + (slice(None,-1),) + full[i+1:j] \
                                + (slice(None,-1),) + full[j+1:]
                    s_c_im_jp = full[:i] + (slice(1,None),)  + full[i+1:j] \
                                + (slice(None,-1),) + full[j+1:]
                    s_c_ip_jm = full[:i] + (slice(None,-1),) + full[i+1:j] \
                                + (slice(1,None),)  + full[j+1:]
                    s_c_im_jm = full[:i] + (slice(1,None),)  + full[i+1:j] \
                                + (slice(1,None),)  + full[j+1:]
                                
                    I_ip_jp = K[s_ip_jp].ravel();  J_ip_jp = K[s_c_ip_jp].ravel()
                    I_im_jp = K[s_im_jp].ravel();  J_im_jp = K[s_c_im_jp].ravel()
                    I_ip_jm = K[s_ip_jm].ravel();  J_ip_jm = K[s_c_ip_jm].ravel()
                    I_im_jm = K[s_im_jm].ravel();  J_im_jm = K[s_c_im_jm].ravel()
                    coeff = (D[i, j] + D[j, i])/(h[i]*h[j])
                    A += csr_matrix(( +coeff*bm.ones_like(I_ip_jp), (J_ip_jp, I_ip_jp)), shape=(NN,NN))
                    A += csr_matrix(( -coeff*bm.ones_like(I_im_jp), (J_im_jp, I_im_jp)), shape=(NN,NN))
                    A += csr_matrix(( -coeff*bm.ones_like(I_ip_jm), (J_ip_jm, I_ip_jm)), shape=(NN,NN))
                    A += csr_matrix(( +coeff*bm.ones_like(I_im_jm), (J_im_jm, I_im_jm)), shape=(NN,NN))
                                              
        return A

    @assemblymethod('fast')
    def assembly_const(self) -> SparseTensor:
        """
        A faster assembly for constant, isotropic diffusion D = d·I.

        In this special case c_i = d/h_i^2, and the matrix has a
        tensor‐product structure exploitable for performance.
        """
        # [Implementation omitted for brevity]
        raise NotImplementedError("Fast constant‐coefficient assembly not yet implemented")