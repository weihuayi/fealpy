import numpy as np
import jax
import jax.numpy as jnp


class LagrangeFESpace():

    def __init__(self, mesh, p=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()


    def basis(self, bc, index=np.s_[:]):
        p = self.p
        phi = self.mesh.shape_function(bc, p=p)
        return phi[..., None, :]

    def grad_basis(self, bc, index=np.s_[:]):
        """
        @brief
        """
        return self.mesh.grad_shape_function(bc, p=self.p, index=index)


    def value(self, uh, bc, index=np.s_[:]):
        """
        @brief
        """
        pass
