
import jax.numpy as jnp
CONTEXT = 'jax'

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer

from fealpy.jax.mesh import TriangleMesh
from fealpy.np.mesh import TriangleMesh as T_

NX, NY = 64, 64

# mesh_ = T_.from_box(nx=NX, ny=NY)
mesh = TriangleMesh.from_box(nx=NX, ny=NY)


