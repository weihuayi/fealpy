from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('INFO')
from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.pde.poisson_3d import CosCosCosData
from fealpy.mesh import TriangleMesh,TetrahedronMesh
from fealpy.fem import PoissonLFEMSolver
import time
from fealpy.sparse import CSRTensor


tmr = timer()
next(tmr)

# Different configurations: coarsest mesh size and number of refinements.
configurations = [
    (2, 6),  
    #(2, 4),   
    #(32, 6)   
]

pde = CosCosData()
domain = pde.domain()

# Iterate over each configuration.
for n, m in configurations:
    mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
    IM = mesh.uniform_refine(n=m, returnim=True)

    p = 1  
    s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)
    tmr.send(f"Running gamg_solve with (n, m) = ({n}, {m})")
    s0.gamg_solve(IM)
    tmr.send(f"Completed gamg_solve for (n, m) = ({n}, {m})")

s0.cg_solve()
#s0.gs_solve()
#s0.jacobi_solve()
s0.minres_solve()
s0.gmres_solve()
tmr.send(None)