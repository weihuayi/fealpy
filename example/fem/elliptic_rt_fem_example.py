
from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.fem import EllipticRTFEMModel
from scipy.sparse.linalg import splu
model = EllipticRTFEMModel()
model.run()
#model.show_mesh()
A,b = model.linear_system()
A,b = model.boundary_apply()
p,u = model.solve()
print(u)
