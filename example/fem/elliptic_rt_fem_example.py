
from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.fem import EllipticRTFEMModel

model = EllipticRTFEMModel()
model.run()
model.show_mesh()
