from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.fem import EllipticRTFEMModel
model = EllipticRTFEMModel()
p,u = model.run()
print(p)
print(u)
error_p,error_u = model.error()
print('error_p:', error_p)
print('error_u:', error_u)