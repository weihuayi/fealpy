from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.fem import OPCRTFEMModel
model = OPCRTFEMModel()
p,u = model.run()
error_p,error_u = model.error()
print('error_p:', error_p)
print('error_u:', error_u)
