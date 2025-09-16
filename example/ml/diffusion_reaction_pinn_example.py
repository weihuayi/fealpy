from fealpy.backend import backend_manager as bm 
bm.set_backend('pytorch')
from fealpy.ml import DiffusionReactionPINNModel
model = DiffusionReactionPINNModel()
model.run()
model.show()