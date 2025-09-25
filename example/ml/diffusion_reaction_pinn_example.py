from fealpy.backend import backend_manager as bm 
bm.set_backend('pytorch')
from fealpy.ml import DiffusionReactionPINNModel
options = DiffusionReactionPINNModel.get_options()
model = DiffusionReactionPINNModel(options)
model.run()
model.show()