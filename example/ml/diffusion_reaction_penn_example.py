from fealpy.backend import backend_manager as bm 
bm.set_backend('pytorch')
from fealpy.ml import DiffusionReactionPENNModel
options = DiffusionReactionPENNModel.get_options()

model = DiffusionReactionPENNModel(options)
model.run()
model.show()
