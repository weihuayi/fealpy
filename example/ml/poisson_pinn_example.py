from fealpy.backend import bm
bm.set_backend('pytorch')  
from fealpy.ml import PoissonPINNModel

options = PoissonPINNModel.get_options()   
model = PoissonPINNModel(options=options)
model.run()   
model.show()  

