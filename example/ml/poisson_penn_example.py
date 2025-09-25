from fealpy.backend import bm
bm.set_backend('pytorch')
from fealpy.ml import PoissonPENNModel

options = PoissonPENNModel.get_options()  
model = PoissonPENNModel(options=options)
model.run()   
model.show()   

