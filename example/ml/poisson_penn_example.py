from fealpy.backend import bm
bm.set_backend('pytorch') # set the backend to pytorch
from fealpy.ml import PoissonPENNModel

options = PoissonPENNModel.get_options()   # Get the default options of the network
model = PoissonPENNModel(options=options)
model.run()   # Train the network
model.show()   # Show the results of the network training

