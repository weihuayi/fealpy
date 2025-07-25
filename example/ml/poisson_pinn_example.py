from fealpy.backend import bm
bm.set_backend('pytorch')   # Set the backend to PyTorch
from fealpy.ml import PoissonPINNModel

options = PoissonPINNModel.get_options()    # Get the default options of the network
options['pde'] = 2   # Set the PDE to Poisson equation
options["epochs"] = 1000   # Set the number of epochs to 1000
model = PoissonPINNModel(options=options)
model.run()   # Train the network
model.show()   # Show the results of the network training


