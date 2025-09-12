from fealpy.backend import bm
bm.set_backend('pytorch')   # Set the backend to PyTorch
from fealpy.ml import PoissonPINNModel

options = PoissonPINNModel.get_options()    # Get the default options of the network
model = PoissonPINNModel(options=options)
model.run()   # Train the network
model.show()   # Show the results of the network training


