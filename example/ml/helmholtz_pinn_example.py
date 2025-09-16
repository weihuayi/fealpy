from fealpy.backend import bm
bm.set_backend('pytorch')  # Set the backend to PyTorch
from fealpy.ml import HelmholtzPINNModel

options = HelmholtzPINNModel.get_options()  # Get the default options of the network
model = HelmholtzPINNModel(options=options)
model.run()   # Train the network
model.show()   # Show the results of the network training

