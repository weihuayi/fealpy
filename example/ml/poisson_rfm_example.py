from fealpy.backend import bm
bm.set_backend('pytorch')   # Set the backend to PyTorch
from fealpy.ml import PoissonRFMModel

options = PoissonRFMModel.get_options()    # Get the default options of the model
model = PoissonRFMModel(options=options)
model.run()   # Train the model
model.show(s=50)   # Show the results
