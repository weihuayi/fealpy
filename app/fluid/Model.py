import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from TriplePointShockInteractionModel import TriplePointShockInteractionModel
from LagrangianHydrodynamicsSimulator import LagrangianHydrodynamicsSimulator

p = 2
model = TriplePointShockInteractionModel()
simulator = LagrangianHydrodynamicsSimulator(model, p, NS=0, NT=10000)
simulator.solve(step=10)


