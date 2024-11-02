import numpy as np
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from fealpy.form.Form import LaplaceSymetricForm, MassForm, SourceForm
from fealpy.boundarycondition import DirichletBC
from fealpy.functionspace.function import FiniteElementFunction 
from fealpy.erroranalysis import L2_error



class TimeIntegratorAlgorithmData(TimeIntegratorAlgorithm):
    def __init__(self, interval, mesh, V, N, model):
   	 self.current_time = interval[0]
         self.stop_time = interval[1]
         self.mesh = mesh
      	 self.V = V
       	 self.N = N
         self.model = model
