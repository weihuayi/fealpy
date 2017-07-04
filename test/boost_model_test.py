from mpi4py import MPI
import pyparmetis 
import numpy as np

comm = MPI.COMM_WORLD
a = np.array([10, 11], dtype=np.float) 
b = 10
pyparmetis.sayhello(a, b, comm)
