
import numpy as np
import sys
from initData import Cmp_mesh_data
from initData import Cmp_pmt_data

################
Nspecies=2
Nblend=1
Nblock=2
Ndeg=100
fA=0.8
chiAB=0.3

################
dim = 2
dtMax = 5e-3
TOL = 1.0e-6
Maxiter = 5000
Initer = 200
################

node = 200


fieldtype = 'fieldmu'
fields = np.random.rand(Nspecies, node) 


test = Cmp_pmt_data(fields, fieldtype,
                    Nspecies, Nblend, Nblock, Ndeg, fA, chiAB,
                    dim, dtMax, TOL, Maxiter, Initer)

# test.initialize(fields, fieldtype)
# print(test.mu.shape)



# c,nn = test.cmp_interval()
# print(c)
# print(nn)
