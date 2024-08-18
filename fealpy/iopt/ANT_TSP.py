from fealpy.experimental.backend import backend_manager as bm

class TSPProblem:
    def __init__(self, city):
        self.city = city

    def calD(self):
       n = self.city.shape[0]
       D = bm.zeros((n, n)) 

       diff = citys[:, bm.newaxis, :] - citys[bm.newaxis, :, :]
       D = bm.sqrt(bm.sum(diff ** 2, axis = -1))
       D[bm.arange(n), bm.arange(n)] = 1e-4

       return D