from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.opt.optimizer_base import opt_alg_options
import matplotlib.pyplot as plt
def ciyt0():

    citys = bm.array([
    [101.7400, 6.5600],
    [112.5300, 37.8700],
    [121.4700, 31.2300],
    [119.3000, 26.0800],
    [106.7100, 26.5700],
    [103.7300, 36.0300],
    [111.6500, 40.8200],
    [120.1900, 30.2600],
    [121.3000, 25.0300],
    [106.5500, 29.5600],
    [106.2700, 38.4700],
    [116.4000, 39.9000],
    [118.7800, 32.0400],
    [114.1700, 22.3200],
    [104.0600, 30.6700],
    [108.9500, 34.2700],
    [117.2000, 39.0800],
    [117.2700, 31.8600],
    [113.5400, 22.1900],
    [102.7300, 25.0400],
    [113.6500, 34.7600],
    [123.3800, 41.8000],
    [114.3100, 30.5200],
    [113.2300, 23.1600],
    [91.1100, 29.9700],
    [117.0000, 36.6500],
    [125.3500, 43.8800],
    [113.0000, 28.2100],
    [110.3500, 20.0200],
    [87.6800, 43.7700],
    [114.4800, 38.0300],
    [126.6300, 45.7500],
    [115.8900, 28.6800],
    [108.3300, 22.8400]
    ])

    return citys


TSP_data = [
    {
        "citys": ciyt0,
        "dim": 34,
    },
]

def soler_tsp_with_algorithm(algorithm, fobj, lb, ub, NP, dim):
    x0 = lb + bm.random.rand(NP, dim) * (ub - lb)
    option = opt_alg_options(x0, fobj, (lb, ub), NP, MaxIters = 10000)
    optimizer = algorithm(option)
    gbest, gbest_f = optimizer.run()
    return gbest, gbest_f

def gbest2route(gbest, gbest_f, citys, alg_name):
    route = bm.argsort(gbest)
    route = bm.concatenate((route, route[0: 1]))
    route_citys = citys[route]
    print(f"The best route obtained by {alg_name} is:", route)
    print(f"The shortest distance found by {alg_name} is:", gbest_f)
    return route, route_citys

def printroute(route_citys, citys, alg_name):
    for i in range(route_citys.shape[0]):
        plt.figure()
        plt.title(f"{alg_name[i]} optimal path")
        plt.scatter(citys[:, 0], citys[:, 1])
        plt.plot(route_citys[i, :, 0], route_citys[i, :, 1])
    plt.show()

class TravellingSalesmanProblem:
    def __init__(self, citys) -> None:
        self.citys = citys
        self.D = bm.zeros((citys.shape[0], citys.shape[0]))
    
    def fitness(self, x):
        index = bm.argsort(x, axis=-1)
        distance = self.D[index[:, -1], index[:, 0]]
        for i in range(x.shape[1] - 1):
            dis = self.D[index[:, i], index[:, i + 1]]
            distance = distance + dis
        return distance

    def calD(self):
        n = self.citys.shape[0] 
        diff = self.citys[:, None, :] - self.citys[None, :, :]
        self.D = bm.sqrt(bm.sum(diff ** 2, axis = -1))
        self.D[bm.arange(n), bm.arange(n)] = 2.2204e-16
        





if __name__ == "__main":
    pass