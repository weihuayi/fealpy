from fealpy.backend import backend_manager as bm
from fealpy.opt.optimizer_base import opt_alg_options
import scipy.sparse as sp
import networkx as nx
from scipy.ndimage import label
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# 二值栅格地图数据
def grid_0():
    Map = bm.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    return Map


MAP_data = [
    {
        'map': grid_0,
        'start': [0, 0],
        'goal': [19,19],
        'dim': 20, 
    },
]


class GridProblem:
    def __init__(self, MAP, start, goal):
        self.MAP = MAP
        self.start = start
        self.goal = goal
        self.data = {}

    def builddata(self):
        self.data["R"] = 1 
        self.data['map'] = self.MAP
        L, _ = label(self.MAP) 

        indices = bm.where(bm.array(L) > 0)
        landmark = bm.concatenate([bm.array(L[i]) for i in indices]) 
        self.data['landmark'] = bm.array(landmark)

        node = [[j, i] for i in range(self.MAP.shape[0]) for j in range(self.MAP.shape[1]) if self.MAP[i, j] == 0]
        self. data['node'] = bm.array(node)

        self.data['D'] = squareform(pdist(self.data['node']))

        p1, p2 = bm.where(bm.array(self.data['D']) <= bm.array(self.data['R']))
        
        D = self.data['D'][(p1, p2)].reshape(-1, 1)
        self.data['net'] = sp.csr_matrix((D.flatten(), (p1, p2)), shape = self.data['D'].shape)

        self.data['noS'] = bm.where((self.data['node'][:, 0] == self.start[0]) & (self.data['node'][:, 1] == self.start[1]))[0][0]
        self.data['noE'] = bm.where((self.data['node'][:, 0] == self.goal[0]) & (self.data['node'][:, 1] == self.goal[1]))[0][0]
        self.data['numLM0'] = 1
        return self.data
    
    def fitness(self, X):
        sorted_numbers = bm.argsort(X)
        G = nx.DiGraph(self.data["net"])
        sorted_numbers_flat = sorted_numbers[:, :self.data['numLM0']]
        noS = bm.full((X.shape[0],), self.data['noS']).reshape(-1, 1)
        noE = bm.full((X.shape[0],), self.data['noE']).reshape(-1, 1)
        path0 = bm.concatenate((noS, sorted_numbers_flat, noE), axis=1)
        distances = bm.zeros((X.shape[0], path0.shape[1] - 1))
        for j in range(0, X.shape[0]):   
            distance = [nx.shortest_path_length(G, source=int(path0[j][i]), target=int(path0[j][i + 1]), weight=None) for i in range(path0.shape[1] - 1)]
            distances[j, :] = bm.tensor(distance)
        fit = bm.sum(distances, axis=1)
        return fit
    
    def calresult(self, X):
        result = {}
        sorted_numbers = bm.argsort(X)
        G = nx.DiGraph(self.data["net"])
        distances = []
        paths = []
        sorted_numbers_flat = sorted_numbers[0:self.data['numLM0']]
        sorted_numbers_flat = [element for element in sorted_numbers_flat]
        path0 = [self.data['noS']] + sorted_numbers_flat + [self.data['noE']]
        for i in range(0, len(path0) - 1):
            source = path0[i]
            target = path0[i + 1]
            source = int(source)
            target = int(target)
            path = nx.shortest_path(G, source = source, target = target)
            distance = nx.shortest_path_length(G, source = source, target = target, weight = None)  
            distances.append(distance)
            paths.append(path)
        fit = sum(distances)
        combined_list = []
        for sublist in paths:
            combined_list.extend(sublist)
        result["fit"] = fit
        result["path"] = combined_list
        return result
    
    def printMAP(self, result):
        b = self.MAP.shape
        self.MAP = 1 - self.MAP

        paths = {key: bm.array(value['path']) for key, value in result.items()}
        result_paths = bm.array([paths[algo] for algo in paths])
        alg_name = list(result.keys())

        for j in range(result_paths.shape[0]):
            plt.scatter(self.start[0], self.start[1], color = 'blue', s = 100)
            plt.scatter(self.goal[0], self.goal[1], color = 'green', s = 100)

            plt.imshow(self.MAP[::1], cmap = 'gray')

            xx = bm.linspace(0,b[1],b[1]) - 0.5
            yy = bm.zeros(b[1]) - 0.5
            for i in range(0, b[0]):
                yy = yy + 1
                plt.plot(xx, yy,'-',color = 'black')
            x = bm.zeros(b[0])-0.5
            y = bm.linspace(0,b[0],b[0])-0.5
            
            for i in range(0, b[1]):
                x = x + 1
                plt.plot(x,y,'-',color = 'black')

            xpath = self.data["node"][result_paths[j], 0]
            ypath = self.data["node"][result_paths[j], 1]
            plt.plot(xpath, ypath, '-', color = 'red')
            plt.plot([xpath[-1], self.goal[0]], [ypath[-1], self.goal[1]], '-', color = 'red')

            plt.xticks([])
            plt.yticks([])
            plt.legend(loc='upper right')
            plt.title(f'Optimal Paths from {alg_name[j]}')
            plt.show()




if __name__ == "__main":
    pass