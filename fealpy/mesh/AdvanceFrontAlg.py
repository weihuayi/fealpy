
import numpy as np


class PolygonDomain():
    def __init__(self, point, facet):
        self.point = point
        self.facet = facet

    def number_of_vertices(self):
        return self.point.shape[0]

    def number_of_facets(self):
        return self.facet.shape[0]

    def discrete_boundary(self, h):
        point = self.point
        facet = self.facet
        NF = self.number_of_facets()
        pp = []
        NN = 0
        for i range(NF):
            p0 = point[facet[i, 0], :]
            p1 = point[facet[i, 1], :]
            fv = p1 - p0 
            fl = np.sqrt(np.sum(fv**2))   
            N = int(fl/h) 
            weight = np.zeros((N-1,2))
            weight[:, 0] = range(1, N)
            weight[:, 1] = range(N-1, 0, -1)
            newPoint = (weight[:, [0]]*p0 + weight[:, [1]]*p1)/N
            pp.append(newPoint)
            NN += N-1

        N = self.number_of_vertices()

        
        


class AdvancingFrontAlg():
    def __init__(self, domain):
        self.domain = domain

    def update_front(self):
        pass

    def smooth_front(self):
        pass

    def insert_points(self):
        pass

    def remove_bad_points(self):
        pass
        

