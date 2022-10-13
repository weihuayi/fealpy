import numpy as np
from .sizing_function import huniform 
from .implicit_surface import TorusSurface

class TorusDomain():
    def __init__(self,fh=huniform):
        self.surface = TorusSurface()
        self.box = self.surface.box
        #self.box = [-6, 6, -6, 6, -6, 6]
        self.fh = fh

        self.facets = {0:None,1:None}
    def __call__(self,p):
        return self.surface(p)

    def signed_dist_function(self,p):
        return self(p)
    
    def sizing_function(self,p):
        return self.fh(p)
    
    def facet(self,dim):
        return self.facets[dim]
    
    def projection(self,p):
        p,d = self.surface.project(p)
        return p

