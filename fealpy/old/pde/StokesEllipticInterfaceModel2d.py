import numpy as np

class CircleSmoothData():
    def __init__(self, betap, betam):
        self.betaplus= betap
        self.betaminus = betam

    def velocity_minus(self, p):
        x = p[:,0]
        y = p[:,1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = np.sin(x-y)
        val[:,1] = val[:,0]
        return val

    def velocity_plus(self, p):
        x = p[:,0]
        y = p[:,1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = np.exp(x+y)
        val[:,1] = np.exp(x-y)
        return val

    def pressure_plus(self, p):
        x = p[:,0]
        y = p[:,1]
        val = np.exp(x)*np.cos(y)
        return val

    def source_plus(self, p):
        x = p[:,0]
        y = p[:,1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = 2*np.sin(x-y)+np.exp(x)*np.cos(y)
        val[:,1] = 2*np.sin(x-y)-np.exp(x)*np.sin(y)
        return val

    def source_minus(self, p):
        x = p[:,0]
        y = p[:,1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = np.exp(x+y) 
        val[:,1] = np.exp(-x-y) 

    def dirichlet(self, p):
        return self.velocity_plus(p)

    def value_jump(self, p):
        x = p[:,0]
        y = p[:,1]
        val = np.zeros((len(x),2), dtype=np.float)
        return val

    def flux_jump(self, p):
        x = p[:,0]
        y = p[:,1]
        val = np.zeros((len(x),2), dtype=np.float)
        return val



        
