import numpy as np

class AllenChanData1:
    def __init__(self,epsilon):
        self.epsilon = epsilon

    def initdata(self,p):
        p = x
        if 0<=x<0.28:
            u0 = np.tanh((0.2-x)/(2*np.sqrt(epsilon**2)))
        elif 0.28<=x<0.4865:
            u0 = np.tanh((x-0.36)/(2*np.sqrt(epsilon**2)))
        elif 0.4865<=x<0.7065:
            u0 = np.tanh((0.613-x)/(2*np.sqrt(epsilon**2)))
        else:
            u0 = np.tanh((x-0.8)/(2*np.sqrt(epsilon**2)))
        return u0

    def solution(self,p):
        """ The exact solution 
        """
        pass

    def gradient(self,p):
        pass

    def laplace(self,p):
        pass

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self,p):
        pass

class AllenChanData2:
    def __init__(self,epsilon):
        self.epsilon = epsilon

    def initdata(self,p):
        p[:,0] = x
        p[:,1] = y
        u0 = 0.05*np.sin(x)*np.sin(y)
        return u0

    def solution(self, p):
        pass

    def gradient(self, p):
        pass

    def laplace(self, p):
        pass

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self,p):
        pass

class AllenChanData3:
    def __init__(self,h,m):
        self.h = h
        self.m = m
    
    def initdata(self,p):
        p[:,0] = x
        p[:,1] = y
        epsilon = (h*m*np.tanh(0.9))/(2*np.sqrt(2))
        u0 = np.tanh((0.25-np.sqrt((x-0.5)**2+(y-0.5)**2))/np.sqrt(2)*epsilon)
        return u0

    def solution(self,p):
        pass

    def gradient(self,p):
        pass

    def laplace(self,p):
        pass

    def dirichlet(self,p):
        """
        Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self,p):
        pass
    
class AllenChanData4:
    def __init__(self,h,m):
        self.h = h
        self.m = m
    
    def initdata(self,p):
        p[:,0] = x
        p[:,1] = y
        epsilon = (h*m*np.tanh(0.9))/(2*np.sqrt(2))
        
        if x>0.5:
            theta = 1/np.tan((y-0.5)/(x-0.5))
        else:
            theta = np.pi + 1/np.tan((y-0.5)/(x-0.5))

        u0 = np.tanh((0.25+0.1*np.cos(7*theta)-np.sqrt((x-0.5)**2+(y-0.5)**2))/np.sqrt(2)*epsilon)
        return u0

    def solution(self,p):
        pass

    def gradient(self,p):
        pass

    def laplace(self,p):
        pass

    def dirichlet(self,p):
        """
        Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self,p):
        pass


class AllenChanData5:
    def __init__(self,h,m):
        self.h = h
        self.m = m

    def initdata(self,p):
        p[:,0] = x
        p[:,1] = y
        p[:,2] = z
        epsilon = (h*m*np.tanh(0.9))/(2*np.sqrt(2))

        u0 = np.tanh((0.4-np.sqrt((x-0.5)**2+(y-0.5)**2+(z-0.5)**2))/np.sqrt(2)*epsilon)
        return u0

    def solution(self,p):
        pass

    def gradient(self,p):
        pass

    def laplace(self,p):
        pass

    def dirichlet(self,p):
        """
        Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self,p):
        pass


class AllenChanData6:
    def __init__(self,h,m):
        self.h = h
        self.m = m

    def initdata(self,p):
        p[:,0] = x
        p[:,1] = y
        p[:,2] = z
        epsilon = (h*m*np.tanh(0.9))/(2*np.sqrt(2))

        if x>0.5:
            theta = 1/np.tan((y-0.5)/(x-0.5))
        else:
            theta = np.pi + 1/np.tan((y-0.5)/(x-0.5))

        phi = 1/np.cos((z-0.5)/np.sqrt((x-0.5)**2+(y-0.5)**2+(z-0.5)**2))
        Y = 
        u0 = np.tanh((0.25+0.1*Y-np.sqrt((x-0.5)**2+(y-0.5)**2)+(z-0.5)**2)/np.sqrt(2)*epsilon)
        return u0

    def solution(self,p):
        pass

    def gradient(self,p):
        pass

    def laplace(self,p):
        pass

    def dirichlet(self,p):
        """
        Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self,p):
        pass



























