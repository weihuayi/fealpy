


class KernelFunctionSpace():

    def __init__(self, mesh, kfun):
        self.mesh = mesh 
        self.kfun = kfun


    def value(self, u, node=None):
        pass

    def grad_value(self, u, node=Node):
        pass

