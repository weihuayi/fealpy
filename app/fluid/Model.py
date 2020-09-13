import numpy as np

from fealpy.mesh import LagrangeQuadrangleMesh
from fealpy.mesh import MeshFactory


class TriplePointShockInteractionModel:

    def __init__(self):
        self.domain = [0, 7, 0, 3]

    def space_mesh(self, NS=0):
        mf = MeshFactory()
        mesh = mf.boxmesh2d(self.domain, nx=70, ny =30, p=p, meshtype='quad') 
        mesh.uniform_refine(NS)
        return mesh

    def time_mesh(self, NT=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, 1, NT)
        return timeline

    def subdomain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.zeros(p.shape[:-1], dtype=p.dtype)
        flag[x < 1] = 1  
        flag[(x > 1) & (y < 1.5)] = 2
        flag[(x > 1) & (y > 1.5)] = 3 
        return flag

    def init_rho(self, p):
        x = p[..., 0]
        y = p[..., 1]

        rho = np.zeros(p.shape[:-1], dtype=p.dtype)
        rho[x < 1] = 1.0
        rho[(x > 1) & (y < 1.5)] = 1
        rho[(x > 1) & (y > 1.5)] = 0.125
        return val

    def init_velocity(self, p):
        val = np.array([0.0, 0.0], dtype=p.dtype)
        shape = (len(p.shape) - 1)*(1, ) + (-1, )
        return val.reshape(shape)
        

    def init_pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.0
        val[(x > 1) & (y < 1.5)] = 0.1
        val[(x > 1) & (y > 1.5)] = 0.1 
        return val

    def adiabatic_index(self, p):
        """
        Notes
        -----
        绝热指数
        """
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.5
        val[(x > 1) & (y < 1.5)] = 1.4 
        val[(x > 1) & (y > 1.5)] = 1.5 
        return val

    def stress(self, p, e):
        gamma = self.adiabatic_index(p)
        rho = self.init_rho(p)
        sigma = -(gamma - 1)*rho*e
        return sigma


class ModelSover():

    def __init__(self, model, p, NS=0, NT=100):
        self.model = model
        self.mesh = model.space_mesh(NS=NS, p=p)
        self.timeline = model.time_mesh(NT=NT)

        self.cspace = ParametricLagrangeFiniteElementSpace(mesh, p=p, spacetype='C')
        self.dspace = ParametricLagrangeFiniteElementSpace(mesh, p=p-1, spacetype='D')

        bc = mesh.entity_barycenter('cell')
        self.rho = model.init_rho(bc)

        self.MV = cspace.mass_matrix(c=self.rho)
        self.ME = dspace.mass_matrix(c=self.rho)

        GD = self.mesh.geo_dimension()
        self.v = cspace.function(dim=GD)
        self.e = dspace.function()



if __name__ == '__main__':
    from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
    model = TriplePointShockInteractionModel()

    flag = model.subdomain(bc)
    mesh.celldata['flag'] = flag
    mesh.to_vtk(fname='test.vtu')
