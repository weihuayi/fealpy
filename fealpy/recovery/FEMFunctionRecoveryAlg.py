import numpy as np
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace 

class FEMFunctionRecoveryAlg():
    def __init__(self):
        pass

    def simple_average(self, uh):
        V = uh.V
        mesh = V.mesh

        node2cell = mesh.ds.node_to_cell()
        valence = node2cell.sum(axis=1)

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        VV = VectorLagrangeFiniteElementSpace(mesh, V.p)
        rguh = VV.function()
        rguh[:] = np.asarray(node2cell@guh)/valence.reshape(-1, 1)
        return rguh

    def area_average(self, uh):
        V = uh.V
        mesh = V.mesh

        node2cell = mesh.ds.node_to_cell()
        area = mesh.area()
        asum = node2cell@area

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        VV = VectorLagrangeFiniteElementSpace(mesh, p=1)
        rguh = VV.function()
        rguh[:] = np.asarray(p2c@(guh*area.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh

    def harmonic_average(self, uh):
        V = uh.V
        mesh = V.mesh

        node2cell = mesh.ds.node_to_cell()
        inva = 1/mesh.area()
        asum = node2cell@inva

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        VV = VectorLagrangeFiniteElementSpace(mesh, p=1)
        rguh = VV.function()
        rguh[:] = np.asarray(p2c@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh
