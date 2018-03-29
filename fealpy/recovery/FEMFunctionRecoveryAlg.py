import numpy as np

class FEMFunctionRecoveryAlg():
    def __init__(self):
        pass

    def simple_average(self, uh):
        V = uh.V
        mesh = V.mesh
        GD = mesh.geo_dimension()

        node2cell = mesh.ds.node_to_cell()
        valence = node2cell.sum(axis=1)

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)
        rguh = V.function(dim=GD)
        rguh[:] = np.asarray(node2cell@guh)/valence.reshape(-1, 1)
        return rguh

    def area_average(self, uh):
        V = uh.V
        mesh = V.mesh
        GD = mesh.geo_dimension()

        node2cell = mesh.ds.node_to_cell()
        area = mesh.area()
        asum = node2cell@area

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        rguh = V.function(dim=GD)
        rguh[:] = np.asarray(node2cell@(guh*area.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh

    def harmonic_average(self, uh):
        V = uh.V
        mesh = V.mesh
        GD = mesh.geo_dimension()

        node2cell = mesh.ds.node_to_cell()
        inva = 1/mesh.area()
        asum = node2cell@inva

        bc = np.array([1/3]*3, dtype=np.float)
        guh = uh.grad_value(bc)

        rguh = V.function(dim=GD)
        rguh[:] = np.asarray(node2cell@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)
        return rguh
