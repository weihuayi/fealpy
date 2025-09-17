from .nodetype import CNodeType, PortConf, DataType

__all__ = ["VPDecoupling"]

class VPDecoupling(CNodeType):
    TITLE: str = "velocity-pressure decoupling"
    PATH: str = "postprocess.decoupling"
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR),
        PortConf("uspace", DataType.SPACE)
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR),
        PortConf("ph", DataType.TENSOR)
    ]

    @staticmethod
    def run(out, uspace):
        ugdof = uspace.number_of_global_dofs()
        uh = out[:ugdof]
        ph = out[ugdof:]

        # uspace.mesh.nodedata['ph'] = ph
        # uspace.mesh.nodedata['uh'] = uh.reshape(2,-1).T
        # uspace.mesh.to_vtk('dld_chip910.vtu')
        return uh ,ph
