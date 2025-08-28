from ..nodetype import CNodeType, PortConf, DataType


class TimobeamAxle3d(CNodeType):
    TITLE: str = "TimobeamAxle 3D"
    PATH: str = "model.timobeam_axle"
    INPUT_SLOTS = []
    
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE),
        PortConf("E", DataType.FLOAT),
        PortConf("nu", DataType.FLOAT),
        PortConf("displacement", DataType.FUNCTION),
        PortConf("body_force", DataType.TENSOR),
        PortConf("displacement_bc", DataType.FUNCTION),
        PortConf("hypo", DataType.STRING)
    ]
    
    @staticmethod
    def run():
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        model = TimobeamAxleData3D()
        return (model.domain(), model.lam(), model.mu()) + tuple(
            getattr(model, name)
            for name in ["displacement", "body_force", "displacement_bc", "hypo"]
        )