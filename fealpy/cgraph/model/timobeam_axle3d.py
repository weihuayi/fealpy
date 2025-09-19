from ..nodetype import CNodeType, PortConf, DataType


class TimobeamAxle3d(CNodeType):
    TITLE: str = "TimobeamAxle 3D"
    PATH: str = "model.timobeam_axle"
    INPUT_SLOTS = []
    
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE),
        PortConf("shear_factors", DataType.FLOAT),
        PortConf("beam_cross_section", DataType.FLOAT),
        PortConf("beam_inertia", DataType.FLOAT),
        PortConf("external_load", DataType.TENSOR),
        PortConf("dirichlet", DataType.FUNCTION)
    ]
    
    @staticmethod
    def run():
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        model = TimobeamAxleData3D()
        return (model.domain()) + tuple(
            getattr(model, name)
            for name in ["shear_factors", "beam_cross_section", "beam_inertia", "external_load", "dirichlet"]
        )