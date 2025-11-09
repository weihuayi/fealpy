from ..nodetype import CNodeType, PortConf, DataType

class LinearElasticityEigen3d(CNodeType):
    TITLE: str = "线弹性特征值模型"
    PATH: str = "模型.线弹性"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE),
        PortConf("material", DataType.NONE),
        PortConf("displacement", DataType.FUNCTION),
        PortConf("body_force", DataType.FUNCTION),
        PortConf("displacement_bc", DataType.FUNCTION),
        PortConf("is_displacement_boundary", DataType.FUNCTION),
    ]
    
    @staticmethod
    def run():
        from fealpy.csm.model import CSMModelManager
        domain = [0,10,0,2,0,2]
        model = CSMModelManager("linear_elasticity").get_example(1)
        return (domain, model.material)+ tuple(
            getattr(model, name)
            for name in ["displacement", "displacement_bc"]
        )


