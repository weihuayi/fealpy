from ..nodetype import CNodeType, PortConf, DataType

class LinearElasticityEigen3d(CNodeType):
    TITLE: str = "线弹性特征值模型"
    PATH: str = "模型.线弹性"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE,title="区域"),
        PortConf("nx", DataType.INT, title="X 分段数"),
        PortConf("ny", DataType.INT, title="Y 分段数"),
        PortConf("nz", DataType.INT, title="Z 分段数"),
        PortConf("material", DataType.NONE,title="材料属性"),
        PortConf("displacement_bc", DataType.FUNCTION),
        PortConf("is_displacement_boundary", DataType.FUNCTION),
    ]
    
    @staticmethod
    def run():
        from fealpy.csm.model import CSMModelManager
        model = CSMModelManager("linear_elasticity").get_example(1)
        domain = model.box
        nx = 40 
        ny = 8
        nz = 8
        return (domain,nx,ny,nz, model.material)+ tuple(
            getattr(model, name)
            for name in [ "displacement_bc",  "is_displacement_boundary"]
        )
