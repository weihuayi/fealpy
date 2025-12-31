from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["LinearElasticMaterial"]

 
class LinearElasticMaterial(CNodeType):
    r"""Linear elastic material property definition node.
    
    Universal linear elastic material for all structural elements (bar, beam, solid, etc.).
    Supports two input modes:
    1. Predefined materials: Select from material database
    2. Custom input: Manually input material properties
    
    Inputs:
        Inputs:
        property (MENU): Material type selection.
        E (FLOAT): Elastic modulus [Pa] (only effective when property='custom-input').
        nu (FLOAT): Poisson's ratio (only effective when property='custom-input').
        rho (FLOAT): Density [kg/m³] (only effective when property='custom-input').

    Outputs:
        mp (DICT): Material properties, containing E, nu, rho, mu, and lambda_. 
    """
    TITLE: str = "线弹性材料"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, 
                desc="材料类型选择", 
                title="材料类型", 
                default="custom-input", 
                items=["structural-steel", "stainless-steel", "aluminum", "aluminum-alloy", 
                       "titanium", "titanium-alloy", "brass", "copper", "concrete", "wood", 
                       "carbon-fiber", "fiberglass", "custom-input"]),
        PortConf("E", DataType.FLOAT, 0, 
                desc="弹性模量(仅当材料类型为'custom-input'时有效)", 
                title="弹性模量", 
                default=2.1e5),
        PortConf("nu", DataType.FLOAT, 0, 
                desc="泊松比 (仅当材料类型为'custom-input'时有效)", 
                title="泊松比", 
                default=0.3),
        PortConf("rho", DataType.FLOAT, 0, 
                desc="密度 [kg/m³] (仅当材料类型为'custom-input'时有效)", 
                title="密度", 
                default=7850.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mp", DataType.DICT, title="材料属性")
    ]
    
    @staticmethod
    def run(**options):
        property_type = options.get("property")
        
        # 材质数据库
        material_database = {
            # 金属材料
            "structural-steel": {"E": 2.0e11, "nu": 0.3, "rho": 7850.0},
            "stainless-steel": {"E": 1.93e11, "nu": 0.31, "rho": 8000.0},
            "aluminum": {"E": 7.0e10, "nu": 0.33, "rho": 2700.0},
            "aluminum-alloy": {"E": 7.2e10, "nu": 0.33, "rho": 2800.0},
            "titanium": {"E": 1.1e11, "nu": 0.34, "rho": 4500.0},
            "titanium-alloy": {"E": 1.14e11, "nu": 0.34, "rho": 4430.0},
            "brass": {"E": 1.0e11, "nu": 0.34, "rho": 8500.0},
            "copper": {"E": 1.2e11, "nu": 0.34, "rho": 8960.0},
            
            # 非金属材料
            "concrete": {"E": 3.0e10, "nu": 0.2, "rho": 2400.0},
            "wood": {"E": 1.0e10, "nu": 0.35, "rho": 600.0},
            
            # 复合材料
            "carbon-fiber": {"E": 1.5e11, "nu": 0.3, "rho": 1600.0},
            "fiberglass": {"E": 3.5e10, "nu": 0.25, "rho": 2000.0}
        }
        
        # 如果选择了预定义材质
        if property_type in material_database:
            material = material_database[property_type]
            E = material["E"]
            nu = material["nu"]
            rho = material["rho"]
        else:
            # 使用自定义输入
            E = options.get("E")
            nu = options.get("nu")
            rho = options.get("rho")
        
        # 剪切模量: mu = E / (2 * (1 + nu))
        mu = E / (2.0 * (1.0 + nu))
        
        # 拉梅第一参数: lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        mp = {
            'E': E,
            'nu': nu,
            'rho': rho,
            'mu': mu,
            'lambda_': lambda_
        }
        return mp