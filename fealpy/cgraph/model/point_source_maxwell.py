from typing import Union, List, Dict, Any, Tuple, Optional
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["PointSourceMaxwell"]

class PointSourceMaxwell(CNodeType):
    r"""Point source Maxwell equations problem model.

    Inputs:
        domain (list): Computational domain [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].
        eps (float): Relative permittivity.
        mu (float): Relative permeability.
        source1_position (list): Position of the first point source.
        source1_component (str): Field component excited by the first source.
        source1_waveform (str): Waveform type of the first source.
        source1_waveform_params (float): Waveform parameters for the first source.
        source1_amplitude (float): Amplitude of the first source.
        source1_spread (int): Spatial spread of the first source.   
        source1_injection (str): Injection type of the first source.
        source2_position (list): Position of the second point source (optional).
        source2_component (str): Field component excited by the second source.
        source2_waveform (str): Waveform type of the second source.
        source2_waveform_params (float): Waveform parameters for the second source.
        source2_amplitude (float): Amplitude of the second source.
        source2_spread (int): Spatial spread of the second source.   
        source2_injection (str): Injection type of the second source.
        object1_box (list): Bounding box of the first object (optional).
        object1_eps (float): Relative permittivity of the first object.
        object1_mu (float): Relative permeability of the first object.
        object2_box (list): Bounding box of the second object (optional).
        object2_eps (float): Relative permittivity of the second object.
        object2_mu (float): Relative permeability of the second object.
        object3_box (list): Bounding box of the third object (optional).
        object3_eps (float): Relative permittivity of the third object.
        object3_mu (float): Relative permeability of the third object.
    """
    TITLE: str = "点源Maxwell方程问题模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点定义点源Maxwell方程模型，支持2D和3D计算域，可以设置背景材料参数、两个个点源激励
                和最多三个物体区域，为FDTD仿真提供物理问题定义。"""
    
    INPUT_SLOTS = [
        PortConf("domain", DataType.LIST, 0, title="计算域", default=[0, 5e-6, 0, 5e-6],),
        PortConf("eps", DataType.FLOAT, 0, title="相对介电常数", default=1.0),
        PortConf("mu", DataType.FLOAT, 0, title="相对磁导率", default=1.0),
        
        # Single source configuration
        PortConf("source1_position", DataType.LIST, 0, title="第一源位置", default=[3e-6,2e-6],
                desc="源位置坐标，格式为(x,y)或(x,y,z)"),
        PortConf("source1_component", DataType.MENU, 0, title="第一源场分量", default="Ez",
                items=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                desc="源激励的电磁场分量"),
        PortConf("source1_waveform", DataType.MENU, 0, title="第一源波形类型", default="gaussian",
                items=["gaussian", "sinusoid", "ricker", "gaussian_enveloped_sine"],
                desc="源的时间波形类型"),
        PortConf("source1_waveform_params", DataType.TEXT, 0, title="第一源波形参数", default=6e14,
                desc="源波形的参数字典，依据波形类型设置"),
        PortConf("source1_amplitude", DataType.FLOAT, 0, title="第一源幅度", default=1.0,
                desc="源的幅度大小"),
        PortConf("source1_spread", DataType.INT, 0, title="第一源扩展半径", default=0,
                desc="源的空间扩展半径（网格点数）"),
        PortConf("source1_injection", DataType.MENU, 0, title="第一源注入方式", default="soft",
                items=["soft", "hard"],
                desc="源的注入方式：soft(叠加)或hard(覆盖)"),
        PortConf("source2_position", DataType.LIST, 0, title="第二源位置", default=None,
                desc="源位置坐标，格式为(x,y)或(x,y,z)，None表示无第二个源"),
        PortConf("source2_component", DataType.MENU, 0, title="第二源场分量", default="Ez",
                items=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                desc="源激励的电磁场分量"),
        PortConf("source2_waveform", DataType.MENU, 0, title="第二源波形类型", default="gaussian",
                items=["gaussian", "sinusoid", "ricker", "gaussian_enveloped_sine"],
                desc="源的时间波形类型"),
        PortConf("source2_waveform_params", DataType.TEXT, 0, title="第二源波形参数", default=3e14,
                desc="源波形的参数字典，依据波形类型设置"),
        PortConf("source2_amplitude", DataType.FLOAT, 0, title="第二源幅度", default=2.0,
                desc="源的幅度大小"),
        PortConf("source2_spread", DataType.INT, 0, title="第二源扩展半径", default=0,
                desc="源的空间扩展半径（网格点数）"),
        PortConf("source2_injection", DataType.MENU, 0, title="第二源注入方式", default="soft",
                items=["soft", "hard"],
                desc="源的注入方式：soft(叠加)或hard(覆盖)"),
        
        # First object configuration
        PortConf("object1_box", DataType.LIST, 0, title="物体1边界框", default=None,
                desc="第一个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]"),
        PortConf("object1_eps", DataType.FLOAT, 0, title="物体1介电常数", default=None,
                desc="第一个物体的相对介电常数，None表示使用背景值"),
        PortConf("object1_mu", DataType.FLOAT, 0, title="物体1磁导率", default=None,
                desc="第一个物体的相对磁导率，None表示使用背景值"),
        
        # Second object configuration
        PortConf("object2_box", DataType.LIST, 0, title="物体2边界框", default=None,
                desc="第二个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]"),
        PortConf("object2_eps", DataType.FLOAT, 0, title="物体2介电常数", default=None,
                desc="第二个物体的相对介电常数，None表示使用背景值"),
        PortConf("object2_mu", DataType.FLOAT, 0, title="物体2磁导率", default=None,
                desc="第二个物体的相对磁导率，None表示使用背景值"),

        # third object configuration
        PortConf("object3_box", DataType.LIST, 0, title="物体3边界框", default=None,
                desc="第三个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]"),
        PortConf("object3_eps", DataType.FLOAT, 0, title="物体3介电常数", default=None,
                desc="第三个物体的相对介电常数，None表示使用背景值"),
        PortConf("object3_mu", DataType.FLOAT, 0, title="物体3磁导率", default=None,
                desc="第三个物体的相对磁导率，None表示使用背景值")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("eps", DataType.FLOAT, title="相对介电常数"),
        PortConf("mu", DataType.FLOAT, title="相对磁导率"),
        PortConf("domain", DataType.LIST, title="计算域"),
        PortConf("source_config", DataType.TEXT, title="源配置"),
        PortConf("object_configs", DataType.TEXT, title="物体配置列表"),
    ]

    @staticmethod
    def run(domain: List[float], eps: float, mu: float,
            source1_position: List[float], source1_component: str,
            source1_waveform: str, source1_waveform_params: float,
            source1_amplitude: float, source1_spread: int, source1_injection: str,
            source2_position: List[float], source2_component: str,
            source2_waveform: str, source2_waveform_params: float,
            source2_amplitude: float, source2_spread: int, source2_injection: str,
            object1_box: Optional[List[float]], object1_eps: Optional[float], object1_mu: Optional[float],
            object2_box: Optional[List[float]], object2_eps: Optional[float], object2_mu: Optional[float],
            object3_box: Optional[List[float]], object3_eps: Optional[float], object3_mu: Optional[float]) -> tuple:
        from fealpy.cem.model.point_source_maxwell import PointSourceMaxwell as MaxwellModel
        
        # 创建Maxwell问题模型
        model = MaxwellModel(eps=eps, mu=mu, domain=domain)
        
        # 添加源
        source_config = []
        if source1_position and source1_component:

            # 配置波形参数（根据波形类型设置默认参数）
            if source1_waveform == "gaussian":
                waveform_params1 = {"t0": 1.0, "tau": 0.2}
            elif source1_waveform == "ricker":
                waveform_params1 = {"t0": 1.0, "f": 1.0}
            elif source1_waveform == "sinusoid":
                waveform_params1 = {"freq": source1_waveform_params, "phase": 0.0}
            elif source1_waveform == "gaussian_enveloped_sine":
                waveform_params1 = {"freq": 1.0, "t0": 1.0, "tau": 0.2}

            source_tag = model.add_source(
                position=tuple(source1_position),
                comp=source1_component,
                waveform=source1_waveform,
                waveform_params=waveform_params1,
                amplitude=source1_amplitude,
                spread=source1_spread,
                injection=source1_injection
            )
            # 获取源配置信息
            source_cfgs = model.list_sources()
            source1_config = next((s for s in source_cfgs if s['tag'] == source_tag), {})
            if source1_config:
                source_config.append(source1_config)

        if source2_position and source2_component:
            # 配置第二个源的波形参数
            waveform_params2 = {}
            if source2_waveform == "gaussian":
                waveform_params2 = {"t0": 1.0, "tau": 0.2}
            elif source2_waveform == "ricker":
                waveform_params2 = {"t0": 1.0, "f": 1.0}
            elif source2_waveform == "sinusoid":
                waveform_params2 = {"freq": source2_waveform_params}
            elif source2_waveform == "gaussian_enveloped_sine":
                waveform_params2 = {"freq": 1.0, "t0": 1.0, "tau": 0.2}
            
            source_tag2 = model.add_source(
                position=tuple(source2_position),
                comp=source2_component,
                waveform=source2_waveform,
                waveform_params=waveform_params2,
                amplitude=source2_amplitude,
                spread=source2_spread,
                injection=source2_injection
            )
            # 获取第二个源配置信息
            source_cfgs = model.list_sources()
            source2_config = next((s for s in source_cfgs if s['tag'] == source_tag2), {})
            if source2_config:
                source_config.append(source2_config)
        
        
        # 添加物体配置
        object_configs = []
        
        # 添加第一个物体（如果提供了边界框）
        if object1_box:
            object1_tag = model.add_object(
                box=object1_box,
                eps=object1_eps,
                mu=object1_mu,
                conductivity=0.0,  # 默认不导电
                tag="object1"
            )
            obj1_cfgs = model.list_objects()
            obj1_config = next((o for o in obj1_cfgs if o['tag'] == object1_tag), {})
            if obj1_config:
                object_configs.append(obj1_config)
        
        # 添加第二个物体（如果提供了边界框）
        if object2_box:
            object2_tag = model.add_object(
                box=object2_box,
                eps=object2_eps,
                mu=object2_mu,
                conductivity=0.0,  # 默认不导电
                tag="object2"
            )
            obj2_cfgs = model.list_objects()
            obj2_config = next((o for o in obj2_cfgs if o['tag'] == object2_tag), {})
            if obj2_config:
                object_configs.append(obj2_config)
            
        # 添加第三个物体（如果提供了边界框）
        if object3_box:
            object3_tag = model.add_object(
                box=object3_box,
                eps=object3_eps,
                mu=object3_mu,
                conductivity=0.0,  # 默认不导电
                tag="object3"
            )
            obj3_cfgs = model.list_objects()
            obj3_config = next((o for o in obj3_cfgs if o['tag'] == object3_tag), {})
            if obj3_config:
                object_configs.append(obj3_config)
        
        return (model.eps, model.mu, model.domain, 
                source_config, object_configs)