from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["PointSource", "ObjectSource"]


class PointSource(CNodeType):
    r"""Point source configuration for electromagnetic simulations.
    
    Inputs:
        source1_position: Position of first source (x,y) or (x,y,z).
        source1_component: Field component excited by first source.
        source1_waveform: Waveform type of first source.
        source1_frequency: Frequency of first source [Hz].
        source1_phase: Phase of first source [degrees].
        source1_amplitude: Amplitude of first source.
        source1_spread: Spatial spread radius of first source.
        source1_injection: Injection type of first source.
        ... (similar for source2)
    
    Outputs:
        source_configs: source configuration dictionaries.
    """
    
    TITLE: str = "点源配置"
    PATH : str = "examples.CEM"
    INPUT_SLOTS = [
        # Single source configuration
        PortConf("source1_position", DataType.LIST, 0, 
                desc="源位置坐标，(x,y)或(x,y,z)",
                title="源1位置", default=[3e-6, 2e-6]),
        PortConf("source1_component", DataType.MENU, 0, 
                desc="源激励的电磁场分量",
                title="源1场分量", default="Ez", 
                items=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]),
        PortConf("source1_waveform", DataType.MENU, 0, 
                desc="源的时间波形类型", 
                title="源1波形类型", default="gaussian",
                items=["gaussian", "sinusoid", "ricker", "gaussian_enveloped_sine"]),
        PortConf("source1_frequency", DataType.FLOAT, 0,
                desc="源频率", title="源1频率", default=6e14),
        PortConf("source1_phase", DataType.FLOAT, 0,
                desc="初始相位", title="源1相位", default=0.0),
        PortConf("source1_amplitude", DataType.FLOAT, 0,
                desc="源的幅度大小", title="源1幅度", default=1.0),
        PortConf("source1_spread", DataType.INT, 0, 
                 desc="源的空间扩展半径（网格点数）,0表示点源",
                 title="源1扩展半径", default=0),
        PortConf("source1_injection", DataType.MENU, 0, 
                 desc="源的注入方式: soft(叠加)或hard(覆盖)",
                title="源1注入方式", default="soft", 
                items=["soft", "hard"]),

        PortConf("source2_position", DataType.LIST, 0, 
                desc="源位置坐标，格式为(x,y)或(x,y,z), None表示无第二个源",
                title="源2位置", default=None),
        PortConf("source2_component", DataType.MENU, 0, 
                desc="源激励的电磁场分量",
                title="源2场分量", default="Ez", 
                items=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]),
        PortConf("source2_waveform", DataType.MENU, 0, 
                desc="源的时间波形类型", 
                title="源2波形类型", default="gaussian", 
                items=["gaussian", "sinusoid", "ricker", "gaussian_enveloped_sine"]),
        PortConf("source2_frequency", DataType.FLOAT, 0,
                desc="源频率", title="源2频率", default=3e14),
        PortConf("source2_phase", DataType.FLOAT, 0,
                desc="初始相位", title="源2相位", default=0.0),
        PortConf("source2_amplitude", DataType.FLOAT, 0, 
                desc="源的幅度大小", title="源2幅度", default=2.0),
        PortConf("source2_spread", DataType.INT, 0, 
                desc="源的空间扩展半径（网格点数）",
                title="源2扩展半径", default=0),
        PortConf("source2_injection", DataType.MENU, 0, 
                 desc="源的注入方式:soft(叠加)或hard(覆盖)",
                title="源2注入方式", default="soft", 
                items=["soft", "hard"])
    ]
    
    OUTPUT_SLOTS = [
        PortConf("source_configs", DataType.DICT, title="点源配置"),
    ]
    @staticmethod
    def run(**options):
        import math
        
        source_configs = []
        
        # 第一个源
        source1_position = options.get("source1_position")
        source1_component = options.get("source1_component")
        
        if source1_position and source1_component:
            source1_waveform = options.get("source1_waveform", "gaussian")
            source1_frequency = options.get("source1_frequency", 6e14)
            source1_phase = options.get("source1_phase", 0.0)
            
            # 配置波形参数（根据波形类型设置默认参数）
            if source1_waveform == "gaussian":
                waveform_params1 = {"t0": 1.0, "tau": 0.2}
            elif source1_waveform == "ricker":
                waveform_params1 = {"t0": 1.0, "f": source1_frequency}
            elif source1_waveform == "sinusoid":
                waveform_params1 = {"freq": source1_frequency, "phase": source1_phase * math.pi / 180.0}
            elif source1_waveform == "gaussian_enveloped_sine":
                waveform_params1 = {"freq": source1_frequency, "t0": 1.0, "tau": 0.2, "phase": source1_phase * math.pi / 180.0}
            else:
                waveform_params1 = {}
            
            source_config1 = {
                'position': tuple(source1_position),
                'comp': source1_component,
                'waveform': source1_waveform,
                'waveform_params': waveform_params1,
                'amplitude': options.get("source1_amplitude", 1.0),
                'spread': options.get("source1_spread", 0),
                'injection': options.get("source1_injection", "soft"),
                'tag': 'source1'
            }
            source_configs.append(source_config1)
        
        # 第二个源
        source2_position = options.get("source2_position")
        source2_component = options.get("source2_component")
        
        if source2_position and source2_component:
            source2_waveform = options.get("source2_waveform", "gaussian")
            source2_frequency = options.get("source2_frequency", 3e14)
            source2_phase = options.get("source2_phase", 0.0)
            
            # 配置第二个源的波形参数
            if source2_waveform == "gaussian":
                waveform_params2 = {"t0": 1.0, "tau": 0.2}
            elif source2_waveform == "ricker":
                waveform_params2 = {"t0": 1.0, "f": source2_frequency}
            elif source2_waveform == "sinusoid":
                waveform_params2 = {"freq": source2_frequency, "phase": source2_phase * math.pi / 180.0}
            elif source2_waveform == "gaussian_enveloped_sine":
                waveform_params2 = {"freq": source2_frequency, "t0": 1.0, "tau": 0.2, "phase": source2_phase * math.pi / 180.0}
            else:
                waveform_params2 = {}
            
            source_config2 = {
                'position': tuple(source2_position),
                'comp': source2_component,
                'waveform': source2_waveform,
                'waveform_params': waveform_params2,
                'amplitude': options.get("source2_amplitude", 1.0),
                'spread': options.get("source2_spread", 0),
                'injection': options.get("source2_injection", "soft"),
                'tag': 'source2'
            }
            source_configs.append(source_config2)
        
        return source_configs


class ObjectSource(CNodeType):
    r"""Object configuration for electromagnetic simulations.
    
    Inputs:
        object1_box: Bounding box of first object.
        object1_eps: Relative permittivity of first object.
        object1_mu: Relative permeability of first object.
        ... (similar for object2 and object3)
    
    Outputs:
        object_configs: object configuration dictionaries.
    """
    TITLE: str = "物体配置"
    PATH : str = "examples.CEM"
    INPUT_SLOTS = [
        # First object configuration
        PortConf("object1_box", DataType.TEXT, 0, 
                desc="第一个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]",
                title="物体1边界框", default=[0,2.10e-6,2.5e-6,2.6e-6]),
        PortConf("object1_eps", DataType.FLOAT, 0, 
                desc="第一个物体的相对介电常数, None表示使用背景值",
                title="物体1介电常数", default=1000),
        PortConf("object1_mu", DataType.FLOAT, 0, 
                desc="第一个物体的相对磁导率, None表示使用背景值",
                title="物体1磁导率", default=1.0),
        
        # Second object configuration
        PortConf("object2_box", DataType.TEXT, 0, 
                desc="第二个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]",
                title="物体2边界框", default=[2.35e-6,2.65e-6,2.5e-6,2.6e-6]),
        PortConf("object2_eps", DataType.FLOAT, 0, 
                desc="第二个物体的相对介电常数, None表示使用背景值",
                title="物体2介电常数", default=1000),
        PortConf("object2_mu", DataType.FLOAT, 0, 
                desc="第二个物体的相对磁导率, None表示使用背景值",
                title="物体2磁导率", default=1.0),

        # third object configuration
        PortConf("object3_box", DataType.TEXT, 0,
                desc="第三个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]",
                title="物体3边界框", default=[2.9e-6,5e-6,2.5e-6,2.6e-6]),
        PortConf("object3_eps", DataType.FLOAT, 0, 
                desc="第三个物体的相对介电常数, None表示使用背景值",
                title="物体3介电常数", default=1000),
        PortConf("object3_mu", DataType.FLOAT, 0, 
                desc="第三个物体的相对磁导率, None表示使用背景值",
                title="物体3磁导率", default=1.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("object_config", DataType.DICT, title="物体配置")
    ]
    
    @staticmethod
    def run(**options):
        import math
        from fealpy.backend import bm
        
        object_configs = []
        
        # 第一个物体
        object1_box = options.get("object1_box")
        if isinstance(object1_box, str):
            box1 = bm.tensor(eval(object1_box, None, vars(math)), dtype=bm.float64)
        else:
            box1 = bm.tensor(object1_box, dtype=bm.float64)
        if object1_box is not None:
            object1_eps = options.get("object1_eps")
            object1_mu = options.get("object1_mu")
            if object1_eps is not None or object1_mu is not None:
                object_configs.append({
                    'box': box1,
                    'eps': object1_eps,
                    'mu': object1_mu,
                    'conductivity': 0.0,
                    'tag': 'object1'
                })
        
        # 第二个物体
        object2_box = options.get("object2_box")
        if isinstance(object2_box, str):
            box2 = bm.tensor(eval(object2_box, None, vars(math)), dtype=bm.float64)
        else:
            box2 = bm.tensor(object2_box, dtype=bm.float64)
        if object2_box is not None:
            object2_eps = options.get("object2_eps")
            object2_mu = options.get("object2_mu")
            if object2_eps is not None or object2_mu is not None:
                object_configs.append({
                    'box': box2,
                    'eps': object2_eps,
                    'mu': object2_mu,
                    'conductivity': 0.0,
                    'tag': 'object2'
                })
        
        # 第三个物体
        object3_box = options.get("object3_box")
        if isinstance(object3_box, str):
            box3 = bm.tensor(eval(object3_box, None, vars(math)), dtype=bm.float64)
        else:
            box3 = bm.tensor(object3_box, dtype=bm.float64)
        if object3_box is not None:
            object3_eps = options.get("object3_eps")
            object3_mu = options.get("object3_mu")
            if object3_eps is not None or object3_mu is not None:
                object_configs.append({
                    'box': box3,
                    'eps': object3_eps,
                    'mu': object3_mu,
                    'conductivity': 0.0,
                    'tag': 'object3'
                })
        
        return object_configs
