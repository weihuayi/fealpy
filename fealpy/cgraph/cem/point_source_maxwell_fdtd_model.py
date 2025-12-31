from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["PointSourceMaxwellFDTDModel"]
    

class PointSourceMaxwellFDTDModel(CNodeType):
    r"""Point source Maxwell equations FDTD solver.

    Inputs:
        mesh: YeeUniformMesher with attached configurations.
        dt: Time step size (optional, auto-calculated if not provided).
        maxstep: Maximum number of time steps.
        save_every: Save field data every N steps.
        boundary: Boundary condition type ('PEC' or 'UPML').
        pml_width: PML layer width for UPML boundary.
        pml_m: PML grading parameter.
        output_dir: Directory to save output files.

    Outputs:
        field_history: List of field snapshots over time.
    """
    
    TITLE: str = "点源Maxwell方程FDTD求解器"
    PATH: str = "simulation.solvers"
    
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1,
            desc="配置完成的Yee网格, 包含材料属性、源配置和物体配置",
            title="Yee网格"),
        PortConf("dt", DataType.FLOAT, 0,
            desc="时间步长, None表示自动计算",
            title="时间步长", default=None),
        PortConf("maxstep", DataType.INT, 0,
            desc="最大时间步数", title="最大时间步数",
            default=1000),
        PortConf("save_every", DataType.INT, 0,
            desc="每N个时间步保存一次场数据", 
            title="保存间隔", default=10),
        PortConf("boundary", DataType.MENU, 0,
            desc="边界条件类型", title="边界条件", default="PEC",
            items=["PEC", "UPML"]),
        PortConf("pml_width", DataType.INT, 0,
            desc="UPML边界层的网格点数",
            title="PML宽度", default=8),
        PortConf("pml_m", DataType.FLOAT, 0,
            desc="UPML grading参数",
            title="PML参数", default=5.0),
        PortConf("output_dir", DataType.STRING, 0,
            desc="输出文件保存目录",
            title="输出目录",default="~/fdtd_output"),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("field_history", dtype=DataType.TEXT, title="场演化历史"),
    ]

    @staticmethod
    def run(**options):
        from fealpy.cem.point_source_maxwell_fdtd_model import PointSourceMaxwellFDTDModel
        from fealpy.cem.model.point_source_maxwell import PointSourceMaxwell
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh import QuadrangleMesh
        from pathlib import Path
        
        mesh = options.get("mesh")
        if mesh is None:
            raise ValueError("必须提供Yee网格对象")
        
        domain = mesh.metadata['domain']
        eps = mesh.eps
        mu = mesh.mu
        n = mesh.metadata['grid_size']
        source_configs = mesh.source_configs
        object_configs = mesh.object_configs
        
        dt = options.get("dt")
        maxstep = options.get("maxstep", 1000)
        save_every = options.get("save_every", 10)
        boundary = options.get("boundary", "PEC")
        pml_width = options.get("pml_width", 8)
        pml_m = options.get("pml_m", 5.0)
        output_dir = options.get("output_dir", "~/fdtd_output")
        
        # 创建PDE模型
        pde = PointSourceMaxwell(eps=eps, mu=mu, domain=domain)
        
        # 添加源配置
        if source_configs and len(source_configs) > 0:
            for src in source_configs:
                source_tag = pde.add_source(
                    position=src.get('position'),
                    comp=src.get('comp'),
                    waveform=src.get('waveform'),
                    waveform_params=src.get('waveform_params', {}),
                    amplitude=src.get('amplitude', 1.0),
                    spread=src.get('spread', 0),
                    injection=src.get('injection', 'soft')
                )
        
        # 添加物体配置
        if object_configs and len(object_configs) > 0:
            for obj in object_configs:
                obj_tag = pde.add_object(
                    box=obj.get('box'),
                    eps=obj.get('eps'),
                    mu=obj.get('mu')
                )
        
        # 配置FDTD选项
        fdtd_options = {
            'dt': dt,
            'maxstep': maxstep,
            'save_every': save_every,
            'boundary': boundary,
            'pml_width': pml_width,
            'pml_m': pml_m
        }
        
        # 创建并运行FDTD模型
        fdtd_model = PointSourceMaxwellFDTDModel(pde, n, fdtd_options)
        field_history = fdtd_model.run(nt=maxstep, save_every=save_every)
        
        # 提取场数据
        H = [field_history[k]['H'] for k in range(len(field_history))]
        Hx = bm.array([K["x"] for K in H])
        Hx = Hx[:, :-1, :] + Hx[:, 1:, :]
        Hx = Hx.reshape(Hx.shape[0], -1)
        
        Hy = bm.array([K["y"] for K in H])
        Hy = Hy[..., :-1] + Hy[..., 1:]
        Hy = Hy.reshape(Hy.shape[0], -1)
        
        H_combined = bm.array([Hx, Hy])
        
        # 创建可视化网格
        vis_mesh = QuadrangleMesh.from_box(domain, nx=n, ny=n)
        
        E = [field_history[k]['E'] for k in range(len(field_history))]
        Ez = bm.array([K["z"] for K in E])
        
        # 保存VTK文件
        export_dir = Path(output_dir).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for i, E_field in enumerate(Ez):
            vis_mesh.nodedata["E"] = E_field.reshape(-1)
            vis_mesh.celldata["H"] = H_combined[:, i, :].T
            fname = export_dir / f"fdtd_{str(i).zfill(10)}.vtu"
            vis_mesh.to_vtk(fname=str(fname))
        return field_history