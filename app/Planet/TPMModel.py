import numpy as np
import meshio

from fealpy.decorator import cartesian, barycentric, timer
from fealpy.geometry.implicit_surface import ScaledSurface, SphereSurface
from fealpy.mesh import LagrangeTriangleMesh, LagrangeWedgeMesh
from fealpy.writer import MeshWriter

class TPMModel():
    def __init__(self, args, options=None):
        self.args = args
        if options is None:
            self.options = self.model_options()

    def model_options(self, 
            A=0.1,             # 邦德反照率
            epsilon=0.9,       # 辐射率
            rho=1400,          # kg/m^3 密度
            c=1200,            # Jkg^-1K^-1 比热容
            kappa=0.02,        # Wm^-1K^-1 热导率
            sigma=5.667e-8,    # Wm^-2K^-4 斯特藩-玻尔兹曼常数
            qs=1367.5,          # W/m^2 太阳辐射通量
            period=0.5,          # 小行星自转周期，单位小时
            sd=[1, 0, 0],      # 指向太阳的方向
            theta=200,         # 初始温度，单位 K
            ):

        """
        Notes
        -----
            设置模型参数信息
        """
        period *= 3600 # 转换为秒
        omega = 2*np.pi/period
        Tss = ((1-A)*qs/(epsilon*sigma))**(1/4) # 日下点温度 T_ss=[(1-A)*r/(epsilon*sigma)]^(1/4)
        Tau = np.sqrt(rho*c*kappa) # 热惯量 Tau = (rho*c*kappa)^(1/2)
        Phi = Tau*np.sqrt(omega)/(epsilon*sigma*Tss**3) # 热参数
        l = np.sqrt(kappa/(rho*c*omega)) # 趋肤深度 l=(kappa/(rho*c*omega))^(1/2)
        options = {
                "A": A,
                "epsilon": epsilon,
                "rho": rho,
                "c": c,
                "kappa": kappa,
                "sigma": sigma,
                "qs": qs,
                "period": period,
                "sd": np.array(sd, dtype=np.float64),
                "omega": omega,
                "Tss": Tss,
                "Tau": Tau,
                "Phi": Phi,
                "theta": theta,
                "l": l,
                }
        return options


    @timer
    def init_rotation_mesh(self):
        print("Generate the init mesh!...")

        args = self.args
        n = args.nrefine # 初始网格加密次数
        p = args.degree # 空间次数 
        h = args.h # 求解区域的厚度
        nh = args.nh # 三棱柱层数
        H = args.scale # 小行星规模

        fname = 'initial/file1.vtu'
        data = meshio.read(fname)
        node = data.points # 无量纲数值
        cell = data.cells[0][1]
        print("number of nodes of surface mesh:", len(node))
        print("number of cells of surface mesh:", len(cell))

        node = node - np.mean(node, axis=0) # 以质心作为原点
        l = self.options['l']
        node *= H # H 小行星的规模
        node /=l # 无量纲化处理

        h /=nh # 单个三棱柱的高度
        h /= l # 无量纲化处理

        mesh = LagrangeTriangleMesh(node, cell, p=p)
        mesh.uniform_refine(n)
        mesh = LagrangeWedgeMesh(mesh, h, nh, p=p)

        print("finish mesh generation!")
        return mesh
    
    @timer
    def test_mesh(self):
        print("Generate the init mesh!...")
        
        args = self.args
        n = args.nrefine # 初始网格加密次数
        p = args.degree # 空间次数 
        h = args.h # 求解区域的厚度
        nh = args.nh # 三棱柱层数
        H = args.scale # 小行星规模

        surface = SphereSurface()
        node, cell = surface.init_mesh(meshtype='tri', returnnc=True, p=p)
        
        node *= H # H 小行星的规模

        h /=nh # 单个三棱柱的高度

        surface = ScaledSurface(surface, H)

        mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
        mesh.uniform_refine(n)
        
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        print('node:', len(node))
        print('cell:', len(cell))
        
        mesh = LagrangeWedgeMesh(mesh, h, nh, p=p)
        
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        print('node:', len(node))
        print('cell:', len(cell))
        
        print("finish mesh generation!")
        return mesh
    
    @timer
    def test_rotation_mesh(self):
        print("Generate the init mesh!...")

        args = self.args
        n = args.nrefine # 初始网格加密次数
        p = args.degree # 空间次数 
        h = args.h # 求解区域的厚度
        nh = args.nh # 三棱柱层数
        H = args.scale # 小行星规模

        surface = SphereSurface()
        node, cell = surface.init_mesh(meshtype='tri', returnnc=True, p=p)
        
        l = self.options['l']
        node *= H # H 小行星的规模
        node /=l # 无量纲化处理

        h /=nh # 单个三棱柱的高度
        h /= l # 无量纲化处理

        surface = ScaledSurface(surface, H/l)

        mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
        mesh.uniform_refine(n)
        
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        print('node:', len(node))
        print('cell:', len(cell))
        
        mesh = LagrangeWedgeMesh(mesh, h, nh, p=p)
        
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        print('node:', len(node))
        print('cell:', len(cell))

        print("finish mesh generation!")
        return mesh
