
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["MicrostripPatchMesher3d"]


class MicrostripPatchMesher3d(CNodeType):
    TITLE: str = "三维微带贴片天线网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("d", DataType.FLOAT, 1, title="基板厚度(mm)", default=1.524),
        PortConf("w_line", DataType.FLOAT, 1, title="50Ω 线宽(mm)", default=3.2),
        PortConf("w_patch", DataType.FLOAT, 1, title="贴片宽度(mm)", default=53.0),
        PortConf("l_patch", DataType.FLOAT, 1, title="贴片长度(mm)", default=52.0),
        PortConf("w_stub", DataType.FLOAT, 1, title="调谐短截线宽度(mm)", default=7.0),
        PortConf("l_stub", DataType.FLOAT, 1, title="调谐短截线长度(mm)", default=15.5),
        PortConf("w_sub", DataType.FLOAT, 1, title="基板宽度(mm)", default=100.0),
        PortConf("l_sub", DataType.FLOAT, 1, title="基板长度(mm)", default=100.0)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("sub", DataType.TENSOR, title="基板单元"),
        PortConf("air", DataType.TENSOR, title="空气单元"),
        PortConf("pec", DataType.TENSOR, title="PEC 边界"),
        PortConf("lumped", DataType.TENSOR, title="集总端口边界")
    ]

    @staticmethod
    def run(d, w_line, w_patch, l_patch, w_stub, l_stub, w_sub, l_sub):
        from fealpy.backend import bm
        from fealpy.mesher.patch_antenna_mesher import PatchAntennaMesher

        R, r = l_sub*1.20, l_sub
        sphere_center=(0.0, 0.0, 0.0)
        plate_center=(0.0, 0.0, 0.0)
        inner_size=(w_patch, l_patch)
        notches=[(-w_line/2-w_stub/2, -l_patch/2+l_stub/2, w_stub, l_stub),
                (w_line/2+w_stub/2,  -l_patch/2+l_stub/2, w_stub, l_stub),]
        n_thk=2
        plate_shell_ratio=2
        n_radial_inner=3
        n_radial_shell=2
        hpi, hps, hsi, hss, hmin, hmax = PatchAntennaMesher.recommend_mesh_sizes(
            t=d, r=r, R=R,
            n_thk=n_thk,                # 厚度方向建议≥2
            plate_shell_ratio=plate_shell_ratio,
            n_radial_inner=n_radial_inner,
            n_radial_shell=n_radial_shell
        )
        mesher = PatchAntennaMesher(
            R=R, r=r, sphere_center=sphere_center,
            Lx=w_sub, Ly=l_sub, t=d, plate_center=plate_center,
            inner_size=inner_size, notches=notches,
            h_sphere_inner=hsi, h_sphere_shell=hss,
            h_plate_inner=hpi, h_plate_shell=hps,
            is_optimize=True, h_min=hmin, h_max=hmax
        )
        mesh, info = mesher.init_mesh(*(True,)*6)

        return mesh, info["plate_whole"], info["air"], \
            bm.concat([info["plate_bottom"], info["plate_inner_top"]]), \
            info["bridge_head_edges"]
