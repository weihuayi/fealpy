from .nodetype import CNodeType, PortConf, DataType

__all__ = ["VPDecoupling"]

class VPDecoupling(CNodeType):
    TITLE: str = "速度-压力解耦"
    PATH: str = "后处理.解耦"
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("u_x", DataType.TENSOR, title="速度x分量数值解"),
        PortConf("u_y", DataType.TENSOR, title="速度y分量数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解")
    ]

    @staticmethod
    def run(out, uspace, mesh):
        ugdof = uspace.number_of_global_dofs()
        NN = mesh.number_of_nodes()
        uh = out[:ugdof]
        uh = uh.reshape(mesh.GD,-1).T
        uh = uh[:NN,:]
        u_x = out[:int(ugdof/2)]
        u_x = u_x[:NN]
        u_y = out[int(ugdof/2):ugdof]
        u_y = u_y[:NN]
        ph = out[ugdof:]

        return uh, u_x, u_y, ph

class OutputVideo(CNodeType):
    TITLE: str = "输出视频"
    PATH: str = "后处理.可视化"
    INPUT_SLOTS = [
        PortConf("T0", DataType.FLOAT, title="初始时间"),
        PortConf("T1", DataType.FLOAT, title="结束时间"),
        PortConf("NL", DataType.INT, title="时间层数"),
        PortConf("domain", DataType.FLOAT, title="显示区域"),
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("out", DataType.TENSOR, title="结果"),
        PortConf("dpi", DataType.FLOAT, 0, title="分辨率", default=300),
        PortConf("bitrate", DataType.FLOAT, 0, title="视频码率", default=5000),
        PortConf("figsize_x", DataType.FLOAT, 0, title="图像长度", default=6.0),
        PortConf("figsize_y", DataType.FLOAT, 0, title="图像高度", default=3.0),
        PortConf("cmap", DataType.MENU, 0, 
                                    title="颜色映射",
                                    default='cividis', 
                                    items=['cividis', 'rainbow', 'turbo', 'inferno', 
                                            'viridis', 'jet', 'plasma', 'coolwarm']),
        PortConf("clim_vmin", DataType.FLOAT, 0, title="颜色最小值", default=0.0),
        PortConf("clim_vmax", DataType.FLOAT, 0, title="颜色最大值", default=1.8),
        PortConf("filename", DataType.STRING, 0, title="文件名", default = None),
        PortConf("title", DataType.STRING, 0, title="标题", default = "数值解")
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果")
    ]
    def run(T0 : float, T1 : float, NL : int, domain, mesh, out, dpi : float, bitrate : float,
            figsize_x : float, figsize_y : float, cmap : str, clim_vmin : float, clim_vmax : float,
            filename : str, title : str):
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from datetime import datetime

        nt = NL - 1
        dt = (T1 - T0)/nt
        box = domain
        ipoints = mesh.interpolation_points(p = 1)
        x = ipoints[:, 0]
        y = ipoints[:, 1]
        cells = mesh.entity('cell')
        triang = tri.Triangulation(x, y, cells)

        fig1, ax1 = plt.subplots(figsize=(figsize_x, figsize_y), dpi = dpi)
        ax1.set_aspect('equal')
        ax1.set_xlim(box[0], box[1])
        ax1.set_ylim(box[2], box[3])
        ax1.set_title(title)

        tpc = ax1.tripcolor(triang, out[0], shading='gouraud', cmap=cmap)
        tpc.set_clim(vmin=clim_vmin, vmax=clim_vmax)

        def update1(frame):
            tpc.set_array(out[frame])
            ax1.set_title(title + "  " + f't={T0 + frame*dt:.2f}')
            return [tpc]
        
        ani1 = FuncAnimation(fig1, update1, frames=nt, blit=True)

        writer = FFMpegWriter(fps=nt/(T1 - T0), bitrate = bitrate)
        if filename is None:
            ani1.save(f'韶峰天工{datetime.now()}.mp4', writer=writer)
        else:
            ani1.save(f'{filename}.mp4', writer=writer)
        plt.close(fig1)
        print("mp4 has been generated!")
        return out
