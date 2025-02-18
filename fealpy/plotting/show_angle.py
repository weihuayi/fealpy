from ..backend import backend_manager as bm
from matplotlib.axes import Axes

from ..mesh import Mesh as _MT
from .classic import MeshPloter

class ShowAngle(MeshPloter[_MT]):
    def draw(self,axes:Axes, *args,**kwargs):
        """
        @brief 显示网格角度的分布直方图
        """
        import numpy as np
        angle = self.mesh.angle()
        if(len(args) == 0):
            pass
        elif(args[0] == 'max'):
            angle = bm.max(angle,axis=1)
        elif(args[0] =='min'):
            angle = bm.min(angle,axis=1)
        angle = bm.to_numpy(angle) 
        hist, bins = np.histogram(angle.flatten('F') * 180 / np.pi, bins=50, range=(0, 180))
        center = (bins[:-1] + bins[1:]) / 2
        axes.bar(center, hist, align='center', width=180 / 50.0)
        axes.set_xlim(0, 180)
        mina = np.min(angle.flatten('F') * 180 / np.pi)
        maxa = np.max(angle.flatten('F') * 180 / np.pi)
        meana = np.mean(angle.flatten('F') * 180 / np.pi)
        axes.annotate('Min angle: {:.4}'.format(mina),
                      xy=(0.41,0.5),xytext=(0.41,0.5),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        axes.annotate('Max angle: {:.4}'.format(maxa), 
                      xy=(0.41,0.45),xytext=(0.41,0.45),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        axes.annotate('Average angle: {:.4}'.format(meana), 
                      xy=(0.41,0.40),xytext=(0.41,0.40),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        return mina, maxa, meana



ShowAngle.register('show_angle')
