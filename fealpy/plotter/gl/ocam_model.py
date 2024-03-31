from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np


@dataclass
class OCAMModel:
    ss: np.ndarray = np.array([-576.3797, 0, 0.0007185556, -3.39907e-07, 5.242219e-10])
    xc: float = 559.875074
    yc: float = 992.836922
    c: float = 1.000938 
    d: float = 0.000132
    e: float = -0.000096

    def world2cam(self, node):
        """
        """
        ps = self.omni3d2pixel(node)
        uv = np.zeros_like(ps)
        uv[:, 0] = ps[:, 0] * c + ps[:, 1] * d + xc
        uv[:, 1] = ps[:, 0] * e + ps[:, 1]     + yc
        return uv 

    def omni3d2pixel(self, node):
        """
        """
        pcoef = np.flip(ss)
        # 解决 node = [0,0,+-1] 的情况
        flag = (node[:, 0] == 0) & (node[:, 1] == 0)
        node[flag, :] = np.finfo(float).eps

        l = np.sqrt(node[:, 0]**2 + node[:, 1]**2)
        m = node[:, 2] / l
        rho = np.zeros(m.shape)
        pcoef_tmp = np.copy(pcoef)
        for i in range(len(m)):
            pcoef_tmp[-2] = pcoef[-2] - m[i]
            r = np.roots(pcoef_tmp)
            flag = (np.imag(r) == 0) & (r > 0)
            res = r[flag]
            if len(res) == 0:
                rho[i] = np.nan
            else:
                rho[i] = np.min(res)

        ps = node[:, 0:2]/l.reshape(-1, 1)*rho.reshape(-1, 1)
        return ps 


