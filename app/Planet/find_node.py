import numpy as np
import meshio
import matplotlib.pyplot as plt

from TPMModel import TPMModel 

class Find_node0():
    def __init__(self):
        pass

    def read_mesh(self):
        fname = 'initial/file1020.vtu'
        data = meshio.read(fname)
        node = data.points # 无量纲数值
        cell = data.cells[0][1]
        print("number of nodes of surface mesh:", len(node))
        print("number of cells of surface mesh:", len(cell))

        node = node - np.mean(node, axis=0) # 以质心作为原点
        print('node:', node.shape)

        special_nodes = self.find_node(node)
        return special_nodes

    def angle(self, x):
        x = x/(np.linalg.norm(x, axis=1)).reshape(len(x), 1)
        y = np.array([[1, 0]], dtype=np.float_)
        theta = np.sign(np.cross(y, x))*np.arccos((x*y).sum(axis=1))
        return theta

    def find_node(self, node):
        flag = np.abs(node[:, 1])>3e-2 #不是经度为 0 的点
        # 计算角度
        th = self.angle(node[:, [0, 2]])
        th[flag] = 100000 #将不是经度为 0 的点排除

        # 0°
        a0 = np.argmin(np.abs(th))

        # 30°
        a30 = np.argmin(np.abs(th-np.pi/6))

        # -30°
#        a_30 = np.argmin(np.abs(th+np.pi/6))

        # 60°
        a60 = np.argmin(np.abs(th-np.pi/3))

        # -60°
#        a_60 = np.argmin(np.abs(th+np.pi/3))

        # 90°
        a90 = np.argmin(np.abs(th-np.pi/2))

        # -90°
#        a_90 = np.argmin(np.abs(th+np.pi/2))

#        print('0:', a0, node[a0, :], '30:', a30, node[a30, :], '-30:', a_30, node[a_30, :], '60:',
#                a60,node[a60, :], '-60:', a_60, node[a_60, :], '90:', a90,
#                node[a90, :], '-90:', a_90, node[a_90, :])
#        return np.array([a0, a30, a_30, a60, a_60, a90, a_90])
        return np.array([a0, a30, a60, a90])
class Find_node():
    def __init__(self):
        pass

    def read_mesh(self):
        fname = 'initial/file560.vtu'
        data = meshio.read(fname)
        node = data.points # 无量纲数值
        cell = data.cells[0][1]
        print("number of nodes of surface mesh:", len(node))
        print("number of cells of surface mesh:", len(cell))

        node = node - np.mean(node, axis=0) # 以质心作为原点
        print('node:', node.shape)

        special_nodes = self.find_node(node)
        return special_nodes

    def angle(self, x):
        x = x/(np.linalg.norm(x, axis=1)).reshape(len(x), 1)
        y = np.array([[1, 0]], dtype=np.float_)
        theta = np.sign(np.cross(y, x))*np.arccos((x*y).sum(axis=1))
        return theta

    def find_node(self, node):
        flag = np.abs(node[:, 1])>3e-2 #不是经度为 0 的点
        # 计算角度
        th = self.angle(node[:, [0, 2]])
        th[flag] = 100000 #将不是经度为 0 的点排除

        # 0°
        a0 = np.argmin(np.abs(th))

        # 30°
        a30 = np.argmin(np.abs(th-np.pi/6))

        # -30°
#        a_30 = np.argmin(np.abs(th+np.pi/6))

        # 60°
        a60 = np.argmin(np.abs(th-np.pi/3))

        # -60°
#        a_60 = np.argmin(np.abs(th+np.pi/3))

        # 90°
        a90 = np.argmin(np.abs(th-np.pi/2))

        # -90°
#        a_90 = np.argmin(np.abs(th+np.pi/2))

#        print('0:', a0, node[a0, :], '30:', a30, node[a30, :], '-30:', a_30, node[a_30, :], '60:',
#                a60,node[a60, :], '-60:', a_60, node[a_60, :], '90:', a90,
#                node[a90, :], '-90:', a_90, node[a_90, :])
#        return np.array([a0, a30, a_30, a60, a_60, a90, a_90])
        return np.array([a0, a30, a60, a90])

#a = Find_node()
#a.read_mesh()




