import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


from data import H


class Roller():
    def __init__(self, center, data):
        self._data = data
        self.center = center 
        # 滚柱轴长度
        self.length_shaft = self._data[48-1]
        # 滚柱轴外径
        self.diameter_outer = self._data[51-1]
        # 滚柱轴内径
        self.diameter_inner = self._data[54-1]
        # 滚柱螺纹长度
        self.length_thread = self._data[57-1]

        # 密度
        self.density = self._data[73-1] 
        # 弹性模量
        self.E = self._data[76-1]
        # 泊松比
        self.nu = self._data[79-1]

    def add_plot(self, axes):
        """画滚柱相关的圆
        """
        r = self.diameter_inner
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'b-')


class Screw():
    def __init__(self, data):
        self._data = data
        # 丝杠轴长度
        self.length_shaft = self._data[46-1]
        # 丝杠外径
        self.diameter_outer = self._data[49-1]
        # 丝杠内径
        self.diameter_inner = self._data[52-1]
        # 丝杠螺纹长度
        self.length_thread = self._data[55-1]

        # 密度
        self.density = self._data[71-1] 
        # 弹性模量
        self.E = self._data[74-1]
        # 泊松比
        self.nu = self._data[77-1]

    def add_plot(self, axes):
        """画丝杠相关的圆
        """
        r = self.diameter_inner
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'b-')



class Nut():
    def __init__(self, data):
        self._data = data
        # 丝杠轴长度
        self.length_shaft = self._data[47-1]
        # 螺母轴外径
        self.diameter_outer = self._data[50-1] 
        # 螺母轴内径
        self.diameter_inner = self._data[53-1]
        # 螺母螺纹长度
        self.length_thread = self._data[56-1]

        # 密度
        self.density = self._data[72-1] 
        # 弹性模量
        self.E = self._data[75-1]
        # 泊松比
        self.nu = self._data[78-1]

    def add_plot(self, axes):
        # 画螺母内啮合圆
        r = self.diameter_outer 
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'b-')



class PlanetaryRollerScrew():

    def __init__(self, data, mesh_outer_data, mesh_inner_data):
        self._data = data
        self._mesh_outer_data = mesh_outer_data
        self._mesh_inner_data = mesh_inner_data

        self.screw = Screw(self._data)
        self.nut = Nut(self._data)
        self.rollers = []
        # 滚柱的个数
        self.number_roller = int(self._data[3-1])
        # 第一个滚柱与 x 轴的夹角
        self.angle_roller = self._data[13-1]
        # 生成所有的滚柱
        for i in range(self.number_roller): 
            self.rollers.append(Roller(np.array([0.0, 0.0]), self._data))
        # 螺距
        self.pitch_screw = data[16-1]

        # 内啮合的有效螺纹长度
        self.length_mesh_inner = data[58-1]
        # 外啮合的有效螺纹长度
        self.length_mesh_outer = data[59-1]

        self.length_mesh = min([self.length_mesh_inner, self.length_mesh_outer])
        self.number =  1 + int(self.length_mesh / self.pitch_screw)

    def circumcenter(self, tri):
        """计算三角形的外接圆圆心
        """
        v0 = tri[2] - tri[1]
        v1 = tri[0] - tri[2]
        v2 = tri[1] - tri[0]
        area = (v1[0]*v2[1]-v1[1]*v2[0])/ 2.0

        x = np.sum(tri**2, axis=1)
        w0 = x[2] + x[1]
        w1 = x[0] + x[2]
        w2 = x[1] + x[0]

        W = np.array([[0, -1], [1, 0]], dtype=np.float64)
        fe0 = w0 * v0 @ W
        fe1 = w1 * v1 @ W
        fe2 = w2 * v2 @ W
        c = 0.25 * (fe0 + fe1 + fe2) / area
        r = np.sqrt(np.sum((c - tri[0]) ** 2))
        return c, r 


    def add_plot(self, axes, data):
        axes.axis('equal')
        axes.grid(True, linestyle='--', color='gray')
        mesh_outer_matrix = data['mesh_outer_matrix']
        mesh_inner_matrix = data['mesh_inner_matrix']
        p0 = mesh_outer_matrix[:, 3:5]
        p1 = mesh_inner_matrix[:, 3:5]
        p = np.vstack((p0, p1))
        axes.plot(p[:, 0], p[:, 1], 'ro')

        # 画丝杠外啮合圆
        r = data['mesh_outer_matrix'][0, 7-1]
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'k-')

        # 画螺母内啮合圆
        r = data['mesh_inner_matrix'][0, 7-1]
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'b-')




prs_data = sio.loadmat('./potential_roller_screw.mat')

data = prs_data['roller_screw_initial'][0]

a = prs_data['mesh_inner_matrix'][0, 6]  - prs_data['mesh_outer_matrix'][0, 6]
b = prs_data['mesh_inner_matrix'][0, 8] + prs_data['mesh_outer_matrix'][0, 8]

print(a, b)

prs = PlanetaryRollerScrew(data)

tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
print(prs.circumcenter(tri))


fig = plt.figure()
axes = fig.add_subplot(111)
prs.add_plot(axes, prs_data)
plt.show()
