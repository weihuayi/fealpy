import sys
import meshio
import vtk

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

from fealpy.mesh import LagrangeTriangleMesh, MeshFactory, LagrangeWedgeMesh

from vtk.numpy_interface import dataset_adapter as dsa

from find_node import Find_node
from TPMModel import TPMModel 

class vtkReader():
    def meshio_read(fname, Mesh):
        data = meshio.read(fname)
        node = data.points
        cell = data.cells[0][1]

        mesh = Mesh(node, cell)
        mesh.to_vtk(fname='write.vtu')

    def vtk_read_data(fname, node_keylist): 
        """
        param[in] fname 文件名
        param[in] node_keylist 节点数据的名字
        param[in] cell_keylist 单元数据的名字
        param[in] Mesh 网格类型
        """
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()
        vmesh = reader.GetOutput()
        wmesh = dsa.WrapDataObject(vmesh)
        
        # 三棱柱网格节点读取错误, 暂未解决
#        node = wmesh.GetPoints() # 节点数据
#        cell = wmesh.GetCells().reshape(-1, 7)[:, 1:] # 单元数据

        nodedata = {}
        for key in node_keylist:
            arr = vmesh.GetPointData().GetArray(key)
            dl = arr.GetNumberOfTuples()
            data = np.zeros(dl, dtype=np.float_)
            for i in range(dl):
                data[i] = arr.GetComponent(i, 0)
            nodedata[key] = data

#        celldata = {}
#        for key in cell_keylist:
#            arr = vmesh.GetCellData().GetArray(key)
#            dl = arr.GetNumberOfTuples()
#            data = np.zeros(dl, dtype=np.float_)
#            for i in range(dl):
#                data[i] = arr.GetComponent(i, 0)
#            celldata[key] = data
        return nodedata

class Aspect():
    def __init__(self):
        self.h = 0.5 # 求解区域的厚度
        self.nh = 100 # 三棱柱层数
#        self.H = 500 # 小行星规模
        self.H = 50 # 小行星规模

        self.Find_node = Find_node() # 寻找特殊角度处的节点值
        self.special_nodes_flag = self.Find_node.read_mesh()
        
        self.angle = np.arange(0, 378, 18) # 绘制随旋转角度的变化情况
        self.layer = np.arange(0, 5) # 绘制 5 层的温度变化情况
#        self.layer = np.arange(0, 101, 20) # 绘制 5 层的温度变化情况
        self.uh_all = self.read_data() # 一旋转周期内分层后的温度值
        self.depth = np.arange(0, 0.505, 0.005) # 每层的深度

    def read_data(self):
        '''

        提取一个自转周期内的 21 张 vtu 文件中的温度数据, 并分层存储
        
        '''
#        uh_all = np.zeros((21, self.nh+1, 4532), dtype=np.float_)
#        uh_all = np.zeros((21, self.nh+1, 512), dtype=np.float_)
        uh_all = np.zeros((21, self.nh+1, 282), dtype=np.float_)
        for i in range(21):
            fname = '5_25_300K_1D_end/test' + str(i).zfill(2) + '.vtu'
            nodedata = vtkReader.vtk_read_data(fname, ['uh'])
            uh = nodedata['uh']
            uh_all[i] = uh[:-1].reshape(self.nh+1, -1)
        return uh_all

    def angle_curve(self, index, degree):
        '''

        绘制某点不同层随自转角度的温度变化曲线
        ---

        每个点绘制选取 5 层, 绘制在同一图中
        x 轴为自 0 相位开始旋转的角度
        y 轴为温度, 单位为开尔文
        ---

        param[in] index 需要绘制的节点的索引
        param[in] degree 需要绘制的节点的纬度
        
        '''
        uh_all = self.uh_all
        angle = self.angle
        layer = self.layer

        fig, axes = plt.subplots()
        for i in range(len(layer)):
            uh0 = np.zeros(uh_all.shape[0], dtype=np.float_)
            uh0[:] = uh_all[:, layer[i], index]

            label = 'Depth' + str(layer[i]*self.h/self.nh)
            axes.plot(angle, uh0, label=label)
        
        title = 'Temperature Curve of 0 Longitude and ' + str(degree) + ' Latitude'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='Rotation Angle (deg)')

        axes.legend()
        fname0 = '50_1000/Latitude' + str(degree) + '.jpg'
        plt.savefig(fname0)

#        plt.show()

    def point_depth_curve(self, index, degree):
        '''

        绘制某点不同层的温度变化曲线
        ---

        每个点绘制选取 5 层, 绘制在同一图中
        x 轴为深度
        y 轴为温度, 单位为开尔文
        ---

        param[in] index 需要绘制的节点的索引
        param[in] degree 需要绘制的节点的纬度
        
        '''
        uh_all = self.uh_all
        depth = self.depth

        fig, axes = plt.subplots()
        for i in range(7):
            uh0 = np.zeros(uh_all.shape[1], dtype=np.float_)
            uh0[:] = uh_all[-1, :, index[i]]

            label = 'Latitude' + str(degree[i])
            axes.plot(depth, uh0, label=label)
        
        title = 'Temperature Curve at different depths'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='Depth (m)')

        axes.legend()
        fname = '50_1000_depth.jpg'
        plt.savefig(fname)

#        plt.show()

    def temperature_data(self):
        '''

        绘制不同层的温度差 平均值 中值曲线
        ---

        x 轴为深度
        y 轴为温度, 单位为开尔文
        ---

        '''
        uh_all = self.uh_all

        fig, axes = plt.subplots()
        dep = np.arange(0, self.nh+1, dtype=np.float_)
        dep *=self.h/self.nh

        uh0 = np.zeros((uh_all.shape[1], uh_all.shape[2]), dtype=np.float_)
        uh0[:] = uh_all[-1, :, :]
        
        y = np.zeros((3, uh_all.shape[1]), dtype=np.float_)
        y[0, :] = uh0.max(axis=1) - uh0.min(axis=1)
        label0 = 'Temperature difference'
        axes.plot(dep, y[0], label=label0)

        y[1, :] = uh0.mean(axis=1)
        label1 = 'Temperature mean'
        axes.plot(dep, y[1], label=label1)

        y[2, :] = np.median(uh0, axis=1)
        label2 = 'Temperature meidan'
        axes.plot(dep, y[2], label=label2)
    
        title = 'Temperature Curve at different depths'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='Depth (m)')

        axes.legend()
        fname = '50_1000_temperature_data.jpg'
        plt.savefig(fname)

#        plt.show()

    def draw_figure(self):
        '''

        绘制 7 个特殊点处的温度变化曲线
        
        '''
        special_degrees = np.array([0, 30, -30, 60, -60, 90, -90])
        
        # 画图程序 1 各点随深度变化的温度图
        self.point_depth_curve(self.special_nodes_flag, special_degrees)
        
         # 画图程序 2 同一点旋转角度变化的温度图
        for j in range(len(special_degrees)):
            self.angle_curve(self.special_nodes_flag[j], special_degrees[j])
            print('Figure of', special_degrees[j], 'degrees have been drawn')

        self.temperature_data() # 画图程序 3 温差 均值 中值图
        print('All figures have been drawn')
    
    def angle_curve_1(self):
        '''

        绘制某点表层随自转角度的温度变化曲线
        ---

        绘制在同一图中
        x 轴为自 0 相位开始旋转的角度
        y 轴为温度, 单位为开尔文
        ---
        
        '''
        index = self.Find_node.read_mesh()
        degree = np.array([0, 30, 60, 90])
        uh_all = self.uh_all
        angle = self.angle

        fig, axes = plt.subplots()
        for i in range(4):
            uh0 = np.zeros(uh_all.shape[0], dtype=np.float_)
            uh0[:] = uh_all[:, 0, index[i]]

            label = 'Latitude' + str(degree[i])
            axes.plot(angle, uh0, label=label)
        
        title = 'A Periodic Surface Temperature Change'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='Rotation Angle (deg)')

        axes.legend()
        fname0 = '5_25_300K_1D_end/1D_25_Surface_Temperature.jpg'
        plt.savefig(fname0)

        plt.show()


    def angle_curve_2(self):
        '''

        绘制某点表层随自转角度的温度变化曲线
        ---

        绘制在同一图中
        x 轴为自 0 相位开始旋转的角度
        y 轴为温度, 单位为开尔文
        ---
        
        '''
        index = self.Find_node.read_mesh()
        degree = np.array([0, 30, 60, 90])
        uh_all = self.uh_all
        angle = self.angle
        color = ['b', 'g', 'r', 'c']

        fig, axes = plt.subplots()
        
        
        ''' 1D 300K'''
        '''
        uh_all0 = np.zeros((21, self.nh+1, 282), dtype=np.float_)
        for i in range(21):
            fname0 = '5_25_300K_1D_end/test' + str(i).zfill(2) + '.vtu'
            nodedata0 = vtkReader.vtk_read_data(fname0, ['uh'])
            uh0 = nodedata0['uh']
            uh_all0[i] = uh0[:-1].reshape(self.nh+1, -1)
        
        for i in range(4):
            uh0 = np.zeros(uh_all0.shape[0], dtype=np.float_)
            uh0[:] = uh_all0[:, 0, index[i]]

            label0 = '1D 300K Latitude' + str(degree[i])
            axes.plot(angle, uh0, dashes=[6, 2], label=label0)
        '''
        
        ''' 1D 150K'''
        uh_all3 = np.zeros((21, self.nh+1, 282), dtype=np.float_)
        for i in range(21):
            fname3 = '5m_100_150K_1D/test' + str(i+1).zfill(2) + '.vtu'
            nodedata3 = vtkReader.vtk_read_data(fname3, ['uh'])
            uh3 = nodedata3['uh']
            uh_all3[i] = uh3[:-1].reshape(self.nh+1, -1)
        
        for i in range(4):
            uh3 = np.zeros(uh_all3.shape[0], dtype=np.float_)
            uh3[:] = uh_all3[:, 0, index[i]]

            label3 = '1D 150K Latitude' + str(degree[i])
            axes.plot(angle, uh3, dashes=[6, 2], label=label3, color=color[i])
        
        '''3D 300K'''
        '''
        uh_all1 = np.zeros((21, self.nh+1, 512), dtype=np.float_)
        for i in range(21):
            fname1 = '50_1000/test' + str(i).zfill(2) + '.vtu'
            nodedata1 = vtkReader.vtk_read_data(fname1, ['uh'])
            uh1 = nodedata1['uh']
            uh_all1[i] = uh1[:-1].reshape(self.nh+1, -1)
        
        for i in range(4):
            uh1 = np.zeros(uh_all1.shape[0], dtype=np.float_)
            uh1[:] = uh_all1[:, 0, index[i]]

            label1 = '3D 50m Latitude' + str(degree[i])
            axes.plot(angle, uh1, label=label1, color=color[i])
        '''
        
        '''3D 150K'''
        uh_all2 = np.zeros((21, self.nh+1, 282), dtype=np.float_)
        for i in range(21):
            fname2 = '5_1008_150K_3D/test' + str(i).zfill(2) + '.vtu'
            nodedata2 = vtkReader.vtk_read_data(fname2, ['uh'])
            uh2 = nodedata2['uh']
            uh_all2[i] = uh2[:-1].reshape(self.nh+1, -1)
        
        for i in range(4):
            uh2 = np.zeros(uh_all2.shape[0], dtype=np.float_)
            uh2[:] = uh_all2[:, 0, index[i]]

            label2 = '3D 150K Latitude' + str(degree[i])
            axes.plot(angle, uh2, dashes=[2, 2], label=label2, color=color[i])
        title = 'A Periodic Surface Temperature Change'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='Rotation Angle (deg)')

        axes.legend(loc='best', ncol=2)
        fname = '1D_3D_Surface_Temperature.jpg'
        plt.savefig(fname)

        plt.show()


    def temperature_comparison(self):
        '''

        对比不同初始温度下温度变化趋势
        ---

        x 轴为自转周期
        y 轴为温度, 单位为开尔文
        ---

        '''
        uh1_inter = np.zeros((1008, 282), dtype=np.float_)
        uh2_inter = np.zeros((1008, 282), dtype=np.float_)
        for i in range(1008):
            fname1 = '5_1000_250K/test' + str(i*1800).zfill(10) + '.vtu'
            nodedata1 = vtkReader.vtk_read_data(fname1, ['uh'])
            uh1 = nodedata1['uh']
            uh1_inter[i] = uh1[-283:-1].reshape(-1)

            fname2 = '5_1000_300K/test' + str(i*1800).zfill(10) + '.vtu'
            nodedata2 = vtkReader.vtk_read_data(fname2, ['uh'])
            uh2 = nodedata2['uh']
            uh2_inter[i] = uh2[-283:-1].reshape(-1)

        fig, axes = plt.subplots()
        x = np.arange(0, 1008, dtype=np.float_)

        label0 = '250K maximum'
        axes.plot(x, uh1_inter.max(axis=1), label=label0)

        label1 = '250K mean'
        axes.plot(x, uh1_inter.mean(axis=1), label=label1)

        label2 = '250K meidan'
        axes.plot(x, np.median(uh1_inter, axis=1), label=label2)
    
        label3 = '300K maximum'
        axes.plot(x, uh2_inter.max(axis=1), label=label3)

        label4 = '300K mean'
        axes.plot(x, uh2_inter.mean(axis=1), label=label4)

        label5 = '300K meidan'
        axes.plot(x, np.median(uh2_inter, axis=1), label=label5)
    
        title = 'Comparison of different initial Temperature'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='period')

        axes.legend()
        fname = '5_temperature_comparison.jpg'
        plt.savefig(fname)

        plt.show()

    def init_mesh(self):
        print("Generate the init mesh!...")

        fname = 'initial/file560.vtu'
        data = meshio.read(fname)
        node = data.points # 无量纲数值
        cell = data.cells[0][1]
        print("number of nodes of surface mesh:", len(node))
        print("number of cells of surface mesh:", len(cell))

        l = np.sqrt(0.02/(1400*1200*4*np.pi)) # 趋肤深度 l=(kappa/(rho*c*omega))^(1/2)
        h = self.h
        nh = self.nh
        H = self.H
        self.NN = len(node)
        self.NC = len(cell)

        node = node - np.mean(node, axis=0) # 以质心作为原点
        node *= H # H 小行星的规模
        node /=l # 无量纲化处理

        h /=nh # 单个三棱柱的高度
        h /= l # 无量纲化处理

        mesh = LagrangeTriangleMesh(node, cell)
        mesh = LagrangeWedgeMesh(mesh, h, nh)

        print("finish mesh generation!")
        return mesh
    
    def surface_vtu(self):
        '''
        绘制选取层的 vtu 文件

        '''
        mesh = self.init_mesh()
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        l = np.sqrt(0.02/(1400*1200*4*np.pi)) # 趋肤深度 l=(kappa/(rho*c*omega))^(1/2)
        
        sd = np.zeros((self.NN, 3), dtype=np.float_)
#        sd0 = [-1.57735, -0.867157, 0]
        sd0 = [-1.76812, -0.337289, 0]
        sd = np.vstack((sd, sd0))

        uh_all = self.uh_all
        layer = self.layer

        for i in range(len(layer)):
            node0 = node[self.NN*layer[i]:self.NN*(layer[i]+1), :]
            cell0 = cell[:self.NC, ::2]
            
            mesh0 = LagrangeTriangleMesh(node0, cell0)
#            mesh0.nodedata['uh'] = uh_all[-1, layer[i], :]
            
            uh0 = uh_all[-1, layer[i], :]
            T = np.zeros(len(uh0)+1, dtype=np.float64)
            T[:-1] = uh0
            mesh0.nodedata['uh'] = T
            
            mesh0.nodedata['sd'] = sd

            mesh0.meshdata['p'] = -sd[-1]*self.H*1.3/(l*1.8)

            fname = '50_1000/surface/surface_depth' + str(i) + '.vtu'
            mesh0.to_vtk(fname=fname) 
            print('The vtu of', layer[i], 'have been completed')
        print('All vtu have been completed')
  
Aspect = Aspect()
Aspect.angle_curve_2()
#Aspect.temperature_comparison()
#Aspect.draw_figure() # 绘制温度变化曲线
#Aspect.surface_vtu() # 生成选取层的 vtu 文件 

#fname = sys.argv[1]
#vtkReader.meshio_read(fname, LagrangeTriangleMesh)
#ndata = vtkReader.vtk_read_data(fname, ['uh']) 
#print(ndata)
#print(cdata)


