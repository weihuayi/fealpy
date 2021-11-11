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
        self.H = 500 # 小行星规模

        self.Find_node = Find_node() # 寻找特殊角度处的节点值
        self.special_nodes_flag = self.Find_node.read_mesh()
        
        self.angle = np.arange(0, 378, 18) # 绘制随旋转角度的变化情况
        self.layer = np.arange(0, 101, 20) # 绘制 5 层的温度变化情况
        self.uh_all = self.read_data() # 一旋转周期内分层后的温度值
        self.depth = np.arange(0, 0.505, 0.005) # 每层的深度

    def read_data(self):
        '''

        提取一个自转周期内的 21 张 vtu 文件中的温度数据, 并分层存储
        
        '''
        uh_all = np.zeros((21, self.nh+1, 4532), dtype=np.float_)
        for i in range(21):
            fname = 'high/test' + str(i).zfill(2) + '.vtu'
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
            uh0 = np.zeros(21, dtype=np.float_)
            uh0[:] = uh_all[:, layer[i], index]

            label = 'Depth' + str(layer[i]*self.h/self.nh)
            axes.plot(angle, uh0, label=label)
        
        title = 'Temperature Curve of 0 Longitude and ' + str(degree) + ' Latitude'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='Rotation Angle (deg)')

        axes.legend()
        fname0 = 'high/Latitude' + str(degree) + '.jpg'
        plt.savefig(fname0)

#        plt.show()

    def depth_curve(self, index, degree):
        '''

        绘制某点不同层的温度变化曲线
        ---

        每个点绘制选取 5 层, 绘制在同一图中
        x 轴为自 0 相位开始旋转的角度
        y 轴为温度, 单位为开尔文
        ---

        param[in] index 需要绘制的节点的索引
        param[in] degree 需要绘制的节点的纬度
        
        '''
        uh_all = self.uh_all
        depth = self.depth

        fig, axes = plt.subplots()
        for i in range(7):
            uh0 = np.zeros(101, dtype=np.float_)
            uh0[:] = uh_all[-1, :, index[i]]

            label = 'Latitude' + str(degree[i])
            axes.plot(depth, uh0, label=label)
        
        title = 'Temperature Curve at different depths'
        axes.set(title=title, ylabel='Temperature (K)', xlabel='Depth (m)')

        axes.legend()
        fname = 'high_depth.jpg'
        plt.savefig(fname)

#        plt.show()

    def draw_figure(self):
        '''

        绘制 7 个特殊点处的温度变化曲线
        
        '''
        special_degrees = np.array([0, 30, -30, 60, -60, 90, -90])
        
        self.depth_curve(self.special_nodes_flag, special_degrees)
        
#        for j in range(len(special_degrees)):
#            self.angle_curve(self.special_nodes_flag[j], special_degrees[j])
#            print('Figure of', special_degrees[j], 'degrees have been drawn')
        print('All figures have been drawn')

    def init_mesh(self):
        print("Generate the init mesh!...")

        fname = 'initial/file1.vtu'
        data = meshio.read(fname)
        node = data.points # 无量纲数值
        cell = data.cells[0][1]
        print("number of nodes of surface mesh:", len(node))
        print("number of cells of surface mesh:", len(cell))

        l = np.sqrt(0.02/(1400*1200*4*np.pi)) # 趋肤深度 l=(kappa/(rho*c*omega))^(1/2)
        h = self.h
        nh = self.nh
        H = self.H

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
        l = np.sqrt(0.02/(1400*1200*4*np.pi)) # 趋肤深度 l=(kappa/(rho*c*omega))^(1/2)
        
        sd = np.zeros((4532, 3), dtype=np.float_)
        sd0 = [-1.57735, -0.867157, 0]
        sd = np.vstack((sd, sd0))

        uh_all = self.uh_all
        layer = self.layer
        mesh = self.init_mesh()
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        for i in range(len(layer)):
            node0 = node[4532*layer[i]:4532*(layer[i]+1), :]
            cell0 = cell[:9060, ::2]
            
            mesh0 = LagrangeTriangleMesh(node0, cell0)
#            mesh0.nodedata['uh'] = uh_all[-1, layer[i], :]
            
            uh0 = uh_all[-1, layer[i], :]
            T = np.zeros(len(uh0)+1, dtype=np.float64)
            T[:-1] = uh0
            mesh0.nodedata['uh'] = T
            
            mesh0.nodedata['sd'] = sd

            mesh0.meshdata['p'] = -sd[-1]*500*1.3/(l*1.8)

            fname = 'high/surface/surface_depth' + str(i) + '.vtu'
            mesh0.to_vtk(fname=fname) 
            print('The vtu of', layer[i], 'have been completed')
        print('All vtu have been completed')
  
Aspect = Aspect()
#Aspect.draw_figure() # 绘制温度变化曲线
Aspect.surface_vtu() # 生成选取层的 vtu 文件 

#fname = sys.argv[1]
#vtkReader.meshio_read(fname, LagrangeTriangleMesh)
#ndata = vtkReader.vtk_read_data(fname, ['uh']) 
#print(ndata)
#print(cdata)


