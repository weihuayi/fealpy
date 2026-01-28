import numpy as np
from typing import Tuple

from fealpy.mesh import EdgeMesh
import matplotlib.pyplot as plt


class Bar942:
    """生成942杆三维桁架网格.
    
    这是一个大型空间桁架结构,包含244个节点和942个杆单元.
    """
    
    def build_truss_3d(self, 
                       d1: float = 2135,
                       d2: float = 5335,
                       d3: float = 7470,
                       d4: float = 9605,
                       r2: float = 4265,
                       r3: float = 6400,
                       r4: float = 8535,
                       l3: float = 43890,
                       l2: float = None,
                       l1: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            d1 : float, optional
                第一层半宽, 默认 4270/2
            d2 : float, optional
                第二层宽度, 默认 5335
            d3 : float, optional
                第三层宽度, 默认 7470
            d4 : float, optional
                第四层宽度, 默认 9605
            r2 : float, optional
                第二层半径, 默认 4265
            r3 : float, optional
                第三层半径, 默认 6400
            r4 : float, optional
                第四层半径, 默认 8535
            l3 : float, optional
                第三段高度, 默认 43890
            l2 : float, optional
                第二段高度, 默认 l3 + 29260
            l1 : float, optional
                第一段高度, 默认 l2 + 21950
        
        Returns
        -------
        nodes : np.ndarray, shape (244, 3)
            节点坐标数组,每行表示一个节点的 (x, y, z) 坐标
        cells : np.ndarray, shape (942, 2)
            单元连接数组,每行表示一个杆单元的两个端点节点编号(从0开始)
        """
        if l2 is None:
            l2 = l3 + 29260
        if l1 is None:
            l1 = l2 + 21950
        
        dl3 = 43890 / 12
        dl2 = 29260 / 8
        dl1 = 21950 / 6
        
        # 初始化节点和单元数组
        nodes = np.zeros((244, 3))
        
        # 定义各层坐标
        coordinate2 = np.array([
            [d2, r2, 0, -r2, -d2, -r2, 0, r2],
            [0, r2, d2, r2, 0, -r2, -d2, -r2]
        ])
        
        coordinate3 = np.array([
            [d3, r3, d1, -d1, -r3, -d3, -d3, -r3, -d1, d1, r3, d3],
            [d1, r3, d3, d3, r3, d1, -d1, -r3, -d3, -d3, -r3, -d1]
        ])
        
        coordinate4 = np.array([
            [d4, r4, d1, -d1, -r4, -d4, -d4, -r4, -d1, d1, r4, d4],
            [d1, r4, d4, d4, r4, d1, -d1, -r4, -d4, -d4, -r4, -d1]
        ])
        
        # 生成节点坐标
        # 第1-24号节点(6层,每层4个节点)
        for i in range(6):
            x = np.array([d1, -d1, -d1, d1])
            y = np.array([d1, d1, -d1, -d1])
            z = np.full(4, l1 - i * dl1)
            nodes[4*i:4*i+4, :] = np.column_stack([x, y, z])
        
        # 第25-88号节点(8层,每层8个节点)
        for i in range(8):
            x = coordinate2[0, :]
            y = coordinate2[1, :]
            z = np.full(8, l2 - i * dl2)
            nodes[24+8*i:24+8*i+8, :] = np.column_stack([x, y, z])
        
        # 第89-232号节点(12层,每层12个节点)
        for i in range(12):
            x = coordinate3[0, :]
            y = coordinate3[1, :]
            z = np.full(12, l3 - i * dl3)
            nodes[88+12*i:88+12*i+12, :] = np.column_stack([x, y, z])
        
        # 第233-244号节点(底层12个节点)
        x = coordinate4[0, :]
        y = coordinate4[1, :]
        z = np.zeros(12)
        nodes[232:244, :] = np.column_stack([x, y, z])
        
        # 生成单元连接(MATLAB索引从1开始,转换为Python的从0开始)
        ele_list = []
        
        # ele0: 2个单元
        ele0 = np.array([[0, 2], [1, 3]])
        ele_list.append(ele0)
        
        # ele1: 24个单元
        ele1 = []
        for i in range(1, 7):
            ele1.extend([
                [4*i-4, 4*i-3],
                [4*i-3, 4*i-2],
                [4*i-2, 4*i-1],
                [4*i-1, 4*i-4]
            ])
        ele1 = np.array(ele1)
        ele_list.append(ele1)
        
        # ele2: 60个单元 和 ele3: 20个单元
        ele2 = []
        ele3 = []
        for i in range(1, 7):
            if i == 6:
                # 特殊情况: ele3
                ele3.extend([
                    [20, 31], [20, 24], [20, 25], [20, 26], [20, 27],
                    [21, 25], [21, 26], [21, 27], [21, 28], [21, 29],
                    [22, 27], [22, 28], [22, 29], [22, 30], [22, 31],
                    [23, 29], [23, 30], [23, 31], [23, 24], [23, 25]
                ])
            else:
                # ele2
                ele2.extend([
                    [4*(i-1), 4*i+3], [4*(i-1), 4*i], [4*(i-1), 4*i+1]
                ])
                for j in range(2, 4):
                    ele2.extend([
                        [4*(i-1)+j-1, 4*i+j-2],
                        [4*(i-1)+j-1, 4*i+j-1],
                        [4*(i-1)+j-1, 4*i+j]
                    ])
                ele2.extend([
                    [4*(i-1)+3, 4*i+2], [4*(i-1)+3, 4*i+3], [4*(i-1)+3, 4*i]
                ])
        
        ele2 = np.array(ele2)
        ele3 = np.array(ele3)
        ele_list.extend([ele2, ele3])
        
        # ele4: 64个单元 和 ele5: 168个单元 和 ele6: 28个单元
        ele4 = []
        ele5 = []
        ele6 = []
        for i in range(1, 9):
            # ele4: 水平环形单元
            for j in range(8):
                ele4.append([24 + 8*i - 8 + j, 24 + 8*i - 8 + (j+1)%8])
            
            if i == 8:
                # ele6: 特殊连接
                ele6.extend([
                    [80, 98], [80, 99], [80, 88], [80, 89]
                ])
                for j in range(1, 4):
                    ele6.extend([
                        [80+2*j, 87+3*j],
                        [80+2*j, 87+3*j+1],
                        [80+2*j, 87+3*j+2],
                        [80+2*j, 87+3*j+3]
                    ])
                for j in range(1, 5):
                    ele6.extend([
                        [79+2*j, 88+3*(j-1)],
                        [79+2*j, 88+3*(j-1)+1],
                        [79+2*j, 88+3*(j-1)+2]
                    ])
            else:
                # ele5: 斜向连接
                ele5.extend([
                    [24+8*(i-1), 24+8*i+7],
                    [24+8*(i-1), 24+8*i],
                    [24+8*(i-1), 24+8*i+1]
                ])
                for j in range(2, 8):
                    ele5.extend([
                        [24+8*(i-1)+j-1, 24+8*i+j-2],
                        [24+8*(i-1)+j-1, 24+8*i+j-1],
                        [24+8*(i-1)+j-1, 24+8*i+j]
                    ])
                ele5.extend([
                    [24+8*(i-1)+7, 24+8*i+6],
                    [24+8*(i-1)+7, 24+8*i+7],
                    [24+8*(i-1)+7, 24+8*i]
                ])
        
        ele4 = np.array(ele4)
        ele5 = np.array(ele5)
        ele6 = np.array(ele6)
        ele_list.extend([ele4, ele5, ele6])
        
        # ele7: 144个单元 和 ele8: 396个单元 和 ele9: 36个单元
        ele7 = []
        ele8 = []
        ele9 = []
        for i in range(1, 13):
            # ele7: 水平环形单元
            for j in range(12):
                ele7.append([88 + 12*i - 13 + j, 88 + 12*i - 13 + (j+1)%12])
            
            if i == 12:
                # ele9: 特殊连接到底部
                ele9.extend([
                    [232, 231], [232, 220], [232, 221]
                ])
                for j in range(2, 12):
                    ele9.extend([
                        [231+j, 219+j],
                        [231+j, 219+j+1],
                        [231+j, 219+j+2]
                    ])
                ele9.extend([
                    [243, 230], [243, 231], [243, 220]
                ])
            else:
                # ele8: 斜向连接
                ele8.extend([
                    [88+12*(i-1), 88+12*i+11],
                    [88+12*(i-1), 88+12*i],
                    [88+12*(i-1), 88+12*i+1]
                ])
                for j in range(2, 12):
                    ele8.extend([
                        [88+12*(i-1)+j-1, 88+12*i+j-2],
                        [88+12*(i-1)+j-1, 88+12*i+j-1],
                        [88+12*(i-1)+j-1, 88+12*i+j]
                    ])
                ele8.extend([
                    [88+12*(i-1)+11, 88+12*i+10],
                    [88+12*(i-1)+11, 88+12*i+11],
                    [88+12*(i-1)+11, 88+12*i]
                ])
        
        ele7 = np.array(ele7)
        ele8 = np.array(ele8)
        ele9 = np.array(ele9)
        ele_list.extend([ele7, ele8, ele9])
        
        # 合并所有单元
        cells = np.vstack(ele_list)
        
        return nodes, cells
    
# bar = Bar942()
# nodes, cells = bar.build_truss_3d()
# mesh = EdgeMesh(nodes, cells)

# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d') 
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True, color='g', markersize=8, fontsize=8, fontcolor='g')
# mesh.find_cell(axes, showindex=True, color='b', markersize=16, fontsize=20, fontcolor='b')
# plt.show()