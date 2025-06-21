import gmsh
import numpy as np
import math
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh import TriangleMesh
class FuelRodMesher:
    def __init__(self, R1, R2, L, w, h, l=None, p=None, meshtype='segmented',modeltype='2D'):
        self.R1 = R1
        self.R2 = R2
        self.L = L
        self.w = w
        self.h = h
        self.l = l
        self.p = p
        self.meshtype = meshtype
        self.modeltype = modeltype
        self.get_mesh = None
        self.cell_tags = None
        self.node_tags = None
        self.get_mesh,self.cell_tags,self.node_tags =self.generate_mesh()

    def generate_mesh(self):
        gmsh.initialize()
        gmsh.model.add("fuel_rod")
        Lc1 = self.h
        Lc2 = self.h / 2.5
        h = self.h
        factory = gmsh.model.geo
        R1, R2, L, w = self.R1, self.R2, self.L, self.w
        l, p = self.l, self.p
        meshtype,modeltype = self.meshtype,self.modeltype
        # 外圈点
        factory.addPoint( -R1 -R2 -L, 0 , 0 , Lc2 , 1 )#圆心1
        factory.addPoint( -R1 -R2 -L, -R1 , 0 , Lc2 , 2)
        factory.addPoint( -R1 -R2 , -R1 , 0 , Lc2 , 3)
        factory.addPoint( -R1 -R2 , -R1 -R2 , 0 , Lc2 , 4)#圆心2
        factory.addPoint( -R1 , -R1 -R2 , 0 , Lc2 , 5)
        factory.addPoint( -R1 , -R1 -R2 -L , 0 , Lc2 , 6)
        factory.addPoint( 0 , -R1 -R2 -L , 0 , Lc2 , 7)#圆心3
        factory.addPoint( R1 , -R1 -R2 -L , 0 , Lc2 , 8)
        factory.addPoint( R1 , -R1 -R2 , 0 , Lc2 , 9)
        factory.addPoint( R1 +R2 , -R1 -R2 , 0, Lc2 , 10)#圆心4
        factory.addPoint( R1 +R2 , -R1 , 0 , Lc2 , 11) 
        factory.addPoint( R1 +R2 +L , -R1 , 0 , Lc2 , 12)
        factory.addPoint( R1 +R2 +L , 0 , 0 , Lc2 , 13)#圆心5
        factory.addPoint( R1 +R2 +L , R1 , 0 , Lc2 , 14)
        factory.addPoint( R1 +R2 , R1 , 0 , Lc2 , 15)
        factory.addPoint( R1 +R2 , R1 +R2 , 0 , Lc2 , 16)#圆心6
        factory.addPoint( R1 , R1 +R2 , 0 , Lc2 , 17)
        factory.addPoint( R1 , R1 +R2 +L , 0 , Lc2 , 18)
        factory.addPoint( 0 , R1 +R2 +L , 0 , Lc2 , 19)#圆心7
        factory.addPoint( -R1 , R1 +R2 +L , 0 , Lc2 , 20)
        factory.addPoint( -R1 , R1 +R2 , 0 , Lc2 , 21)
        factory.addPoint( -R1 -R2 , R1 +R2 , 0 , Lc2 , 22)#圆心8
        factory.addPoint( -R1 -R2 , R1 , 0 , Lc2 , 23)
        factory.addPoint( -R1 -R2 -L , R1 , 0 , Lc2 , 24)
      
        # 外圈线
        line_list_out = []
        for i in range(8):
            if i == 0:
                factory.addCircleArc(24 , 3*i+1 , 3*i+2, 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            else:
                factory.addCircleArc(3*i , 3*i+1 , 3*i+2 , 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            line_list_out.append(2*i+1)
            line_list_out.append(2*(i+1))
        factory.addCurveLoop(line_list_out, 17)
        # 内圈点
        factory.addPoint( -R1 -R2 -L, -R1 +w , 0 , Lc1 , 25)
        factory.addPoint( -R1 -R2 , -R1 +w , 0 , Lc1 , 26)
        factory.addPoint( -R1 +w , -R1 -R2 , 0 , Lc1 , 27)
        factory.addPoint( -R1 +w , -R1 -R2 -L , 0 , Lc1 , 28)
        factory.addPoint( R1 -w , -R1 -R2 -L , 0 , Lc1 , 29)
        factory.addPoint( R1 -w , -R1 -R2 , 0 , Lc1 , 30)
        factory.addPoint( R1 +R2 , -R1 +w , 0 , Lc1 , 31) 
        factory.addPoint( R1 +R2 +L , -R1 +w , 0 , Lc1 , 32)
        factory.addPoint( R1 +R2 +L , R1 -w , 0 , Lc1 , 33)
        factory.addPoint( R1 +R2 , R1 -w , 0 , Lc1 , 34)
        factory.addPoint( R1 -w , R1 +R2 , 0 , Lc1 , 35)
        factory.addPoint( R1 -w , R1 +R2 +L , 0 , Lc1 , 36)
        factory.addPoint( -R1 +w , R1 +R2 +L , 0 , Lc1 , 37)
        factory.addPoint( -R1 +w , R1 +R2 , 0 , Lc1 , 38)
        factory.addPoint( -R1 -R2 , R1 -w, 0 , Lc1 , 39)
        factory.addPoint( -R1 -R2 -L , R1 -w, 0 , Lc1 , 40)

        # 内圈线
        line_list_in = []
        for j in range(8):
            if j == 0:
                factory.addCircleArc(40 , 3*j+1 , 25+2*j , 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            else:
                factory.addCircleArc(24+2*j , 3*j+1 , 25+2*j, 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            line_list_in.append(18+2*j)
            line_list_in.append(19+2*j)
        factory.addCurveLoop(line_list_in, 34)
        # 燃料面
        factory.addPlaneSurface([34], 35)
        # 包壳面
        factory.addPlaneSurface([17, 34], 36)
        factory.synchronize()

        if modeltype == '2D':
            if meshtype == '2D_refine':
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmmf = gmsh.model.mesh.field
                gmmf.add("Distance",1)
                gmmf.setNumbers(1, "CurvesList",line_list_in)
                gmmf.setNumber(1,"Sampling",1000)
                gmmf.add("Threshold",2)
                gmmf.setNumber(2, "InField", 1)
                gmmf.setNumber(2, "SizeMin", Lc1/5)
                gmmf.setNumber(2, "SizeMax", Lc1)
                gmmf.setNumber(2, "DistMin", w)
                gmmf.setNumber(2, "DistMax", 2*w)
                gmmf.setAsBackgroundMesh(2)
            gmsh.model.mesh.generate(2)
            # 获取节点信息
            node_coords = gmsh.model.mesh.getNodes()[1]
            node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)[:, 0:2].copy()
            # 获取三角形单元信息
            cell_type = 2  # 三角形单元的类型编号为 2
            cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)
            cell = np.array(cell_connectivity, dtype=np.int_).reshape(-1, 3) -1
            # 获得正确的节点标签
            node_tags = np.unique(cell_connectivity)
            NN = len(node)
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            # 去除未三角化的点
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]
            print(f"Number of nodes: {node.shape[0]}")
            print(f"Number of cells: {cell.shape[0]}")
            return TriangleMesh(node,cell),cell_tags,node_tags

        elif modeltype == '3D':
            # 旋转拉伸次数
            N = math.ceil((2*l) / p)
            # 每次旋转角度
            angle = ((2*l) / p * math.pi) / N
            # 每次拉伸段数
            nsection = math.ceil(l / (N * 0.42 * h))
            ov1 = [[0, 35]]
            ov2 = [[0, 36]]
            if meshtype == 'unsegmented':
                for i in range(N):
                    ov1 = factory.twist([(2, ov1[0][1])], 0, 0, 0, 0, 0, l / N, 0, 0, 1, angle)
                    ov2 = factory.twist([(2, ov2[0][1])], 0, 0, 0, 0, 0, l / N, 0, 0, 1, angle)
            else:
                for i in range(N):
                    ov1 = factory.twist([(2, ov1[0][1])], 0, 0, 0, 0, 0, l / N, 0, 0, 1, angle, [nsection], [], False)
                    ov2 = factory.twist([(2, ov2[0][1])], 0, 0, 0, 0, 0, l / N, 0, 0, 1, angle, [nsection], [], False)
            
            factory.synchronize()
            gmsh.model.mesh.generate(3)
            # 获取节点信息
            node_coords = gmsh.model.mesh.getNodes()[1]
            node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

            # 获取四面体单元信息
            tetrahedron_type = 4  
            tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
            cell = np.array(tetrahedron_connectivity, dtype=np.int_).reshape(-1, 4) -1

            # 获得正确的节点标签
            node_tags = np.unique(tetrahedron_connectivity)

            NN = len(node)
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            # 去除未三角化的点
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]
            print(f"Number of nodes: {node.shape[0]}")
            print(f"Number of cells: {cell.shape[0]}")
            return TetrahedronMesh(node, cell), tetrahedron_tags,node_tags
        else:
            raise ValueError("Invalid model type. Must be '2D' or '3D'.")

    
    def get_2D_cnidx_bdnidx(self):
        """
        获得燃料和包壳共享节点编号和外边界节点编号
        """
        if self.modeltype == '3D':
            raise ValueError("Invalid model type. Must be '2D'")
        node_tags = self.node_tags
        NN = len(node_tags)
        # 标签到节点的映射
        tag2nidx = np.zeros(2*NN,dtype=np.int_)
        tag2nidx[node_tags] = np.arange(NN)
        dimtags1 = gmsh.model.getEntities(1)
        # 共享节点编号
        cnidx = []
        # 边界节点编号
        bdnidx = []
        for dim, tag in dimtags1:
            ntags = gmsh.model.mesh.get_elements(dim,tag)[2][0]
            idx = tag2nidx[ntags]
            if tag < 17:
                bdnidx.extend(idx)
            else:
                cnidx.extend(idx)
        cnidx = np.unique(cnidx)
        bdnidx = np.unique(bdnidx)
        return cnidx,bdnidx
    
    def get_2D_fcidx_cacidx(self):
        """
        获得燃料单元编号和包壳单元编号
        """
        if self.modeltype == '3D':
            raise ValueError("Invalid model type. Must be '2D'.")
        cell_tags = self.cell_tags
        # 标签到单元的映射
        NC = len(cell_tags)
        tag2cidx = np.zeros(2*NC,dtype=np.int_)
        tag2cidx[cell_tags] = np.arange(NC)
        # 内部节点编号和包壳节点编号
        inctags = gmsh.model.mesh.get_elements(2,35)[1][0]
        cactags = gmsh.model.mesh.get_elements(2,36)[1][0]
        # 内部单元索引  
        fcidx = tag2cidx[inctags]
        # 外部单元索引  
        cacidx = tag2cidx[cactags]
        return fcidx,cacidx
    
    def get_3D_cnidx_bdnidx(self):
        """
        获得燃料和包壳共享节点编号和外边界节点编号
        """
        if self.modeltype == '2D':
            raise ValueError("Invalid model type. Must be '3D'.")
        node_tags = self.node_tags
        NN = len(node_tags)
        tag2nidx = np.zeros(2*NN,dtype=np.int_)
        tag2nidx[node_tags] = np.arange(NN)
        # 二维面片
        dimtags2 = gmsh.model.getEntities(2)
        # 删去前后和中间插入的面片，剩下两边包裹的
        del dimtags2[0],dimtags2[0],dimtags2[16:len(dimtags2):17]
        # 内边界面片集合
        in_dimtags2 = sum([dimtags2[i:i+16] for i in range(0,len(dimtags2),32)],[])
        # 存储共享边界点
        cntags = []
        for dim,tag in in_dimtags2:
            idx = gmsh.model.mesh.get_elements(dim,tag)[2][0]
            cntags.extend(idx)
        cntags = np.unique(cntags)
        cnidx = tag2nidx[cntags]
        # 外边界面片集合
        out_dimtags2 = sum([dimtags2[i:i+16] for i in range(16,len(dimtags2),32)],[])
        # 存储外边界点
        bdntags = []
        for dim,tag in out_dimtags2:
            idx = gmsh.model.mesh.get_elements(dim,tag)[2][0]
            bdntags.extend(idx)
        bdntags = np.unique(bdntags)
        bdnidx = tag2nidx[bdntags]
        return cnidx , bdnidx
    
    def get_3D_fcidx_cacidx(self):
        """
        获得燃料单元编号和包壳单元编号
        """
        if self.modeltype == '2D':
            raise ValueError("Invalid model type. Must be '3D'.")
        cell_tags = self.cell_tags
        NC = len(cell_tags)
        tag2cidx = np.zeros(3*NC,dtype=np.int_)
        tag2cidx[cell_tags] = np.arange(NC)
        dimtags3 = gmsh.model.getEntities(3)
        # 存储燃料节点编号
        fctags = []
        # 存储包壳节点编号
        cactags = []
        for dim,tag in dimtags3:
            ctags = gmsh.model.mesh.get_elements(dim,tag)[1][0]      
            if tag%2 == 1:
                fctags.extend(ctags)
            else:
                cactags.extend(ctags)
        # 燃料单元
        fcidx = tag2cidx[fctags]
        # 包壳单元
        cacidx = tag2cidx[cactags]
        return fcidx,cacidx
    
    def finalize(self):
        """
        清理 Gmsh 资源
        """
        gmsh.finalize()

    def run(self):
        """
        Gmsh 画图
        """
        gmsh.fltk.run()