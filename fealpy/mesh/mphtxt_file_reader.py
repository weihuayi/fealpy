import numpy as np

"""
http://victorsndvg.github.io/FEconv/formats/mphtxt.xhtml
"""

class MPHTxtFileReader:
    """ 
    @brief 该类负责处理来自 Comsol .mphtxt 文件的数据
    """
    def __init__(self, fname):
        with open(fname, 'r') as f:
            contents = f.read()
        self.contents = contents.split('\n')
        self.cline = 0
        self.version = None
        self.tags = [] 
        self.types = []
        self.mesh = {}
        self.geometry = {}

    def get_data_line(self):
        """
        @brief 获取下一个数据行
        """
        if self.cline < len(self.contents):
            line = self.contents[self.cline].strip()
            while (not line) or line.startswith('#') :
                self.cline += 1
                if self.cline < len(self.contents):
                    line = self.contents[self.cline].strip()
                else:
                    return None
            return line
        else:
            return None

    def parse(self):
        """
        @brief 解析文件中的数据
        """
        self.parse_header_data()
        self.cline += 1
        self.parse_mesh_data()
        self.cline +=1
        self.parse_geometry_data()

    def parse_header_data(self):
        """
        @brief 解析文件开头的数据
        """
        line = self.get_data_line()
        self.version = line.replace(' ', '.')

        self.cline += 1
        line = self.get_data_line()
        n = int(line.split('#')[0]) # number of tags
        for i in range(n):
            self.cline += 1
            line = self.get_data_line()
            ws = line.split(' ')
            self.tags.append(ws[1][0:int(ws[0])])

        self.cline += 1
        line = self.get_data_line()
        n = int(line.split('#')[0]) # number of types
        for i in range(n):
            self.cline += 1
            line = self.get_data_line()
            ws = line.split(' ')
            self.types.append(ws[1][0:int(ws[0])])

    def parse_mesh_data(self):
        """
        @brief 解析网格数据
        """
        line = self.get_data_line()
        self.cline += 1 
        line = self.get_data_line()
        ws = line.split(' ')
        s = ws[1][0:int(ws[0])]
        assert s == "Mesh" # 必须是网格类

        self.cline += 1
        line = self.get_data_line()
        self.mesh['version'] = line.split('#')[0] # version

        self.cline += 1
        line = self.get_data_line()
        self.mesh['sdim'] = int(line.split('#')[0])

        self.cline += 1
        self.parse_vertices_data()

        self.cline +=1
        self.parse_element_data()

    def parse_vertices_data(self):
        """
        @brief 解析节点数据
        """
        line = self.get_data_line()
        NV = int(line.split('#')[0])

        self.cline += 1
        line = self.get_data_line()
        lidx = int(line.split('#')[0]) # lowest mesh vertex index

        self.cline += 1
        line = self.get_data_line()

        vertices = np.array([list(map(float, s.split())) for s in
            self.contents[self.cline:self.cline + NV]])
        self.cline += NV

        self.mesh['vertices'] = vertices

    def parse_element_data(self):
        """
        @brief 解析单元数据
        """
        line = self.get_data_line()
        self.mesh['element'] = {}
        
        NET = int(line.split('#')[0])
        self.mesh['element']['NET'] = NET 
        element_types = np.arange(NET)
        
        self.cline += 1
        line = self.get_data_line()
        ws = line.split(' ')
        s = ws[1][0:int(ws[0])] # type name
        self.mesh['element'][s]= {}

        self.cline += 1 
        line = self.get_data_line()
        NVE = int(line.split('#')[0]) # number of vertices per element
        
        self.cline += 1
        line = self.get_data_line()
        NE = int(line.split('#')[0]) # number of elements
        self.mesh['element'][s]['NE'] = NE
        self.cline += 1
        line = self.get_data_line()
        idx = [list((map(int,t.split()))) for t in
                self.contents[self.cline:self.cline + NE]] # Elements
        idx = tuple(item for sublist in idx for item in sublist) 
        self.mesh['element'][s]['idx'] = idx
        self.cline += NE

        self.cline += 1
        line = self.get_data_line()
        NGEI = int(line.split('#')[0]) # number of geometric entity indices
        
        self.cline +=1
        line = self.get_data_line()
        geo_idx = [list((map(int,t.split()))) for t in
                self.contents[self.cline:self.cline + NGEI]]# Geometric entity indices
        geo_idx = tuple(item for sublist in geo_idx for item in sublist)
        self.mesh['element'][s]['geo_idx']=geo_idx
        self.cline += NGEI 
        
        for i in range(1, NET):
            self.cline += 1
            line = self.get_data_line()
            ws = line.split(' ')
            s = ws[1][0:int(ws[0])] # type name
            self.mesh['element'][s]= {}
            
            self.cline += 1 
            line = self.get_data_line()
            NVE = int(line.split('#')[0]) # number of vertices per element
            
            self.cline += 1
            line = self.get_data_line()
            NE = int(line.split('#')[0]) # number of elements
            self.mesh['element'][s]['NE'] = NE

            self.cline += 1
            line = self.get_data_line()
            Element = np.array([list(map(int, t.split())) for t in
                self.contents[self.cline:self.cline + NE]]) #Element
            self.mesh['element'][s]['Element'] = Element
            self.cline += NE

            self.cline +=1
            line = self.get_data_line()
            NGEI = int(line.split('#')[0]) # number of geometric entity indices

            self.cline +=1
            line = self.get_data_line()
            geo_idx = [list((map(int,t.split()))) for t in
                    self.contents[self.cline:self.cline + NGEI]]# Geometric entity indices
            geo_idx = tuple(item for sublist in geo_idx for item in sublist)
            self.mesh['element'][s]['geo_idx']=geo_idx
            self.cline += NGEI            
    def parse_geometry_data(self):
        """
        @brief 解析几何信息数据
        """
    
        for i in range(1,len(self.tags)):
            s = self.tags[i]
            self.geometry[s] = {}
            line = self.get_data_line()
            self.cline +=1
            line = self.get_data_line()
            ws = line.split(' ')
            s1 = ws[1][0:int(ws[0])]
            assert s1 =="Selection" #必须是Selection类
            
            self.cline += 1
            line = self.get_data_line()
            self.geometry[s]['version'] = line.split('#')[0] # version
            
            self.cline += 1
            line = self.get_data_line()
            ws = line.split(' ')
            s2 = ws[1][0:int(ws[0])]
            self.geometry[s]['Label'] = ws[1][0:int(ws[0])] # label

            self.cline += 1
            line = self.get_data_line()
            ws = line.split(' ')
            s2 = ws[1][0:int(ws[0])]
            self.geometry[s]['meshtag'] = ws[1][0:int(ws[0])]# Geometry/mesh tag

            self.cline += 1
            line = self.get_data_line()
            self.geometry[s]['dimension'] = int(line.split('#')[0])# Dimension

            self.cline += 1
            line = self.get_data_line()
            NE = int(line.split('#')[0])
            self.geometry[s]['NE'] = NE # Number of geometry edge
            
            self.cline += 1
            line = self.get_data_line()
            idx = [list((map(int,t.split()))) for t in
                    self.contents[self.cline:self.cline+NE]] #Entities
            idx = tuple(item for sublist in idx for item in sublist)
            self.geometry[s]['entities'] = idx # The index of geometry edge
            self.cline += NE
        
    def print(self):
        """
        """
        print("Version: ", self.version)
        print("Tags:\n", self.tags)
        print("Types: \n", self.types)
        #print("Mesh:\n", self.mesh)
        print("Geometric entity indices of tri:\n",self.mesh['element']['tri']['geo_idx'])
        print("Number of TetElements:\n",self.mesh['element']['tet']['NE'])
        print("Element of TetMesh:\n",self.mesh['element']['tet']['Element'])
        print("Geometry Information:\n",self.geometry)


