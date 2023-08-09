import numpy as np
import ipdb

"""

http://victorsndvg.github.io/FEconv/formats/mphtxt.xhtml
"""

class MPHTxtFileReader:
    """ 
    @brief 该类负责处理来自 Abaqus .inp 文件的数据
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

    def parse_header_data(self):
        """
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


    def parse_vertices_data(self):
        """
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
        """



    def print(self):
        """
        """
        print("Version: ", self.version)
        print("Tags:\n", self.tags)
        print("Types: \n", self.types)
        print("Mesh:\n", self.mesh)



if __name__ == "__main__":
    reader = MPHTxtFileReader("/home/why/Downloads/E_gnd_L2_msh.mphtxt")
    reader.parse()
    reader.print()

