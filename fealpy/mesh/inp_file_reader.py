import numpy as np

"""
Abaqus 有限元模型简介

两大类数据：
 + Model data
 + History data
   - 基于 step 的概念

三种类型的行:
 + Keyword line
 + Data line
 + Comment line

Assembly 模型的三级结构：
 + Part（部件）
   - 部件之间相互独立，并且有自己的坐标系，并不知道其它的部件
   - Solid Section 用来指定部件部分单元或全部单元的材料属性
 + Instance（部件的实例）
 + Assembly（组合体）
   - 在组合体模块里创建 Part 对应的 Instance
   - 组合体有一个全局坐标系，Instance 定义了 Part 从局部坐标系变换到全局坐标系的信息 
   - 可以在组合体中创建其它 Model 的实例
   - 可以创建独立、或不独立于原来 Part 的 Instance
     独立的 Instance，才可以 add partitions, create virtual topology and
     mesh the instance
   - 接触、载荷、边界条件通过 Interaction 和  Load 模块在 Assembly 模块中定义


 一个 Part 可以对应多个 Instance,
 比如一个机器设备可以有很多螺栓（bolt)，在建模时只需要一个 Part 来代表这个螺栓，
 在组合时，可以通过平移和旋转生成多个 Part Instance。


 *INSTANCE 
 *END INSTANCE
 https://abaqus-docs.mit.edu/2017/English/SIMACAEKEYRefMap/simakey-r-instance.htm

"""

class InpFileReader:
    """ 
    @brief 该类负责处理来自 Abaqus .inp 文件的数据
    """
    def __init__(self, fname):
        with open(fname, 'r') as f:
            contents = f.read()
        self.contents = contents.split('\n')
        self.cline = 0
        self.parts = {}
        self.materials = {}
        self.assembly = {}
        self.step = {}

    def parse(self):
        """
        @brief 解析文件中的数据
        """
        line = self.get_keyword_line() # 拿到下一个还没有处理的 keyword 行
        while line is not None:
            if line.startswith('*Part'):
                self.parse_part_data()
            elif line.startswith('*Assembly'):
                self.parse_assembly_data()
            elif line.startswith('*Step'):
                self.parse_step_data()
            elif line.startswith('*Material'):
                self.parse_material_data()

            self.cline += 1
            line = self.get_keyword_line() # 拿到下一个还没有处理的 keyword 行

    def parse_keyword_line(self, line):
        """
        @brief 解析一个 keyword line
        @param line 要解析的 keyword line
        @return 包含解析结果的字典
        """
        d = {}
        words  = line.split(',')
        d['keyword'] = words[0].strip()
        if len(words) > 1:
            for word in words[1:]:
                s = word.strip().split('=')
                d[s[0]] = s[1]
        return d


    def get_keyword_line(self):
        """
        @brief 获取还没有处理的 keyword  行
        @return 下一个未处理的 keyword 行，如果没有则返回 None
        """
        if self.cline < len(self.contents):
            line = self.contents[self.cline]
            while (not line.startswith('*')) or line.startswith('**'):
                self.cline += 1
                if self.cline < len(self.contents):
                    line = self.contents[self.cline]
                else:
                    return None
            return line
        else:
            return None

    def parse_part_data(self):
        """
        @brief 解析 part 数据
        """

        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        # print("part_data:", d)

        self.parts[d['name']] = {'node':None, 'elem':{}, 'nset':{}, 'elset':{},
                                 'orientation':{}, 'solid_section':{}, 'beam_section':{}}
        self.cline += 1
        line = self.get_keyword_line()

        while line is not None and not line.startswith('*End Part'):
            if line.startswith('*Node'):
                self.parse_node_data(self.parts, d['name'])
            elif line.startswith('*Element'):
                self.parse_elem_data(self.parts, d['name'])
            elif line.startswith('*Nset'):
                self.parse_nset_data(self.parts, d['name'])
            elif line.startswith('*Elset'):
                self.parse_elset_data(self.parts, d['name'])
            elif line.startswith('*Orientation'):
                self.parse_orientation_data(self.parts, d['name'])
            elif line.startswith('*Solid Section'):
                self.parse_solid_section_data(self.parts, d['name'])
            elif line.startswith('*Beam Section'):
                self.parse_beam_section_data(self.parts, d['name'])
            else:
                print("we pass the keyword line:" + line)
                self.cline += 1

            line = self.get_keyword_line() # 拿到下一个还没有处理的 keyword 行


    def parse_node_data(self, parts, name):
        """
        @brief 处理节点坐标数据
        @param parts 用于存储节点数据的字典
        @param name 当前部分的名称
        """
        i = 0
        node = []
        nmap = {}

        self.cline += 1
        line = self.contents[self.cline]
        while not line.startswith('*'):
            s = line.split(',')
            node.append((float(s[1]), float(s[2]), float(s[3])))
            nmap[int(s[0])] = i # 原始节点编号 1 被映射为目前节点编号 0，依次
            i += 1

            self.cline += 1
            line = self.contents[self.cline]

        parts[name]['node'] = (np.array(node, dtype=np.float64), nmap)
        # print("node:", parts[name]['node'])

    def parse_elem_data(self, parts, name):
        """
        @brief 解析单元数据
        @param parts 用于存储单元数据的字典
        @param name 当前部分的名称
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)

        i = 0
        elem = []
        emap = {}

        nmap = parts[name]['node'][1] # 原始节点编号 1 被映射为目前节点编号 0，依次

        self.cline += 1
        line = self.contents[self.cline]
        while not line.startswith('*'):
            ss = line.split(',')
            elem.append([nmap[int(s)] for s in ss[1:]])
            emap[int(ss[0])] = i # 原始单元编号 1 被映射为原始单元编号 0，依次
            i += 1
            self.cline += 1
            line = self.contents[self.cline]

        parts[name]['elem'][d['type']] = (np.array(elem, dtype=np.int_), emap)
        # print("elem:", parts[name]['elem'][d['type']])

    def parse_nset_data(self, parts, name):
        """
        @brief 解析节点集合数据，可以用于应用边界条件、载荷等

        @param parts 用于存储节点集合数据的字典
        @param name 当前部分的名称
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        # print("nset_data:", d)

        nmap = parts[name]['node'][1]
        idx = []

        self.cline += 1
        line = self.contents[self.cline]
        while not line.startswith('*'):
            ss = line.split(',')
            idx += [nmap[int(s)] for s in ss] # 原始节点集合编号 1 被映射为目前节点集合编号 0，依次
            self.cline += 1
            line = self.contents[self.cline]

        parts[name]['nset'][d['nset']] = (np.array(idx, dtype=np.int_), nmap)
        # print("nset:", parts[name]['nset'][d['nset']])


    def parse_elset_data(self, parts, name):
        """
        @brief 解析单元集合数据，可以用于应用边界条件、载荷等

        @param parts 用于存储单元集合数据的字典
        @param name 当前部分的名称

        @note 注意这里我们没有把原始单元的编号映射成当前连续的编号
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        # print("elset_d:", d)

        idx = []

        self.cline += 1
        line = self.contents[self.cline]
        # print("elset_line:", line)
        
        while not line.startswith('*'):
            ss = line.split(',')
            # print("elset_ss:", ss)
            idx += [int(s) for s in ss]
            self.cline += 1
            line = self.contents[self.cline]

        parts[name]['elset'][d['elset']] = np.array(idx, dtype=np.int_) 
        # print("elset:", parts[name]['elset'][d['elset']])


    def parse_orientation_data(self, parts, name):
        """
        @brief 解析 parts 的坐标系数据
        数值用于定义材料坐标系的方向
        '1., 0., 1.'定义主方向向量，沿着 x 轴的方向
        '0., 1., 0.'定义第二个向量，是在主方向向量所定义的平面内的向量，沿着 y 轴的方向
        '1., 0.' 定义主方向向量和第二个向量之间的角度，主方向向量和第二个向量之间的角度是 1 弧度
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)

        idx = []

        self.cline += 1
        line = self.contents[self.cline]
        # print("orientation_line:", line)

        while not line.startswith('*'):
            ss = line.split(',')
            # print("orientation_ss:", ss)
            idx += [float(s) for s in ss]
            self.cline += 1
            line = self.contents[self.cline]

        parts[name]['orientation'][d['name']] = np.array(idx, dtype=np.float64) 
        # print("orientation:", parts[name]['orientation'][d['name']])


    def parse_solid_section_data(self, parts, name):
        """
        @brief 解析 parts 的实体截面数据
        数值定义了一个实体截面，对应于元素集合 "Set-3"，材料为 "Material-1"
        '1.' 代表实体截面的厚度
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        # print("solid_section_d:", d)

        idx = []

        self.cline += 1
        line = self.contents[self.cline]
        # print("solid_section_data:", line)

        while not line.startswith('*'): 
            ss = line.split(',') 
            # print("solid_section_ss:", ss) 
            idx += [float(s) for s in ss if s]
            self.cline += 1

            line = self.contents[self.cline]

        solid_section_key = (d['elset'], d['material'])
        parts[name]['solid_section'][solid_section_key] = np.array(idx, dtype=np.float64) 
        # print("solid_section:", parts[name]['solid_section'][solid_section_key])


    def parse_beam_section_data(self, parts, name):
        """
        @brief 解析 parts 的梁截面数据
        数值定义了一个梁截面，对应于元素集合 "beam2"，材料为 "Material-1"，温度梯度为 "GRADIENTS"，剖面形状为 "PIPE"
        '80., 10.' 梁截面的几何参数，具体含义取决于剖面形状， "PIPE" 表示管状剖面，80 是外径，10 是壁厚
        '0.,0.,-1.' 梁元素的局部 z 轴方向，定义了截面的方向
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        # print("beam_section_d:", d)

        idx = []

        self.cline += 1
        line = self.contents[self.cline]
        # print("beam_section_data:", line)

        while not line.startswith('*'):
            ss = line.split(',')
            # print("beam_section_ss:", ss)
            idx += [float(s) for s in ss]
            self.cline += 1

            line = self.contents[self.cline]

        beam_section_key = (d['elset'], d['material'], d['temperature'], d['section'])
        parts[name]['beam_section'][beam_section_key] = np.array(idx, dtype=np.float64) 
        # print("beam_section:", parts[name]['beam_section'][beam_section_key])


    def parse_assembly_data(self):
        """
        @brief 解析 assembly 数据
        用于将所有的部件（Parts）组合在一起，创建整个模型
        """

        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        print("assembly_d:", d)

        self.assembly[d['name']] = {'instance':{}, 'nset':{}}
        print("assembly_data:", self.assembly)
        self.cline += 1
        line = self.get_keyword_line()

        while line is not None and not line.startswith('*End Assembly'):
            if line.startswith('*Instance'):
                self.parse_instance_data(self.assembly, d['name'])
            elif line.startswith('*Nset'):
                self.parse_nset_assembly_data(self.assembly, d['name'])
            else:
                print("we pass the keyword line:" + line)
                self.cline += 1

            line = self.get_keyword_line() # 拿到下一个还没有处理的 keyword 行


    def parse_instance_data(self, assemblys, name):
        """
        @brief 解析 instance 数据
        每一个部件在组装过程中都被称为一个实例

        @param parts 用于存储节点集合数据的字典
        @param name 当前部分的名称
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        # print("instance_d:", d)

        self.cline += 1
        line = self.get_keyword_line()

        instance_key = (d['name'], d['part'])
        assemblys[name]['instance'][instance_key] = None
        # print("instance:", assemblys[name]['instance'][instance_key])


    def parse_nset_assembly_data(self, assemblys, name):
        """
        @brief 解析 nset 数据

        @param parts 用于存储单元集合数据的字典
        @param name 当前部分的名称

        @note 注意这里我们没有把原始节点的编号映射成当前连续的编号
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)
        # print("nset_assembly_d:", d)

        idx = []

        self.cline += 1
        line = self.contents[self.cline]
        # print("nset_assembly_line:", line)
        
        while not line.startswith('*'):
            ss = line.split(',')
            idx += [int(s) for s in ss]
            self.cline += 1
            line = self.contents[self.cline]

        nset_assembly_key = (d['nset'], d['instance'])
        assemblys[name]['nset'][nset_assembly_key] = np.array(idx, dtype=np.int_)
        # print("nset_assembly:", assemblys[name]['nset'][nset_assembly_key])



    def parse_step_data(self):
        line = self.contents[self.cline]
        print(line)

    def parse_material_data(self):
        line = self.contents[self.cline]
        print(line)

