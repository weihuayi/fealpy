import numpy as np
import re
import ipdb

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
    def __init__(self, fname):
        with open(fname, 'r') as f:
            contents = f.read()
        self.contents = contents.split('\n')
        self.cline = 0
        self.parts = {}
        self.materials = {}
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
        """
        d={}
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
        """

        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)

        self.parts[d['name']] = {'elem':{}}
        self.cline += 1
        line = self.get_keyword_line()
        while not line.startswith('*End Part'):
            if line.startswith('*Node'):
                self.parse_node_data(self.parts, d['name'])
            elif line.startswith('*Element'):
                self.parse_elem_data(self.parts, d['name'])
            elif line.startswith('*Nset'):
                self.parse_nset_data(self.parts, d['name'])
            elif line.startswith('*Elset'):
                self.parse_elset_data(self.parts, d['name'])
            else:
                print("we pass the keyword line:" + line)
            self.cline += 1
            line = self.get_keyword_line() # 拿到下一个还没有处理的 keyword 行

    def parse_node_data(self, parts, name):
        """
        @brief 处理节点坐标数据
        """
        i = 0
        node = []
        nmap = {}

        self.cline += 1
        line = self.contents[self.cline]
        while not line.startswith('*'):
            s = line.split(',')
            node.append((float(s[1]), float(s[2]), float(s[3])))
            nmap[int(s[0])] = i
            i += 1

            self.cline += 1
            line = self.contents[self.cline]

        parts[name]['node'] = (np.array(node, dtype=np.float64), nmap)


    def parse_elem_data(self, parts, name):
        """
        """
        line = self.contents[self.cline]
        d = self.parse_keyword_line(line)

        i = 0
        elem = []
        emap = {}

        nmap = parts[name]['node'][1]

        self.cline += 1
        line = self.contents[self.cline]
        while not line.startswith('*'):
            ss = line.split(',')
            elem.append([nmap[int(s)] for s in ss[1:]])
            emap[int(ss[0])] = i
            i += 1
            self.cline += 1
            line = self.contents[self.cline]

        parts[name]['elem'][d['type']] = (np.array(elem, dtype=np.int_), emap)


    def parse_nset_data(self, parts, name):
        pass

    def parse_elset_data(self, parts, name):
        pass

    def parse_assembly_data(self):
        line = self.contents[self.cline]
        print(line)

    def parse_step_data(self):
        line = self.contents[self.cline]
        print(line)

    def parse_material_data(self):
        line = self.contents[self.cline]
        print(line)

