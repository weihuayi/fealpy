
import sys
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

from .triangle_mesh import TriangleMesh

class FABFileReader:
    def __init__(self, fname):
        try:
            with open(fname, 'r') as f:
                self.lines = f.read().split('\n')
        except EnvironmentError:
            print("Warning! open file failed!")
        self.cline = 0

    def read(self):
        NL = len(self.lines)
        while self.cline < NL:
            line = self.lines[self.cline]
            words = [word.replace(' ', '') for word in line.split(' ')]
            if words[0] == 'BEGIN':
                if words[1] == 'FORMAT':
                    self.read_format()
                elif words[1] == 'PROPERTIES':
                    self.read_properties()
                elif words[1] == 'SETS':
                    self.read_sets()
                elif words[1] == 'FRACTURE':
                    self.read_fracture()
                elif words[1] == 'TESSFRACTURE':
                    self.read_tessfracture()
                elif words[1] == 'ROCKBLOCK':
                    self.read_rockblock()
                else:
                    raise ValueError('I do not code for {}!'.format(words[1]))
            else:
                self.cline += 1

    def read_format(self):
        self.cline += 1
        line = self.lines[self.cline]
        self.format = {}
        while line.find('END') == -1:
            words = line.split()
            assert words[1] == '='
            self.format[words[0]] = words[2]
            self.cline += 1
            line = self.lines[self.cline]

    def read_properties(self):
        self.cline += 1
        line = self.lines[self.cline]
        self.properties = {}
        while line.find('END') == -1:
            words = line.split()
            assert words[1] == '='
            self.properties[words[0]] = words[2]
            self.cline += 1
            line = self.lines[self.cline]

    def read_fracture(self):
        NF = int(self.format['No_Fractures']) # 裂缝个数
        NN = int(self.format['No_Nodes']) # 节点个数
        NP = int(self.format['No_Properties']) # 性质个数
        self.unknown = np.zeros((NF, 3), dtype=np.float) # 未知属性
        self.node = np.zeros((NN, 3), dtype=np.float)
        self.fracture = np.zeros(NN, dtype=np.int)
        self.fractureLocation = np.zeros(NF+1, dtype=np.int)
        self.prop = [] 
        start = 0
        for i in range(NF):
            self.cline += 1 # 移到下一行
            line = self.lines[self.cline]
            words = line.split()
            index = words[0] # index of the fracture
            self.unknown[i, 0] = float(words[3])
            self.unknown[i, 1] = float(words[4])
            self.unknown[i, 2] = float(words[5])

            NV = int(words[1]) 
            n = int(words[2]) # 属性组数

            for j in range(NV):
                self.cline += 1
                line = self.lines[self.cline]
                words = line.split()
                self.node[start, 0] = float(words[1]) # drop words[0]
                self.node[start, 1] = float(words[2])
                self.node[start, 2] = float(words[3])
                self.fracture[start] = start
                start += 1
            self.fractureLocation[i+1] = self.fractureLocation[i] + NV

            prop = np.zeros((n, NP), dtype=np.float)
            for j in range(n):
                self.cline += 1
                line = self.lines[self.cline]
                words = line.split()
                prop[j, 0] = float(words[1])
                prop[j, 1] = float(words[2])
                prop[j, 2] = float(words[3])
            self.prop.append(prop)

    def read_sets(self):
        self.cline += 1
        line = self.lines[self.cline]
        self.sets = {}
        while line.find('END') == -1:
            words = line.split()
            assert words[1] == '='
            self.sets[words[0]] = words[2]
            self.cline += 1
            line = self.lines[self.cline]

    def read_tessfracture(self):
        self.cline += 1
        line = self.lines[self.cline]
        self.format = {}
        while line.find('END') == -1:
            self.cline += 1
            line = self.lines[self.cline]

    def read_rockblock(self):
        self.cline += 1
        line = self.lines[self.cline]
        self.format = {}
        while line.find('END') == -1:
            self.cline += 1
            line = self.lines[self.cline]
