import numpy as np
import re

class PolyFileReader():
    def __init__(self, fname):
        with open(fname, 'r') as f:
            contents = f.read() 
        contents = re.sub('#.*\n', '', contents) 
        contents = contents.split('\n')
        self.contents = contents
        self.cline = 0
        self.data = {'vertices':None, 'segments':None, 'holes':None,
                'regions':None}

    def read(self):
        self.read_vertices()
        self.read_segments()
        self.read_holes()
        self.contents = None
        print(self.data)

    def read_vertices(self):
        head = np.fromstring(self.contents[self.cline], np.int, sep=' ')
        NV = head[0]
        nc = 3 + head[2] + head[3]
        data = np.zeros((NV, nc), dtype=np.float)
        self.read_data(data)
        self.data['vertices'] = {'index': data[:, 0].astype(np.int), 'xy': data[:, 1:3]} 
        self.data['vertices']['attribute'] = data[:, 3:3+head[2]] if head[2] > 0 else None
        self.data['vertices']['bdmarker'] = data[:, 3+head[2]:].astype(np.int) if head[3] == 1 else None

    def read_segments(self):
        head = np.fromstring(self.contents[self.cline], np.int, sep=' ')
        NS = head[0]
        nc = 3 + head[1]
        data = np.zeros((NS, nc), dtype=np.int)
        self.read_data(data)
        self.data['segments'] = {'index': data[:, 0], 'endpoint': data[:, 1:3]-1}
        self.data['segments']['bdmarker'] = data[:, 3:] if head[1] == 1 else None

    def read_holes(self):
        head = np.fromstring(self.contents[self.cline], np.int, sep=' ')
        NH = head[0]
        nc = 3
        if NH > 0:
            data = np.zeros((NH, nc), dtype=np.float)
            self.read_data(data)
            self.data['holes']={'index': data[:, 0].astype(np.int), 'xy':data[:, 1:]}

    def read_data(self, data):
        N = data.shape[0]
        self.cline += 1
        for i in range(N):
            data[i, :] = np.fromstring(self.contents[self.cline + i], np.float, sep=' ')
        self.cline += N

if __name__ == '__main__':
    import sys

    fname = sys.argv[1]
    reader = PolyFileReader(fname)
    reader.read()
