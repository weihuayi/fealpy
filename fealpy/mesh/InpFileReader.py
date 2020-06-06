import numpy as np


class InpFileReader():

    def __init__(self, fname):
        with open(fname, 'r') as f:
            contents = f.read() 
        contents = re.sub('**.*\n', '', contents) 
        contents = contents.split('\n')
        self.contents = contents
        self.cline = 0

if __name__ == '__main__':
    import sys

    fname = sys.argv[1]
    reader = InpFileReader(fname)
    reader.read()
