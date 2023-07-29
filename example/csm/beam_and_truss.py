import sys
from fealpy.mesh import InpFileReader


fname = sys.argv[1]
reader = InpFileReader(fname)
reader.parse()

print(reader.parts)
