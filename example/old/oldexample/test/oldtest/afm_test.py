
import numpy as np
from fealpy.mesh.AdvancingFrontAlg import PolygonDomain, AdvancingFrontAlg


point = np.array([(0,0), (1,0), (1,1), (0,1)], dtype=np.float)
facet = np.array([(0,1), (1,2), (2,3), (3,0)], dtype=np.int)             


pdomain = PolygonDomain(point, facet)

alg = AdvancingFrontAlg(pdomain)

