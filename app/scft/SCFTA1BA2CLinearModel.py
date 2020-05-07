
import numpy as np
from fealpy.functionspace import FourierSpace
from fealpy.timeintegratoralg.timeline_new import UniformTimeLine

def model_options(
        nspecies = 3,
        nblend = 1,
        nblock = 4,
        ndeg = 100,
        fA1 = 0.25,
        fA2 = 0.25,
        fB = 0.25,
        fC = 0.25,
        maxdt = 0.01,
        chiAB = 0.25,
        chiAC = 0.25,
        chiBC = 0.25,
        dim = 2,
        NS = 16,
        maxdt = 0.01):
        # the parameter for scft model
        options = {
                'nspecies': nspecies,
                'nblend': nblend,
                'nblock': nblock,
                'ndeg': ndeg,
                'fA1': fA1,
                'fA2': fA2,
                'fB': fB,
                'fC': fC,
                'maxit': maxdt,
                'chiAB': chiAB,
                'chiAC': chiAC,
                'chiBC': chiBC,
                'dim': dim,
                'NS' : NS,
                'maxdt' : = maxdt
                }
        return options


class SCFTA1BA2CLinearModel():
    def __init__(self, options=None):
        if options == None:
            options = pscftmodel_options()
        self.options = options
        dim = options['dim']
        box = np.diag(dim*[2*np.pi])
        self.space = FourierSpace(box,  options['NS'])

        fA1 = options['fA1']
        fB  = options['fB']
        fA2 = options['fA2']
        fC  = options['fC']
        maxdt = options['maxdt']

        self.timeline0 = UniformTimeLine(0, fA1, int(fA1//maxdt))
        self.timeline1 = UniformTimeLine(0, fB, int(fB//maxdt))
        self.timeline2 = UniformTimeLine(0, fA2, int(fA2//maxdt))
        self.timeline3 = UniformTimeLine(0, fC, int(fC//maxdt))

        self.pdesolver0 = ParabolicFourierModel(self.space, self.timeline0) 


