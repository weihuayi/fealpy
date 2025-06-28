
from ..decorator import variantmethod

class BoxDomainMesher:


    def __init__(self, box):

        self.box = box


    @variantmethod('tet')
    def init_mesh(self):
        pass
