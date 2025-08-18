
from ..backend import bm
from ..decorator import variantmethod


class DLDGeometryModeler:

    def __init__(self, options):
        self.options = options

    @variantmethod('circle')
    def build(self, gmsh=None):
        pass


    @build.register('ellipse')
    def build(self):
        pass
