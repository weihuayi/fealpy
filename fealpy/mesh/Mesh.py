import numpy as np

class Mesh:
    def number_of_cells(self):
        raise NotImplementedError

    def number_of_nodes(self):
        raise NotImplementedError

    def number_of_edges(self):
        raise NotImplementedError

    def number_of_faces(self):
        raise NotImplementedError

    def geo_dimension(self):
        raise NotImplementedError

    def top_dimension(self):
        raise NotImplementedError

    def integrator(self, k):
        raise NotImplementedError

    def entity(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_barycenter(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_measure(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def shape_function(self, p):
        raise NotImplementedError

    def grad_shape_function(self, p, index=np.s_[:])
        raise NotImplementedError

    def interpolation_points(self)
        raise NotImplementedError

    def cell_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def edge_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def face_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def node_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def add_plot(self):
        raise NotImplementedError

    def find_node(self):
        raise NotImplementedError

    def find_edge(self):
        raise NotImplementedError

    def find_face(self):
        raise NotImplementedError

    def find_cell(self):
        raise NotImplementedError
