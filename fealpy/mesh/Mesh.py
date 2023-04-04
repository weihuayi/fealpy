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

    def uniform_refine(self):
        raise NotImplementedError

    def integrator(self, k):
        raise NotImplementedError

    def number_of_entities(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_barycenter(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_measure(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def shape_function(self, p):
        raise NotImplementedError

    def grad_shape_function(self, p, index=np.s_[:]):
        raise NotImplementedError

    def interpolation_points(self):
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

    def find_node(self, axes, node=None, index=np.s_[:],
            showindex=False, color='r',
            markersize=20, fontsize=24, fontcolor='k', multiindex=None):

        if node is None:
            node = self.entity('node')

        GD = self.geo_dimension()

        if GD == 1:
            node = np.r_['1', node, np.zeros_like(node)]
            GD = 2

        if index == np.s_[:]:
            index = range(node.shape[0])
        elif (type(index) is np.int_):
            index = np.array([index], dtype=np.int_)
        elif (type(index) is np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
        elif (type(index) is list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
        else:
            raise ValueError("the type of index is not correct!")


        if (type(color) is np.ndarray) and (np.isreal(color[0])):
            umax = color.max()
            umin = color.min()
            norm = colors.Normalize(vmin=umin, vmax=umax)
            mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
            color = mapper.to_rgba(color)

        bc = node[index]
        if GD == 2:
            axes.scatter(bc[..., 0], bc[..., 1], c=color, s=markersize)
            if showindex:

                if multiindex is not None:
                    if (type(multiindex) is np.ndarray) and (len(multiindex.shape) > 1):
                        for i, idx in enumerate(multiindex):
                            s = str(idx).replace('[', '(')
                            s = s.replace(']', ')')
                            s = s.replace(' ', ',')
                            axes.text(bc[i, 0], bc[i, 1], s,
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor)
                    else:
                        for i, idx in enumerate(multiindex):
                            axes.text(bc[i, 0], bc[i, 1], str(idx),
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor) 
                else:
                    for i in range(len(index)):
                        axes.text(bc[i, 0], bc[i, 1], str(index[i]),
                                multialignment='center', fontsize=fontsize, 
                                color=fontcolor) 
        else:
            axes.scatter(bc[..., 0], bc[..., 1], bc[..., 2], c=color, s=markersize)
            if showindex:
                if multiindex is not None:
                    if (type(multiindex) is np.ndarray) and (len(multiindex.shape) > 1):
                        for i, idx in enumerate(multiindex):
                            s = str(idx).replace('[', '(')
                            s = s.replace(']', ')')
                            s = s.replace(' ', ',')
                            axes.text(bc[i, 0], bc[i, 1], bc[i, 2], s,
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor)
                    else:
                        for i, idx in enumerate(multiindex):
                            axes.text(bc[i, 0], bc[i, 1], bc[i, 2], str(idx),
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor) 
                else:
                    for i in range(len(index)):
                        axes.text(bc[i, 0], bc[i, 1], bc[i, 2], str(index[i]),
                                 multialignment='center', fontsize=fontsize, color=fontcolor) 

    def find_edge(self, axes, index=np.s_[:], 
            showindex=False,
            color='r', markersize=20,
            fontsize=24, fontcolor='k'):
        return self.find_entity(axes, 'edge', 
                showindex=showindex,
                color=color, 
                markersize=markersize,
                fontsize=fontsize, 
                fontcolor=fontcolor)


    def find_face(self):
        return self.find_entity(axes, 'face', 
                showindex=showindex,
                color=color, 
                markersize=markersize,
                fontsize=fontsize, 
                fontcolor=fontcolor)

    def find_cell(self):
        return self.find_entity(axes, 'cell', 
                showindex=showindex,
                color=color, 
                markersize=markersize,
                fontsize=fontsize, 
                fontcolor=fontcolor)

    def find_entity(self, axes, 
            etype, 
            index=np.s_[:], 
            showindex=False,
            color='r', markersize=20,
            fontsize=24, fontcolor='k'):

        GD = self.geo_dimension()
        bc = self.entity_barycenter(etype, index=index)

        if GD == 1:
            node = np.r_['1', bc, np.zeros_like(bc)]
            GD = 2

        if index == np.s_[:]:
            index = range(bc.shape[0])
        elif (type(index) is np.int_):
            index = np.array([index], dtype=np.int_)
        elif (type(index) is np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
        elif (type(index) is list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
        else:
            raise ValueError("the type of index is not correct!")

        if (type(color) is np.ndarray) & (np.isreal(color[0])):
            umax = color.max()
            umin = color.min()
            norm = colors.Normalize(vmin=umin, vmax=umax)
            mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
            color = mapper.to_rgba(color)

        bc = bc[index]
        if GD == 2:
            axes.scatter(bc[:, 0], bc[:, 1], c=color, s=markersize)
            if showindex:
                for i in range(len(index)):
                    axes.text(bc[i, 0], bc[i, 1], str(index[i]),
                            multialignment='center', fontsize=fontsize, 
                            color=fontcolor) 
        else:
            axes.scatter(bc[:, 0], bc[:, 1], bc[:, 2], c=color, s=markersize)
            if showindex:
                for i in range(len(index)):
                    axes.text(
                            bc[i, 0], bc[i, 1], bc[i, 2],
                            str(index[i]),
                            multialignment='center',
                            fontsize=fontsize, color=fontcolor)
