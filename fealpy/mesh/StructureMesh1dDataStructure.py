import numpy as np
from scipy.sparse import csr_matrix

class StructureMesh1dDataStructure():
    def __init__(self, NN, itype):
        """
        @brief Initialize 1D structured mesh data structure

        @param NN    : int, number of nodes
        @param itype : dtype, data type of indices
        """
        self.nx = NN-1
        self.NN = NN
        self.NC = NN-1 
        self.itype = itype 

    @property
    def cell(self):
        """
        @brief Generate all the cells in the mesh

        @return cell : ndarray, shape (NC, 2), contains node indices of each cell
        """
        NC = self.NC
        cell = np.zeros((NC, 2), dtype=self.itype)
        cell[:, 0] = range(0, NC)
        cell[:, 1] = range(1, NC+1) 
        return cell

    def cell_to_node(self):
        """
        @brief Get cell to node relationship

        @return cell : ndarray, shape (NC, 2), contains node indices of each cell
        """
        return self.cell

    def cell_to_cell(self):
        """
        @brief Get cell to cell relationship

        @return cell2cell : ndarray, shape (NC, 2), contains indices of adjacent cells
        """
        cell2cell = np.zeros((NC, 2), dtype=self.itype)

        cell2cell[0, 0] = 0
        cell2cell[1:, 0] = range(0, NC-1) 

        cell2cell[0:-1, 1] = range(1, NC)
        cell2cell[-1, 1] = NC-1
        
        return cell2cell

    def node_to_cell(self):
        """
        @brief Get node to cell relationship

        @return node2cell : ndarray, shape (NN, 4), contains cell indices connected to each node
        """
        node2cell = np.zeros((NN, 4), dtype=self.itype)

        node2cell[0, 0] = 0
        node2cell[1:, 0] = range(0, NC)
        node2cell[:, 2] = 0

        node2cell[0:-1, 1] = range(0, NC)
        node2cell[-1, 1] = NC-1

        return node2cell

    def face_to_cell(self):
        """
        @brief Get face to cell relationship (same as node_to_cell in 1D mesh)

        @return node2cell : ndarray, shape (NN, 4), contains cell indices connected to each node
        """
        return self.node_to_cell() 

    def node_to_node(self):
        """
        @brief Get the node-to-node relationship in the 1D structure mesh.

        This function returns a csr_matrix representing the relationship
        between nodes in the mesh.

        @return node2cell : csr_matrix, shape (NN, NN), dtype=np.bool_
            The csr_matrix representing the node-to-node relationship.
        """
        NC = self.NC
        val = np.ones((2*NC,), dtype=np.bool_)
        cell = self.cell
        I = cell.flat
        J = cell[:,[1,0]].flat
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN),dtype=np.bool_)
        return node2node

    def boundary_node_flag(self):
        """
        @brief Determine boundary nodes in the 1D structure mesh.

        @return isBdNode : np.array, dtype=np.bool_
            An array of booleans where True indicates a boundary node.
        """
        isBdNode = np.zeros(self.NN, dtype=np.bool_)
        isBdNode[0] = True
        isBdNode[-1] = True
        return isBdNode 

    def boundary_cell_flag(self):
        """
        @brief Determine boundary cells in the 1D structure mesh.

        @return isBdCell : np.array, dtype=np.bool_
            An array of booleans where True indicates a boundary cell.
        """
        isBdCell = np.zeros((NC,), dtype=np.bool_)
        isBdCell[0] = True
        isBdCell[-1] = True
        return isBdCell 

    def boundary_node_index(self):
        """
        @brief Get the indices of boundary nodes in the 1D structure mesh.

        @return boundary_node_indices : np.array, dtype=self.itype
            An array containing the indices of the boundary nodes.
        """
        return np.array([0, self.NN-1], dtype=self.itype) 

    def boundary_cell_index(self):
        """
        @brief Get the indices of boundary cells in the 1D structure mesh.

        This function returns an array containing the indices of the
        boundary cells in the mesh.

        @return boundary_cell_indices : np.array, dtype=self.itype
            An array containing the indices of the boundary cells.
        """ 
        return np.array([0, self.NC-1], dtype=self.itype) 
