classdef TriangleMeshDataStructure < MeshDataStructure
    properties
        localFace = [
            2, 3, 1
            3, 1, 2]

        localEdge = [
            2, 3, 1
            3, 1, 2]

        index = [
            1, 2, 3
            2, 3, 1
            3, 1, 2]

        F = 3 
        E = 3
        V = 3
    end
    methods
        function obj = TriangleMeshDataStructure(NN, cell)
            TD = 2;
            obj = obj@MeshDataStructure(TD, NN, cell);
        end

        function edge2cell = edge_to_cell(obj)
            edge2cell = obj.face2cell;
        end

        function cell2edge = cell_to_edge(obj)
            cell2edge = obj.cell2face;
        end
    end
end
