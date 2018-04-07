classdef TetrahedronMeshDataStructure < Mesh3dDataStructure
    properties
        localFace = [
            2, 1, 1, 1
            3, 4, 2, 3
            4, 3, 4, 2]
        localEdge = [
            1, 1, 1, 2, 2, 3
            2, 3, 4, 3, 4, 4]
        localFace2edge = [
            6, 6, 5, 4
            5, 2, 3, 1
            4, 3, 1, 2]
        index = [
            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4
            2, 3, 4, 3, 1, 4, 1, 2, 4, 1, 3, 2
            3, 4, 2, 1, 4, 3, 2, 4, 1, 3, 2, 1
            4, 2, 3, 4, 3, 1, 4, 1, 2, 2, 1, 3
            ]
        F = 4
        E = 6
        V = 4
    end
    methods
        function obj = TetrahedronMeshDataStructure(NN, cell)
            obj = obj@Mesh3dDataStructure(NN, cell);
        end
    end
    
end

