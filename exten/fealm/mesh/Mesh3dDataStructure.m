classdef Mesh3dDataStructure
    properties
        NN
        NC
        cell
        face
        edge
        face2cell
        cell2edge
        localFace
        localEdge
    end
    methods
        function obj = Mesh3dDataStructure(NN, cell)
            obj.NN = NN;
            obj.NC = length(cell);
            obj.cell = cell;
            obj.construct();
        end
        function construct(obj)
        end
        function total_edge(obj)
        end
        function total_face(obj)
        end
        
        function reinit(obj, NN, cell)
            obj.NN = NN;
            obj.NC = length(cell);
            obj.cell = cell;
            obj.construct();
        end
    end