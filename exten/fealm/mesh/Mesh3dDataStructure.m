classdef Mesh3dDataStructure
    properties
        NN
        NE
        NF
        NC
        cell
        face
        edge
        face2cell
        cell2edge        
    end
    methods
        function obj = Mesh3dDataStructure(NN, cell)
            obj.NN = NN;
            obj.NC = size(cell, 1);
            obj.cell = cell;
            obj = obj.construct();
        end
        function obj = construct(obj)
            totalFace = obj.total_face();
            [~, i1, j] = unique(sort(totalFace, 2), 'rows');
            obj.NF = length(i1);
            i2 = zeros(obj.NF, 1);
            i2(j) = 1:obj.F*obj.NC;
            obj.face2cell = zeros(obj.NF, 4, 'int32');
            obj.face2cell(:, 1) = ceil(i1/obj.F);
            obj.face2cell(:, 2) = ceil(i2/obj.F);
            obj.face2cell(:, 3) = rem(i1-1, obj.F)+1;
            obj.face2cell(:, 4) = rem(i2-1, obj.F)+1;
            obj.face = totalFace(i1, :);
            totalEdge = obj.total_edge();
            [~, i1, j] = unique(sort(totalEdge, 2), 'rows');
            obj.edge = totalEdge(i1, :);
            obj.NE = length(i1);
            
            obj.cell2edge = int32(reshape(j, obj.E, []))';
        end
        function te = total_edge(obj)
            c = obj.cell';
            te = reshape(c(obj.localEdge, :), 2, [])';
        end
        function tf = total_face(obj)
            c = obj.cell';
            FV = size(obj.localFace, 1);
            tf = reshape(c(obj.localFace, :), FV, [])';
        end
        
        function obj = reinit(obj, NN, cell)
            obj.NN = NN;
            obj.NC = size(cell, 2);
            obj.cell = cell;
            obj.construct();
        end
    end
end