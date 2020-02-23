classdef MeshDataStructure < handle
    properties
        TD % the toplogy dimenstion

        NN % the number of nodes
        NE % the number of edges
        NF % the number of faces
        NC % the number of cells

        cell
        face
        edge

        face2cell
        cell2face

        cell2edge
    end
    methods
        function obj = MeshDataStructure(TD, NN, cell)
            obj.TD = TD;
            obj.NN = NN;
            obj.NC = size(cell, 1);
            obj.cell = cell;
            obj = obj.construct_face();
            if obj.TD == 3
                obj = obj.construct_edge();
            end
        end

        function obj = construct_face(obj)
            totalFace = obj.total_face();
            [~, i1, j] = unique(sort(totalFace, 2), 'rows');
            obj.NF = length(i1);
            obj.cell2face = int32(reshape(j, obj.F, []))';
            i2 = zeros(obj.NF, 1);
            i2(j) = 1:obj.F*obj.NC;
            obj.face2cell = zeros(obj.NF, 4, 'int32');
            obj.face2cell(:, 1) = ceil(i1/obj.F);
            obj.face2cell(:, 2) = ceil(i2/obj.F);
            obj.face2cell(:, 3) = rem(i1-1, obj.F)+1;
            obj.face2cell(:, 4) = rem(i2-1, obj.F)+1;
            obj.face = totalFace(i1, :);
            
            if obj.TD == 2
                obj.NE = obj.NF;
            end
        end

        function obj = construct_edge(obj)
            totalEdge = obj.total_edge();
            [~, i1, j] = unique(sort(totalEdge, 2), 'rows');
            obj.NE = length(i1);
            obj.cell2edge = int32(reshape(j, obj.E, []))';
            obj.edge = totalEdge(i1, :);
        end

        function tf = total_face(obj)
            c = obj.cell';
            FV = size(obj.localFace, 1);
            tf = reshape(c(obj.localFace, :), FV, [])';
        end

        function te = total_edge(obj)
            c = obj.cell';
            te = reshape(c(obj.localEdge, :), 2, [])';
        end
        
        function obj = reinit(obj, NN, cell)
            obj.NN = NN;
            obj.NC = size(cell, 1);
            obj.cell = cell;
            obj = obj.construct_face();
            if obj.TD == 3
                obj = obj.construct_edge()
            end
        end

        function isBdFace = boundary_face_flag(obj)
            isBdFace = obj.face2cell(:, 1) == obj.face2cell(:, 2);
        end

        function isBdEdge = boundary_edge_flag(obj)
            if obj.TD == 3 
                disp("Don't implement it for 3D case!"); %TODO
            elseif obj.TD == 2
                isBdEdge = obj.face2cell(:, 1) == obj.face2cell(:, 2);
            end
        end

        function isBdNode = boundary_node_flag(obj)
            NN = obj.NN;
            isBdNode = false(NN, 1);
            isBdFace = obj.face2cell(:, 1) == obj.face2cell(:, 2);
            isBdNode(obj.face(isBdFace, :)) = true;
        end
    end
end
