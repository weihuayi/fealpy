classdef TriangleMesh < Mesh
methods
    function obj = TriangleMesh(node, cell)
        NN = size(node, 1);
        ds = TriangleMeshDataStructure(NN, cell);
        obj = obj@Mesh(node, ds);
        obj.meshtype='triangle';
    end

    function uniform_refine(obj, n, surface)
        if nargin == 1
            n = 1;
            surface = [];
        elseif nargin == 2
            surface = [];
        end

        for i=1:n
            NN = obj.number_of_nodes();
            NE = obj.number_of_edges();
            NC = obj.number_of_cells();

            node = obj.entity('node');
            edge = obj.entity('edge');
            cell = obj.entity('cell');
            cell2edge = obj.ds.cell_to_edge();
            edge2cell = obj.ds.edge_to_cell();

            newNode = (node(edge(:, 1), :) + node(edge(:, 2), :))/2.0;
            if isempty(surface)
                obj.node(NN+1:NN+NE, :) = newNode;
            else
                obj.node(NN+1:NN+NE, :) = surface.project(newNode);
            end
            edge2newNode = uint32((NN+1:NN+NE)');

            %% Refine each triangle into four triangles as follows
            %     3
            %    / \
            %   5 - 4
            %  / \ / \
            % 1 - 6 - 2
            t = 1:NC;
            p(t,1:3) = cell(t,1:3);
            p(t,4:6) = edge2newNode(cell2edge(t, 1:3));
            cell(t,:) = [p(t,1), p(t,6), p(t,5)];
            cell(NC+1:2*NC,:) = [p(t,6), p(t,2), p(t,4)];
            cell(2*NC+1:3*NC,:) = [p(t,5), p(t,4), p(t,3)];
            cell(3*NC+1:4*NC,:) = [p(t,4), p(t,5), p(t,6)];
            obj.ds.reinit(NN+NE, cell);
        end
    end
        
    function a = cell_measure(obj)
        GD = obj.geo_dimension();
        v12 = obj.node(obj.ds.cell(:, 2), :) - obj.node(obj.ds.cell(:, 1), :);
        v13 = obj.node(obj.ds.cell(:, 3), :) - obj.node(obj.ds.cell(:, 1), :);
        if GD == 2
            a = (v12(:, 1).*v13(:, 2)-v12(:, 2).*v13(:, 1))/2;
        elseif  GD == 3     % surface triangles
            normal = cross(v12, v13, 2);
            a = sqrt(sum(normal.^2,2))/2;
        end
    end

    function glambda = grad_lambda(obj)
        NC = obj.number_of_cells();
        node = obj.entity('node');
        cell = obj.entity('cell');
        v1 = node(cell(:, 3), :) - node(cell(:, 2), :);
        v2 = node(cell(:, 1), :) - node(cell(:, 3), :);
        v3 = node(cell(:, 2), :) - node(cell(:, 1), :);
        GD = obj.geo_dimension();
        glambda = zeros(NC, GD, 3);
        if GD == 2
            error("We have not implement it!");
        elseif GD == 3
            nv = cross(v3, -v2, 2);
            l = sqrt(sum(nv.^2, 2));
            n = nv./l;
            glambda(:, :, 1) = cross(n, v1)./l;
            glambda(:, :, 2) = cross(n, v2)./l;
            glambda(:, :, 3) = cross(n, v3)./l;
        end
    end

    function p = bc_to_point(obj, bc, surface)
        node = obj.entity('node');
        cell = obj.entity('cell');
        p = bc(1)*node(cell(:, 1), :) + bc(2)*node(cell(:, 2), :) + bc(3)*ndoe(cell(:, 3), :);
        if nargin == 3
            p = surface.project(p);
        end
    end
end
end
        
