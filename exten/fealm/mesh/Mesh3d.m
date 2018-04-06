classdef Mesh3d
    properties
        node
        ds
    end
    methods
        function NN = number_of_nodes(obj)
            NN = obj.ds.NN;
        end
        function NE = number_of_edges(obj)
            NE = length(obj.ds.edge);
        end
        function NF = number_of_faces(obj)
            NF = length(obj.ds.face);
        end
        function NC = number_of_cells(obj)
            NC = obj.ds.NC;
        end
        
        function GD = geo_dimension(obj)
            GD = 3;
        end
        
        function TD = top_dimension(obj)
            TD = 3;
        end
        
        function e = entity(obj, etype)
            if etype == 3
                e = obj.ds.cell;
            elseif etype == 2
                e = obj.ds.face;
            elseif etype == 1
                e = obj.ds.edge;
            elseif etype == 0
                e = obj.node;
            else
                error('etype is not [3, 2, 1, 0]!')
            end
        end
        
        function a = entity_measure(obj, etype)
            if etype == 3
                a = obj.cell_volumn();
            elseif etype == 2
                a = obj.face_area();
            elseif etype == 1
                a = obj.edge_length();
            elseif etype == 0
                a = 0;
            else
                error('etype is not [3, 2, 1, 0]!')
            end
        end
        
        function l = edge_length(obj)
            v = obj.node(obj.ds.edge(:, 1), :) - obj.node(obj.ds.edge(:, 2), :);
            l = sqrt(sum(v.^2, 2));
        end
            
            
    end  
end

