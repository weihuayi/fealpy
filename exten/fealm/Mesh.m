classdef Mesh < handle
    properties
        node
        ds
        nodedata
        edgedata
        facedata
        celldata

        GD % geometry dimenstion
        meshdata
        meshtype
        name
    end
    methods
        function obj = Mesh(node, ds)
            obj.GD = size(node, 2);
            obj.node = node;
            obj.ds = ds;
            obj.nodedata = containers.Map;
            obj.edgedata = containers.Map;
            obj.facedata = containers.Map;
            obj.celldata = containers.Map;

            obj.meshdata = containers.Map;
        end
        
        function NN = number_of_nodes(obj)
            NN = double(obj.ds.NN);
        end

        function NE = number_of_edges(obj)
            NE = double(obj.ds.NE);
        end

        function NF = number_of_faces(obj)
            NF = double(obj.ds.NF);
        end

        function NC = number_of_cells(obj)
            NC = double(obj.ds.NC);
        end
        
        function GD = geo_dimension(obj)
            GD = obj.GD;
        end
        
        function TD = top_dimension(obj)
            TD = obj.ds.TD;
        end

        function e = entity(obj, etype)
            switch etype
                case 'cell'
                    e = obj.ds.cell;
                case 'face'
                    e = obj.ds.face;
                case 'edge'
                    if obj.ds.TD == 2
                        e = obj.ds.face;
                    elseif obj.ds.TD == 3
                        e = obj.ds.edge;
                    end
                case 'node'
                    e = obj.node;
                otherwise
                    warning('Unexpected entity type. return a empty array');
                    e = [];
            end
        end

        function a = entity_measure(obj, etype)
            switch etype
                case 'cell'
                    a = obj.cell_measure();
                case 'face'
                    a = obj.face_measure();
                case 'edge'
                    if obj.ds.TD == 2
                        a = obj.face_measure();
                    elseif obj.ds.TD == 3
                        a = obj.edge_measure();
                    end
                case 'node'
                    NN = obj.number_of_nodes();
                    a = zeros(NN, 1);
                otherwise
                    a = [];
                    warning('Unexpected entity type. return a empty array');
            end
        end

        function l = edge_measure(obj)
            if obj.ds.TD == 2
                v = obj.node(obj.ds.face(1, :), :) - obj.node(obj.ds.face(2, :), :);
            else
                v = obj.node(obj.ds.edge(1, :), :) - obj.node(obj.ds.edge(2, :), :);
            end
            l = sqrt(sum(v.^2, 1));
        end 
    end
end
