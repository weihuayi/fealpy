classdef Mesh3d
    properties
        node
        ds
    end
    methods
        function obj = Mesh3d(node, ds)
            obj.node = node;
            obj.ds = ds;
        end
        
        function NN = number_of_nodes(obj)
            NN = obj.ds.NN;
        end
        function NE = number_of_edges(obj)
            NE = obj.ds.NE;
        end
        function NF = number_of_faces(obj)
            NF = obj.ds.NF;
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
                a = obj.cell_volume();
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
            v = obj.node(obj.ds.edge(1, :), :) - obj.node(obj.ds.edge(2, :), :);
            l = sqrt(sum(v.^2, 1));
        end 

        function find_node(obj, range, varargin)
            hold on
            if (nargin==1)
                NN = obj.number_of_nodes();
                range = (1:NN)';
            end
            if islogical(range)
                range = find(range);
            end
            if size(range,2)>size(range,1)
                range = range';
            end
            h = plot3(obj.node(range, 1),obj.node(range, 2), obj.node(range, 3),'k.','MarkerSize', 20);
            if nargin>2
                if strcmp(varargin{1},'noindex') || strcmp(varargin{1},'index')
                    startidx = 2;
                else
                    startidx = 1;
                end
                set(h,varargin{startidx:end});
            end
            if (nargin<3) || ~(strcmp(varargin{1},'noindex'))
                text(obj.node(range,1)+0.015,obj.node(range,2)+0.015,obj.node(range,3)+0.015,int2str(range), ...
                    'FontSize',16,'FontWeight','bold');
            end
            hold off;
        end
        function find_edge(obj)
        end
        function find_face(obj)
        end
        function find_cell(obj)
        end
    end
end

