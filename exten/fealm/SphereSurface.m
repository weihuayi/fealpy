classdef SphereSurface < Surface 
    properties
        C 
        R 
    end
    methods
        function obj = SphereSurface(c, r)
            obj = obj@Surface('sphere');
            obj.C = c;
            obj.R = r;
        end

        function mesh = init_mesh(obj)
            t = (sqrt(5)-1)/2;
            node=[0, 1, t
                  0, 1,-t
                  1, t, 0
                  1,-t, 0
                  0,-1,-t
                  0,-1, t
                  t, 0, 1
                  -t,0, 1
                  t, 0,-1
                  -t,0,-1
                  -1,t, 0
                  -1,-t,0];
             cell=[ 6, 2,0
                    3, 2, 6
                    5, 3, 6
                    5, 6, 7
                    6, 0, 7
                    3, 8, 2
                    2, 8, 1
                    2, 1, 0
                    0, 1, 10
                    1, 9, 10
                    8, 9, 1
                    4, 8, 3
                    4, 3, 5
                    4, 5, 11
                    7, 10, 11
                    0, 10, 7
                    4, 11, 9
                    8, 4, 9
                    5, 7, 11
                    10, 9, 11]+1;
             node = obj.project(node);
             mesh = TriangleMesh(node, cell);
        end

        function val = phi(obj, p)
            val = sqrt(sum(( p - obj.C).^2, 2)) - obj.R;
        end

        function n = gradient(obj, p)
            r = sqrt(sum((p - obj.C).^2, 2));
            n = (p - obj.C)./r;
        end

        function [node, sd] = project(obj, p)
            d = obj.phi(p);
            node = p - d.*obj.gradient(p);
            if nargout == 2
                sd = d;
            end
        end
    end
end
