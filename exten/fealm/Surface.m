classdef Surface < handle
    properties
        name
    end
    methods
        function obj = Surface(name)
            obj.name = name;
        end

        function [node, sd] = project(obj, p)
            s = sign(obj.phi(p));
            node = p;

            normalAtNode = obj.gradient(node);
            valueAtNode = obj.phi(node);
            node = node - valueAtNode*ones(1,3).*normalAtNode./(dot(normalAtNode,normalAtNode,2)*ones(1,3));

            vector = (-s)*ones(1,3).*(node - p);

            d = s.*sqrt(dot(vector,vector,2));

            normalAtNode = obj.gradient(node);

            node = p - d*ones(1,3).*normalAtNode./(sqrt(dot(normalAtNode,normalAtNode,2))*ones(1,3));

            valueAtNode = obj.phi(node);

            normalAtNode = obj.gradient(node);

            vector = (-s)*ones(1,3).*(node - p);

            d = s.*sqrt(dot(vector,vector,2));

            e1 = normalAtNode./(sqrt(dot(normalAtNode, normalAtNode,2))*ones(1,3))-vector./(sqrt(dot(vector,vector,2))*ones(1,3));
            error=sqrt(valueAtNode.^2./(dot(normalAtNode, normalAtNode,2))+dot(e1,e1,2));

            k=1;
            while max(abs(error)) > 1e-12 && k<200
                
                k=k+1;
                
                node = node - valueAtNode*ones(1,3).*normalAtNode./(dot(normalAtNode,normalAtNode,2)*ones(1,3));
                
                vector = -s*ones(1,3).*(node - p);
                d = s.*sqrt(dot(vector,vector,2));
                normalAtNode = obj.gradient(node);
                
                node = p - d*ones(1,3).*normalAtNode./(sqrt(dot(normalAtNode,normalAtNode,2))*ones(1,3));
                
                valueAtNode = obj.phi(node);
                
                normalAtNode = obj.gradient(node);
                
                vector = (-s)*ones(1,3).*(node - p);
                
                d = s.*sqrt(dot(vector,vector,2));
                e1 = normalAtNode./(sqrt(dot(normalAtNode, normalAtNode,2))*ones(1,3))-vector./(sqrt(dot(vector,vector,2))*ones(1,3));
                error=sqrt(valueAtNode.^2./(dot(normalAtNode, normalAtNode,2))+dot(e1,e1,2));   
            end


            if nargout == 2
                sd = d;
            end
        end
    end
end
