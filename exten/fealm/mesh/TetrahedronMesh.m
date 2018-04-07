classdef TetrahedronMesh < Mesh3d
    methods
        function obj = TetrahedronMesh(node, cell)
            NN = size(node, 2);
            ds = TetrahedronMeshDataStructure(NN, cell);
            obj = obj@Mesh3d(node, ds);
        end
        function show(obj)
            h = tetramesh(,node,ones(size(elem,1),1));
            set(h,'facecolor',[0.35 0.75 0.35],'edgecolor','k');
            set(h,'FaceAlpha',0.4);
            view(3);
            axis off; axis equal; axis tight
        end
    end  
end