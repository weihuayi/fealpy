classdef TetrahedronMesh < Mesh3d
    methods
        function obj = TetrahedronMesh(node, cell)
            NN = size(node, 1);
            ds = TetrahedronMeshDataStructure(NN, cell);
            obj = obj@Mesh3d(node, ds);
        end
        function show(obj)
            NC = obj.number_of_cells();
            h = tetramesh(obj.ds.cell', obj.node, ones(NC,1));
            set(h,'facecolor',[0.35 0.75 0.35],'edgecolor','k');
            set(h,'FaceAlpha',0.4);
            view(3);
            axis off; axis equal; axis tight
        end
        function show_boundary(obj, color)
            isBdFace = obj.ds.face2cell(1, :) == obj.ds.face2cell(2, :);
            bdFace = obj.ds.face(:, isBdFace);
            h = trisurf(bdFace',obj.node(:,1),obj.node(:,2),obj.node(:,3));
            set(h,'facecolor', color,'edgecolor','k','FaceAlpha',0.75);
            view(3); axis equal; axis off; axis tight;
        end
    end  
end