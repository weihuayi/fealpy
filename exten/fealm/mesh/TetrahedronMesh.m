classdef TetrahedronMesh < Mesh3d
    methods
        function obj = TetrahedronMesh(node, cell)
            NN = size(node, 1);
            ds = TetrahedronMeshDataStructure(NN, cell);
            obj = obj@Mesh3d(node, ds);
        end
        function grad = grad_lambda()
            
            
        localFace = self.ds.localFace
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        Dlambda = np.zeros((NC, 4, 3), dtype=self.dtype)
        volume = self.volume()
        for i in range(4):
            j,k,m = localFace[i]
            vjk = node[cell[:,k],:] - node[cell[:,j],:]
            vjm = node[cell[:,m],:] - node[cell[:,j],:]
            Dlambda[:,i,:] = np.cross(vjm, vjk)/(6*volume.reshape(-1,1))
        return Dlambda

        end
        
        function v = cell_volume(obj)
            v01 = obj.node(obj.cell(:, 2), :) - obj.node(obj.cell(:, 1), :);
            v02 = obj.node(obj.cell(:, 3), :) - obj.node(obj.cell(:, 1), :);
            v03 = obj.node(obj.cell(:, 4), :) - obj.node(obj.cell(:, 1), :);
            v = sum(v03.*cross(v01, v02), 2)/6.0;
        end
        
        function show(obj)
            NC = obj.number_of_cells();
            h = tetramesh(obj.ds.cell, obj.node, ones(NC,1));
            set(h,'facecolor',[0.35 0.75 0.35],'edgecolor','k');
            set(h,'FaceAlpha',0.4);
            view(3);
            axis off; axis equal; axis tight
        end
        function show_boundary(obj, color)
            isBdFace = obj.ds.face2cell(:, 1) == obj.ds.face2cell(:, 2);
            bdFace = obj.ds.face(isBdFace, :);
            h = trisurf(bdFace,obj.node(:,1),obj.node(:,2),obj.node(:,3));
            set(h,'facecolor', color,'edgecolor','k','FaceAlpha',0.75);
            view(3); axis equal; axis off; axis tight;
        end
    end  
end