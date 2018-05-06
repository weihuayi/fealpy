% node = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1];
% elem = [1,2,3,7; 1,6,2,7; 1,5,6,7; 1,8,5,7; 1,4,8,7; 1,3,4,7]';
% 
% node = [0, 0, 0; 1, 0, 0; 0, 1, 0; 0, 0, 1; 0, 0, -1];
% elem = [1, 1; 2, 3; 3, 2; 4, 5];
% mesh = TetrahedronMesh(node, elem);
% mesh.show()
% mesh.find_node()
% disp('Face:')
% disp(mesh.ds.face)
% disp('Face2cell:')
% disp(mesh.ds.face2cell)
% disp('Edge:')
% disp(mesh.ds.edge)
% disp('cell2edge:')
% disp(mesh.ds.cell2edge)


mio = MeshIO();
meshes = mio.read_inp('data2.inp');
cmap = hsv(double(meshes.Count));
i = 1;
fig = figure();
hold on 
for mesh=meshes.values
    c = cmap(i, :);
    mesh{1}.show_boundary(c);
    i = i+1;
end

