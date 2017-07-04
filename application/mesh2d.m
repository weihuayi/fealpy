
h0 = 0.035
fd=inline('drectangle(p,-1,1,-1,1)','p');
box=[-1,-1;1,1];
fix=[-1,-1;-1,1;1,-1;1,1];
[point, cell,q,u]=odtmesh2d(fd, @huniform, h0,box,fix,1);
save('square.mat', 'point', 'cell')

h0 = 0.05
fd=inline('dpoly(p,fix)','p','fix');
n=6;
phi=(0:n)'/n*2*pi;
box=[-1,-1;1,1];
fix=[cos(phi),sin(phi)];
[point, cell,q,u]=odtmesh2d(fd, @huniform, h0, box, fix, 1,fix);
save('polygon.mat', 'point', 'cell')
