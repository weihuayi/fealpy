clear all;
close all;

r = 1.0;
C = [0.0, 0.0, 0.0];
R = 1.0;
model = SpherePFCModelData(r, C, R);

mesh = model.init_mesh();
sphere = model.surface;
mesh.uniform_refine(4, sphere);

space = LagrangeFiniteElementSpace(mesh);

A = space.stiff_matrix();
M = space.mass_matrix();

gdof = space.number_of_global_dofs();
uh = ones(gdof, 1);
F0 = space.cross_matrix(uh, @model.nonlinear_term);
F1 = space.cross_matrix(uh, @model.grad_nonlinear_term);


% plot 
cell = mesh.entity('cell');
node = mesh.entity('node');
showmesh(node, cell);
