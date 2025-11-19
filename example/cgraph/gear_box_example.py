
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH
mesher = cgraph.create("InpMeshReader") 
matrixer = cgraph.create("MatMatrixReader")              
spacer = cgraph.create("TensorFunctionSpace")        
eig_eq = cgraph.create("GearBox")
eigensolver = cgraph.create("SLEPcEigenSolver")
postprocess = cgraph.create("GearboxPostprocess")

mesher(input_inp_file ='/home/hk/下载/box_case3.inp') 
matrixer(input_mat_file ='/home/hk/下载/shaft_case3.mat') 
spacer(mesh = mesher(), gd = 3)
eig_eq(mesh=mesher(),shaftmatrix = matrixer(), space=spacer(),q = 3)


eigensolver(
    S=eig_eq().stiffness,
    M=eig_eq().mass,
    neigen=6,
)

postprocess(
    vecs=eigensolver().vec,
    vals=eigensolver().val,
    NS=eig_eq().NS,
    G=eig_eq().G,
    mesh=eig_eq().mesh,
)

WORLD_GRAPH.output(freqs=postprocess().freqs, vecs=postprocess().eigvecs)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())