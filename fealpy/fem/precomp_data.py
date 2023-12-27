import numpy as np
"""
算子编码：

BL0
BL1


网格编码规则：

基函数类型编码规则：

"""
data = {"":0}

operator_coding = {
        "BL0":"ScalarDiffusionIntegrator",  # 
        "BL1":"ScalarBoundarySourceIntegrator", 
        "BL2":"ScalarConvectionIntegrator", 
        "BL3":"ScalarMassIntegrator", 
        "BL4":"ScalarNeumannBCIntegrator", 
        "BL5":"ScalarPGLSConvectionIntegrator", 
        "BL6":"ScalarRobinBoundaryIntegrator", 
        "BL7":"ScalarSourceIntegrator", 
        "BL8":"VectorBoundarySourceIntegrator", 
        "BL9":"VectorConvectionIntegrator", 
        "BL10":"VectorDiffusionIntegrator", 
        "BL11":"VectorMassIntegrator", 
        "BL12":"VectorNeumannBCIntegrator", 
        "BL13":"VectorSourceIntegrator", 
        "BL14":"VectorViscousWorkIntegrator", 
        "BL15":"TrussStructureIntegrator", 
        "BL16":"ProvidesSymmetricTangentOperatorIntegrator", 
        "BL17":"PressWorkIntegrator", 
        "BL18":"LinearElasticityOperatorIntegrator", 
        "BL19":"FluidBoundaryFrictionIntegrator", 
        "BL20":"BeamStructureIntegrator", 
        "BL21":"DiffusionIntegrator", 
        }

mesh_coding = {
        "TRI": "TriangleMesh",
        "TET": "TetrahedronMesh",
        "HEX":  "HexahedronMesh",
        "QUAD" : "QuadrangleMesh",
        "INT":  "IntervalMesh",
        "U1D":  "UniformMesh1d",
        "U2D":  "UniformMesh2d",
        "U3D":  "UniformMesh3d"
        }

basis_coding = {
        "LS": "Lagrange basis on simplex",
        "LR": "Lagrange basis on rectangle",
        "LC": "Lagrange basis on cuboid",
        "BS": "Bernstein on simplex",
        "BR": "Bernstein on rectangle",
        "BC": "Bernstein on cuboid",
        }



