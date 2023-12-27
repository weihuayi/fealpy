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
        "BL1":""
        }

mesh_coding = {
        "TRI": "TriangleMesh",
        "TET":  "",
        "HEX":  "",
        "QUAD" : "",
        "INT":  "",
        "U1D":  "",
        "U2D":  "",
        "U3D":  ""
        }

basis_coding = {
        "LS": "Lagrange basis on simplex",
        "LR": "Lagrange basis on rectangle",
        "LC": "Lagrange basis on cuboid",
        "BS": "Be   on simplex",
        "BR": "     on rectangle",
        "BC": "     on cuboid",
        }



