import numpy as np

# Eagle Ford Shale
# 
# Basic infomation
# ----------------
# 
# Phenomina
# ---------
#
# interwell interference 井间干扰问题
#
# Notes
# -----
#
# In the study area in the Eagle Ford Shale, a normal
# faulting regime is the extensional stress regime
# 在Eagle Ford页岩研究区，正常的断层状态是张应力状态。
#
# $ S_v > S_{Hmax} > S_{hmin} $
# $ S_{hmin} > 0.6 S_v $
# $ S_{hmin} > p$
# 
# 
# 储层模型(Reservoir Model) 中的各种参数, 是通过历史匹配(history matching)进行校
# 准得到的.
# 
# Concepts
# --------
#   Differential stress: the difference between original $S_{Hmax}$ and $S_{hmin}
#   Overburden stress:
#   Maximum horizontal stress:
#   Minimum horizontal stress:
#   Bottomhole pressure:
#   Interstitial velocity term for water flow
#
# Assumptions
# -----------
#   1. paren-well fractures are planar.
#   2. infill-well fracture growth simulated by the hydrauli-fracture model is
#      nonplanar.
#
# Reference
# ---------
# 
# Guo X, Wu K, An C, et al. Numerical Investigation of Effects of Subsequent 
# Parent-Well Injection on Interwell Fracturing Interference Using Reservoir-
# Geomechanics-Fracturing Modeling[J]. Spe Journal, 2019, 24(04): 1884-1902.
#

ReservoirParameters = {
        'Reservoir mesh dimension (x–y–z)': ([466, 810, 15], 'm'),
        'Matrix permeability': (4.6e-19,  'm2, 470 nd'),
        'Matrix porosity': (0.12, None),
        'Initial reservoir pressure': (56.02, 'MPa'),
        'Initial water saturation': (0.17, None),
        'Parent-well bottomhole pressure': (20.7, 'MPa'),
        'Subsequent parent-well injection pressure': (60, 'MPa'),
        'Oil viscosity': (3e-4, 'Pa⋅s'),
        'Water viscosity': (6e-4, 'Pa⋅s'),
        'Fracture spacing within one stage': (23, 'm'),
        'Half-length of short parent-well fractures': (50, 'm'),
        'Half-length of long parent-well fractures': (150, 'm'),
        'Parent well spacing': (400, 'm'),
        'Stage number of a parent well': (3, None),
        'Fracture number in one stage of the parent well': (4, None),
        'Fracture conductivity': (8.06e-13, 'm2⋅m'),
        'Boundaries of the reservoir mesh': 'No flow (Neumann boundary)' 
        }

GeomechanicalParameters = {
        'Young’s modulus': (20, 'GPa'),
        'Poisson’s ratio': (0.22, None),
        'Biot’s coefficient': (0.7, None),
        'Overburden stress': (75, 'MPa'),
        'Initial maximum horizontal stress': (68, 'MPa'),
        'Initial minimum horizontal stress': (65, 'MPa'),
        'Differential stress (stress contrast)': (3, 'MPa')
        } 
