#!/usr/bin/env python3

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import HalfEdgeMesh2d
from fealpy.opt.saddleoptalg import SteepestDescentAlg

from SCFTVEMModel2d import scftmodel2d_options, SCFTVEMModel2d
from vem2d_problem import halfedgemesh


class HalfEdgeAVEMTest():
    def __init__(self, mesh, fieldstype, moptions, optoptions):
        self.optoptions = optoptions
        obj = SCFTVEMModel2d(mesh, options=moptions)
        mu = obj.init_value(fieldstype=fieldstype)
        self.problem = {'objective': obj, 'mesh': mesh, 'x0': mu}

    def run(self, estimator='mix'):
        problem = self.problem
        options = self.optoptions
        model = problem['objective']
        mesh = problem['mesh']
        mesh = HalfEdgeMesh2d.from_mesh(mesh)

        #fig = plt.figure()
        #axes = fig.gca()
        #mesh.add_plot(axes)
        #mesh.find_node(axes, showindex=True)
        #mesh.find_cell(axes, showindex=True)
        #plt.show()

        aopts = mesh.adaptive_options(maxcoarsen=3, HB=True)
        while True:
            while True:
                print('NN', mesh.number_of_nodes())
                NC = mesh.number_of_cells()

                mu = problem['x0']
                if estimator == 'mix':
                    eta = model.mix_estimate(mu,w=1)
                if estimator == 'grad':
                    eta = model.estimate(q)

                aopts['data'] = {'mu':mu}
                S0 = model.vemspace.project_to_smspace(aopts['data']['mu'][:,0])
                S1 = model.vemspace.project_to_smspace(aopts['data']['mu'][:,1])

                mesh.adaptive(eta, aopts)

                model.reinit(mesh)
                aopts['data']['mu'] = np.zeros((model.gdof,2))
                aopts['data']['mu'][:,0] = model.vemspace.interpolation(S0, aopts['HB'])
                aopts['data']['mu'][:,1] = model.vemspace.interpolation(S1, aopts['HB'])
                problem['x0'] = aopts['data']['mu']

                optalg = SteepestDescentAlg(problem, options)
                x, f, g, diff = optalg.run()
                problem['mesh'] = mesh
                problem['x0'] = x
                problem['rho'] = model.rho
                self.problem = problem

                if diff < options['FunValDiff']:
                   if (np.max(problem['rho'][:,0]) < 1) and (np.min(problem['rho'][:,0]) >0):
                       break
                pass


options = {
        'MaxIters': 5000,
        'MaxFunEvals': 5000,
        'NormGradTol': 1e-7,
        'FunValDiff': 1e-6,
        'StepLength': 2,
        'StepTol': 1e-14,
        'Output': True
        }

moptions = scftmodel2d_options(
        nspecies= 2,
        nblend = 1,
        nblock = 2,
        ndeg = 100,
        fA = 0.2,
        chiAB = 0.25,
        dim = 2,
        T0 = 10,
        T1 = 40,
        nupdate = 1,
        order = 2)


mesh = halfedgemesh(n=1)
Halftest = HalfEdgeAVEMTest(mesh, fieldstype=3, moptions=moptions,
        optoptions=options)
Halftest.run()

