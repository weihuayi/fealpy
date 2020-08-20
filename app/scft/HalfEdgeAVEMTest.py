#!/usr/bin/env python3

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import HalfEdgeMesh2d, PolygonMesh
from fealpy.opt.saddleoptalg import SteepestDescentAlg

from SCFTVEMModel2d import scftmodel2d_options, SCFTVEMModel2d
from vem2d_problem import halfedgemesh, init_mesh, complex_mesh


class HalfEdgeAVEMTest():
    def __init__(self, mesh, fieldstype, moptions, optoptions):
        print('NN', mesh.number_of_nodes())
        self.optoptions = optoptions
        obj = SCFTVEMModel2d(mesh, options=moptions)
        mu = obj.init_value(fieldstype=fieldstype)
        self.problem = {'objective': obj, 'mesh': mesh, 'x0': mu}

    def uni_run(self):
        problem = self.problem
        options = self.optoptions
        model = problem['objective']
        mesh = problem['mesh']

        optalg = SteepestDescentAlg(problem, options)
        optalg.run()

    def run(self, estimator='mix'):
        problem = self.problem
        options = self.optoptions
        model = problem['objective']


        optalg = SteepestDescentAlg(problem, options)
        x, f, g, diff = optalg.run(maxit=1)
        while True:
            mesh = problem['mesh']
            hmesh = HalfEdgeMesh2d.from_mesh(mesh)
            cell, cellLocation = hmesh.entity('cell')
            print('2', cellLocation)
            aopts = hmesh.adaptive_options(method='mean', maxcoarsen=3, HB=True)
            print('NN', mesh.number_of_nodes())
            mu = problem['x0']
            if estimator == 'mix':
                eta = model.mix_estimate(mu, w=1)
            if estimator == 'grad':
                eta = model.estimate(q)

            aopts['data'] = {'mu':mu}
            S0 = model.vemspace.project_to_smspace(aopts['data']['mu'][:,0])
            S1 = model.vemspace.project_to_smspace(aopts['data']['mu'][:,1])


            hmesh.adaptive(eta, aopts)
            #fig = plt.figure()
            #axes = fig.gca()
            #hmesh.add_plot(axes)
            #hmesh.find_cell(axes, showindex=True)
            #plt.show()

            cell, cellLocation = hmesh.entity('cell')
            print('3', cellLocation)
            mesh = PolygonMesh.from_halfedgemesh(hmesh)

            model.reinit(mesh)
            aopts['data']['mu'] = np.zeros((model.gdof,2))
            aopts['data']['mu'][:,0] = model.vemspace.interpolation(S0, aopts['HB'])
            aopts['data']['mu'][:,1] = model.vemspace.interpolation(S1, aopts['HB'])
            problem['x0'] = aopts['data']['mu']

            optalg = SteepestDescentAlg(problem, options)
            x, f, g, diff = optalg.run(maxit=1)
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
        'NormGradTol': 1e-6,
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
        fA = 0.5,
        chiAB = 0.15,
        dim = 2,
        T0 = 20,
        T1 = 80,
        nupdate = 1,
        order = 2,
        rdir = sys.argv[3])
if sys.argv[1] =='quadtree':
    mesh = init_mesh(n=5, h=12)
    mesh = mesh.to_pmesh()
elif sys.argv[1] =='halfedge':
    mesh = halfedgemesh(n=6, h=12)
elif sys.argv[1] =='complex':
    mesh = complex_mesh(r=0.006, filename = sys.argv[2], n=1)

Halftest = HalfEdgeAVEMTest(mesh, fieldstype=4, moptions=moptions,
        optoptions=options)

Halftest.uni_run()
#Halftest.run()
