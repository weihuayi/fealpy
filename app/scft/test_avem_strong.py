#!/usr/bin/env python3

import sys
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from fealpy.mesh import HalfEdgeMesh2d, PolygonMesh
from fealpy.opt.saddleoptalg import SteepestDescentAlg

from SCFTVEMModel2d import scftmodel2d_options, SCFTVEMModel2d
from vem2d_problem import halfedgemesh, init_mesh, complex_mesh


class HalfEdgeAVEMTest():
    def __init__(self, fieldstype, moptions, optoptions):
        self.optoptions = optoptions
        self.moptions = moptions
        ##多边形网格－空间

        data = scio.loadmat('results_lam/15.mat')
        rho = data['rho']
        mu = data['mu']
        mesh= open('results_lam/15mesh.bin','rb')
        mesh= pickle.load(mesh)
        obj = SCFTVEMModel2d(mesh, options=self.moptions)
        self.problem = {'objective': obj, 'x0': mu, 'mesh': mesh, 'rho': rho}

    def run(self, estimator='mix'):
        problem = self.problem
        options = self.optoptions
        moptions = self.moptions
        model = problem['objective']

        optalg = SteepestDescentAlg(problem, options)
        x, f, g, diff = optalg.run(maxit=10)

        q = np.zeros(model.rho.shape)
        q[:,0] = model.q0[:,-1]
        q[:,1] = model.q1[:,-1]
        problem['x0'] = x
        while True:
            print('chiN', moptions['chiN'])
            while True:
                mesh = problem['mesh']##多边形网格
                hmesh = HalfEdgeMesh2d.from_mesh(mesh)##半边网格
                aopts = hmesh.adaptive_options(method='mean', maxcoarsen=3, HB=True)
                print('NN', mesh.number_of_nodes())

                mu = problem['x0']
                if estimator == 'mix':
                    eta = model.mix_estimate(q, w=1)
                if estimator == 'grad':
                    eta = model.estimate(q)

                aopts['data'] = {'mu':mu}
                S0 = model.vemspace.project_to_smspace(aopts['data']['mu'][:,0])
                S1 = model.vemspace.project_to_smspace(aopts['data']['mu'][:,1])


                hmesh.adaptive(eta, aopts)###半边网格做自适应
                mesh = PolygonMesh.from_halfedgemesh(hmesh)###多边形网格

                model.reinit(mesh)###多边形网格给进空间
                aopts['data']['mu'] = np.zeros((model.gdof,2))
                aopts['data']['mu'][:,0] = model.vemspace.interpolation(S0, aopts['HB'])
                aopts['data']['mu'][:,1] = model.vemspace.interpolation(S1, aopts['HB'])
                problem['x0'] = aopts['data']['mu']

                optalg = SteepestDescentAlg(problem, options)
                x, f, g, diff = optalg.run(maxit=200)
                problem['mesh'] = mesh###多边形网格
                problem['x0'] = x
                problem['rho'] = model.rho
                self.problem = problem
                q = np.zeros(model.rho.shape)

                q[:,0] = model.q0[:,-1]
                q[:,1] = model.q1[:,-1]


                if diff < options['FunValDiff']:
                   if (np.max(problem['rho'][:,0]) < 1) and (np.min(problem['rho'][:,0]) >0):
                       break

            myfile=open(moptions['rdir'] +'/'+str(int(moptions['chiN']))+'mesh.bin','wb')
            import pickle
            pickle.dump(problem['mesh'], myfile)
            myfile.close()
            model.save_data(moptions['rdir']+'/'+ str(int(moptions['chiN']))+'.mat')

            moptions['chiN'] +=5
            if moptions['chiN'] >60:
                break


options = {
        'MaxIters': 5000,
        'MaxFunEvals': 5000,
        'NormGradTol': 1e-6,
        'FunValDiff': 1e-6,
        'StepLength': 2,
        'StepTol': 1e-14,
        'etarefTol':0.1,
        'Output': True
        }

moptions = scftmodel2d_options(
        nspecies= 2,
        nblend = 1,
        nblock = 2,
        ndeg = 100,
        fA = 0.5,
        chiAB = 0.20,
        dim = 2,
        T0 = 40,
        T1 = 160,
        nupdate = 1,
        order = 2,
        rdir = sys.argv[1])


Halftest = HalfEdgeAVEMTest(fieldstype=4, moptions=moptions,
        optoptions=options)

Halftest.run()
