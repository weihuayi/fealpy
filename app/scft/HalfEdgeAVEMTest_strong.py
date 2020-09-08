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
        self.moptions = moptions
        ##多边形网格－空间
        obj = SCFTVEMModel2d(mesh, options=self.moptions)
        mu = obj.init_value(fieldstype=fieldstype)
        self.problem = {'objective': obj, 'mesh': mesh, 'x0': mu}

    def run(self, estimator='grad'):
        problem = self.problem
        options = self.optoptions
        moptions = self.moptions
        model = problem['objective']
        mesh = problem['mesh']
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes,cellcolor='w')
        plt.show()
        #plt.savefig('flower15.png')
        #plt.savefig('flower15.pdf')
        #plt.close()



        optalg = SteepestDescentAlg(problem, options)
        x, f, g, diff = optalg.run(maxit=500)

        q = np.zeros(model.rho.shape)
        while True:
            print('chiN', moptions['chiN'])
            if moptions['chiN'] > 15:
                optalg = SteepestDescentAlg(problem, options)
                x, f, g, diff = optalg.run(maxit=10, eta_ref='etamaxmin')
            q = np.zeros(model.rho.shape)
            q[:,0] = model.q0[:,-1]
            q[:,1] = model.q1[:,-1]
            problem['x0'] = x
            while True:
                mesh = problem['mesh']
                hmesh = HalfEdgeMesh2d.from_mesh(mesh)
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


                hmesh.adaptive(eta, aopts)
                mesh = PolygonMesh.from_halfedgemesh(hmesh)

                model.reinit(mesh)
                aopts['data']['mu'] = np.zeros((model.gdof,2))
                aopts['data']['mu'][:,0] = model.vemspace.interpolation(S0, aopts['HB'])
                aopts['data']['mu'][:,1] = model.vemspace.interpolation(S1, aopts['HB'])
                problem['x0'] = aopts['data']['mu']

                optalg = SteepestDescentAlg(problem, options)
                x, f, g, diff = optalg.run(maxit=100)
                problem['mesh'] = mesh
                problem['x0'] = x
                problem['rho'] = model.rho
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

            moptions['chiN'] +=1
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
        chiAB = 0.15,
        dim = 2,
        T0 = 40,
        T1 = 160,
        nupdate = 1,
        order = 2,
        rdir = sys.argv[2])

mesh = complex_mesh(r=20, filename = sys.argv[1])

Halftest = HalfEdgeAVEMTest(mesh, fieldstype=4, moptions=moptions,
        optoptions=options)

Halftest.run()
