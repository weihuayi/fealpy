#!/usr/bin/env python3

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.opt.saddleoptalg import SteepestDescentAlg

from vem2d_problem import adaptive_model,converge_model,converge_apt_model
from SCFTVEMModel2d import scftmodel2d_options


moptions = scftmodel_options(
        nspecies=2,
        nblend=1,
        nblock=2,
        ndeg=100,
        fA=0.2,
        chiAB=0.25,
        dim=2,
        T0=20,
        T1=80,
        nupdate=1,
        order=1,
        )

problem = converge_model(fieldstype=3, options=moptions, dataname=sys.argv[1])


options = {
        'MaxIters': 5000,
        'MaxFunEvals': 5000,
        'NormGradTol': 1e-7,
        'FunValDiff': 1e-6,
        'StepLength': 2,
        'StepTol': 1e-14,
        'Output': True
        }


model = problem['objective']
quadtree = problem['quadtree']
q = problem['q']

while True:
    while True:
        print('NN', quadtree.number_of_nodes())
        aopts = quadtree.adaptive_options(method='mean',maxcoarsen= 10,HB=True)
        mesh =quadtree.to_pmesh()
        h = np.sqrt(mesh.entity_measure('cell'))
        NC = mesh.number_of_cells()

        mu = problem['x0']
        w = np.zeros((mu.shape))
        w[:, 0] = mu[:, 0] - mu[:, 1]
        w[:, 1] = mu[:, 0] + mu[:, 1]
        estimator = sys.argv[2]

        if estimator == 'mix':
            eta = model.mix_estimate(q,w=1)
        if estimator == 'grad':
            eta = model.estimate(q)

        aopts['data'] = {'mu':mu}
        S0=model.vemspace.project_to_smspace(aopts['data']['mu'][:,0])
        S1=model.vemspace.project_to_smspace(aopts['data']['mu'][:,1])
        quadtree.adaptive(eta, aopts)
        print('NN', quadtree.number_of_nodes())
        mesh = quadtree.to_pmesh()
        model.reinit(mesh)
        aopts['data']['mu'] = np.zeros((model.gdof,2))
        aopts['data']['mu'][:,0] =model.vemspace.interpolation(S0, aopts['HB'])
        aopts['data']['mu'][:,1] =model.vemspace.interpolation(S1,aopts['HB'])
        problem['x0'] = aopts['data']['mu']

        optalg = SteepestDescentAlg(problem, options)
        x, f, g, diff = optalg.run()
        problem['rho'] = model.rho
        problem['quadtree'] = quadtree
        problem['x0'] = x
        q = model.rho.copy()
        q[:,0] = model.q0[:,-1]
        q[:,1] = model.q1[:,-1]


        if diff < options['FunValDiff']:
           if (np.max(problem['rho'][:,0]) < 1) and (np.min(problem['rho'][:,0]) >0):
               break

    myfile=open(str(int(moptions['chiN']))+'quadtree.bin','wb')
    import pickle
    pickle.dump(problem['quadtree'], myfile)
    myfile.close()
    model.save_data(str(int(moptions['chiN']))+'.mat')
    np.save(str(int(moptions['chiN']))+'.npy', x)

    moptions['chiN'] += 5
    if moptions['chiN'] >60:
        break


