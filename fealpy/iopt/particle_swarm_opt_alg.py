import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage
from scipy.ndimage import label
import matplotlib.pyplot as plt

class PSOProblem:
    def __init__(self,MAP,dataS,dataE):
        self.MAP=MAP
        self.dataS=dataS
        self.dataE=dataE
        self.data={}

    def builddata(self):
        self.data["R"]=1
        L, N = label(self.MAP)
        landmark = np.vstack([np.argwhere(L == i) for i in range(1, np.max(L) + 1)])
        self.data['landmark']=np.array(landmark)
        self.data['map']=self.MAP
        node = [[j, i] for i in range(self.MAP.shape[0]) for j in range(self.MAP.shape[1]) if self.MAP[i, j] == 0]
        self. data['node']=np.array(node)
        self.data['D']=squareform(pdist(self.data['node']))
        p1,p2=np.where(self.data['D'] <= self.data['R'])
        D=self.data['D'][(p1,p2)].reshape(-1, 1)
        self.data['net']=sp.csr_matrix((D.flatten(),(p1, p2)),shape=self.data['D'].shape)
        self.data['noS']=np.where((self.data['node'][:, 0]==self.dataS[0]) & (self.data['node'][:, 1]==self.dataS[1]))[0][0]
        self.data['noE']=np.where((self.data['node'][:, 0]==self.dataE[0]) & (self.data['node'][:, 1]==self.dataE[1]))[0][0]
        self.data['numLM0']=1
        return self.data
    
    def fitness(self,X):
        result={}
        sorted_numbers=np.argsort(X)
        G=nx.DiGraph(self.data["net"])
        # 初始化距离列表和路径列表
        distances=[]
        paths=[]
        sorted_numbers_flat=sorted_numbers[0:self.data['numLM0']]
        sorted_numbers_flat = [element for element in sorted_numbers_flat]
        path0=[self.data['noS']]+sorted_numbers_flat+[self.data['noE']]
        for i in range(0, len(path0)-1):
            source=path0[i]
            target=path0[i+1]
            path=nx.shortest_path(G, source=source, target=target)
            distance=nx.shortest_path_length(G,source=source,target=target,weight=None)  
            distances.append(distance)
            paths.append(path)
        fit=sum(distances)
        combined_list=[]
        for sublist in paths:
            combined_list.extend(sublist)
        result["fit"]=fit
        result["path"]=combined_list
        return fit,result
    
    def printMAP(self,result):
        b=self.MAP.shape
        self.MAP = 1 - self.MAP
        plt.scatter(self.dataS[0], self.dataS[1], color='blue', s=100)
        plt.scatter(self.dataE[0], self.dataE[1], color='green', s=100)
        plt.imshow(self.MAP[::1], cmap='gray')
        xx=np.linspace(0,b[1],b[1])-0.5
        yy=np.zeros(b[1])-0.5
        x=np.zeros(b[0])-0.5
        y=np.linspace(0,b[0],b[0])-0.5
        for i in range(0, b[0]):
            yy=yy+1
            plt.plot(xx,yy,'-',color='black')
        for i in range(0, b[1]):
            x=x+1
            plt.plot(x,y,'-',color='black')
        plt.xticks([])
        plt.yticks([])
        plt.title('The optimal path')
        xpath=self.data["node"][result["path"], 0]
        ypath=self.data["node"][result["path"], 1]
        print("The opimal path coordinates：")
        for x, y in zip(xpath, ypath):
            print("({}, {})".format(x, y))
        plt.plot(xpath,ypath,'-',color='red')
        plt.plot([xpath[-1], self.dataE[0]],[ypath[-1], self.dataE[1]],'-',color='red')
        plt.show()

class PSO:
    def __init__ (self, N, dim, ub, lb, MaxIT, fobj):
        self.N = N
        self.dim = dim
        self.ub = ub
        self.vub = ub * 0.2
        self.lb = lb
        self.vlb = lb * 0.2
        self.MaxIT = MaxIT
        self.fobj = fobj
        self.best = np.zeros(self.MaxIT)
        self.gbest = np.zeros((1,self.dim))
        self.gbest_f = 0

    def initialize(self):
        #种群
        a=np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb 
        fit=np.zeros((self.N,1))
        for i in range(0, self.N):
            fit[i,0], _ = self.fobj(a[i,:])
        #个体最优
        pbest=a.copy()
        pbest_f=fit.copy()
        #全局最优
        gbest_idx = np.argmin(pbest_f)
        self.gbest_f = pbest_f[gbest_idx]
        self.gbest = pbest[gbest_idx]
        return fit, a, pbest, pbest_f
    
    def updatePGbest(self,fit,x,pbest_f,pbest):
        pbest_f, pbest = (fit, x) if fit < pbest_f else (pbest_f, pbest)
        gbest_f,gbest = (pbest_f, pbest) if pbest_f < self.gbest_f else (self.gbest_f, self.gbest)
        return pbest_f, pbest, gbest_f, gbest

    def cal(self):
        c1=2
        c2=2
        fit, x, pbest, pbest_f = self.initialize()
        v=np.zeros([self.N,self.dim])
        for it in range(0, self.MaxIT):
            w=0.9-0.4*(it/self.MaxIT)
            for i in range(0, self.N):
                v[i,:]=w*v[i,:]+c1*np.random.rand()*(pbest[i,:]-x[i,:])+c2*np.random.rand()*(self.gbest-x[i,:])
                v[i,:] = np.clip(v[i,:], self.vlb, self.vub)
                x[i,:]=x[i,:]+v[i,:]
                x[i,:] = np.clip(x[i,:], self.lb, self.ub)
                fit[i,0],_ = self.fobj(x[i,:])
                pbest_f[i, 0],pbest[i, :],self.gbest_f,self.gbest=self.updatePGbest(fit[i,0],x[i,:],pbest_f[i, 0],pbest[i, :])
            self.best[it] = self.gbest_f   

class QPSO(PSO):
    def cal(self):
        fit, a, pbest, pbest_f = self.initialize()
        #主循环
        for it in range(0,self.MaxIT):
            alpha=1-(it+1)/(2*self.MaxIT)
            mbest=sum(pbest)/self.N
            for i in range(0,self.N):
                phi=np.random.rand(1, self.dim)
                p=phi*pbest[i,:]+(1-phi)*self.gbest
                u=np.random.rand(1,self.dim)
                rand=np.random.rand()
                a[i,:]=p+alpha*np.abs(mbest-a[i,:])*np.log(1./u)*(1-2*(rand>=0.5))
                a[i]=np.clip(a[i],self.lb,self.ub)
                fit[i,0], _ = self.fobj(a[i,:])
                pbest_f[i, 0],pbest[i, :],self.gbest_f,self.gbest=self.updatePGbest(fit[i,0],a[i,:],pbest_f[i, 0],pbest[i, :])   
            self.best[it] = self.gbest_f   