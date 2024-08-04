import math
import matplotlib.pyplot as plt
import random
from fealpy.experimental.iopt import initialize
from ..backend import backend_manager as bm

class COA:
    def __init__(self, N, dim,  ub, lb, Max_iter, fobj):
        """
        
        """
        self.N = N
        self.Max_iter = Max_iter
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.fobj = fobj
        #self.init_x = init_x
    
    def p_obj(self, x):
        y =  0.2 * ( 1 / (math.sqrt(2 * bm.pi) * 3)) * math.exp(-(x-25) ** 2 / (2*3 ** 2))
        return y 
    
    def cal(self):
        cuve_f = bm.zeros((1, self.Max_iter))

        #I = initialize_2(self.N, self.dim, self.ub, self.lb, self.Max_iter, self.fobj, self.init_x)
        
        I = initialize(self.N, self.dim, self.ub, self.lb, self.Max_iter, self.fobj)
        fitness_f, X,  best_position, Best_fitness = I.initialize()
        
        fitness_f = fitness_f.reshape((1,self.N))
        Xnew = bm.copy(X)
        global_position = best_position
        global_fitness = Best_fitness
        cuve_f[:, 0]=Best_fitness
        t=0

        while t < self.Max_iter:
            C = 2-(t / self.Max_iter)
            temp = random.random() *15 + 20
            xf = (best_position + global_position) / 2
            Xfood = best_position


            """
            for i in range(self.N):            
                if temp > 30:
                    #summer resort stage
                    rand = random.random()
                    if rand < 0.5:
                        Xnew_i = X[i,:] + C * bm.random.rand(1, self.dim) * (xf - X[i,:])
                        for j in range(self.dim):
                            Xnew[i][j] = Xnew_i[:,j]           
                    else:
                        #competition stage
                        for j in range(self.dim):
                            z = round(random.random() * (self.N - 1)) 
                            Xnew[i][j] = X[i][j] - X[z][j] + xf[j]
                else:
                    # foraging stage
                    P = 3 * random.random() * fitness_f[:,i] / self.fobj( Xfood )                   
                    if P > 2:
                        Xfood = bm.exp(-1/P) * Xfood
                        
                        for j in range(self.dim):                        
                            Xnew[i][j] = X[i][j] + math.cos(2*bm.pi*random.random()) * Xfood[j] * self.p_obj(temp) - math.sin(2 * bm.pi* random.random()) * Xfood[j] * self.p_obj(temp)
                    else:
                        Xnew_i = (X[i,:]-Xfood)* self.p_obj(temp)+ self.p_obj(temp) * bm.random.rand(1, self.dim) * X[i,:]
                        for j in range(self.dim):
                            Xnew[i][j] = Xnew_i[:,j]
            """
            
            for i in range(self.N):  
                for j in range(self.dim):  
                    if isinstance(self.ub, (int, float)):  
                        Xnew[i][j] = max(min(self.ub, Xnew[i][j]), self.lb)  
                    else:  
                        Xnew[i][j] = max(min(self.ub[j], Xnew[i][j]), self.lb[j])
       
            global_position = Xnew[0,:]
            global_fitness = self.fobj(global_position)

            for i in range(self.N):     
                #Obtain the optimal solution for the updated population
                new_fitness = self.fobj(Xnew[i,:])
                if new_fitness<global_fitness:
                    global_fitness = new_fitness
                    global_position = Xnew[i,:]
                #Update the population to a new location
                if new_fitness < fitness_f[:,i] :
                    fitness_f[:,i]  = new_fitness

                    for j in range(self.dim):
                        #X[i][j] = Xnew[i][j].copy()
                        X[i][j] = bm.copy(Xnew[i][j])

                    if  fitness_f[:,i]  < Best_fitness:
                        Best_fitness= fitness_f[:,i] 
                        best_position = X[i,:]
           
            cuve_f[:,t] = Best_fitness
            t = t + 1
            if t % 50 == 0:
                print("COA" + " iter" , t , ":", Best_fitness)

        return Best_fitness,best_position,cuve_f






