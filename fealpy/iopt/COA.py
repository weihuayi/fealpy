import numpy as np 
import matplotlib.pyplot as plt
import random
from initialize import initialize
from Get_fun import Get_F
import time

class COA:
    def __init__(self, N, T, dim, UB, LB, fobj):
        """
        """
        self.N = N
        self.T = T
        self.dim = dim
        self.UB = UB
        self.LB = LB
        self.fobj = fobj
    
    def p_obj(self, x):
        y = y = 0.2 * ( 1 / (np.sqrt(2 * np.pi) * 3)) * np.exp(-(x-25) ** 2 / (2*3 ** 2))
        return y 
    
    def cal(self):
        cuve_f = np.zeros((1, self.T))

        #X = self.initialization()
        I = initialize(self.N, self.dim, self.UB, self.LB, self.T, self.fobj)
        fitness_f, X, global_Cov, best_position, Best_fitness = I.initialize()
        
        fitness_f = fitness_f.reshape((1,self.N))
        global_Cov = global_Cov.reshape((1, self.T))

        Xnew = X.copy()


        global_position = best_position
        global_fitness = Best_fitness
        cuve_f[:,0]=Best_fitness
        t=0

        while t < self.T:
            C = 2-(t / self.T)
            temp = random.random() *15 + 20
            xf = (best_position + global_position) / 2
            Xfood = best_position
            

            for i in range(self.N):
             
                if temp > 30:
                    #summer resort stage
                    rand = random.random()
                    if rand < 0.5:
                        Xnew_i = X[i,:] + C * np.random.rand(1, self.dim) * (xf - X[i,:])
                
                        for j in range(self.dim):
                            Xnew[i][j] = Xnew_i[:,j]
                  
                    
                    else:
                        #competition stage
                        for j in range(self.dim):
                            z = round(random.random() * (self.N - 1)) 
                            Xnew[i][j] = X[i][j] - X[z][j] + xf[j]
                else:
                    # foraging stage
                    P = 3 * random.random() * fitness_f[:,i] / self.fobj( Xfood.reshape((1, self.dim)) )
                    #print(fitness_f)
                    #print(fitness_f[:,i])
                   
                    if P > 2:
                        Xfood = np.exp(-1/P) * Xfood
                        #print("Xfood",Xfood)
                        
                        for j in range(self.dim):
                          
                            Xnew[i][j] = X[i][j] + np.cos(2*np.pi*random.random()) * Xfood[j] * self.p_obj(temp) - np.sin(2 * np.pi* random.random()) * Xfood[j] * self.p_obj(temp)
                    else:
                        Xnew_i = (X[i,:]-Xfood)* self.p_obj(temp)+ self.p_obj(temp) * np.random.rand(1, self.dim) * X[i,:]
                        for j in range(self.dim):
                            Xnew[i][j] = Xnew_i[:,j]

            for i in range(self.N):
                for j in range(self.dim):
                    if isinstance(self.UB, int) :
                        Xnew[i][j] = min(self.UB, Xnew[i][j])
                        Xnew[i][j] = max(self.LB, Xnew[i][j])
                    
                    else:
                        Xnew[i][j] = min(self.UB[j], Xnew[i][j])
                        Xnew[i][j] = max(self.LB[j], Xnew[i][j])  
            
       
            global_position = Xnew[0,:]
            global_fitness = self.fobj(global_position.reshape((1, self.dim)))

            for i in range(self.N):     
                #Obtain the optimal solution for the updated population
                new_fitness = self.fobj(Xnew[i,:].reshape((1, self.dim)))
                if new_fitness<global_fitness:
                    global_fitness = new_fitness
                    global_position = Xnew[i,:]
                #Update the population to a new location
                if new_fitness < fitness_f[:,i] :
                    fitness_f[:,i]  = new_fitness

                    for j in range(self.dim):
                        X[i][j] = Xnew[i][j].copy()

                    #X[i,:] = Xnew[i,:]
                    if  fitness_f[:,i]  < Best_fitness:
                        Best_fitness= fitness_f[:,i] 
                        best_position = X[i,:]
            global_Cov[:,t] = global_fitness
            cuve_f[:,t] = Best_fitness
            t = t + 1
            #if np.mod(t, 50) == 0:
            #    print("COA" + " iter" , t , ":", Best_fitness)
        best_fun = Best_fitness

        return best_fun,best_position,cuve_f,global_Cov





F_name = 'F1'
f = Get_F(F_name)
fobj, LB, UB, dim = f.G_F()


N = 100
T  = 200
dim = 10
#UB = 100
#LB = -100

#UB = np.array([100,100,100,100,100,100,100,100,100,90])
#LB = np.array([-100,-100,-100,-100,-100,-100,-100,-100,-100,-90])


start_time=time.perf_counter()
C = COA(N, T, dim, UB, LB, fobj)
#X = C.initialization()
best_fun,best_position,cuve_f,global_Cov = C.cal()
end_time=time.perf_counter()
running_time=end_time-start_time
print("running_time：",running_time)
print('The best-obtained solution by COA is : ' , best_position[0])
print('The best optimal value of the objective funciton found by COA is : ' , best_fun[0])







  


  
# plt.subplot(1, 2, 1)  
plt.semilogy(cuve_f[0], linewidth=3)  
  
# Third subplot: Convergence plot for global_Cov  
# plt.subplot(1, 2, 2)  
# plt.semilogy(global_Cov)  
# plt.xlabel('Iteration#')  
# plt.ylabel('Best fitness so far')  
# plt.legend(['COA'])  
  
# Adjust the spacing between subplots  
plt.tight_layout()  
  
# Show the figure  
plt.show() 