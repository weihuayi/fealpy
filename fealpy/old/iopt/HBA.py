import numpy as np 
import random
import matplotlib.pyplot as plt  
  
def initialization(N, dim, ub, lb):  
    # 检查up和down的维度  
    # if len(up.shape) == 1:  
        # 如果up和down是一维数组，直接进行广播操作  
    X = np.random.rand(N, dim) * (ub - lb) + lb 
    # elif up.shape[1] > 1:  
    #     # 如果up和down是多列矩阵或二维数组  
    #     X = np.zeros((N, dim))  # 初始化X, X是二维的  
    #     for i in range(dim):  
    #         high = up[i]  
    #         low = down[i]  
    #         # 对于每一维，生成随机数组并缩放到指定范围  
    #         X[:, i] = np.random.rand(N) * (high - low) + low  
    # else:  
    #     # 如果up或down的列数不是1且不大于1，则抛出错误  
    #     raise ValueError("The dimensions of 'up' and 'down' are not compatible.")  
      
    return X  


def fun_calcobjfunc(func, X):  
    # 获取X的行数  
    N = X.shape[0]  
    # print(N)  
    # 初始化结果数组Y  
    Y = np.zeros(N)  
      
    # 对X的每一行应用函数func，并将结果存储在Y中  
    for i in range(N):  
        Y[i] = func(X[i, :])  
      
    return Y  

def Intensity(N, Xprey, X):  
    eps = 0.00001
    # 初始化距离平方的数组  
    di = np.zeros(N)  
    S = np.zeros(N)  
      
    # 计算每个点到Xprey的距离平方  
    for i in range(N):  
        di[i] = np.linalg.norm(X[i, :] - Xprey + eps)**2  
      
    # 计算每个点到下一个点的距离平方，最后一个点到第一个点的距离平方  
    for i in range(N - 1):  
        S[i] = np.linalg.norm(X[i, :] - X[i + 1, :] + eps)**2  

    S[N - 1] = np.linalg.norm(X[N - 1, :] - X[0, :] + eps)**2  
      
    # 初始化强度数组  
    I = np.zeros(N)  
      
    # 计算每个点的强度值  
    for i in range(N):  
        r2 = random.random()  # 生成[0, 1)之间的随机数  
        I[i] = r2 * S[i] / (4 * np.pi * di[i])  
      
    return I  


def HBA(objfunc, dim, lb, ub, tmax, N):  
    beta = 6  # 捕食能力  
    C = 2     # 密度因子的随机数，默认为2
    vec_flag = [1, -1]  
  
    # 种群初始化  
    X = initialization(N, dim, ub, lb)  
  
    # 种群评估  
    fitness = fun_calcobjfunc(objfunc, X)  
    GYbest = np.min(fitness)  
    gbest = np.argmin(fitness)  
    Xprey = X[gbest, :]  
  
     
    CNVG = np.zeros(tmax)  
  
    for t in range(tmax):  
        alpha = C * np.exp(-t / tmax)  # 密度因子 
        I = Intensity(N, Xprey, X)  # 气味强度  
  
        # 创建新的种群  
        Xnew = np.zeros((N, dim))  
        for i in range(N):  
            r = np.random.rand()  
            F = vec_flag[np.random.randint(2)]  #-1和1之间随机选择一个
            for j in range(dim):  
                di = Xprey[j] - X[i, j]   #猎物与蜜獾之间的距离
                if r < 0.5:  
                    r3, r4, r5 = np.random.rand(3) 
                    #Digging phase 更新策略 
                    Xnew[i, j] = Xprey[j] + F * beta * I[i] * Xprey[j] + F * r3 * alpha * di * np.abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))  
                else:  
                    r7 = np.random.rand()
                    #Honey phase 更新策略
                    Xnew[i, j] = Xprey[j] + F * r7 * alpha * di
                    
  
            # 确保更新后的位置不超过边界范围  
            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)  
            # 计算更新后的位置的适应度
            tempFitness = fun_calcobjfunc(objfunc, Xnew)
        
            if tempFitness[i] < fitness[i]:  #更新之后更优的情况
                fitness[i] = tempFitness[i]  
                X[i, :] = Xnew[i, :]  
  
 
        X = np.clip(X, lb, ub)  
        Ybest = np.min(fitness)  
        CNVG[t] = Ybest   #每一次迭代都获得当次最优的
        # print(CNVG) 
        if Ybest < GYbest:  
            GYbest = Ybest  
            Xprey = X[np.argmin(fitness), :]  
  
    Food_Score = GYbest  
    return Xprey, Food_Score, CNVG  
  












'''
def f2(x):
    d = len(x)
    sum = 0
    for i in range(d):
        x_i = x[i]
        sum += i * x_i ** 2 

    y = sum
    return y
'''

def sumqu(x):
    d = len(x)
    i = np.arange(0,d)
    y = i @ (x ** 2)
    
    return y

def f1(x):
    d = len(x)
    sum = 0
    for i in range(d):
        x_i = x[i] ** 2
        sum += x_i

    return sum




if __name__ == "__main__": 
    dim = 30  
    T = 1000 
    Lb = -100  
    Ub = 100  
    N = 30  

    # 调用HBA函数进行优化  
    xmin, fmin, CNVG = HBA(sumqu, dim, Lb, Ub, T, N)  
    # print(CNVG)  
    # 绘制收敛曲线  
    plt.figure()  
    plt.semilogy(CNVG, 'r')  
    plt.xlim([0, T])  
    plt.title('Convergence curve')  
    plt.xlabel('Iteration')  
    plt.ylabel('Best fitness obtained so far')  
    plt.legend(['HBA'])  
    plt.show()  

    # 显示最佳位置和适应度得分  
    print(f"The best location= {xmin}")  
    print(f"The best fitness score = {fmin}")
