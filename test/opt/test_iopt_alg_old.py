import time

from Function import Function
from Function_plot import Function_plot
from fealpy.opt import COA
from fealpy.opt import HBA
from fealpy.opt import SAO
from fealpy.opt import QPSO
from fealpy.backend import backend_manager as bm


#运行
def test_alg(alg_name):
    for i in range(11, 12):
        F = 'F' + str(i) 

        N = 100
        T = 1000

        f = Function(F)
        fobj, lb, ub, dim = f.Functions()

        f_plot = Function_plot(F)
        # X, Y, Z = f_plot.Functions_plot()

        start_time = time.time()

        algorithm_dict = {  
            'HBA': HBA,  
            'COA': COA,  
            'QPSO': QPSO,  
            'SAO': SAO  
        }  
        if alg_name in algorithm_dict:   
            C = algorithm_dict[alg_name](N, dim, ub, lb, T, fobj)  
        else:  
            print(f"Unrecognized algorithm name: {alg_name}")  

        #C = HBA(N,dim,  ub, lb, T,fobj)#N, dim,  ub, lb, Max_iter, fobj
        Best_pos, Best_score, Convergence_curve = C.cal()
        end_time = time.time()

        #结果展示
        print('F', i,'----------------------------------------')
        print('The best solution obtained is:', Best_pos)
        print('The best optimal value of the objective function found by HBA is:', Best_score)
        print('HBA execution time:', end_time - start_time)
        print('----------------------------------------')

# test("HBA")
'''
    # 绘制参数空间图像、收敛曲线
    fig = plt.figure(figsize=(10, 5))

    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.plot_surface(X, Y, Z, cmap='viridis')
    ax_3d.set_title('Function: ' + F)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    ax_conv = fig.add_subplot(122)
    #print(Convergence_curve[0])
    if F not in ["F6","F15","F16","F17","F20","F23"]:
        ax_conv.semilogy(Convergence_curve[0], 'r-', linewidth=2)
    else:
        x = list(range(0, T))
        cuve_f_list = []
        cuve_f_list = list(Convergence_curve[0])
        ax_conv.plot(list(x), cuve_f_list,'r-', linewidth=2)
    #ax_conv.semilogy(Convergence_curve, 'g-', linewidth=2)
    ax_conv.set_title('Objective space')
    ax_conv.set_xlabel('iter')
    ax_conv.set_ylabel('Best score obtained so far')
    ax_conv.grid(True)

    plt.subplots_adjust(wspace= 0.8)

    # 保存图像
    save_folder = 'plots1'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, 'Function_' + F + '_plot.png'))

    #plt.show()
'''
