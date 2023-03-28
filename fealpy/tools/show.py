import sys
import numpy as np
from types import ModuleType



def show_error_table(N, errorType, errorMatrix, 
        f='e', pre=4, sep=' & ',
        out=sys.stdout, end='\n'):

    flag = False
    if type(out) == type(''):
        flag = True
        out = open(out, 'w')

    string = ''
    n = errorMatrix.shape[1] + 1
    print('\\begin{table}[!htdp]', file=out, end='\n')
    print('\\begin{tabular}[c]{|'+ n*'c|' + '}\hline', file=out, end='\n')

    s = 'Dof' + sep + np.array2string(N, separator=sep,
            )
    s = s.replace('\n', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    print(s, file=out, end=end)
    print('\\\\\\hline', file=out)

    n = len(errorType)
    ff = '%.'+str(pre)+f
    for i in range(n):
        first = errorType[i]
        line = errorMatrix[i]
        s = first + sep + np.array2string(line, separator=sep,
                precision=pre, formatter=dict( float = lambda x: ff % x ))
        
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=out, end=end)
        print('\\\\\\hline', file=out)

        order = np.log(line[0:-1]/line[1:])/np.log(2)
        s = 'Order' + sep + '--' + sep + np.array2string(order,
                separator=sep, precision=2)
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=out, end=end)
        print('\\\\\\hline', file=out)

    print('\\end{tabular}', file=out, end='\n')
    print('\\end{table}', file=out, end='\n')

    if flag:
        out.close()

def showmultirate(plot, k, N, errorMatrix, labellist, optionlist=None, lw=1,
        ms=4, propsize=10, computerate=True):
    if isinstance(plot, ModuleType):
        fig = plot.figure()
        fig.set_facecolor('white')
        axes = fig.gca()
       
    else:
        axes = plot
    if optionlist is None:
        optionlist = ['k-*', 'r-o', 'b-D', 'g-->', 'k--8', 'm--x','r-.x', 'b-.+', 'b-.h', 'm:s', 'm:p', 'm:h']

    m, n = errorMatrix.shape
    for i in range(m):
        if len(N.shape) == 1:
            showrate(axes, k, N, errorMatrix[i], optionlist[i], label=labellist[i], lw=lw, ms=ms, computerate=computerate)
        else:
            showrate(axes, k, N[i], errorMatrix[i], optionlist[i], label=labellist[i], lw=lw, ms=ms, computerate=computerate)
    axes.legend(loc=3, framealpha=0.2, fancybox=True, prop={'size': propsize})
    return axes

def showrate(axes, k, N, error, option, label=None, lw=1, ms=4, computerate=True):
    axes.set_xlim(left=N[0]/2, right=N[-1]*2)
    line0, = axes.loglog(N, error, option, lw=lw, ms=ms, label=label)
    if computerate:
        if isinstance(k, int):
            c = np.polyfit(np.log(N[k:]), np.log(error[k:]), 1)
            s = 0.75*error[k]/N[k]**c[0]
            line1, = axes.loglog(N[k:], s*N[k:]**c[0], label='C$N^{%0.4f}$'%(c[0]),
                    lw=lw, ls=line0.get_linestyle(), color=line0.get_color())
        else:
            c = np.polyfit(np.log(N[k]), np.log(error[k]), 1)
            s = 0.75*error[k[0]]/N[k[0]]**c[0]
            line1, = axes.loglog(N[k], s*N[k]**c[0], label='C$N^{%0.4f}$'%(c[0]),
                    lw=lw, ls=line0.get_linestyle(), color=line0.get_color())

