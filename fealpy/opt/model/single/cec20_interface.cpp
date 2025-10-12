#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include "cec20_common.h"

double *OShift = nullptr;
double *M = nullptr;
double *y = nullptr;
double *z = nullptr;
double *x_bound = nullptr;
int ini_flag = 0;
int n_flag = 0;
int func_flag = 0;
int *SS = nullptr;

CEC20_API void cec20_init(int func_num, int dim) {
    if (dim != 2 && dim != 5 && dim != 10 && dim != 15 && dim != 20 && dim != 30 && dim != 50 && dim != 100) {
        throw std::invalid_argument("Dimension must be 2, 5, 10, 15, 20, 30, 50 or 100");
    }

    if (ini_flag == 1 && (n_flag != dim || func_flag != func_num)) {
        ini_flag = 0;
        cec20_cleanup();
    }
    
    if (ini_flag == 0) {
        OShift = (double*)malloc(dim * sizeof(double));
        M = (double*)malloc(dim * dim * sizeof(double));
        y = (double*)malloc(dim * sizeof(double));
        z = (double*)malloc(dim * sizeof(double));
        x_bound = (double*)malloc(dim * sizeof(double));
        
        if (!OShift || !M || !y || !z || !x_bound) {
            cec20_cleanup();
            throw std::bad_alloc();
        }
        
        cec20_test_func(nullptr, nullptr, dim, 0, func_num);
        ini_flag = 1;
        n_flag = dim;
        func_flag = func_num;
    }
}

CEC20_API double cec20_evaluate(const double *x, int dim) {
    if (!x || dim != n_flag) return INFINITY;
    if (ini_flag == 0) return INFINITY;
    
    double f;
    cec20_test_func(const_cast<double*>(x), &f, dim, 1, func_flag);
    return f;
}

CEC20_API void cec20_cleanup() {
    free(M); M = nullptr;
    free(OShift); OShift = nullptr;
    free(y); y = nullptr;
    free(z); z = nullptr;
    free(x_bound); x_bound = nullptr;
    
    if (SS) {
        free(SS);
        SS = nullptr;
    }
    
    ini_flag = 0;
    n_flag = 0;
    func_flag = 0;
}