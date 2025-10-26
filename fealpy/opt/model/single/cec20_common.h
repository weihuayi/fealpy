#ifndef CEC20_COMMON_H
#define CEC20_COMMON_H

#include <cfloat>
#include <cstddef> 

#ifndef INFINITY
#define INFINITY DBL_MAX
#endif

#ifdef _WIN32
#define CEC20_API __declspec(dllexport)
#else
#define CEC20_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double *OShift, *M, *y, *z, *x_bound;
extern int ini_flag, n_flag, func_flag, *SS;

CEC20_API void cec20_init(int func_num, int dim);
CEC20_API double cec20_evaluate(const double *x, int dim);
CEC20_API void cec20_cleanup(void);
void cec20_test_func(double *, double *, int, int, int);

#ifdef __cplusplus
}
#endif

#endif // CEC20_COMMON_H