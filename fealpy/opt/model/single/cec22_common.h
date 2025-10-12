#ifndef CEC22_COMMON_H
#define CEC22_COMMON_H

#include <cfloat>
#include <cstddef>  

#ifndef INFINITY
#define INFINITY DBL_MAX
#endif

#ifdef _WIN32
#define CEC22_API __declspec(dllexport)
#else
#define CEC22_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double *OShift, *M, *y, *z, *x_bound;
extern int ini_flag, n_flag, func_flag, *SS;

CEC22_API void cec22_init(int func_num, int dim);
CEC22_API double cec22_evaluate(const double *x, int dim);
CEC22_API void cec22_cleanup(void);
void cec22_test_func(double *, double *, int, int, int);

#ifdef __cplusplus
}
#endif

#endif // CEC22_COMMON_H