#ifndef CEC17_COMMON_H
#define CEC17_COMMON_H

#include <cfloat>
#include <cstddef>  

#ifndef INFINITY
#define INFINITY DBL_MAX
#endif

#ifdef _WIN32
#define CEC17_API __declspec(dllexport)
#else
#define CEC17_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double *OShift, *M, *y, *z, *x_bound;
extern int ini_flag, n_flag, func_flag, *SS;


CEC17_API void cec17_init(int func_num, int dim);
CEC17_API double cec17_evaluate(const double *x, int dim);
CEC17_API void cec17_cleanup(void);
void cec17_test_func(double *, double *, int, int, int);

#ifdef __cplusplus
}
#endif

#endif // CEC17_COMMON_H