#ifndef AlgebraKernel_h
#define AlgebraKernel_h

#include "Matrix.h"
#include "Array.h"
namespace WHYSC {

template<typename F=double, typename I=int>
class Algebra_kernel
{
public:
    typedef typename AlgebraObject::Matrix<F, 2, 2> Matrix22;
    typedef typename AlgebraObject::Matrix<F, 3, 3> Matrix33;
    typedef typename AlgebraObject::Matrix<F, 4, 4> Marrix44;
    typedef typename AlgebraObject::Array<F, I> Array2d;

};

} // end of namespace WHYSC
#endif // end of AlgebraKernel_h
