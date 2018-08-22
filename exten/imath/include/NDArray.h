#ifndef NDArray_h
#define NDArray_h

namespace iMath {

namespace LinearAlgebra {

template<class I, class F>
class NDArray 
{
public:
    std::vector<I> shape;
    std::vector<I> strides;
    I ndim;
    I itemsize;
    I size;

    F * data;
    char order;
    bool from_other;
public:
    NDArray(F * _data, I _size)
    {
        data = _data;
        size = _size;
        itemsize = sizeof(F);
        ndim = 1;
        shape.push_back(_size);
    }
};

} // end of namespace LinearAlgebra

} // end of namespace iMath
#endif // end of NDArray_h
