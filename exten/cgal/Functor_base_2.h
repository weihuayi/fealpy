#ifndef FUNCTOR_BASE_2_H
#define FUNCTOR_BASE_2_H

/**
 * @file Functor_base_2.h
 * @author Huayi Wei
 * @date 2014-03-17 19:36:16
 *
 * @brief 该文件提供一个仿函数的基类
 *
 * 
 */

namespace CGAL{

template<class K>
class A11_functor
{
public:
    typedef typename K::FT FT;
    typedef typename K::Point_2 Point_2;
public:
    inline FT operator ()(FT x, FT y) const{
        return 1.0+100*x*x;
    }

    inline FT operator ()(Point_2 & p) const{
        return operator ()(p[0],p[1]);
    }
};

template<class K>
class A22_functor
{
public:
    typedef typename K::FT FT;
    typedef typename K::Point_2 Point_2;
public:
    inline FT operator ()(FT x, FT y) const{
        return 1.0+100*y*y;
    }

    inline FT operator ()(Point_2 & p) const{
        return operator ()(p[0],p[1]);
    }
};

template<class K>
class A12_functor
{
public:
    typedef typename K::FT FT;
    typedef typename K::Point_2 Point_2;

public:
    inline FT operator ()(FT x, FT y) const{
        return 0.0;
    }

    inline FT operator ()(Point_2 & p) const{
        return operator ()(p[0],p[1]);
    }
};

 
}

#endif // end of FUNCTOR_BASE_2_H
