#ifndef Level_set_function_h
#define Level_set_function_h

#include <vector>
#include <algorithm>
#include <cmath>

namespace iMath {

namespace LevelSetFunction {


template<class GK>
class Level_set_function
{
public:
    typedef typename GK::Point_2 Point_2;
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Vector_2 Vector_2;
    typedef typename GK::Vector_3 Vector_3;
public:
    double operator () (const Point_2 &p){return 0.0;}
    int sign(const Point_2 &p) {return 0;}
    double operator () (const Point_3 &p){return 0.0;}
    int sign(const Point_3 &p){return 0;}

    int sign(const double val)
    {
        int s;
        if(val > 0)
            s = 1;
        else if(val < 0)
            s = -1;

        if(std::abs(val) < GK::eps())
            s = 0;
        return s;
    }
};

template<class GK>
class Circle: public Level_set_function<GK>
{
public:
    typedef typename GK::Point_2 Point_2;
    typedef Level_set_function<GK> Base;
public:
    /*
     * Constructor
     *
     */
    Circle(): _center(0, 0), _r(1)
    {
    }

    Circle(const double x, const double y, const double r): _center(x, y), _r(r)
    {
    }

    Circle(const Circle & c)
    {
        _center = c._center;
        _r = c._r;
    }

    double operator () (const Point_2 & p)
    {

        return  (p-_center).squared_length() - _r*_r;
    }

    int sign(const Point_2 & p)
    {
        double val = this->operator () (p);
        return Base::sign(val);
    }

    Point_2 point(const double * p)
    {
        return GK::point_2(p);
    }

private:
    Point_2 _center;
    double _r;
    
};

template<class GK>
class Signed_distance_circle: public Level_set_function<GK>
{
public:
    typedef typename GK::Point_2 Point_2;
    typedef Level_set_function<GK> Base;
public:
    /*
     * Constructor
     *
     */
    Signed_distance_circle(): _center(0, 0), _r(1)
    {
    }

    Signed_distance_circle(const double x, const double y, const double r): _center(x, y), _r(r)
    {
    }

    Signed_distance_circle(const Signed_distance_circle & c)
    {
        _center = c._center;
        _r = c._r;
    }

    double operator () (const Point_2 & p)
    {

        return  std::sqrt((p-_center).squared_length()) - _r;
    }

    int sign(const Point_2 & p)
    {
        double val = this->operator () (p);
        return Base::sign(val);
    }

    Point_2 point(const double * p)
    {
        return GK::point_2(p);
    }

private:
    Point_2 _center;
    double _r;
    
};


template<class GK>
class Sphere:public Level_set_function<GK>
{
public:
    typedef typename GK::Point_3 Point_3;
    typedef Level_set_function<GK> Base;
public:
    Sphere():_center(0.0, 0.0, 0.0), _r(1) {}
    Sphere(const double x, const double y, const double z, const double r):_center(x, y, z), _r(r){}
    Sphere(const Sphere & s)
    {
        _center = s._center;
        _r = s._r;
    }

    double operator () (const Point_3 & p)
    {

        return  (p-_center).squared_length() - _r*_r;
    }

    int sign(const Point_3 & p)
    {
        double val = this->operator () (p);
        return Base::sign(val);
    }

    Point_3 point(const double *p)
    {
        return GK::point_3(p);
    }

private:
    Point_3 _center;
    double _r;
};

template<class GK>
class Signed_distance_sphere:public Level_set_function<GK>
{
public:
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Vector_3 Vector_3;
    typedef Level_set_function<GK> Base;
public:
    Signed_distance_sphere():_center(0.0, 0.0, 0.0), _r(1.0) {}
    Signed_distance_sphere(const double x, const double y, const double z, const double r):_center(x, y, z), _r(r){}
    Signed_distance_sphere(const Point_3 & c, const double r):_center(c), _r(r){}
    template<class S>
    Signed_distance_sphere(const S & s)
    {
        _center = s._center;
        _r = s._r;
    }

    double operator () (const Point_3 & p)
    {
        return  std::sqrt((p-_center).squared_length()) - _r;
    }

    int sign(const Point_3 & p)
    {
        double val = this->operator () (p);
        return Base::sign(val);
    }

    Vector_3 gradient(const Point_3 & p)
    {
        double d = std::sqrt((p - _center).squared_length());
        Vector_3 v = (p - _center)/d;
        return v;
    }

    Point_3 project(const Point_3 & p)
    {
        double d = this->operator ()(p);
        Vector_3 v = gradient(p);
        Point_3 q = p - d*v;
        return q;
    }

    Point_3 center() { return _center;}
    double radius() { return _r;}
private:
    Point_3 _center;
    double _r;
};


template<class GK, class Point>
class Union: public Level_set_function<GK>
{
public:
    typedef Level_set_function<GK> Base;
public:
    Union(Base & lsf_1, Base & lsf_2)
    {
        _lsfs.push_back(lsf_1);
        _lsfs.push_back(lsf_2);
    }

    Union(const std::vector<Base> & lsfs): _lsfs(lsfs)
    {

    }

    double operator () (const Point & p)
    {
        int n = _lsfs.size();
        std::vector<double> vals(n);
        for(int i = 0; i < n; i++)
            vals[i] = _lsfs[i](p);

        double max_val = std::min_element(vals.begin(), vals.end());
        return max_val;
    }

    int sign(const Point & p)
    {
        double val = this->operator () (p);
        return Base::sign(val);
    }
private:
    std::vector<Base> _lsfs;
};

} // end of namespace LevelSetFunction

} // end of namespace iMath
#endif // end of Level_set_function_h
