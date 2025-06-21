#ifndef METRIC_FIELD_BASE_2_H
#define METRIC_FIELD_BASE_2_H

namespace CGAL{

template<class K>
class Metric_field_base_2{

public:
    typedef typename K::FT FT;

public:
    inline FT a11(FT x, FT y) const{
        return 1.0;
    }

    inline FT a12(FT x, FT y) const{
        return 0.0;
    }

    inline FT a22(FT x, FT y) const{
        return 1.0;
    }
};

}

#endif // METRIC_FIELD_BASE_2_H
