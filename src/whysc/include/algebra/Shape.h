#ifndef Shape_h
#define Shape_h

#include <initializer_list>
#include <numeric>
#include <vector>
#include <string>

namespace WHYSC {

namespace Algebra {

class Shape
{
public:
    typedef std::vector<int>::size_type size_type;
private:
    std::vector<int> m_shape;
public:

    Shape(): m_shape{0}
    {
    }

    Shape(int a): m_shape{a}
    {
    }

    Shape(int a, int b):m_shape{a, b}
    {
    }

    Shape(int a, int b, int c):m_shape{a, b, c}
    {
    }

    Shape(int a, int b, int c, int d):m_shape{a, b, c, d}
    {
    }

    Shape(const std::initializer_list<int> &l): m_shape(l)
    { 
    }

    void reshape(int a)
    {
        auto NN = size();
        if( a == -1 || a == NN)
        {
            m_shape.resize(1);
            m_shape[0] = NN;
        }
        else
        {
            std::string err = "cannot reshape array of size " + 
                    std::to_string(NN) + " into shape (" 
                    + std::to_string(a) + ", )";
            throw err.c_str();
        }
    }

    void reshape(int a, int b)
    {
        //TODO: a == -1 or b == -1
        auto NN = size();
        if( a*b == NN)
        {
            m_shape.resize(2);
            m_shape[0] = a;
            m_shape[1] = b;
        }
        else
        {
            std::string err = "cannot reshape array of size " + 
                    str() + " into shape (" 
                    + std::to_string(a) + "," 
                    + std::to_string(b) + ",)";
            throw err.c_str();
        }
    }

    void reshape(int a, int b, int c)
    {
        //TODO: a == -1 or b == -1
        auto NN = size();
        if( a*b*c == NN)
        {
            m_shape.resize(3);
            m_shape[0] = a;
            m_shape[1] = b;
            m_shape[2] = c;
        }
        else
        {
            std::string err = "cannot reshape array of size "; 
            throw err.c_str();
        }
    }

    void reshape(const std::initializer_list<int> &l)
    {
        auto NN = size();
        auto N = std::accumulate(l.begin(), l.end(), 1, std::multiplies<int>());
        if( N == NN)
        {
            m_shape.insert(m_shape.begin(), l.begin(), l.end());
        }
    }

    size_type size()
    {
        auto NN = std::accumulate(m_shape.begin(),  m_shape.end(), 1, std::multiplies<int>());
        return NN;
    }

    std::string str() const noexcept
    {
        std::string out = "(";
        for(auto val:m_shape)
        {
            out += std::to_string(val) + ",";
        }
        out += ")\n";

        return out;
    }

    int & operator [] (const int i)
    {
        return m_shape[i];
    }

    const int & operator [] (const int i) const
    {
        return m_shape[i];
    }
};

template<typename OS>
OS & operator<<(OS& os, const Shape& shape) 
{
    os << shape.str();
    return os;
}

} // end of namespace Algebra

} // end of namespace WHYSC

#endif // end of Shape_h
