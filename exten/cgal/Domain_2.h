#ifndef DOMAIN_2_H
#define DOMAIN_2_H

#include "/home/why/software/itaps/include/GeometryQueryTool.hpp"
#include "/home/why/software/itaps/include/GeometryModifyTool.hpp"
#include "/home/why/software/itaps/include/Body.hpp"
#include "/home/why/software/itaps/include/RefEdge.hpp"
#include "/home/why/software/itaps/include/RefVertex.hpp"
#include "/home/why/software/itaps/include/CubitDefines.h"
#include "/home/why/software/itaps/include/CubitBox.hpp"
#include "/home/why/software/itaps/include/InitCGMA.hpp"

#include <iostream>

namespace CGAL {

template<class Gt>
class Domain_2 {

public:
    typedef typename Gt::FT FT;
    typedef typename Gt::Point_2 Point_2;


private:
    std::vector<Point_2> m_dpts; //!标记区域内部的点集

public:

    Domain_2()
    {
        CubitStatus s = InitCGMA::initialize_cgma("OCC");
        if (CUBIT_SUCCESS != s) return 1;
    }


    void read_domain(std::istream & in)
    {

       int num_points,num_curves,num_holes;
       int l,r,num_c;
       FT x,y;
       char c;

       in>> num_points >> num_curves >> num_holes;

       std::vector<RefVertex*> vts(num_points);
       for(int i=0 ; i < num_points; i++)
       {
           in>>x>>y;
           vts[i] = GeometryModifyTool::instance()->make_RefVertex(CubitVector(x,y,0));
       }

       char c;
       int
       for(int i=0; i < num_curves; i++)
       {
           DLIList<CubitVector*> v;
           CubitVector cv(0.0,0.0,0.0);
           in>>c;
           switch(c)
           {
           case 's'://spline
               GeometryType gt = SPLINE_CURVE_TYPE;
               v.clean_out();
               in>>l>>r>>num_c;
               for(int k =0; k < num_c; k++)
               {
                   in>>cv[0]>>cv[1];
                   vector.append(cv);
               }
               RefEdge* curve = GeometryModifyTool::instance()->make_RefEdge(gt,vts[l],vts[r],v);
               if(!curve)
               {
                 std::cerr<<"failed to make curve\n"<<std::endl;
               }
               break;
           case 'c'://circle
                GeometryModifyTool::instance()->create_arc_radius()
               break;
           case 'l'://secgment
               ;
               break;
           default:
               std::cerr<<c<<" is not a support curve type!"<<std::endl;
               break;
           }

       }

    }

};
}
#endif // DOMAIN_2_H
