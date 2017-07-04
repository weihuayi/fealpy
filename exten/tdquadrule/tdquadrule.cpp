# include <cstdlib>
# include <iostream>
# include <fstream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <cstring>

using namespace std;

# include "tdquadrule.hpp"



#include <boost/python.hpp>

BOOST_PYTHON_MODULE(tdquadrule)
{
    using namespace boost::python;
    def("order_num", dunavant_order_num);
    def("rule", dunavant_rule);
    def("degree", dunavant_degree);
    def("rule_num", dunavant_rule_num);
}


//****************************************************************************80

int dunavant_degree ( int rule )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_DEGREE returns the degree of a Dunavant rule for the triangle.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Output, int DUNAVANT_DEGREE, the polynomial degree of exactness of
//    the rule.
//
{
  int degree;

  if ( 1 <= rule && rule <= 20 )
  {
    degree = rule;
  }
  else
  {
    degree = -1;
    cout << "\n";
    cout << "DUNAVANT_DEGREE - Fatal error!\n";
    cout << "  Illegal RULE = " << rule << "\n";
    exit ( 1 );
  }

  return degree;
}
//****************************************************************************80

int dunavant_order_num ( int rule )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_ORDER_NUM returns the order of a Dunavant rule for the triangle.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Output, int DUNAVANT_ORDER_NUM, the order (number of points) of the rule.
//
{
  int order;
  int order_num;
  int *suborder;
  int suborder_num;

  suborder_num = dunavant_suborder_num ( rule );

  suborder = dunavant_suborder ( rule, suborder_num );

  order_num = 0;
  for ( order = 0; order < suborder_num; order++ )
  {
    order_num = order_num + suborder[order];
  }

  delete [] suborder;

  return order_num;
}
//****************************************************************************80

void dunavant_rule ( int rule, int order_num, double xy[], double w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_RULE returns the points and weights of a Dunavant rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Input, int ORDER_NUM, the order (number of points) of the rule.
//
//    Output, double XY[2*ORDER_NUM], the points of the rule.
//
//    Output, double W[ORDER_NUM], the weights of the rule.
//
{
  int k;
  int o;
  int s;
  int *suborder;
  int suborder_num;
  double *suborder_w;
  double *suborder_xyz;
//
//  Get the suborder information.
//
  suborder_num = dunavant_suborder_num ( rule );

  suborder_xyz = new double[3*suborder_num];
  suborder_w = new double[suborder_num];

  suborder = dunavant_suborder ( rule, suborder_num );

  dunavant_subrule ( rule, suborder_num, suborder_xyz, suborder_w );
//
//  Expand the suborder information to a full order rule.
//
  o = 0;

  for ( s = 0; s < suborder_num; s++ )
  {
    if ( suborder[s] == 1 )
    {
      xy[0+o*2] = suborder_xyz[0+s*3];
      xy[1+o*2] = suborder_xyz[1+s*3];
      w[o] = suborder_w[s];
      o = o + 1;
    }
    else if ( suborder[s] == 3 )
    {
      for ( k = 0; k < 3; k++ )
      {
        xy[0+o*2] = suborder_xyz [ i4_wrap(k,  0,2) + s*3 ];
        xy[1+o*2] = suborder_xyz [ i4_wrap(k+1,0,2) + s*3 ];
        w[o] = suborder_w[s];
        o = o + 1;
      }
    }
    else if ( suborder[s] == 6 )
    {
      for ( k = 0; k < 3; k++ )
      {
        xy[0+o*2] = suborder_xyz [ i4_wrap(k,  0,2) + s*3 ];
        xy[1+o*2] = suborder_xyz [ i4_wrap(k+1,0,2) + s*3 ];
        w[o] = suborder_w[s];
        o = o + 1;
      }

      for ( k = 0; k < 3; k++ )
      {
        xy[0+o*2] = suborder_xyz [ i4_wrap(k+1,0,2) + s*3 ];
        xy[1+o*2] = suborder_xyz [ i4_wrap(k,  0,2) + s*3 ];
        w[o] = suborder_w[s];
        o = o + 1;
      }
    }
    else
    {
      cout << "\n";
      cout << "DUNAVANT_RULE - Fatal error!\n;";
      cout << "  Illegal SUBORDER(" << s << ") = " << suborder[s] << "\n";
      exit ( 1 );
    }
  }

  delete [] suborder;
  delete [] suborder_xyz;
  delete [] suborder_w;

  return;
}
//****************************************************************************80

int dunavant_rule_num ( )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_RULE_NUM returns the number of Dunavant rules available.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Output, int DUNAVANT_RULE_NUM, the number of rules available.
//
{
  int rule_num;

  rule_num = 20;

  return rule_num;
}
//****************************************************************************80

int *dunavant_suborder ( int rule, int suborder_num )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBORDER returns the suborders for a Dunavant rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, int DUNAVANT_SUBORDER[SUBORDER_NUM], the suborders of the rule.
//
{
  int *suborder;

  suborder = new int[suborder_num];

  if ( rule == 1 )
  {
    suborder[0] = 1;
  }
  else if ( rule == 2 )
  {
    suborder[0] = 3;
  }
  else if ( rule == 3 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
  }
  else if ( rule == 4 )
  {
    suborder[0] = 3;
    suborder[1] = 3;
  }
  else if ( rule == 5 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
  }
  else if ( rule == 6 )
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 6;
  }
  else if ( rule == 7 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 6;
  }
  else if ( rule == 8 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 6;
  }
  else if ( rule == 9 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 6;
  }
  else if ( rule == 10 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 6;
    suborder[4] = 6;
    suborder[5] = 6;
  }
  else if ( rule == 11 )
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 6;
    suborder[6] = 6;
  }
  else if ( rule == 12 )
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 6;
    suborder[6] = 6;
    suborder[7] = 6;
  }
  else if ( rule == 13 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 6;
    suborder[8] = 6;
    suborder[9] = 6;
  }
  else if ( rule == 14 )
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 6;
    suborder[7] = 6;
    suborder[8] = 6;
    suborder[9] = 6;
  }
  else if ( rule == 15 )
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 6;
    suborder[7] = 6;
    suborder[8] = 6;
    suborder[9] = 6;
    suborder[10] = 6;
  }
  else if ( rule == 16 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 6;
    suborder[9] = 6;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
  }
  else if ( rule == 17 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 6;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
  }
  else if ( rule == 18 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 3;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
    suborder[15] = 6;
    suborder[16] = 6;
  }
  else if ( rule == 19 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 6;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
    suborder[15] = 6;
    suborder[16] = 6;
  }
  else if ( rule == 20 )
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 3;
    suborder[10] = 3;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
    suborder[15] = 6;
    suborder[16] = 6;
    suborder[17] = 6;
    suborder[18] = 6;
  }
  else
  {
    cout << "\n";
    cout << "DUNAVANT_SUBORDER - Fatal error!\n";
    cout << "  Illegal RULE = " << rule << "\n";
    exit ( 1 );
  }

  return suborder;
}
//****************************************************************************80

int dunavant_suborder_num ( int rule )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBORDER_NUM returns the number of suborders for a Dunavant rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Output, int DUNAVANT_SUBORDER_NUM, the number of suborders of the rule.
//
{
  int suborder_num;

  if ( rule == 1 )
  {
    suborder_num = 1;
  }
  else if ( rule == 2 )
  {
    suborder_num = 1;
  }
  else if ( rule == 3 )
  {
    suborder_num = 2;
  }
  else if ( rule == 4 )
  {
    suborder_num = 2;
  }
  else if ( rule == 5 )
  {
    suborder_num = 3;
  }
  else if ( rule == 6 )
  {
    suborder_num = 3;
  }
  else if ( rule == 7 )
  {
    suborder_num = 4;
  }
  else if ( rule == 8 )
  {
    suborder_num = 5;
  }
  else if ( rule == 9 )
  {
    suborder_num = 6;
  }
  else if ( rule == 10 )
  {
    suborder_num = 6;
  }
  else if ( rule == 11 )
  {
    suborder_num = 7;
  }
  else if ( rule == 12 )
  {
    suborder_num = 8;
  }
  else if ( rule == 13 )
  {
    suborder_num = 10;
  }
  else if ( rule == 14 )
  {
    suborder_num = 10;
  }
  else if ( rule == 15 )
  {
    suborder_num = 11;
  }
  else if ( rule == 16 )
  {
    suborder_num = 13;
  }
  else if ( rule == 17 )
  {
    suborder_num = 15;
  }
  else if ( rule == 18 )
  {
    suborder_num = 17;
  }
  else if ( rule == 19 )
  {
    suborder_num = 17;
  }
  else if ( rule == 20 )
  {
    suborder_num = 19;
  }
  else
  {
    suborder_num = -1;
    cout << "\n";
    cout << "DUNAVANT_SUBORDER_NUM - Fatal error!\n";
    cout << "  Illegal RULE = " << rule << "\n";
    exit ( 1 );
  }

  return suborder_num;
}
//****************************************************************************80

void dunavant_subrule ( int rule, int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE returns a compressed Dunavant rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  if ( rule == 1 )
  {
    dunavant_subrule_01 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 2 )
  {
    dunavant_subrule_02 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 3 )
  {
    dunavant_subrule_03 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 4 )
  {
    dunavant_subrule_04 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 5 )
  {
    dunavant_subrule_05 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 6 )
  {
    dunavant_subrule_06 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 7 )
  {
    dunavant_subrule_07 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 8 )
  {
    dunavant_subrule_08 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 9 )
  {
    dunavant_subrule_09 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 10 )
  {
    dunavant_subrule_10 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 11 )
  {
    dunavant_subrule_11 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 12 )
  {
    dunavant_subrule_12 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 13 )
  {
    dunavant_subrule_13 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 14 )
  {
    dunavant_subrule_14 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 15 )
  {
    dunavant_subrule_15 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 16 )
  {
    dunavant_subrule_16 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 17 )
  {
    dunavant_subrule_17 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 18 )
  {
    dunavant_subrule_18 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 19 )
  {
    dunavant_subrule_19 ( suborder_num, suborder_xyz, suborder_w );
  }
  else if ( rule == 20 )
  {
    dunavant_subrule_20 ( suborder_num, suborder_xyz, suborder_w );
  }
  else
  {
    cout << "\n";
    cout << "DUNAVANT_SUBRULE - Fatal error!\n";
    cout << "  Illegal RULE = " << rule << "\n";
    exit ( 1 );
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_01 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_01 returns a compressed Dunavant rule 1.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_01[3*1] = {
      0.333333333333333,  0.333333333333333, 0.333333333333333
  };
  double suborder_w_rule_01[1] = {
      1.000000000000000
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_01[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_01[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_01[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_01[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_02 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_02 returns a compressed Dunavant rule 2.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_02[3*1] = {
      0.666666666666667, 0.166666666666667, 0.166666666666667
  };
  double suborder_w_rule_02[1] = {
      0.333333333333333
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_02[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_02[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_02[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_02[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_03 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_03 returns a compressed Dunavant rule 3.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_03[3*2] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.600000000000000, 0.200000000000000, 0.200000000000000
  };
  double suborder_w_rule_03[2] = {
      -0.562500000000000,
       0.520833333333333
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_03[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_03[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_03[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_03[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_04 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_04 returns a compressed Dunavant rule 4.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_04[3*2] = {
      0.108103018168070, 0.445948490915965, 0.445948490915965,
      0.816847572980459, 0.091576213509771, 0.091576213509771
  };
  double suborder_w_rule_04[2] = {
      0.223381589678011,
      0.109951743655322
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_04[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_04[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_04[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_04[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_05 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_05 returns a compressed Dunavant rule 5.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_05[3*3] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.059715871789770, 0.470142064105115, 0.470142064105115,
      0.797426985353087, 0.101286507323456, 0.101286507323456
  };
  double suborder_w_rule_05[3] = {
      0.225000000000000,
      0.132394152788506,
      0.125939180544827
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_05[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_05[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_05[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_05[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_06 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_06 returns a compressed Dunavant rule 6.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_06[3*3] = {
      0.501426509658179, 0.249286745170910, 0.249286745170910,
      0.873821971016996, 0.063089014491502, 0.063089014491502,
      0.053145049844817, 0.310352451033784, 0.636502499121399
  };
  double suborder_w_rule_06[3] = {
      0.116786275726379,
      0.050844906370207,
      0.082851075618374
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_06[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_06[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_06[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_06[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_07 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_07 returns a compressed Dunavant rule 7.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_07[3*4] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.479308067841920, 0.260345966079040, 0.260345966079040,
      0.869739794195568, 0.065130102902216, 0.065130102902216,
      0.048690315425316, 0.312865496004874, 0.638444188569810
  };
  double suborder_w_rule_07[4] = {
     -0.149570044467682,
      0.175615257433208,
      0.053347235608838,
      0.077113760890257
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_07[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_07[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_07[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_07[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_08 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_08 returns a compressed Dunavant rule 8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_08[3*5] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.081414823414554, 0.459292588292723, 0.459292588292723,
      0.658861384496480, 0.170569307751760, 0.170569307751760,
      0.898905543365938, 0.050547228317031, 0.050547228317031,
      0.008394777409958, 0.263112829634638, 0.728492392955404
  };
  double suborder_w_rule_08[5] = {
      0.144315607677787,
      0.095091634267285,
      0.103217370534718,
      0.032458497623198,
      0.027230314174435
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_08[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_08[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_08[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_08[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_09 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_09 returns a compressed Dunavant rule 9.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_09[3*6] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.020634961602525, 0.489682519198738, 0.489682519198738,
      0.125820817014127, 0.437089591492937, 0.437089591492937,
      0.623592928761935, 0.188203535619033, 0.188203535619033,
      0.910540973211095, 0.044729513394453, 0.044729513394453,
      0.036838412054736, 0.221962989160766, 0.741198598784498
  };
  double suborder_w_rule_09[6] = {
      0.097135796282799,
      0.031334700227139,
      0.077827541004774,
      0.079647738927210,
      0.025577675658698,
      0.043283539377289
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_09[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_09[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_09[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_09[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_10 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_10 returns a compressed Dunavant rule 10.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_10[3*6] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.028844733232685, 0.485577633383657, 0.485577633383657,
      0.781036849029926, 0.109481575485037, 0.109481575485037,
      0.141707219414880, 0.307939838764121, 0.550352941820999,
      0.025003534762686, 0.246672560639903, 0.728323904597411,
      0.009540815400299, 0.066803251012200, 0.923655933587500
  };
  double suborder_w_rule_10[6] = {
      0.090817990382754,
      0.036725957756467,
      0.045321059435528,
      0.072757916845420,
      0.028327242531057,
      0.009421666963733
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_10[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_10[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_10[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_10[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_11 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_11 returns a compressed Dunavant rule 11.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_11[3*7] = {
     -0.069222096541517, 0.534611048270758, 0.534611048270758,
      0.202061394068290, 0.398969302965855, 0.398969302965855,
      0.593380199137435, 0.203309900431282, 0.203309900431282,
      0.761298175434837, 0.119350912282581, 0.119350912282581,
      0.935270103777448, 0.032364948111276, 0.032364948111276,
      0.050178138310495, 0.356620648261293, 0.593201213428213,
      0.021022016536166, 0.171488980304042, 0.807489003159792
  };
  double suborder_w_rule_11[7] = {
      0.000927006328961,
      0.077149534914813,
      0.059322977380774,
      0.036184540503418,
      0.013659731002678,
      0.052337111962204,
      0.020707659639141
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_11[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_11[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_11[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_11[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_12 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_12 returns a compressed Dunavant rule 12.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_12[3*8] = {
      0.023565220452390, 0.488217389773805, 0.488217389773805,
      0.120551215411079, 0.439724392294460, 0.439724392294460,
      0.457579229975768, 0.271210385012116, 0.271210385012116,
      0.744847708916828, 0.127576145541586, 0.127576145541586,
      0.957365299093579, 0.021317350453210, 0.021317350453210,
      0.115343494534698, 0.275713269685514, 0.608943235779788,
      0.022838332222257, 0.281325580989940, 0.695836086787803,
      0.025734050548330, 0.116251915907597, 0.858014033544073
  };
  double suborder_w_rule_12[8] = {
      0.025731066440455,
      0.043692544538038,
      0.062858224217885,
      0.034796112930709,
      0.006166261051559,
      0.040371557766381,
      0.022356773202303,
      0.017316231108659
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_12[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_12[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_12[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_12[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_13 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_13 returns a compressed Dunavant rule 13.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_13[3*10] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.009903630120591, 0.495048184939705, 0.495048184939705,
      0.062566729780852, 0.468716635109574, 0.468716635109574,
      0.170957326397447, 0.414521336801277, 0.414521336801277,
      0.541200855914337, 0.229399572042831, 0.229399572042831,
      0.771151009607340, 0.114424495196330, 0.114424495196330,
      0.950377217273082, 0.024811391363459, 0.024811391363459,
      0.094853828379579, 0.268794997058761, 0.636351174561660,
      0.018100773278807, 0.291730066734288, 0.690169159986905,
      0.022233076674090, 0.126357385491669, 0.851409537834241
  };
  double suborder_w_rule_13[10] = {
      0.052520923400802,
      0.011280145209330,
      0.031423518362454,
      0.047072502504194,
      0.047363586536355,
      0.031167529045794,
      0.007975771465074,
      0.036848402728732,
      0.017401463303822,
      0.015521786839045
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_13[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_13[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_13[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_13[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_14 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_14 returns a compressed Dunavant rule 14.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_14[3*10] = {
      0.022072179275643, 0.488963910362179, 0.488963910362179,
      0.164710561319092, 0.417644719340454, 0.417644719340454,
      0.453044943382323, 0.273477528308839, 0.273477528308839,
      0.645588935174913, 0.177205532412543, 0.177205532412543,
      0.876400233818255, 0.061799883090873, 0.061799883090873,
      0.961218077502598, 0.019390961248701, 0.019390961248701,
      0.057124757403648, 0.172266687821356, 0.770608554774996,
      0.092916249356972, 0.336861459796345, 0.570222290846683,
      0.014646950055654, 0.298372882136258, 0.686980167808088,
      0.001268330932872, 0.118974497696957, 0.879757171370171
  };
  double suborder_w_rule_14[10] = {
      0.021883581369429,
      0.032788353544125,
      0.051774104507292,
      0.042162588736993,
      0.014433699669777,
      0.004923403602400,
      0.024665753212564,
      0.038571510787061,
      0.014436308113534,
      0.005010228838501
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_14[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_14[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_14[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_14[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_15 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_15 returns a compressed Dunavant rule 15.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_15[3*11] = {
     -0.013945833716486, 0.506972916858243, 0.506972916858243,
      0.137187291433955, 0.431406354283023, 0.431406354283023,
      0.444612710305711, 0.277693644847144, 0.277693644847144,
      0.747070217917492, 0.126464891041254, 0.126464891041254,
      0.858383228050628, 0.070808385974686, 0.070808385974686,
      0.962069659517853, 0.018965170241073, 0.018965170241073,
      0.133734161966621, 0.261311371140087, 0.604954466893291,
      0.036366677396917, 0.388046767090269, 0.575586555512814,
     -0.010174883126571, 0.285712220049916, 0.724462663076655,
      0.036843869875878, 0.215599664072284, 0.747556466051838,
      0.012459809331199, 0.103575616576386, 0.883964574092416
  };
  double suborder_w_rule_15[11] = {
      0.001916875642849,
      0.044249027271145,
      0.051186548718852,
      0.023687735870688,
      0.013289775690021,
      0.004748916608192,
      0.038550072599593,
      0.027215814320624,
      0.002182077366797,
      0.021505319847731,
      0.007673942631049
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_15[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_15[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_15[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_15[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_16 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_16 returns a compressed Dunavant rule 16.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_16[3*13] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.005238916103123, 0.497380541948438, 0.497380541948438,
      0.173061122901295, 0.413469438549352, 0.413469438549352,
      0.059082801866017, 0.470458599066991, 0.470458599066991,
      0.518892500060958, 0.240553749969521, 0.240553749969521,
      0.704068411554854, 0.147965794222573, 0.147965794222573,
      0.849069624685052, 0.075465187657474, 0.075465187657474,
      0.966807194753950, 0.016596402623025, 0.016596402623025,
      0.103575692245252, 0.296555596579887, 0.599868711174861,
      0.020083411655416, 0.337723063403079, 0.642193524941505,
     -0.004341002614139, 0.204748281642812, 0.799592720971327,
      0.041941786468010, 0.189358492130623, 0.768699721401368,
      0.014317320230681, 0.085283615682657, 0.900399064086661
  };
  double suborder_w_rule_16[13] = {
      0.046875697427642,
      0.006405878578585,
      0.041710296739387,
      0.026891484250064,
      0.042132522761650,
      0.030000266842773,
      0.014200098925024,
      0.003582462351273,
      0.032773147460627,
      0.015298306248441,
      0.002386244192839,
      0.019084792755899,
      0.006850054546542
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_16[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_16[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_16[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_16[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_17 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_17 returns a compressed Dunavant rule 17.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_17[3*15] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.005658918886452, 0.497170540556774, 0.497170540556774,
      0.035647354750751, 0.482176322624625, 0.482176322624625,
      0.099520061958437, 0.450239969020782, 0.450239969020782,
      0.199467521245206, 0.400266239377397, 0.400266239377397,
      0.495717464058095, 0.252141267970953, 0.252141267970953,
      0.675905990683077, 0.162047004658461, 0.162047004658461,
      0.848248235478508, 0.075875882260746, 0.075875882260746,
      0.968690546064356, 0.015654726967822, 0.015654726967822,
      0.010186928826919, 0.334319867363658, 0.655493203809423,
      0.135440871671036, 0.292221537796944, 0.572337590532020,
      0.054423924290583, 0.319574885423190, 0.626001190286228,
      0.012868560833637, 0.190704224192292, 0.796427214974071,
      0.067165782413524, 0.180483211648746, 0.752351005937729,
      0.014663182224828, 0.080711313679564, 0.904625504095608
  };
  double suborder_w_rule_17[15] = {
      0.033437199290803,
      0.005093415440507,
      0.014670864527638,
      0.024350878353672,
      0.031107550868969,
      0.031257111218620,
      0.024815654339665,
      0.014056073070557,
      0.003194676173779,
      0.008119655318993,
      0.026805742283163,
      0.018459993210822,
      0.008476868534328,
      0.018292796770025,
      0.006665632004165
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_17[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_17[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_17[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_17[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_18 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_18 returns a compressed Dunavant rule 18.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_18[3*17] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.013310382738157, 0.493344808630921, 0.493344808630921,
      0.061578811516086, 0.469210594241957, 0.469210594241957,
      0.127437208225989, 0.436281395887006, 0.436281395887006,
      0.210307658653168, 0.394846170673416, 0.394846170673416,
      0.500410862393686, 0.249794568803157, 0.249794568803157,
      0.677135612512315, 0.161432193743843, 0.161432193743843,
      0.846803545029257, 0.076598227485371, 0.076598227485371,
      0.951495121293100, 0.024252439353450, 0.024252439353450,
      0.913707265566071, 0.043146367216965, 0.043146367216965,
      0.008430536202420, 0.358911494940944, 0.632657968856636,
      0.131186551737188, 0.294402476751957, 0.574410971510855,
      0.050203151565675, 0.325017801641814, 0.624779046792512,
      0.066329263810916, 0.184737559666046, 0.748933176523037,
      0.011996194566236, 0.218796800013321, 0.769207005420443,
      0.014858100590125, 0.101179597136408, 0.883962302273467,
     -0.035222015287949, 0.020874755282586, 1.014347260005363
  };
  double suborder_w_rule_18[17] = {
      0.030809939937647,
      0.009072436679404,
      0.018761316939594,
      0.019441097985477,
      0.027753948610810,
      0.032256225351457,
      0.025074032616922,
      0.015271927971832,
      0.006793922022963,
     -0.002223098729920,
      0.006331914076406,
      0.027257538049138,
      0.017676785649465,
      0.018379484638070,
      0.008104732808192,
      0.007634129070725,
      0.000046187660794
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_18[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_18[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_18[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_18[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_19 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_19 returns a compressed Dunavant rule 19.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_19[3*17] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.020780025853987, 0.489609987073006, 0.489609987073006,
      0.090926214604215, 0.454536892697893, 0.454536892697893,
      0.197166638701138, 0.401416680649431, 0.401416680649431,
      0.488896691193805, 0.255551654403098, 0.255551654403098,
      0.645844115695741, 0.177077942152130, 0.177077942152130,
      0.779877893544096, 0.110061053227952, 0.110061053227952,
      0.888942751496321, 0.055528624251840, 0.055528624251840,
      0.974756272445543, 0.012621863777229, 0.012621863777229,
      0.003611417848412, 0.395754787356943, 0.600633794794645,
      0.134466754530780, 0.307929983880436, 0.557603261588784,
      0.014446025776115, 0.264566948406520, 0.720987025817365,
      0.046933578838178, 0.358539352205951, 0.594527068955871,
      0.002861120350567, 0.157807405968595, 0.839331473680839,
      0.223861424097916, 0.075050596975911, 0.701087978926173,
      0.034647074816760, 0.142421601113383, 0.822931324069857,
      0.010161119296278, 0.065494628082938, 0.924344252620784
  };
  double suborder_w_rule_19[17] = {
      0.032906331388919,
      0.010330731891272,
      0.022387247263016,
      0.030266125869468,
      0.030490967802198,
      0.024159212741641,
      0.016050803586801,
      0.008084580261784,
      0.002079362027485,
      0.003884876904981,
      0.025574160612022,
      0.008880903573338,
      0.016124546761731,
      0.002491941817491,
      0.018242840118951,
      0.010258563736199,
      0.003799928855302
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_19[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_19[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_19[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_19[s];
  }

  return;
}
//****************************************************************************80

void dunavant_subrule_20 ( int suborder_num, double suborder_xyz[],
  double suborder_w[] )

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_20 returns a compressed Dunavant rule 20.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_20[3*19] = {
      0.333333333333333, 0.333333333333333, 0.333333333333333,
     -0.001900928704400, 0.500950464352200, 0.500950464352200,
      0.023574084130543, 0.488212957934729, 0.488212957934729,
      0.089726636099435, 0.455136681950283, 0.455136681950283,
      0.196007481363421, 0.401996259318289, 0.401996259318289,
      0.488214180481157, 0.255892909759421, 0.255892909759421,
      0.647023488009788, 0.176488255995106, 0.176488255995106,
      0.791658289326483, 0.104170855336758, 0.104170855336758,
      0.893862072318140, 0.053068963840930, 0.053068963840930,
      0.916762569607942, 0.041618715196029, 0.041618715196029,
      0.976836157186356, 0.011581921406822, 0.011581921406822,
      0.048741583664839, 0.344855770229001, 0.606402646106160,
      0.006314115948605, 0.377843269594854, 0.615842614456541,
      0.134316520547348, 0.306635479062357, 0.559048000390295,
      0.013973893962392, 0.249419362774742, 0.736606743262866,
      0.075549132909764, 0.212775724802802, 0.711675142287434,
     -0.008368153208227, 0.146965436053239, 0.861402717154987,
      0.026686063258714, 0.137726978828923, 0.835586957912363,
      0.010547719294141, 0.059696109149007, 0.929756171556853
  };
  double suborder_w_rule_20[19] = {
      0.033057055541624,
      0.000867019185663,
      0.011660052716448,
      0.022876936356421,
      0.030448982673938,
      0.030624891725355,
      0.024368057676800,
      0.015997432032024,
      0.007698301815602,
     -0.000632060497488,
      0.001751134301193,
      0.016465839189576,
      0.004839033540485,
      0.025804906534650,
      0.008471091054441,
      0.018354914106280,
      0.000704404677908,
      0.010112684927462,
      0.003573909385950
  };

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_20[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_20[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_20[2+s*3];
  }

  for ( s = 0; s < suborder_num; s++ )
  {
    suborder_w[s] = suborder_w_rule_20[s];
  }

  return;
}
//****************************************************************************80

int i4_max ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MAX returns the maximum of two I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 October 1998
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, are two integers to be compared.
//
//    Output, int I4_MAX, the larger of I1 and I2.
//
{
  int value;

  if ( i2 < i1 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
//****************************************************************************80

int i4_min ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MIN returns the smaller of two I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 October 1998
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, two integers to be compared.
//
//    Output, int I4_MIN, the smaller of I1 and I2.
//
{
  int value;

  if ( i1 < i2 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
//****************************************************************************80

int i4_modp ( int i, int j )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MODP returns the nonnegative remainder of I4 division.
//
//  Formula:
//
//    If
//      NREM = I4_MODP ( I, J )
//      NMULT = ( I - NREM ) / J
//    then
//      I = J * NMULT + NREM
//    where NREM is always nonnegative.
//
//  Discussion:
//
//    The MOD function computes a result with the same sign as the
//    quantity being divided.  Thus, suppose you had an angle A,
//    and you wanted to ensure that it was between 0 and 360.
//    Then mod(A,360) would do, if A was positive, but if A
//    was negative, your result would be between -360 and 0.
//
//    On the other hand, I4_MODP(A,360) is between 0 and 360, always.
//
//  Example:
//
//        I         J     MOD  I4_MODP   I4_MODP Factorization
//
//      107        50       7       7    107 =  2 *  50 + 7
//      107       -50       7       7    107 = -2 * -50 + 7
//     -107        50      -7      43   -107 = -3 *  50 + 43
//     -107       -50      -7      43   -107 =  3 * -50 + 43
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 May 1999
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I, the number to be divided.
//
//    Input, int J, the number that divides I.
//
//    Output, int I4_MODP, the nonnegative remainder when I is
//    divided by J.
//
{
  int value;

  if ( j == 0 )
  {
    cout << "\n";
    cout << "I4_MODP - Fatal error!\n";
    cout << "  I4_MODP ( I, J ) called with J = " << j << "\n";
    exit ( 1 );
  }

  value = i % j;

  if ( value < 0 )
  {
    value = value + abs ( j );
  }

  return value;
}
//****************************************************************************80*

int i4_wrap ( int ival, int ilo, int ihi )

//****************************************************************************80*
//
//  Purpose:
//
//    I4_WRAP forces an integer to lie between given limits by wrapping.
//
//  Example:
//
//    ILO = 4, IHI = 8
//
//    I   Value
//
//    -2     8
//    -1     4
//     0     5
//     1     6
//     2     7
//     3     8
//     4     4
//     5     5
//     6     6
//     7     7
//     8     8
//     9     4
//    10     5
//    11     6
//    12     7
//    13     8
//    14     4
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 August 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int IVAL, an integer value.
//
//    Input, int ILO, IHI, the desired bounds for the integer value.
//
//    Output, int I4_WRAP, a "wrapped" version of IVAL.
//
{
  int jhi;
  int jlo;
  int value;
  int wide;

  jlo = i4_min ( ilo, ihi );
  jhi = i4_max ( ilo, ihi );

  wide = jhi + 1 - jlo;

  if ( wide == 1 )
  {
    value = jlo;
  }
  else
  {
    value = jlo + i4_modp ( ival - jlo, wide );
  }

  return value;
}
//****************************************************************************80

double r8_huge ( )

//****************************************************************************80
//
//  Purpose:
//
//    R8_HUGE returns a "huge" R8.
//
//  Discussion:
//
//    HUGE_VAL is the largest representable legal double precision number,
//    and is usually defined in math.h, or sometimes in stdlib.h.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    31 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_HUGE, a "huge" R8 value.
//
{
  return HUGE_VAL;
}
//****************************************************************************80

int r8_nint ( double x )

//****************************************************************************80
//
//  Purpose:
//
//    R8_NINT returns the nearest integer to an R8.
//
//  Example:
//
//        X         Value
//
//      1.3         1
//      1.4         1
//      1.5         1 or 2
//      1.6         2
//      0.0         0
//     -0.7        -1
//     -1.1        -1
//     -1.6        -2
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the value.
//
//    Output, int R8_NINT, the nearest integer to X.
//
{
  int s;
  int value;

  if ( x < 0.0 )
  {
    s = -1;
  }
  else
  {
    s = 1;
  }
  value = s * ( int ) ( fabs ( x ) + 0.5 );

  return value;
}
//****************************************************************************80

void reference_to_physical_t3 ( double t[], int n, double ref[], double phy[] )

//****************************************************************************80
//
//  Purpose:
//
//    REFERENCE_TO_PHYSICAL_T3 maps T3 reference points to physical points.
//
//  Discussion:
//
//    Given the vertices of an order 3 physical triangle and a point
//    (XSI,ETA) in the reference triangle, the routine computes the value
//    of the corresponding image point (X,Y) in physical space.
//
//    Note that this routine may also be appropriate for an order 6
//    triangle, if the mapping between reference and physical space
//    is linear.  This implies, in particular, that the sides of the
//    image triangle are straight and that the "midside" nodes in the
//    physical triangle are literally halfway along the sides of
//    the physical triangle.
//
//  Reference Element T3:
//
//    |
//    1  3
//    |  |\
//    |  | \
//    S  |  \
//    |  |   \
//    |  |    \
//    0  1-----2
//    |
//    +--0--R--1-->
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2005
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double T[2*3], the coordinates of the vertices.
//    The vertices are assumed to be the images of (0,0), (1,0) and
//    (0,1) respectively.
//
//    Input, int N, the number of objects to transform.
//
//    Input, double REF[2*N], points in the reference triangle.
//
//    Output, double PHY[2*N], corresponding points in the
//    physical triangle.
//
{
  int i;
  int j;

  for ( i = 0; i < 2; i++ )
  {
    for ( j = 0; j < n; j++ )
    {
      phy[i+j*2] = t[i+0*2] * ( 1.0 - ref[0+j*2] - ref[1+j*2] )
                 + t[i+1*2] *       + ref[0+j*2]
                 + t[i+2*2] *                    + ref[1+j*2];
    }
  }

  return;
}
//****************************************************************************80

int s_len_trim ( char *s )

//****************************************************************************80
//
//  Purpose:
//
//    S_LEN_TRIM returns the length of a string to the last nonblank.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 April 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, char *S, a pointer to a string.
//
//    Output, int S_LEN_TRIM, the length of the string to the last nonblank.
//    If S_LEN_TRIM is 0, then the string is entirely blank.
//
{
  int n;
  char *t;

  n = strlen ( s );
  t = s + strlen ( s ) - 1;

  while ( 0 < n )
  {
    if ( *t != ' ' )
    {
      return n;
    }
    t--;
    n--;
  }

  return n;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
//****************************************************************************80

double triangle_area ( double t[2*3] )

//****************************************************************************80
//
//  Purpose:
//
//    TRIANGLE_AREA computes the area of a triangle.
//
//  Discussion:
//
//    If the triangle's vertices are given in counter clockwise order,
//    the area will be positive.  If the triangle's vertices are given
//    in clockwise order, the area will be negative!
//
//    An earlier version of this routine always returned the absolute
//    value of the computed area.  I am convinced now that that is
//    a less useful result!  For instance, by returning the signed
//    area of a triangle, it is possible to easily compute the area
//    of a nonconvex polygon as the sum of the (possibly negative)
//    areas of triangles formed by node 1 and successive pairs of vertices.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 October 2005
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double T[2*3], the vertices of the triangle.
//
//    Output, double TRIANGLE_AREA, the area of the triangle.
//
{
  double area;

  area = 0.5 * (
    t[0+0*2] * ( t[1+1*2] - t[1+2*2] ) +
    t[0+1*2] * ( t[1+2*2] - t[1+0*2] ) +
    t[0+2*2] * ( t[1+0*2] - t[1+1*2] ) );

  return area;
}

