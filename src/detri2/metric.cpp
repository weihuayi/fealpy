#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

//#define USING_GMP

#ifdef USING_GMP
  #include <gmpxx.h>
  #include <mpfr.h>
#endif

using  namespace detri2;

int metric_use_gmp;

//==============================================================================

REAL Triangulation::get_innerproduct(Vertex *v1, Vertex *v2)
{
  REAL product = 0.;

  REAL vx = v2->crd[0] - v1->crd[0];
  REAL vy = v2->crd[1] - v1->crd[1];
  product = (vx * vx + vy * vy);

  /*
  if (op_metric == METRIC_Euclidean) {
    REAL vx = v2->crd[0] - v1->crd[0];
    REAL vy = v2->crd[1] - v1->crd[1];
    product = (vx * vx + vy * vy);
  } else if (op_metric == METRIC_Riemannian) {
    // = (v2-v1)^T A (v2-v1), where A is a 2x2 symmetric metrix.

  } else if (op_metric == METRIC_HDE) {
    REAL vec[3];
    vec[0] = (v2->crd[0] - v1->crd[0]);
    vec[1] = (v2->crd[1] - v1->crd[1]);
    //vec[2] = (v2->crd[2] - v1->crd[2]);
    //vec[2] = (v2->mtr[0] - v1->mtr[0]) * op_hde_s1; // function value
    vec[2] = (v2->val - v1->val);
    for (int i = 0; i < 3; i++) {
      product += (vec[i] * vec[i]);
    }
    // TODO... add option to use gradient.
  } else {
    // Default is Euclidean metric
    REAL vx = v2->crd[0] - v1->crd[0];
    REAL vy = v2->crd[1] - v1->crd[1];
    product = (vx * vx + vy * vy);
  }
  */

  return product;
}

REAL Triangulation::get_distance(Vertex *v1, Vertex *v2)
{
  REAL distsum = 0.0;

  distsum = sqrt(get_innerproduct(v1, v2));

  return distsum;
}

// angle at v0 in [0,pi].
REAL Triangulation::get_angle(Vertex* v0, Vertex* v1, Vertex* v2)
{
  // Use the law of cosines: cos(C) = (a^2 + b^2 - c^2) / (2ab);
  // see https://en.wikipedia.org/wiki/Law_of_cosines

  REAL a = get_distance(v0, v1);
  REAL b = get_distance(v0, v2);
  REAL c = get_distance(v1, v2);

  REAL cosC = (a*a + b*b - c*c) / (2.0*a*b);
  if (cosC > 1.0) cosC = 1.0;
  if (cosC < -1.0) cosC = -1.0;

  return acos(cosC);
}

REAL Triangulation::get_cosangle(Vertex* v0, Vertex* v1, Vertex* v2)
{
  // Use the law of cosines: cos(C) = (a^2 + b^2 - c^2) / (2ab);
  // see https://en.wikipedia.org/wiki/Law_of_cosines

  REAL a = get_distance(v0, v1);
  REAL b = get_distance(v0, v2);
  REAL c = get_distance(v1, v2);

  REAL cosC = (a*a + b*b - c*c) / (2.0*a*b);
  if (cosC > 1.0) cosC = 1.0;
  if (cosC < -1.0) cosC = -1.0;

  return cosC;
  //return acos(cosC);
}

//==============================================================================

REAL Triangulation::get_tri_area(Vertex* pa, Vertex* pb, Vertex* pc)
{
  //REAL detleft = (pa->crd[0] - pc->crd[0]) * (pb->crd[1] - pc->crd[1]);
  //REAL detright = (pa->crd[1] - pc->crd[1]) * (pb->crd[0] - pc->crd[0]);
  //REAL det = detleft - detright;
  //return 0.5 * fabs(det);
  
  // Use Heron's Formula
  REAL a = get_distance(pb, pc);
  REAL b = get_distance(pc, pa);
  REAL c = get_distance(pa, pb);
  REAL s = (a + b + c) / 2.;

  REAL delta = s * (s - a) * (s - b) * (s - c);

  if (delta > 0) {
    return sqrt(delta);
  } else {
    return 0.; // Wrong triangle (triangle inequality failed).
  }
}

//==============================================================================

// dot() returns the dot product: v1 dot v2.
static REAL Dot(REAL* v1, REAL* v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// cross() computes the cross product: n = v1 cross v2.
static void Cross(REAL* v1, REAL* v2, REAL* n)
{
  n[0] =   v1[1] * v2[2] - v2[1] * v1[2];
  n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
  n[2] =   v1[0] * v2[1] - v2[0] * v1[1];
}

// Get the face normal of [a,b,c].
// The direction of this normal is v1 x v2, where
//   v1 = pa->pb, and v2 = pa->pc, (using the right handed rule).
//   The returned normal is unnormalised.
int Triangulation::get_tri_normal(Vertex* pa, Vertex* pb, Vertex* pc, REAL normal[3])
{
  REAL v1[3], v2[3], v3[3], *pv1, *pv2;
  REAL L1, L2, L3;

  v1[0] = pb->crd[0] - pa->crd[0];  // edge vector v1: a->b
  v1[1] = pb->crd[1] - pa->crd[1];
  v1[2] = pb->crd[2] - pa->crd[2];
  v2[0] = pa->crd[0] - pc->crd[0];  // edge vector v2: c->a
  v2[1] = pa->crd[1] - pc->crd[1];
  v2[2] = pa->crd[2] - pc->crd[2];

  // Default, normal is calculated by: v1 x (-v2) (see Fig. fnormal).
  // Choose edge vectors by Burdakov's algorithm to improve numerical accuracy.
  v3[0] = pc->crd[0] - pb->crd[0];  // edge vector v3: b->c
  v3[1] = pc->crd[1] - pb->crd[1];
  v3[2] = pc->crd[2] - pb->crd[2];
  L1 = Dot(v1, v1);
  L2 = Dot(v2, v2);
  L3 = Dot(v3, v3);
  // Sort the three edge lengths.
  if (L1 < L2) {
    if (L2 < L3) {
      pv1 = v1; pv2 = v2; // n = v1 x (-v2).
    } else {
      pv1 = v3; pv2 = v1; // n = v3 x (-v1).
    }
  } else {
    if (L1 < L3) {
      pv1 = v1; pv2 = v2; // n = v1 x (-v2).
    } else {
      pv1 = v2; pv2 = v3; // n = v2 x (-v3).
    }
  }
  
  // Calculate the face normal.
  Cross(pv1, pv2, normal);
  // Inverse the direction;
  normal[0] = -normal[0];
  normal[1] = -normal[1];
  normal[2] = -normal[2];

  return 1;
}

// Get the dihedral angle (in degree) at the edge [a,b], it is the angle
//   between the two normals of the faces [a,b,c] and [a,b,d].
//   The range of this angle is in [0, 359.999...] degree.
REAL Triangulation::get_dihedral(Vertex* pa, Vertex* pb, Vertex* pc,Vertex* pd)
{
  REAL n1[3], n2[3];
  get_tri_normal(pa, pb, pc, n1);
  get_tri_normal(pa, pb, pd, n2);

  REAL L1 = sqrt(Dot(n1, n1));
  REAL L2 = sqrt(Dot(n2, n2));

  REAL cosang = Dot(n1, n2) / (L1 * L2);
  if (fabs(cosang) > 1.) { // Rounding error.
    cosang = cosang > 0. ? 1.0 : -1.0;
  }

  REAL ang = acos(cosang); // ang is in [0,pi]

  if (Orient3d(pa, pb, pc, pd) > 0) {
    // d lies below plane [a,b,c].
    ang = 2 * PI - ang;
  }

  return ang / PI * 180.0; // Return the angle in degree
}

REAL Triangulation::get_min_cosangle(Vertex* pa, Vertex* pb, Vertex* pc)
{
  REAL l_ab = get_distance(pa, pb); //HDE_dist(pa, pb);
  REAL l_bc = get_distance(pb, pc); //HDE_dist(pb, pc);
  REAL l_ca = get_distance(pc, pa); //HDE_dist(pc, pa);

  if (op_db_verbose > 3) {
    printf("  Lengths AB=%g, BC=%g, CA=%g\n", l_ab, l_bc, l_ca);
  }

  // Check if there is a bad angle.
  REAL cosA = (l_ca*l_ca + l_ab*l_ab - l_bc*l_bc) / (2.* l_ca * l_ab);
  REAL cosB = (l_ab*l_ab + l_bc*l_bc - l_ca*l_ca) / (2.* l_ab * l_bc);
  REAL cosC = (l_bc*l_bc + l_ca*l_ca - l_ab*l_ab) / (2.* l_bc * l_ca);
    
  if (cosA >  1.0) cosA =  1.0;
  if (cosA < -1.0) cosA = -1.0;
  if (cosB >  1.0) cosB =  1.0;
  if (cosB < -1.0) cosB = -1.0;
  if (cosC >  1.0) cosC =  1.0;
  if (cosC < -1.0) cosC = -1.0;
    
  if (op_db_verbose > 3) {
    printf("  Angles (degree) A=%g, B=%g, C=%g\n", acos(cosA) / PI * 180.0,
           acos(cosB) / PI * 180.0, acos(cosC) / PI * 180.0);
  }

  // Get the minimum angle.
  //TriEdge EminAng = E;
  REAL cosMinAngle = cosA;
  if (cosB > cosMinAngle) {
    if (cosC > cosB) {
      cosMinAngle = cosC;
      //EminAng = E.eprev(); // [c,a]
    } else {
      cosMinAngle = cosB;
      //EminAng = E.enext(); // [b,c]
    }
  } else if (cosC > cosMinAngle) {
    cosMinAngle = cosC;
    //EminAng = E.eprev(); // [c,a]
  }

  if (op_db_verbose > 3) {
    //printf("  Smallest angle at [%d] %g (degree)\n",
    //       EminAng.org()->idx, acos(cosMinAngle) / PI * 180.0);
    printf("  Smallest angle = %g (degree)\n", acos(cosMinAngle) / PI * 180.0);
  }

  return cosMinAngle;
}

//==============================================================================
// BAMG, Metric.cpp, MatVVP2x2::MatVVP2x2(const MetricAnIso M)

REAL detri2::get_MatVVP2x2(double a11, double a21, double a22,
               double *lambda1, double *lambda2, double vec1[2], double vec2[2])
{
    double c11 = a11*a11, c22 = a22*a22, c21= a21*a21;
    double b=-a11-a22,c=-c21+a11*a22;
    double delta = b*b - 4 * c ;
    double n2=(c11+c22+c21);

    if ( n2 < 1.e-30) {
       *lambda1 = *lambda2 = 0.;
       vec1[0] = 1; vec1[1] = 0; // v.x=1,v.y=0;
      }
    else if (delta < 1.e-5*n2) // eps=1e-5
      {
        *lambda1 = *lambda2 = -b/2;
        vec1[0] = 1; vec1[1] = 0; // v.x=1,v.y=0;
      }
    else
      {  //    ---  construction  de 2 vecteur dans (Im ( A - D(i) Id) ortogonal
         // construction of 2 vectors in (Im ( A - D(i) Id) ortogonal
        delta = sqrt(delta);
        *lambda1 = (-b-delta)/2.0;
        *lambda2 = (-b+delta)/2.0;
        double v0 = a11- *lambda1, v1 = a21, v2 = a22 - *lambda1;
        double s0 = v0*v0 + v1*v1, s1 = v1*v1 +v2*v2;

        if (s1 < s0) {
          s0 = sqrt(s0);
          vec1[0]=v1/s0; vec1[1]=-v0/s0;  //v.x=v1/s0,v.y=-v0/s0;
        }
        else
        {
          s1=sqrt(s1);
          vec1[0]=v2/s1; vec1[1]=-v1/s1; //v.x=v2/s1,v.y=-v1/s1;
        }
    }
    // The eigenvector of lambda2.
    vec2[0] = -vec1[1];
    vec2[1] =  vec1[0];

    return *lambda1 * (*lambda2);
}

REAL detri2::get_rotation_angle(double vec[2])
{
    REAL theta;

    if (fabs(vec[0]) > 1.e-8) {
      theta = atan(fabs(vec[1]/vec[0]));
      if (vec[0] > 0.) {
        if (vec[1] >= 0.) {
          theta += 0.;
        } else {
          theta += (PI / 2. * 3.);
        }
      } else {
        if (vec[1] >= 0) {
          theta += (PI / 2.);
        } else {
          theta += PI;
        }
      }
    } else {
      if (vec[1] >= 0.) {
        theta = (PI / 2.);
      } else {
        theta = (PI / 2. * 3.);
      }
    }

    return theta;
}

//==============================================================================

static
int line_line_intersection_fp(double X0, double Y0, double X1, double Y1,
                              double X2, double Y2, double X3, double Y3,
                              double *t1, double *t2)
{
  double Ux = X1 - X0;
  double Uy = Y1 - Y0;
  double Vx = X3 - X2;
  double Vy = Y3 - Y2;
  double Wx = X2 - X0;
  double Wy = Y2 - Y0;
  double det = Ux*Vy - Uy*Vx;

  double absolut = fabs(Ux*Vy) + fabs(Uy*Vx);
  if (fabs(det) / absolut < 1e-6) {
    printf("Warning: Two lines are nearly parallel.\n");
    *t1 = *t2 = 0.0;
    return 0;
  }

  *t1 = (Wx*Vy - Wy*Vx) / det;
  *t2 = (Ux*Wy - Uy*Wx) / det;
  return 1;
}

int detri2::
  line_line_intersection(double X0, double Y0, double X1, double Y1,
                         double X2, double Y2, double X3, double Y3,
                         double *t1, double *t2)
{
#ifdef USING_GMP
  if (metric_use_gmp) { //if (op_use_gmp) {
    //mpf_set_default_prec(PRECISION);
    mpf_class x0, x1, x2, x3, y0, y1, y2, y3;
    x0=X0; x1=X1; x2=X2; x3=X3;
    y0=Y0; y1=Y1; y2=Y2; y3=Y3;

    mpf_class Ux = x1 - x0;
    mpf_class Uy = y1 - y0;
    mpf_class Vx = x3 - x2;
    mpf_class Vy = y3 - y2;
    mpf_class Wx = x2 - x0;
    mpf_class Wy = y2 - y0;
    mpf_class det = Ux*Vy - Uy*Vx;

    if(det == 0.0) {
      //cout<<""<<endl;
      *t1 = 0.;
      *t2 = 0.;
      return 0;
    }

    mpf_class r1 = (Wx*Vy - Wy*Vx) / det;
    mpf_class r2 = (Ux*Wy - Uy*Wx) / det;
    *t1 = r1.get_d();
    *t2 = r2.get_d();

    //cout<<"PRECISION: "<<r1.get_prec()<<endl;
    return 1;
  } else {
    return line_line_intersection_fp(X0, Y0, X1, Y1,
                                     X2, Y2, X3, Y3,
                                     t1, t2);
  }
#else
  // Not using GMP library
  return line_line_intersection_fp(X0, Y0, X1, Y1,
                                   X2, Y2, X3, Y3,
                                   t1, t2);
  /*
  double Ux = X1 - X0;
  double Uy = Y1 - Y0;
  double Vx = X3 - X2;
  double Vy = Y3 - Y2;
  double Wx = X2 - X0;
  double Wy = Y2 - Y0;
  double det = Ux*Vy - Uy*Vx;

  double absolut = fabs(Ux*Vy) + fabs(Uy*Vx);
  if (fabs(det) / absolut < 1e-6) {
    printf("Warning: Two lines are nearly parallel.\n");
    *t1 = *t2 = 0.0;
    return 0;
  }

  *t1 = (Wx*Vy - Wy*Vx) / det;
  *t2 = (Ux*Wy - Uy*Wx) / det;
  return 1;
  */
#endif
}

//==============================================================================

/*
// Invert a 3x3 matrix
// https://www.thecrazyprogrammer.com/2017/02/c-c-program-find-inverse-matrix.html
//
// Example:
//   mat[0][0] = 3, mat[0][1] = 0, mat[0][1] = 2;
//   mat[1][0] = 2, mat[1][1] = 0, mat[1][1] = -2;
//   mat[2][0] = 0, mat[2][1] = 1, mat[2][1] = 1;
//
// Given matrix is:
//  3  0  2
//  2  0  -2
//  0  1  1
//
// determinant: 10.00000
//
// Inverse of matrix is:
//   0.20  0.20  0.00
//  -0.20  0.30  1.00
//   0.20 -0.30  0.00

static int invert3x3m(double mat[][3], double inv[][3])
{
    int i, j;

    printf("\nGiven matrix is:");
    for(i = 0; i < 3; i++){
        printf("\n");
        for(j = 0; j < 3; j++)
            printf("%f\t", mat[i][j]);
    }

    double determinant = 0.;

    for(i = 0; i < 3; i++) {
      determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));
    }

    printf("\n\ndeterminant: %f\n", determinant);

    printf("\nInverse of matrix is: \n");
    for(i = 0; i < 3; i++){
      for(j = 0; j < 3; j++) {
          inv[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
          printf("%.2f\t", inv[i][j]);
      }
      printf("\n");
    }

    return 1;
}

static void test_invmat()
{
    // Test
    double mat[3][3], inv[3][3];
       mat[0][0] = 3, mat[0][1] = 0, mat[0][2] = 2;
       mat[1][0] = 2, mat[1][1] = 0, mat[1][2] = -2;
       mat[2][0] = 0, mat[2][1] = 1, mat[2][2] = 1;
    invert3x3m(mat, inv);
}
*/

//==============================================================================

static
int get_orthocenter_fp(REAL Ux, REAL Uy, REAL U_weight,
                                      REAL Vx, REAL Vy, REAL V_weight,
                                      REAL Wx, REAL Wy, REAL W_weight,
                                      REAL* Cx, REAL* Cy, REAL* r2,
                                      double a11, double a21, double a22)
{
  double mat[3][3], inv[3][3], rhs[3];
  double a = a11, b = a22, c = a21;

  mat[0][0] = 2.*a*Ux+2.*c*Uy; mat[0][1] = 2.*b*Uy+2.*c*Ux; mat[0][2]=-1.0;
  mat[1][0] = 2.*a*Vx+2.*c*Vy; mat[1][1] = 2.*b*Vy+2.*c*Vx; mat[1][2]=-1.0;
  mat[2][0] = 2.*a*Wx+2.*c*Wy; mat[2][1] = 2.*b*Wy+2.*c*Wx; mat[2][2]=-1.0;

  //invert3x3m(mat, inv);
  double determinant = 0.;
  int i, j;
  for(int i = 0; i < 3; i++) {
    determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));
  }

  if (determinant == 0) return 0;

  for(i = 0; i < 3; i++){
    for(j = 0; j < 3; j++) {
      inv[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
    }
  }

  rhs[0] = a*Ux*Ux + 2.*c*Ux*Uy + b*Uy*Uy - U_weight;
  rhs[1] = a*Vx*Vx + 2.*c*Vx*Vy + b*Vy*Vy - V_weight;
  rhs[2] = a*Wx*Wx + 2.*c*Wx*Wy + b*Wy*Wy - W_weight;

  double cx = inv[0][0]*rhs[0] + inv[0][1]*rhs[1] + inv[0][2]*rhs[2];
  double cy = inv[1][0]*rhs[0] + inv[1][1]*rhs[1] + inv[1][2]*rhs[2];
  double  h = inv[2][0]*rhs[0] + inv[2][1]*rhs[1] + inv[2][2]*rhs[2];

  double C_weight = a*cx*cx + 2.*c*cx*cy +  b*cy*cy - h;

  *Cx = cx;
  *Cy = cy;
  *r2 = C_weight;

  return 1;

  /*
  REAL Px = op_lambda1 * (Vx - Ux);
  REAL Py = op_lambda2 * (Vy - Uy);
  REAL Qx = op_lambda1 * (Wx - Ux);
  REAL Qy = op_lambda2 * (Wy - Uy);
  REAL Px2_plus_Py2 = op_lambda1 * Px * Px + op_lambda2 * Py * Py + U_weight - V_weight; // + w_0 - w_p
  REAL Qx2_plus_Qy2 = op_lambda1 * Qx * Qx + op_lambda2 * Qy * Qy + U_weight - W_weight; // + w_0 - w_q

  REAL det = Px * Qy - Qx * Py;
  REAL dx, dy;

  if ((fabs(det) / (fabs(Px2_plus_Py2) + fabs(Qx2_plus_Qy2))) < 1e-8) {
    // The triangle is nearly degenerated.
    det = 0;
    //if (op_db_verbose) {
    //  printf("!! Warning: triangle is (nearly) degenerated.\n");
    //  printf("  The mass center is used.\n");
    //}
    dx = (Ux + Vx + Wx) / 3.0;
    dy = (Uy + Vy + Wy) / 3.0;
    dx -= Ux;
    dy -= Uy;
  } else {
    REAL denominator = 0.5 / det; //  0.5 / (Px * Qy - Qx * Py);
    dx = (Qy * Px2_plus_Py2 - Py * Qx2_plus_Qy2) * denominator;
    dy = (Px * Qx2_plus_Qy2 - Qx * Px2_plus_Py2) * denominator;
  }

  // The power (of radius).
  REAL rr2 = op_lambda1 * dx * dx + op_lambda2 * dy * dy - U_weight;

  if (Cx != NULL) {
    *Cx = Ux + dx;
    *Cy = Uy + dy;
    *r2 = rr2;
    //if (op_db_verbose > 2) {
    //  printf("  (Cx, Cy): %g %g r(%g) r2(%g)\n", *Cx, *Cy, fabs(rr2), rr2);
    //}
  } else {
    // Only print the result (for debug).
    REAL cx, cy, r, h;
    cx = Ux + dx;
    cy = Uy + dy;
    r = sqrt(fabs(rr2));
    h = cx*cx + cy*cy - rr2;
    printf("  (Cx, Cy): %g %g r(%g) r2(%g) h(%g)\n", cx, cy, r, rr2, h);
  }

  return det != 0.0;
  */
}

// Get the circumcenter (or orthocenter) of 3 (weighted) vertices, U, V, W.
int detri2::get_orthocenter(REAL Ux, REAL Uy, REAL U_weight,
                            REAL Vx, REAL Vy, REAL V_weight,
                            REAL Wx, REAL Wy, REAL W_weight,
                            REAL* Cx, REAL* Cy, REAL* r2,
                            double a11, double a21, double a22)
{
#ifdef USING_GMP
  if (metric_use_gmp) { //if (op_use_gmp) {
    mpf_class UUx, UUy, UU_w, VVx, VVy, VV_w, WWx, WWy, WW_w;
    UUx = Ux; UUy = Uy; UU_w = U_weight;
    VVx = Vx; VVy = Vy; VV_w = V_weight;
    WWx = Wx; WWy = Wy; WW_w = W_weight;

    mpf_class mat[3][3], inv[3][3], rhs[3];
    mpf_class a = a11;
    mpf_class b = a22;
    mpf_class c = a21;

    mat[0][0] = 2.*a*UUx+2.*c*UUy; mat[0][1] = 2.*b*UUy+2.*c*UUx; mat[0][2]=-1.0;
    mat[1][0] = 2.*a*VVx+2.*c*VVy; mat[1][1] = 2.*b*VVy+2.*c*VVx; mat[1][2]=-1.0;
    mat[2][0] = 2.*a*WWx+2.*c*WWy; mat[2][1] = 2.*b*WWy+2.*c*WWx; mat[2][2]=-1.0;

    rhs[0] = a*UUx*UUx + 2.*c*UUx*UUy + b*UUy*UUy - UU_w;
    rhs[1] = a*VVx*VVx + 2.*c*VVx*VVy + b*VVy*VVy - VV_w;
    rhs[2] = a*WWx*WWx + 2.*c*WWx*WWy + b*WWy*WWy - WW_w;

    mpf_class  determinant = 0.;
    int i, j;
    for(i = 0; i < 3; i++) {
      determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));
    }

    if (determinant.get_d() == 0) return 0;

    for(i = 0; i < 3; i++){
      for(j = 0; j < 3; j++) {
        inv[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
      }
    }

    mpf_class cx, cy, h, CC_w;

    cx = inv[0][0]*rhs[0] + inv[0][1]*rhs[1] + inv[0][2]*rhs[2];
    cy = inv[1][0]*rhs[0] + inv[1][1]*rhs[1] + inv[1][2]*rhs[2];
     h = inv[2][0]*rhs[0] + inv[2][1]*rhs[1] + inv[2][2]*rhs[2];

    CC_w = a*cx*cx + 2.*c*cx*cy + b*cy*cy - h;

    *Cx = cx.get_d();
    *Cy = cy.get_d();
    *r2 = CC_w.get_d();

    return 1;
    /*
    mpf_class Px = VVx - UUx;
    mpf_class Py = VVy - UUy;
    mpf_class Qx = WWx - UUx;
    mpf_class Qy = WWy - UUy;
    mpf_class Px2_plus_Py2 = Px * Px + Py * Py + UU_w - VV_w; // + w_0 - w_p
    mpf_class Qx2_plus_Qy2 = Qx * Qx + Qy * Qy + UU_w - WW_w; // + w_0 - w_q
    mpf_class det = Px * Qy - Qx * Py;

    mpf_class denominator = 0.5 / det; //  0.5 / (Px * Qy - Qx * Py);

    if (det != 0.0) {
      mpf_class dx, dy;
      dx = (Qy * Px2_plus_Py2 - Py * Qx2_plus_Qy2) * denominator;
      dy = (Px * Qx2_plus_Qy2 - Qx * Px2_plus_Py2) * denominator;

      // The power (of radius).
      mpf_class rr2 = dx * dx + dy * dy - UU_w;

      if (Cx != NULL) {
        mpf_class tempX = UUx + dx;
        mpf_class tempY = UUy + dy;
        *Cx = tempX.get_d();
        *Cy = tempY.get_d();
        *r2 = rr2.get_d();
      }
      return 1;
    } else {
      return 0;
    }
    */
  } else {
    return get_orthocenter_fp(Ux, Uy, U_weight,
                              Vx, Vy, V_weight,
                              Wx, Wy, W_weight,
                              Cx, Cy, r2,
                              a11, a21, a22);
  }
#else
  // Not using GMP
  return get_orthocenter_fp(Ux, Uy, U_weight,
                            Vx, Vy, V_weight,
                            Wx, Wy, W_weight,
                            Cx, Cy, r2,
                            a11, a21, a22);
#endif
}

//==============================================================================

static
int get_bissector_fp(REAL Ux, REAL Uy, REAL U_weight,
                     REAL Vx, REAL Vy, REAL V_weight,
                     REAL* Cx, REAL* Cy, REAL* r2, // Cx, Cy, radius^2
                     double a11, double a21, double a22)
{
  double mat[3][3], inv[3][3], rhs[3];
  double a = a11, b = a22, c = a21;

  mat[0][0] = 2.*a*Ux+2.*c*Uy; mat[0][1] = 2.*b*Uy+2.*c*Ux; mat[0][2]=-1.0;
  mat[1][0] = 2.*a*Vx+2.*c*Vy; mat[1][1] = 2.*b*Vy+2.*c*Vx; mat[1][2]=-1.0;
  mat[2][0] = Uy - Vy;         mat[2][1] = Vx - Ux;         mat[2][2]= 0.;

  //invert3x3m(mat, inv);
  double determinant = 0.;
  int i, j;
  for(int i = 0; i < 3; i++) {
    determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));
  }

  if (determinant == 0) return 0;

  for(i = 0; i < 3; i++){
    for(j = 0; j < 3; j++) {
      inv[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3])) / determinant;
    }
  }

  rhs[0] = a*Ux*Ux + 2.*c*Ux*Uy + b*Uy*Uy - U_weight;
  rhs[1] = a*Vx*Vx + 2.*c*Vx*Vy + b*Vy*Vy - V_weight;
  rhs[2] = Vx*Uy - Ux*Vy;

  double cx = inv[0][0]*rhs[0] + inv[0][1]*rhs[1] + inv[0][2]*rhs[2];
  double cy = inv[1][0]*rhs[0] + inv[1][1]*rhs[1] + inv[1][2]*rhs[2];
  double  h = inv[2][0]*rhs[0] + inv[2][1]*rhs[1] + inv[2][2]*rhs[2];

  double C_weight = a*cx*cx + 2.*c*cx*cy + b*cy*cy - h;

  *Cx = cx;
  *Cy = cy;
  *r2 = C_weight;

  return 1;

  /*
  REAL Qx = (Vx - Ux);
  REAL Qy = (Vy - Uy);

  REAL Qx2_plus_Qy2 = Qx*Qx + Qy*Qy;
  REAL t = 0.5 * (1 + (U_weight - V_weight) / Qx2_plus_Qy2);

  // (cx, cy) is the bisect point (midpoint) on the line [e1->e2].
  REAL cx = Ux + t * Qx;
  REAL cy = Uy + t * Qy;
  REAL rr2 = t*t * (Qx*Qx + Qy*Qy) - U_weight;

  if (Cx != NULL) {
    *Cx = cx;
    *Cy = cy;
    *r2 = rr2;
  } else {
    // Only print the result.
    REAL r, h;
    r = sqrt(fabs(rr2));
    h = cx*cx + cy*cy - rr2;
    printf("  (Cx, Cy): %g %g r(%g) r2(%g) h(%g)\n", cx, cy, r, rr2, h);
  }
  
  return 1;
  */
}

int detri2::get_bissector(REAL Ux, REAL Uy, REAL U_weight,
                          REAL Vx, REAL Vy, REAL V_weight,
                          REAL* Cx, REAL* Cy, REAL* r2, // Cx, Cy, radius^2
                          double a11, double a21, double a22)
{
#ifdef USING_GMP
  if (metric_use_gmp) { //if (op_use_gmp) {
    mpf_class UUx, UUy, UU_w, VVx, VVy, VV_w;
    UUx = Ux; UUy = Uy; UU_w = U_weight;
    VVx = Vx; VVy = Vy; VV_w = V_weight;

    mpf_class mat[3][3], inv[3][3], rhs[3];
    mpf_class a = a11;
    mpf_class b = a22;
    mpf_class c = a21;

    mat[0][0] = 2.*a*UUx+2.*c*UUy; mat[0][1] = 2.*b*UUy+2.*c*UUx; mat[0][2]=-1.0;
    mat[1][0] = 2.*a*VVx+2.*c*VVy; mat[1][1] = 2.*b*VVy+2.*c*VVx; mat[1][2]=-1.0;
    mat[2][0] = UUy-VVy;           mat[2][1] = VVx-UUx;           mat[2][2]= 0.0;

    rhs[0] = a*UUx*UUx + 2.*c*UUx*UUy + b*UUy*UUy - UU_w;
    rhs[1] = a*VVx*VVx + 2.*c*VVx*VVy + b*VVy*VVy - VV_w;
    rhs[2] = VVx * UUy - UUx * VVy;

    mpf_class  determinant = 0.;
    int i, j;
    for(i = 0; i < 3; i++) {
      determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));
    }

    if (determinant.get_d() == 0) return 0;

    for(i = 0; i < 3; i++){
      for(j = 0; j < 3; j++) {
        inv[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
      }
    }

    mpf_class cx, cy, h, CC_w;

    cx = inv[0][0]*rhs[0] + inv[0][1]*rhs[1] + inv[0][2]*rhs[2];
    cy = inv[1][0]*rhs[0] + inv[1][1]*rhs[1] + inv[1][2]*rhs[2];
     h = inv[2][0]*rhs[0] + inv[2][1]*rhs[1] + inv[2][2]*rhs[2];

    CC_w = a*cx*cx + 2.*c*cx*cy + b*cy*cy - h;

    *Cx = cx.get_d();
    *Cy = cy.get_d();
    *r2 = CC_w.get_d();

    return 1;

    /*
    mpf_class UUx, UUy, UU_w, VVx,VVy,VV_w;
    UUx = Ux; UUy = Uy; UU_w = U_weight;
    VVx = Vx; VVy = Vy; VV_w = V_weight;

    mpf_class Qx = VVx - UUx;
    mpf_class Qy = VVy - UUy;

    mpf_class Qx2_plus_Qy2 = Qx*Qx + Qy*Qy;
    mpf_class t = 0.5 * (1 + (UU_w - VV_w) / Qx2_plus_Qy2);

    // (cx, cy) is the bisect point (midpoint) on the line [e1->e2].
    mpf_class cx = UUx + t * Qx;
    mpf_class cy = UUy + t * Qy;
    mpf_class rr2 = t*t * (Qx*Qx + Qy*Qy) - UU_w;

    if (Cx != NULL) {
      *Cx = cx.get_d();
      *Cy = cy.get_d();
      *r2 = rr2.get_d();
    }
    //cout<<"Bisector GMP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl
    return 1;
    */
  } else {
    return get_bissector_fp(Ux, Uy, U_weight,
                            Vx, Vy, V_weight,
                            Cx, Cy, r2,
                            a11, a21, a22);
  }
#else
  return get_bissector_fp(Ux, Uy, U_weight,
                          Vx, Vy, V_weight,
                          Cx, Cy, r2,
                          a11, a21, a22);
#endif
}
