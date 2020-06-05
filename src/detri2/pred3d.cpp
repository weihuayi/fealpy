/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-point Arithmetic               */
/*  and Fast Robust Geometric Predicates                                     */
/*  (predicates.c)                                                           */
/*                                                                           */
/*  May 18, 1996                                                             */
/*                                                                           */
/*  Placed in the public domain by                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*  This file contains C implementation of algorithms for exact addition     */
/*    and multiplication of floating-point numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    Point Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef CPU86
#include <float.h>
#endif /* CPU86 */
#ifdef LINUX
#include <fpu_control.h>
#endif /* LINUX */

#include "detri2.h"
//#define REAL double

#ifdef USE_CGAL_PREDICATES
  #include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
  typedef CGAL::Exact_predicates_inexact_constructions_kernel cgalEpick;
  typedef cgalEpick::Point_3 Point;
  cgalEpick cgal_pred_obj;
#endif // #ifdef USE_CGAL_PREDICATES

/* On some machines, the exact arithmetic routines might be defeated by the  */
/*   use of internal extended precision floating-point registers.  Sometimes */
/*   this problem can be fixed by defining certain values to be volatile,    */
/*   thus forcing them to be stored to memory and rounded off.  This isn't   */
/*   a great solution, though, as it slows the arithmetic down.              */
/*                                                                           */
/* To try this out, write "#define INEXACT volatile" below.  Normally,       */
/*   however, INEXACT should be defined to be nothing.  ("#define INEXACT".) */

#define INEXACT                          /* Nothing */
/* #define INEXACT volatile */

/* Which of the following two methods of finding the absolute values is      */
/*   fastest is compiler-dependent.  A few compilers can inline and optimize */
/*   the fabs() call; but most will incur the overhead of a function call,   */
/*   which is disastrously slow.  A faster way on IEEE machines might be to  */
/*   mask the appropriate bit, but that's difficult to do in C.              */

//#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))
#define Absolute(a)  fabs(a)

/* Many of the operations are broken up into two pieces, a main part that    */
/*   performs an approximate operation, and a "tail" that computes the       */
/*   roundoff error of that operation.                                       */
/*                                                                           */
/* The operations Fast_Two_Sum(), Fast_Two_Diff(), Two_Sum(), Two_Diff(),    */
/*   Split(), and Two_Product() are all implemented as described in the      */
/*   reference.  Each of these macros requires certain variables to be       */
/*   defined in the calling routine.  The variables `bvirt', `c', `abig',    */
/*   `_i', `_j', `_k', `_l', `_m', and `_n' are declared `INEXACT' because   */
/*   they store the result of an operation that may incur roundoff error.    */
/*   The input parameter `x' (or the highest numbered `x_' parameter) must   */
/*   also be declared `INEXACT'.                                             */

#define Fast_Two_Sum_Tail(a, b, x, y) \
  bvirt = x - a; \
  y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
  x = (REAL) (a + b); \
  Fast_Two_Sum_Tail(a, b, x, y)

#define Fast_Two_Diff_Tail(a, b, x, y) \
  bvirt = a - x; \
  y = bvirt - b

#define Fast_Two_Diff(a, b, x, y) \
  x = (REAL) (a - b); \
  Fast_Two_Diff_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
  bvirt = (REAL) (x - a); \
  avirt = x - bvirt; \
  bround = b - bvirt; \
  around = a - avirt; \
  y = around + bround

#define Two_Sum(a, b, x, y) \
  x = (REAL) (a + b); \
  Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
  bvirt = (REAL) (a - x); \
  avirt = x + bvirt; \
  bround = bvirt - b; \
  around = a - avirt; \
  y = around + bround

#define Two_Diff(a, b, x, y) \
  x = (REAL) (a - b); \
  Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
  c = (REAL) (splitter * a); \
  abig = (REAL) (c - a); \
  ahi = c - abig; \
  alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
  Split(a, ahi, alo); \
  Split(b, bhi, blo); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3

#define Two_Product(a, b, x, y) \
  x = (REAL) (a * b); \
  Two_Product_Tail(a, b, x, y)

/* Two_Product_Presplit() is Two_Product() where one of the inputs has       */
/*   already been split.  Avoids redundant splitting.                        */

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
  x = (REAL) (a * b); \
  Split(a, ahi, alo); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3

/* Two_Product_2Presplit() is Two_Product() where both of the inputs have    */
/*   already been split.  Avoids redundant splitting.                        */

#define Two_Product_2Presplit(a, ahi, alo, b, bhi, blo, x, y) \
  x = (REAL) (a * b); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3

/* Square() can be done more quickly than Two_Product().                     */

#define Square_Tail(a, x, y) \
  Split(a, ahi, alo); \
  err1 = x - (ahi * ahi); \
  err3 = err1 - ((ahi + ahi) * alo); \
  y = (alo * alo) - err3

#define Square(a, x, y) \
  x = (REAL) (a * a); \
  Square_Tail(a, x, y)

/* Macros for summing expansions of various fixed lengths.  These are all    */
/*   unrolled versions of Expansion_Sum().                                   */

#define Two_One_Sum(a1, a0, b, x2, x1, x0) \
  Two_Sum(a0, b , _i, x0); \
  Two_Sum(a1, _i, x2, x1)

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
  Two_Diff(a0, b , _i, x0); \
  Two_Sum( a1, _i, x2, x1)

#define Two_Two_Sum(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Sum(a1, a0, b0, _j, _0, x0); \
  Two_One_Sum(_j, _0, b1, x3, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Diff(a1, a0, b0, _j, _0, x0); \
  Two_One_Diff(_j, _0, b1, x3, x2, x1)

#define Four_One_Sum(a3, a2, a1, a0, b, x4, x3, x2, x1, x0) \
  Two_One_Sum(a1, a0, b , _j, x1, x0); \
  Two_One_Sum(a3, a2, _j, x4, x3, x2)

#define Four_Two_Sum(a3, a2, a1, a0, b1, b0, x5, x4, x3, x2, x1, x0) \
  Four_One_Sum(a3, a2, a1, a0, b0, _k, _2, _1, _0, x0); \
  Four_One_Sum(_k, _2, _1, _0, b1, x5, x4, x3, x2, x1)

#define Four_Four_Sum(a3, a2, a1, a0, b4, b3, b1, b0, x7, x6, x5, x4, x3, x2, \
                      x1, x0) \
  Four_Two_Sum(a3, a2, a1, a0, b1, b0, _l, _2, _1, _0, x1, x0); \
  Four_Two_Sum(_l, _2, _1, _0, b4, b3, x7, x6, x5, x4, x3, x2)

#define Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b, x8, x7, x6, x5, x4, \
                      x3, x2, x1, x0) \
  Four_One_Sum(a3, a2, a1, a0, b , _j, x3, x2, x1, x0); \
  Four_One_Sum(a7, a6, a5, a4, _j, x8, x7, x6, x5, x4)

#define Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, x9, x8, x7, \
                      x6, x5, x4, x3, x2, x1, x0) \
  Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b0, _k, _6, _5, _4, _3, _2, \
                _1, _0, x0); \
  Eight_One_Sum(_k, _6, _5, _4, _3, _2, _1, _0, b1, x9, x8, x7, x6, x5, x4, \
                x3, x2, x1)

#define Eight_Four_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b4, b3, b1, b0, x11, \
                       x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0) \
  Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, _l, _6, _5, _4, _3, \
                _2, _1, _0, x1, x0); \
  Eight_Two_Sum(_l, _6, _5, _4, _3, _2, _1, _0, b4, b3, x11, x10, x9, x8, \
                x7, x6, x5, x4, x3, x2)

/* Macros for multiplying expansions of various fixed lengths.               */

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
  Split(b, bhi, blo); \
  Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
  Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x1); \
  Fast_Two_Sum(_j, _k, x3, x2)

#define Four_One_Product(a3, a2, a1, a0, b, x7, x6, x5, x4, x3, x2, x1, x0) \
  Split(b, bhi, blo); \
  Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
  Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x1); \
  Fast_Two_Sum(_j, _k, _i, x2); \
  Two_Product_Presplit(a2, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x3); \
  Fast_Two_Sum(_j, _k, _i, x4); \
  Two_Product_Presplit(a3, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x5); \
  Fast_Two_Sum(_j, _k, x7, x6)

#define Two_Two_Product(a1, a0, b1, b0, x7, x6, x5, x4, x3, x2, x1, x0) \
  Split(a0, a0hi, a0lo); \
  Split(b0, bhi, blo); \
  Two_Product_2Presplit(a0, a0hi, a0lo, b0, bhi, blo, _i, x0); \
  Split(a1, a1hi, a1lo); \
  Two_Product_2Presplit(a1, a1hi, a1lo, b0, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, _1); \
  Fast_Two_Sum(_j, _k, _l, _2); \
  Split(b1, bhi, blo); \
  Two_Product_2Presplit(a0, a0hi, a0lo, b1, bhi, blo, _i, _0); \
  Two_Sum(_1, _0, _k, x1); \
  Two_Sum(_2, _k, _j, _1); \
  Two_Sum(_l, _j, _m, _2); \
  Two_Product_2Presplit(a1, a1hi, a1lo, b1, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _n, _0); \
  Two_Sum(_1, _0, _i, x2); \
  Two_Sum(_2, _i, _k, _1); \
  Two_Sum(_m, _k, _l, _2); \
  Two_Sum(_j, _n, _k, _0); \
  Two_Sum(_1, _0, _j, x3); \
  Two_Sum(_2, _j, _i, _1); \
  Two_Sum(_l, _i, _m, _2); \
  Two_Sum(_1, _k, _i, x4); \
  Two_Sum(_2, _i, _k, x5); \
  Two_Sum(_m, _k, x7, x6)

/* An expansion of length two can be squared more quickly than finding the   */
/*   product of two different expansions of length two, and the result is    */
/*   guaranteed to have no more than six (rather than eight) components.     */

#define Two_Square(a1, a0, x5, x4, x3, x2, x1, x0) \
  Square(a0, _j, x0); \
  _0 = a0 + a0; \
  Two_Product(a1, _0, _k, _1); \
  Two_One_Sum(_k, _1, _j, _l, _2, x1); \
  Square(a1, _j, _1); \
  Two_Two_Sum(_j, _1, _l, _2, x5, x4, x3, x2)

/* splitter = 2^ceiling(p / 2) + 1.  Used to split floats in half.           */
static REAL splitter;
static REAL epsilon;         /* = 2^(-p).  Used to estimate roundoff errors. */
/* A set of coefficients used to calculate maximum roundoff errors.          */
static REAL resulterrbound;
static REAL ccwerrboundA, ccwerrboundB, ccwerrboundC;
static REAL o3derrboundA, o3derrboundB, o3derrboundC;
static REAL iccerrboundA, iccerrboundB, iccerrboundC;
static REAL isperrboundA, isperrboundB, isperrboundC;

// Options to choose types of geometric computtaions. 
// Added by H. Si, 2012-08-23.
static int  _use_inexact_arith; // -X option.
static int  _use_static_o3d_filter; 
static int  _use_static_isp_filter; 

// Static filters for orient3d() and insphere(). 
// They are pre-calcualted and set in exactinit().
// Added by H. Si, 2012-08-23.
static REAL o3dstaticfilter;
static REAL ispstaticfilter;


// The following codes were part of "IEEE 754 floating-point test software"
//          http://www.math.utah.edu/~beebe/software/ieee/
// The original program was "fpinfo2.c".

double fppow2(int n)
{
  double x, power;
  x = (n < 0) ? ((double)1.0/(double)2.0) : (double)2.0;
  n = (n < 0) ? -n : n;
  power = (double)1.0;
  while (n-- > 0)
	power *= x;
  return (power);
}

#ifdef SINGLE

float fstore(float x)
{
  return (x);
}

int test_float(int verbose)
{
  float x;
  int pass = 1;

  //(void)printf("float:\n");

  if (verbose) {
    (void)printf("  sizeof(float) = %2u\n", (unsigned int)sizeof(float));
#ifdef CPU86  // <float.h>
    (void)printf("  FLT_MANT_DIG = %2d\n", FLT_MANT_DIG);
#endif
  }

  x = (float)1.0;
  while (fstore((float)1.0 + x/(float)2.0) != (float)1.0)
    x /= (float)2.0;
  if (verbose)
    (void)printf("  machine epsilon = %13.5e  ", x);

  if (x == (float)fppow2(-23)) {
    if (verbose)
      (void)printf("[IEEE 754 32-bit macheps]\n");
  } else {
    (void)printf("[not IEEE 754 conformant] !!\n");
    pass = 0;
  }

  x = (float)1.0;
  while (fstore(x / (float)2.0) != (float)0.0)
    x /= (float)2.0;
  if (verbose)
    (void)printf("  smallest positive number =  %13.5e  ", x);

  if (x == (float)fppow2(-149)) {
    if (verbose)
      (void)printf("[smallest 32-bit subnormal]\n");
  } else if (x == (float)fppow2(-126)) {
    if (verbose)
      (void)printf("[smallest 32-bit normal]\n");
  } else {
	(void)printf("[not IEEE 754 conformant] !!\n");
    pass = 0;
  }

  return pass;
}

# else

double dstore(double x)
{
  return (x);
}

int test_double(int verbose)
{
  double x;
  int pass = 1;

  // (void)printf("double:\n");
  if (verbose) {
    (void)printf("  sizeof(double) = %2u\n", (unsigned int)sizeof(double));
#ifdef CPU86  // <float.h>
    (void)printf("  DBL_MANT_DIG = %2d\n", DBL_MANT_DIG);
#endif
  }

  x = 1.0;
  while (dstore(1.0 + x/2.0) != 1.0)
    x /= 2.0;
  if (verbose) 
    (void)printf("  machine epsilon = %13.5le ", x);

  if (x == (double)fppow2(-52)) {
    if (verbose)
      (void)printf("[IEEE 754 64-bit macheps]\n");
  } else {
    (void)printf("[not IEEE 754 conformant] !!\n");
    pass = 0;
  }

  x = 1.0;
  while (dstore(x / 2.0) != 0.0)
    x /= 2.0;
  //if (verbose)
  //  (void)printf("  smallest positive number = %13.5le ", x);

  if (x == (double)fppow2(-1074)) {
    //if (verbose)
    //  (void)printf("[smallest 64-bit subnormal]\n");
  } else if (x == (double)fppow2(-1022)) {
    //if (verbose)
    //  (void)printf("[smallest 64-bit normal]\n");
  } else {
    (void)printf("[not IEEE 754 conformant] !!\n");
    pass = 0;
  }

  return pass;
}

#endif

REAL exactinit()
{
  REAL half;
  REAL check, lastcheck;
  int every_other;
#ifdef LINUX
  int cword;
#endif /* LINUX */

#ifdef CPU86
#ifdef SINGLE
  _control87(_PC_24, _MCW_PC); /* Set FPU control word for single precision. */
#else /* not SINGLE */
  _control87(_PC_53, _MCW_PC); /* Set FPU control word for double precision. */
#endif /* not SINGLE */
#endif /* CPU86 */
#ifdef LINUX
#ifdef SINGLE
  /*  cword = 4223; */
  cword = 4210;                 /* set FPU control word for single precision */
#else /* not SINGLE */
  /*  cword = 4735; */
  cword = 4722;                 /* set FPU control word for double precision */
#endif /* not SINGLE */
  _FPU_SETCW(cword);
#endif /* LINUX */

  every_other = 1;
  half = 0.5;
  epsilon = 1.0;
  splitter = 1.0;
  check = 1.0;
  /* Repeatedly divide `epsilon' by two until it is too small to add to    */
  /*   one without causing roundoff.  (Also check if the sum is equal to   */
  /*   the previous sum, for machines that round up instead of using exact */
  /*   rounding.  Not that this library will work on such machines anyway. */
  do {
    lastcheck = check;
    epsilon *= half;
    if (every_other) {
      splitter *= 2.0;
    }
    every_other = !every_other;
    check = 1.0 + epsilon;
  } while ((check != 1.0) && (check != lastcheck));
  splitter += 1.0;

  /* Error bounds for orientation and incircle tests. */
  resulterrbound = (3.0 + 8.0 * epsilon) * epsilon;
  ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon;
  ccwerrboundB = (2.0 + 12.0 * epsilon) * epsilon;
  ccwerrboundC = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
  o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon;
  o3derrboundB = (3.0 + 28.0 * epsilon) * epsilon;
  o3derrboundC = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
  iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon;
  iccerrboundB = (4.0 + 48.0 * epsilon) * epsilon;
  iccerrboundC = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
  isperrboundA = (16.0 + 224.0 * epsilon) * epsilon;
  isperrboundB = (5.0 + 72.0 * epsilon) * epsilon;
  isperrboundC = (71.0 + 1408.0 * epsilon) * epsilon * epsilon;

  return epsilon; /* Added by H. Si 30 Juli, 2004. */
}

/*****************************************************************************/
/*                                                                           */
/*  exactinit()   Initialize the variables used for exact arithmetic.        */
/*                                                                           */
/*  `epsilon' is the largest power of two such that 1.0 + epsilon = 1.0 in   */
/*  floating-point arithmetic.  `epsilon' bounds the relative roundoff       */
/*  error.  It is used for floating-point error analysis.                    */
/*                                                                           */
/*  `splitter' is used to split floating-point numbers into two half-        */
/*  length significands for exact multiplication.                            */
/*                                                                           */
/*  I imagine that a highly optimizing compiler might be too smart for its   */
/*  own good, and somehow cause this routine to fail, if it pretends that    */
/*  floating-point arithmetic is too much like real arithmetic.              */
/*                                                                           */
/*  Don't change this routine unless you fully understand it.                */
/*                                                                           */
/*****************************************************************************/

void detri2::exactinit(int verbose, int noexact, int o3dfilter, int ispfilter,
                        REAL maxx, REAL maxy, REAL maxz)
{
  REAL half;
  REAL check, lastcheck;
  int every_other;
#ifdef LINUX
  int cword;
#endif /* LINUX */

#ifdef CPU86
#ifdef SINGLE
  _control87(_PC_24, _MCW_PC); /* Set FPU control word for single precision. */
#else /* not SINGLE */
  _control87(_PC_53, _MCW_PC); /* Set FPU control word for double precision. */
#endif /* not SINGLE */
#endif /* CPU86 */
#ifdef LINUX
#ifdef SINGLE
  /*  cword = 4223; */
  cword = 4210;                 /* set FPU control word for single precision */
#else /* not SINGLE */
  /*  cword = 4735; */
  cword = 4722;                 /* set FPU control word for double precision */
#endif /* not SINGLE */
  _FPU_SETCW(cword);
#endif /* LINUX */

  if (verbose) {
    printf("  Initializing robust predicates.\n");
  }

#ifdef USE_CGAL_PREDICATES
  if (cgal_pred_obj.Has_static_filters) {
    printf("  Use static filter.\n");
  } else {
    printf("  No static filter.\n");
  }
#endif // USE_CGAL_PREDICATES

#ifdef SINGLE
  test_float(verbose);
#else
  test_double(verbose);
#endif

  every_other = 1;
  half = 0.5;
  epsilon = 1.0;
  splitter = 1.0;
  check = 1.0;
  /* Repeatedly divide `epsilon' by two until it is too small to add to    */
  /*   one without causing roundoff.  (Also check if the sum is equal to   */
  /*   the previous sum, for machines that round up instead of using exact */
  /*   rounding.  Not that this library will work on such machines anyway. */
  do {
    lastcheck = check;
    epsilon *= half;
    if (every_other) {
      splitter *= 2.0;
    }
    every_other = !every_other;
    check = 1.0 + epsilon;
  } while ((check != 1.0) && (check != lastcheck));
  splitter += 1.0;

  /* Error bounds for orientation and incircle tests. */
  resulterrbound = (3.0 + 8.0 * epsilon) * epsilon;
  ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon;
  ccwerrboundB = (2.0 + 12.0 * epsilon) * epsilon;
  ccwerrboundC = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
  o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon;
  o3derrboundB = (3.0 + 28.0 * epsilon) * epsilon;
  o3derrboundC = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
  iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon;
  iccerrboundB = (4.0 + 48.0 * epsilon) * epsilon;
  iccerrboundC = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
  isperrboundA = (16.0 + 224.0 * epsilon) * epsilon;
  isperrboundB = (5.0 + 72.0 * epsilon) * epsilon;
  isperrboundC = (71.0 + 1408.0 * epsilon) * epsilon * epsilon;

  // Set TetGen options.  Added by H. Si, 2012-08-23.
  _use_inexact_arith = noexact;
  _use_static_o3d_filter = o3dfilter;
  _use_static_isp_filter = ispfilter;

  // Calculate the two static filters for orient3d() and insphere() tests.
  // Added by H. Si, 2012-08-23.

  // Sort maxx < maxy < maxz. Re-use 'half' for swapping.
  if (maxx > maxz) {
    half = maxx; maxx = maxz; maxz = half;
  }
  if (maxy > maxz) {
    half = maxy; maxy = maxz; maxz = half;
  }
  else if (maxy < maxx) {
    half = maxy; maxy = maxx; maxx = half;
  }

  o3dstaticfilter = 5.1107127829973299e-15 * maxx * maxy * maxz;
  ispstaticfilter = 1.2466136531027298e-13 * maxx * maxy * maxz * (maxz * maxz);

}

/*****************************************************************************/
/*                                                                           */
/*  fast_expansion_sum()   Sum two expansions.                               */
/*                                                                           */
/*  Sets h = e + f.  See the long version of my paper for details.           */
/*                                                                           */
/*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
/*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
/*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
/*  properties.                                                              */
/*                                                                           */
/*****************************************************************************/

int fast_expansion_sum(int elen, REAL *e, int flen, REAL *f, REAL *h)
/* h cannot be e or f. */
{
  REAL Q;
  INEXACT REAL Qnew;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  int eindex, findex, hindex;
  REAL enow, fnow;

  enow = e[0];
  fnow = f[0];
  eindex = findex = 0;
  if ((fnow > enow) == (fnow > -enow)) {
    Q = enow;
    enow = e[++eindex];
  } else {
    Q = fnow;
    fnow = f[++findex];
  }
  hindex = 0;
  if ((eindex < elen) && (findex < flen)) {
    if ((fnow > enow) == (fnow > -enow)) {
      Fast_Two_Sum(enow, Q, Qnew, h[0]);
      enow = e[++eindex];
    } else {
      Fast_Two_Sum(fnow, Q, Qnew, h[0]);
      fnow = f[++findex];
    }
    Q = Qnew;
    hindex = 1;
    while ((eindex < elen) && (findex < flen)) {
      if ((fnow > enow) == (fnow > -enow)) {
        Two_Sum(Q, enow, Qnew, h[hindex]);
        enow = e[++eindex];
      } else {
        Two_Sum(Q, fnow, Qnew, h[hindex]);
        fnow = f[++findex];
      }
      Q = Qnew;
      hindex++;
    }
  }
  while (eindex < elen) {
    Two_Sum(Q, enow, Qnew, h[hindex]);
    enow = e[++eindex];
    Q = Qnew;
    hindex++;
  }
  while (findex < flen) {
    Two_Sum(Q, fnow, Qnew, h[hindex]);
    fnow = f[++findex];
    Q = Qnew;
    hindex++;
  }
  h[hindex] = Q;
  return hindex + 1;
}

/*****************************************************************************/
/*                                                                           */
/*  fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero     */
/*                                  components from the output expansion.    */
/*                                                                           */
/*  Sets h = e + f.  See the long version of my paper for details.           */
/*                                                                           */
/*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
/*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
/*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
/*  properties.                                                              */
/*                                                                           */
/*****************************************************************************/

int fast_expansion_sum_zeroelim(int elen, REAL *e, int flen, REAL *f, REAL *h)
/* h cannot be e or f. */
{
  REAL Q;
  INEXACT REAL Qnew;
  INEXACT REAL hh;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  int eindex, findex, hindex;
  REAL enow, fnow;

  enow = e[0];
  fnow = f[0];
  eindex = findex = 0;
  if ((fnow > enow) == (fnow > -enow)) {
    Q = enow;
    enow = e[++eindex];
  } else {
    Q = fnow;
    fnow = f[++findex];
  }
  hindex = 0;
  if ((eindex < elen) && (findex < flen)) {
    if ((fnow > enow) == (fnow > -enow)) {
      Fast_Two_Sum(enow, Q, Qnew, hh);
      enow = e[++eindex];
    } else {
      Fast_Two_Sum(fnow, Q, Qnew, hh);
      fnow = f[++findex];
    }
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
    while ((eindex < elen) && (findex < flen)) {
      if ((fnow > enow) == (fnow > -enow)) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
      } else {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
      }
      Q = Qnew;
      if (hh != 0.0) {
        h[hindex++] = hh;
      }
    }
  }
  while (eindex < elen) {
    Two_Sum(Q, enow, Qnew, hh);
    enow = e[++eindex];
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
  }
  while (findex < flen) {
    Two_Sum(Q, fnow, Qnew, hh);
    fnow = f[++findex];
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
  }
  if ((Q != 0.0) || (hindex == 0)) {
    h[hindex++] = Q;
  }
  return hindex;
}

/*****************************************************************************/
/*                                                                           */
/*  scale_expansion()   Multiply an expansion by a scalar.                   */
/*                                                                           */
/*  Sets h = be.  See either version of my paper for details.                */
/*                                                                           */
/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
/*  properties as well.  (That is, if e has one of these properties, so      */
/*  will h.)                                                                 */
/*                                                                           */
/*****************************************************************************/

int scale_expansion(int elen, REAL *e, REAL b, REAL *h)
/* e and h cannot be the same. */
{
  INEXACT REAL Q;
  INEXACT REAL sum;
  INEXACT REAL product1;
  REAL product0;
  int eindex, hindex;
  REAL enow;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;

  Split(b, bhi, blo);
  Two_Product_Presplit(e[0], b, bhi, blo, Q, h[0]);
  hindex = 1;
  for (eindex = 1; eindex < elen; eindex++) {
    enow = e[eindex];
    Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
    Two_Sum(Q, product0, sum, h[hindex]);
    hindex++;
    Two_Sum(product1, sum, Q, h[hindex]);
    hindex++;
  }
  h[hindex] = Q;
  return elen + elen;
}

/*****************************************************************************/
/*                                                                           */
/*  scale_expansion_zeroelim()   Multiply an expansion by a scalar,          */
/*                               eliminating zero components from the        */
/*                               output expansion.                           */
/*                                                                           */
/*  Sets h = be.  See either version of my paper for details.                */
/*                                                                           */
/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
/*  properties as well.  (That is, if e has one of these properties, so      */
/*  will h.)                                                                 */
/*                                                                           */
/*****************************************************************************/

int scale_expansion_zeroelim(int elen, REAL *e, REAL b, REAL *h)
/* e and h cannot be the same. */
{
  INEXACT REAL Q, sum;
  REAL hh;
  INEXACT REAL product1;
  REAL product0;
  int eindex, hindex;
  REAL enow;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;

  Split(b, bhi, blo);
  Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
  hindex = 0;
  if (hh != 0) {
    h[hindex++] = hh;
  }
  for (eindex = 1; eindex < elen; eindex++) {
    enow = e[eindex];
    Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
    Two_Sum(Q, product0, sum, hh);
    if (hh != 0) {
      h[hindex++] = hh;
    }
    Fast_Two_Sum(product1, sum, Q, hh);
    if (hh != 0) {
      h[hindex++] = hh;
    }
  }
  if ((Q != 0.0) || (hindex == 0)) {
    h[hindex++] = Q;
  }
  return hindex;
}

/*****************************************************************************/
/*                                                                           */
/*  compress()   Compress an expansion.                                      */
/*                                                                           */
/*  See the long version of my paper for details.                            */
/*                                                                           */
/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
/*  with IEEE 754), then any nonoverlapping expansion is converted to a      */
/*  nonadjacent expansion.                                                   */
/*                                                                           */
/*****************************************************************************/

int compress(int elen, REAL *e, REAL *h)
/* e and h may be the same. */
{
  REAL Q, q;
  INEXACT REAL Qnew;
  int eindex, hindex;
  INEXACT REAL bvirt;
  REAL enow, hnow;
  int top, bottom;

  bottom = elen - 1;
  Q = e[bottom];
  for (eindex = elen - 2; eindex >= 0; eindex--) {
    enow = e[eindex];
    Fast_Two_Sum(Q, enow, Qnew, q);
    if (q != 0) {
      h[bottom--] = Qnew;
      Q = q;
    } else {
      Q = Qnew;
    }
  }
  top = 0;
  for (hindex = bottom + 1; hindex < elen; hindex++) {
    hnow = h[hindex];
    Fast_Two_Sum(hnow, Q, Qnew, q);
    if (q != 0) {
      h[top++] = q;
    }
    Q = Qnew;
  }
  h[top] = Q;
  return top + 1;
}

/*****************************************************************************/
/*                                                                           */
/*  estimate()   Produce a one-word estimate of an expansion's value.        */
/*                                                                           */
/*  See either version of my paper for details.                              */
/*                                                                           */
/*****************************************************************************/

REAL estimate(int elen, REAL *e)
{
  REAL Q;
  int eindex;

  Q = e[0];
  for (eindex = 1; eindex < elen; eindex++) {
    Q += e[eindex];
  }
  return Q;
}

/*****************************************************************************/
/*                                                                           */
/*  orient2dfast()   Approximate 2D orientation test.  Nonrobust.            */
/*  orient2dexact()   Exact 2D orientation test.  Robust.                    */
/*  orient2dslow()   Another exact 2D orientation test.  Robust.             */
/*  orient2d()   Adaptive exact 2D orientation test.  Robust.                */
/*                                                                           */
/*               Return a positive value if the points pa, pb, and pc occur  */
/*               in counterclockwise order; a negative value if they occur   */
/*               in clockwise order; and zero if they are collinear.  The    */
/*               result is also a rough approximation of twice the signed    */
/*               area of the triangle defined by the three points.           */
/*                                                                           */
/*  Only the first and last routine should be used; the middle two are for   */
/*  timings.                                                                 */
/*                                                                           */
/*  The last three use exact arithmetic to ensure a correct answer.  The     */
/*  result returned is the determinant of a matrix.  In orient2d() only,     */
/*  this determinant is computed adaptively, in the sense that exact         */
/*  arithmetic is used only to the degree it is needed to ensure that the    */
/*  returned value has the correct sign.  Hence, orient2d() is usually quite */
/*  fast, but will run more slowly when the input points are collinear or    */
/*  nearly so.                                                               */
/*                                                                           */
/*****************************************************************************/

REAL orient2dadapt(REAL *pa, REAL *pb, REAL *pc, REAL detsum)
{
  INEXACT REAL acx, acy, bcx, bcy;
  REAL acxtail, acytail, bcxtail, bcytail;
  INEXACT REAL detleft, detright;
  REAL detlefttail, detrighttail;
  REAL det, errbound;
  REAL B[4], C1[8], C2[12], D[16];
  INEXACT REAL B3;
  int C1length, C2length, Dlength;
  REAL u[4];
  INEXACT REAL u3;
  INEXACT REAL s1, t1;
  REAL s0, t0;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  acx = (REAL) (pa[0] - pc[0]);
  bcx = (REAL) (pb[0] - pc[0]);
  acy = (REAL) (pa[1] - pc[1]);
  bcy = (REAL) (pb[1] - pc[1]);

  Two_Product(acx, bcy, detleft, detlefttail);
  Two_Product(acy, bcx, detright, detrighttail);

  Two_Two_Diff(detleft, detlefttail, detright, detrighttail,
               B3, B[2], B[1], B[0]);
  B[3] = B3;

  det = estimate(4, B);
  errbound = ccwerrboundB * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(pa[0], pc[0], acx, acxtail);
  Two_Diff_Tail(pb[0], pc[0], bcx, bcxtail);
  Two_Diff_Tail(pa[1], pc[1], acy, acytail);
  Two_Diff_Tail(pb[1], pc[1], bcy, bcytail);

  if ((acxtail == 0.0) && (acytail == 0.0)
      && (bcxtail == 0.0) && (bcytail == 0.0)) {
    return det;
  }

  errbound = ccwerrboundC * detsum + resulterrbound * Absolute(det);
  det += (acx * bcytail + bcy * acxtail)
       - (acy * bcxtail + bcx * acytail);
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Product(acxtail, bcy, s1, s0);
  Two_Product(acytail, bcx, t1, t0);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
  u[3] = u3;
  C1length = fast_expansion_sum_zeroelim(4, B, 4, u, C1);

  Two_Product(acx, bcytail, s1, s0);
  Two_Product(acy, bcxtail, t1, t0);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
  u[3] = u3;
  C2length = fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

  Two_Product(acxtail, bcytail, s1, s0);
  Two_Product(acytail, bcxtail, t1, t0);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
  u[3] = u3;
  Dlength = fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

  return(D[Dlength - 1]);
}

REAL orient2d(REAL *pa, REAL *pb, REAL *pc)
{
  REAL detleft, detright, det;
  REAL detsum, errbound;

  detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
  detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
  det = detleft - detright;

  if (detleft > 0.0) {
    if (detright <= 0.0) {
      return det;
    } else {
      detsum = detleft + detright;
    }
  } else if (detleft < 0.0) {
    if (detright >= 0.0) {
      return det;
    } else {
      detsum = -detleft - detright;
    }
  } else {
    return det;
  }

  errbound = ccwerrboundA * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  return orient2dadapt(pa, pb, pc, detsum);
}

REAL detri2::Orient2d(Vertex *pa, Vertex *pb, Vertex *pc)
{
  return orient2d(pa->crd, pb->crd, pc->crd);
}

/*****************************************************************************/
/*                                                                           */
/*  orient3dfast()   Approximate 3D orientation test.  Nonrobust.            */
/*  orient3dexact()   Exact 3D orientation test.  Robust.                    */
/*  orient3dslow()   Another exact 3D orientation test.  Robust.             */
/*  orient3d()   Adaptive exact 3D orientation test.  Robust.                */
/*                                                                           */
/*               Return a positive value if the point pd lies below the      */
/*               plane passing through pa, pb, and pc; "below" is defined so */
/*               that pa, pb, and pc appear in counterclockwise order when   */
/*               viewed from above the plane.  Returns a negative value if   */
/*               pd lies above the plane.  Returns zero if the points are    */
/*               coplanar.  The result is also a rough approximation of six  */
/*               times the signed volume of the tetrahedron defined by the   */
/*               four points.                                                */
/*                                                                           */
/*  Only the first and last routine should be used; the middle two are for   */
/*  timings.                                                                 */
/*                                                                           */
/*  The last three use exact arithmetic to ensure a correct answer.  The     */
/*  result returned is the determinant of a matrix.  In orient3d() only,     */
/*  this determinant is computed adaptively, in the sense that exact         */
/*  arithmetic is used only to the degree it is needed to ensure that the    */
/*  returned value has the correct sign.  Hence, orient3d() is usually quite */
/*  fast, but will run more slowly when the input points are coplanar or     */
/*  nearly so.                                                               */
/*                                                                           */
/*****************************************************************************/

REAL orient3dexact(REAL *pa, REAL *pb, REAL *pc, REAL *pd)
{
  INEXACT REAL axby1, bxcy1, cxdy1, dxay1, axcy1, bxdy1;
  INEXACT REAL bxay1, cxby1, dxcy1, axdy1, cxay1, dxby1;
  REAL axby0, bxcy0, cxdy0, dxay0, axcy0, bxdy0;
  REAL bxay0, cxby0, dxcy0, axdy0, cxay0, dxby0;
  REAL ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
  REAL temp8[8];
  int templen;
  REAL abc[12], bcd[12], cda[12], dab[12];
  int abclen, bcdlen, cdalen, dablen;
  REAL adet[24], bdet[24], cdet[24], ddet[24];
  int alen, blen, clen, dlen;
  REAL abdet[48], cddet[48];
  int ablen, cdlen;
  REAL deter[96];
  int deterlen;
  int i;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  Two_Product(pa[0], pb[1], axby1, axby0);
  Two_Product(pb[0], pa[1], bxay1, bxay0);
  Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);

  Two_Product(pb[0], pc[1], bxcy1, bxcy0);
  Two_Product(pc[0], pb[1], cxby1, cxby0);
  Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]);

  Two_Product(pc[0], pd[1], cxdy1, cxdy0);
  Two_Product(pd[0], pc[1], dxcy1, dxcy0);
  Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]);

  Two_Product(pd[0], pa[1], dxay1, dxay0);
  Two_Product(pa[0], pd[1], axdy1, axdy0);
  Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]);

  Two_Product(pa[0], pc[1], axcy1, axcy0);
  Two_Product(pc[0], pa[1], cxay1, cxay0);
  Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]);

  Two_Product(pb[0], pd[1], bxdy1, bxdy0);
  Two_Product(pd[0], pb[1], dxby1, dxby0);
  Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]);

  templen = fast_expansion_sum_zeroelim(4, cd, 4, da, temp8);
  cdalen = fast_expansion_sum_zeroelim(templen, temp8, 4, ac, cda);
  templen = fast_expansion_sum_zeroelim(4, da, 4, ab, temp8);
  dablen = fast_expansion_sum_zeroelim(templen, temp8, 4, bd, dab);
  for (i = 0; i < 4; i++) {
    bd[i] = -bd[i];
    ac[i] = -ac[i];
  }
  templen = fast_expansion_sum_zeroelim(4, ab, 4, bc, temp8);
  abclen = fast_expansion_sum_zeroelim(templen, temp8, 4, ac, abc);
  templen = fast_expansion_sum_zeroelim(4, bc, 4, cd, temp8);
  bcdlen = fast_expansion_sum_zeroelim(templen, temp8, 4, bd, bcd);

  alen = scale_expansion_zeroelim(bcdlen, bcd, pa[2], adet);
  blen = scale_expansion_zeroelim(cdalen, cda, -pb[2], bdet);
  clen = scale_expansion_zeroelim(dablen, dab, pc[2], cdet);
  dlen = scale_expansion_zeroelim(abclen, abc, -pd[2], ddet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, deter);

  return deter[deterlen - 1];
}

REAL orient3dadapt(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL permanent)
{
  INEXACT REAL adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
  REAL det, errbound;

  INEXACT REAL bdxcdy1, cdxbdy1, cdxady1, adxcdy1, adxbdy1, bdxady1;
  REAL bdxcdy0, cdxbdy0, cdxady0, adxcdy0, adxbdy0, bdxady0;
  REAL bc[4], ca[4], ab[4];
  INEXACT REAL bc3, ca3, ab3;
  REAL adet[8], bdet[8], cdet[8];
  int alen, blen, clen;
  REAL abdet[16];
  int ablen;
  REAL *finnow, *finother, *finswap;
  REAL fin1[192], fin2[192];
  int finlength;


  REAL adxtail, bdxtail, cdxtail;
  REAL adytail, bdytail, cdytail;
  REAL adztail, bdztail, cdztail;
  INEXACT REAL at_blarge, at_clarge;
  INEXACT REAL bt_clarge, bt_alarge;
  INEXACT REAL ct_alarge, ct_blarge;
  REAL at_b[4], at_c[4], bt_c[4], bt_a[4], ct_a[4], ct_b[4];
  int at_blen, at_clen, bt_clen, bt_alen, ct_alen, ct_blen;
  INEXACT REAL bdxt_cdy1, cdxt_bdy1, cdxt_ady1;
  INEXACT REAL adxt_cdy1, adxt_bdy1, bdxt_ady1;
  REAL bdxt_cdy0, cdxt_bdy0, cdxt_ady0;
  REAL adxt_cdy0, adxt_bdy0, bdxt_ady0;
  INEXACT REAL bdyt_cdx1, cdyt_bdx1, cdyt_adx1;
  INEXACT REAL adyt_cdx1, adyt_bdx1, bdyt_adx1;
  REAL bdyt_cdx0, cdyt_bdx0, cdyt_adx0;
  REAL adyt_cdx0, adyt_bdx0, bdyt_adx0;
  REAL bct[8], cat[8], abt[8];
  int bctlen, catlen, abtlen;
  INEXACT REAL bdxt_cdyt1, cdxt_bdyt1, cdxt_adyt1;
  INEXACT REAL adxt_cdyt1, adxt_bdyt1, bdxt_adyt1;
  REAL bdxt_cdyt0, cdxt_bdyt0, cdxt_adyt0;
  REAL adxt_cdyt0, adxt_bdyt0, bdxt_adyt0;
  REAL u[4], v[12], w[16];
  INEXACT REAL u3;
  int vlength, wlength;
  REAL negate;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j, _k;
  REAL _0;


  adx = (REAL) (pa[0] - pd[0]);
  bdx = (REAL) (pb[0] - pd[0]);
  cdx = (REAL) (pc[0] - pd[0]);
  ady = (REAL) (pa[1] - pd[1]);
  bdy = (REAL) (pb[1] - pd[1]);
  cdy = (REAL) (pc[1] - pd[1]);
  adz = (REAL) (pa[2] - pd[2]);
  bdz = (REAL) (pb[2] - pd[2]);
  cdz = (REAL) (pc[2] - pd[2]);

  Two_Product(bdx, cdy, bdxcdy1, bdxcdy0);
  Two_Product(cdx, bdy, cdxbdy1, cdxbdy0);
  Two_Two_Diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc3, bc[2], bc[1], bc[0]);
  bc[3] = bc3;
  alen = scale_expansion_zeroelim(4, bc, adz, adet);

  Two_Product(cdx, ady, cdxady1, cdxady0);
  Two_Product(adx, cdy, adxcdy1, adxcdy0);
  Two_Two_Diff(cdxady1, cdxady0, adxcdy1, adxcdy0, ca3, ca[2], ca[1], ca[0]);
  ca[3] = ca3;
  blen = scale_expansion_zeroelim(4, ca, bdz, bdet);

  Two_Product(adx, bdy, adxbdy1, adxbdy0);
  Two_Product(bdx, ady, bdxady1, bdxady0);
  Two_Two_Diff(adxbdy1, adxbdy0, bdxady1, bdxady0, ab3, ab[2], ab[1], ab[0]);
  ab[3] = ab3;
  clen = scale_expansion_zeroelim(4, ab, cdz, cdet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  finlength = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1);

  det = estimate(finlength, fin1);
  errbound = o3derrboundB * permanent;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(pa[0], pd[0], adx, adxtail);
  Two_Diff_Tail(pb[0], pd[0], bdx, bdxtail);
  Two_Diff_Tail(pc[0], pd[0], cdx, cdxtail);
  Two_Diff_Tail(pa[1], pd[1], ady, adytail);
  Two_Diff_Tail(pb[1], pd[1], bdy, bdytail);
  Two_Diff_Tail(pc[1], pd[1], cdy, cdytail);
  Two_Diff_Tail(pa[2], pd[2], adz, adztail);
  Two_Diff_Tail(pb[2], pd[2], bdz, bdztail);
  Two_Diff_Tail(pc[2], pd[2], cdz, cdztail);

  if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0)
      && (adytail == 0.0) && (bdytail == 0.0) && (cdytail == 0.0)
      && (adztail == 0.0) && (bdztail == 0.0) && (cdztail == 0.0)) {
    return det;
  }

  errbound = o3derrboundC * permanent + resulterrbound * Absolute(det);
  det += (adz * ((bdx * cdytail + cdy * bdxtail)
                 - (bdy * cdxtail + cdx * bdytail))
          + adztail * (bdx * cdy - bdy * cdx))
       + (bdz * ((cdx * adytail + ady * cdxtail)
                 - (cdy * adxtail + adx * cdytail))
          + bdztail * (cdx * ady - cdy * adx))
       + (cdz * ((adx * bdytail + bdy * adxtail)
                 - (ady * bdxtail + bdx * adytail))
          + cdztail * (adx * bdy - ady * bdx));
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  finnow = fin1;
  finother = fin2;

  if (adxtail == 0.0) {
    if (adytail == 0.0) {
      at_b[0] = 0.0;
      at_blen = 1;
      at_c[0] = 0.0;
      at_clen = 1;
    } else {
      negate = -adytail;
      Two_Product(negate, bdx, at_blarge, at_b[0]);
      at_b[1] = at_blarge;
      at_blen = 2;
      Two_Product(adytail, cdx, at_clarge, at_c[0]);
      at_c[1] = at_clarge;
      at_clen = 2;
    }
  } else {
    if (adytail == 0.0) {
      Two_Product(adxtail, bdy, at_blarge, at_b[0]);
      at_b[1] = at_blarge;
      at_blen = 2;
      negate = -adxtail;
      Two_Product(negate, cdy, at_clarge, at_c[0]);
      at_c[1] = at_clarge;
      at_clen = 2;
    } else {
      Two_Product(adxtail, bdy, adxt_bdy1, adxt_bdy0);
      Two_Product(adytail, bdx, adyt_bdx1, adyt_bdx0);
      Two_Two_Diff(adxt_bdy1, adxt_bdy0, adyt_bdx1, adyt_bdx0,
                   at_blarge, at_b[2], at_b[1], at_b[0]);
      at_b[3] = at_blarge;
      at_blen = 4;
      Two_Product(adytail, cdx, adyt_cdx1, adyt_cdx0);
      Two_Product(adxtail, cdy, adxt_cdy1, adxt_cdy0);
      Two_Two_Diff(adyt_cdx1, adyt_cdx0, adxt_cdy1, adxt_cdy0,
                   at_clarge, at_c[2], at_c[1], at_c[0]);
      at_c[3] = at_clarge;
      at_clen = 4;
    }
  }
  if (bdxtail == 0.0) {
    if (bdytail == 0.0) {
      bt_c[0] = 0.0;
      bt_clen = 1;
      bt_a[0] = 0.0;
      bt_alen = 1;
    } else {
      negate = -bdytail;
      Two_Product(negate, cdx, bt_clarge, bt_c[0]);
      bt_c[1] = bt_clarge;
      bt_clen = 2;
      Two_Product(bdytail, adx, bt_alarge, bt_a[0]);
      bt_a[1] = bt_alarge;
      bt_alen = 2;
    }
  } else {
    if (bdytail == 0.0) {
      Two_Product(bdxtail, cdy, bt_clarge, bt_c[0]);
      bt_c[1] = bt_clarge;
      bt_clen = 2;
      negate = -bdxtail;
      Two_Product(negate, ady, bt_alarge, bt_a[0]);
      bt_a[1] = bt_alarge;
      bt_alen = 2;
    } else {
      Two_Product(bdxtail, cdy, bdxt_cdy1, bdxt_cdy0);
      Two_Product(bdytail, cdx, bdyt_cdx1, bdyt_cdx0);
      Two_Two_Diff(bdxt_cdy1, bdxt_cdy0, bdyt_cdx1, bdyt_cdx0,
                   bt_clarge, bt_c[2], bt_c[1], bt_c[0]);
      bt_c[3] = bt_clarge;
      bt_clen = 4;
      Two_Product(bdytail, adx, bdyt_adx1, bdyt_adx0);
      Two_Product(bdxtail, ady, bdxt_ady1, bdxt_ady0);
      Two_Two_Diff(bdyt_adx1, bdyt_adx0, bdxt_ady1, bdxt_ady0,
                  bt_alarge, bt_a[2], bt_a[1], bt_a[0]);
      bt_a[3] = bt_alarge;
      bt_alen = 4;
    }
  }
  if (cdxtail == 0.0) {
    if (cdytail == 0.0) {
      ct_a[0] = 0.0;
      ct_alen = 1;
      ct_b[0] = 0.0;
      ct_blen = 1;
    } else {
      negate = -cdytail;
      Two_Product(negate, adx, ct_alarge, ct_a[0]);
      ct_a[1] = ct_alarge;
      ct_alen = 2;
      Two_Product(cdytail, bdx, ct_blarge, ct_b[0]);
      ct_b[1] = ct_blarge;
      ct_blen = 2;
    }
  } else {
    if (cdytail == 0.0) {
      Two_Product(cdxtail, ady, ct_alarge, ct_a[0]);
      ct_a[1] = ct_alarge;
      ct_alen = 2;
      negate = -cdxtail;
      Two_Product(negate, bdy, ct_blarge, ct_b[0]);
      ct_b[1] = ct_blarge;
      ct_blen = 2;
    } else {
      Two_Product(cdxtail, ady, cdxt_ady1, cdxt_ady0);
      Two_Product(cdytail, adx, cdyt_adx1, cdyt_adx0);
      Two_Two_Diff(cdxt_ady1, cdxt_ady0, cdyt_adx1, cdyt_adx0,
                   ct_alarge, ct_a[2], ct_a[1], ct_a[0]);
      ct_a[3] = ct_alarge;
      ct_alen = 4;
      Two_Product(cdytail, bdx, cdyt_bdx1, cdyt_bdx0);
      Two_Product(cdxtail, bdy, cdxt_bdy1, cdxt_bdy0);
      Two_Two_Diff(cdyt_bdx1, cdyt_bdx0, cdxt_bdy1, cdxt_bdy0,
                   ct_blarge, ct_b[2], ct_b[1], ct_b[0]);
      ct_b[3] = ct_blarge;
      ct_blen = 4;
    }
  }

  bctlen = fast_expansion_sum_zeroelim(bt_clen, bt_c, ct_blen, ct_b, bct);
  wlength = scale_expansion_zeroelim(bctlen, bct, adz, w);
  finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                          finother);
  finswap = finnow; finnow = finother; finother = finswap;

  catlen = fast_expansion_sum_zeroelim(ct_alen, ct_a, at_clen, at_c, cat);
  wlength = scale_expansion_zeroelim(catlen, cat, bdz, w);
  finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                          finother);
  finswap = finnow; finnow = finother; finother = finswap;

  abtlen = fast_expansion_sum_zeroelim(at_blen, at_b, bt_alen, bt_a, abt);
  wlength = scale_expansion_zeroelim(abtlen, abt, cdz, w);
  finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                          finother);
  finswap = finnow; finnow = finother; finother = finswap;

  if (adztail != 0.0) {
    vlength = scale_expansion_zeroelim(4, bc, adztail, v);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdztail != 0.0) {
    vlength = scale_expansion_zeroelim(4, ca, bdztail, v);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdztail != 0.0) {
    vlength = scale_expansion_zeroelim(4, ab, cdztail, v);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }

  if (adxtail != 0.0) {
    if (bdytail != 0.0) {
      Two_Product(adxtail, bdytail, adxt_bdyt1, adxt_bdyt0);
      Two_One_Product(adxt_bdyt1, adxt_bdyt0, cdz, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (cdztail != 0.0) {
        Two_One_Product(adxt_bdyt1, adxt_bdyt0, cdztail, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
    if (cdytail != 0.0) {
      negate = -adxtail;
      Two_Product(negate, cdytail, adxt_cdyt1, adxt_cdyt0);
      Two_One_Product(adxt_cdyt1, adxt_cdyt0, bdz, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (bdztail != 0.0) {
        Two_One_Product(adxt_cdyt1, adxt_cdyt0, bdztail, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
  }
  if (bdxtail != 0.0) {
    if (cdytail != 0.0) {
      Two_Product(bdxtail, cdytail, bdxt_cdyt1, bdxt_cdyt0);
      Two_One_Product(bdxt_cdyt1, bdxt_cdyt0, adz, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (adztail != 0.0) {
        Two_One_Product(bdxt_cdyt1, bdxt_cdyt0, adztail, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
    if (adytail != 0.0) {
      negate = -bdxtail;
      Two_Product(negate, adytail, bdxt_adyt1, bdxt_adyt0);
      Two_One_Product(bdxt_adyt1, bdxt_adyt0, cdz, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (cdztail != 0.0) {
        Two_One_Product(bdxt_adyt1, bdxt_adyt0, cdztail, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
  }
  if (cdxtail != 0.0) {
    if (adytail != 0.0) {
      Two_Product(cdxtail, adytail, cdxt_adyt1, cdxt_adyt0);
      Two_One_Product(cdxt_adyt1, cdxt_adyt0, bdz, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (bdztail != 0.0) {
        Two_One_Product(cdxt_adyt1, cdxt_adyt0, bdztail, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
    if (bdytail != 0.0) {
      negate = -cdxtail;
      Two_Product(negate, bdytail, cdxt_bdyt1, cdxt_bdyt0);
      Two_One_Product(cdxt_bdyt1, cdxt_bdyt0, adz, u3, u[2], u[1], u[0]);
      u[3] = u3;
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                              finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (adztail != 0.0) {
        Two_One_Product(cdxt_bdyt1, cdxt_bdyt0, adztail, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
    }
  }

  if (adztail != 0.0) {
    wlength = scale_expansion_zeroelim(bctlen, bct, adztail, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdztail != 0.0) {
    wlength = scale_expansion_zeroelim(catlen, cat, bdztail, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdztail != 0.0) {
    wlength = scale_expansion_zeroelim(abtlen, abt, cdztail, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }

  return finnow[finlength - 1];
}

#ifdef USE_CGAL_PREDICATES

REAL orient3d(REAL *pa, REAL *pb, REAL *pc, REAL *pd)
{
  return (REAL) 
    - cgal_pred_obj.orientation_3_object()
        (Point(pa[0], pa[1], pa[2]), 
         Point(pb[0], pb[1], pb[2]),
         Point(pc[0], pc[1], pc[2]),
         Point(pd[0], pd[1], pd[2]));
}

#else

REAL orient3d(REAL *pa, REAL *pb, REAL *pc, REAL *pd)
{
  REAL adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
  REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
  REAL det;


  adx = pa[0] - pd[0];
  ady = pa[1] - pd[1];
  adz = pa[2] - pd[2];
  bdx = pb[0] - pd[0];
  bdy = pb[1] - pd[1];
  bdz = pb[2] - pd[2];
  cdx = pc[0] - pd[0];
  cdy = pc[1] - pd[1];
  cdz = pc[2] - pd[2];

  bdxcdy = bdx * cdy;
  cdxbdy = cdx * bdy;

  cdxady = cdx * ady;
  adxcdy = adx * cdy;

  adxbdy = adx * bdy;
  bdxady = bdx * ady;

  det = adz * (bdxcdy - cdxbdy) 
      + bdz * (cdxady - adxcdy)
      + cdz * (adxbdy - bdxady);

  if (_use_inexact_arith) {
    return det;
  }

  if (_use_static_o3d_filter) {
    //if (fabs(det) > o3dstaticfilter) return det;
    if (det > o3dstaticfilter) return det;
    if (det < -o3dstaticfilter) return det;
  }


  REAL permanent, errbound;

  permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute(adz)
            + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute(bdz)
            + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute(cdz);
  errbound = o3derrboundA * permanent;
  if ((det > errbound) || (-det > errbound)) {
    return det;
  }

  return orient3dadapt(pa, pb, pc, pd, permanent);
}

#endif // #ifdef USE_CGAL_PREDICATES

REAL detri2::Orient3d(Vertex *pa, Vertex *pb, Vertex *pc, Vertex *pd)
{
  return orient3d(pa->crd, pb->crd, pc->crd, pd->crd);
}

/*****************************************************************************/
/*                                                                           */
/*  orient4d()   Return a positive value if the point pe lies above the      */
/*               hyperplane passing through pa, pb, pc, and pd; "above" is   */
/*               defined in a manner best found by trial-and-error.  Returns */
/*               a negative value if pe lies below the hyperplane.  Returns  */
/*               zero if the points are co-hyperplanar (not affinely         */
/*               independent).  The result is also a rough approximation of  */
/*               24 times the signed volume of the 4-simplex defined by the  */
/*               five points.                                                */
/*                                                                           */
/*  Uses exact arithmetic if necessary to ensure a correct answer.  The      */
/*  result returned is the determinant of a matrix.  This determinant is     */
/*  computed adaptively, in the sense that exact arithmetic is used only to  */
/*  the degree it is needed to ensure that the returned value has the        */
/*  correct sign.  Hence, orient4d() is usually quite fast, but will run     */
/*  more slowly when the input points are hyper-coplanar or nearly so.       */
/*                                                                           */
/*  See my Robust Predicates paper for details.                              */
/*                                                                           */
/*****************************************************************************/

REAL orient4dexact(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe,
                   REAL aheight, REAL bheight, REAL cheight, REAL dheight, 
                   REAL eheight)
{
  INEXACT REAL axby1, bxcy1, cxdy1, dxey1, exay1;
  INEXACT REAL bxay1, cxby1, dxcy1, exdy1, axey1;
  INEXACT REAL axcy1, bxdy1, cxey1, dxay1, exby1;
  INEXACT REAL cxay1, dxby1, excy1, axdy1, bxey1;
  REAL axby0, bxcy0, cxdy0, dxey0, exay0;
  REAL bxay0, cxby0, dxcy0, exdy0, axey0;
  REAL axcy0, bxdy0, cxey0, dxay0, exby0;
  REAL cxay0, dxby0, excy0, axdy0, bxey0;
  REAL ab[4], bc[4], cd[4], de[4], ea[4];
  REAL ac[4], bd[4], ce[4], da[4], eb[4];
  REAL temp8a[8], temp8b[8], temp16[16];
  int temp8alen, temp8blen, temp16len;
  REAL abc[24], bcd[24], cde[24], dea[24], eab[24];
  REAL abd[24], bce[24], cda[24], deb[24], eac[24];
  int abclen, bcdlen, cdelen, dealen, eablen;
  int abdlen, bcelen, cdalen, deblen, eaclen;
  REAL temp48a[48], temp48b[48];
  int temp48alen, temp48blen;
  REAL abcd[96], bcde[96], cdea[96], deab[96], eabc[96];
  int abcdlen, bcdelen, cdealen, deablen, eabclen;
  REAL adet[192], bdet[192], cdet[192], ddet[192], edet[192];
  int alen, blen, clen, dlen, elen;
  REAL abdet[384], cddet[384], cdedet[576];
  int ablen, cdlen;
  REAL deter[960];
  int deterlen;
  int i;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;


  Two_Product(pa[0], pb[1], axby1, axby0);
  Two_Product(pb[0], pa[1], bxay1, bxay0);
  Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);

  Two_Product(pb[0], pc[1], bxcy1, bxcy0);
  Two_Product(pc[0], pb[1], cxby1, cxby0);
  Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]);

  Two_Product(pc[0], pd[1], cxdy1, cxdy0);
  Two_Product(pd[0], pc[1], dxcy1, dxcy0);
  Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]);

  Two_Product(pd[0], pe[1], dxey1, dxey0);
  Two_Product(pe[0], pd[1], exdy1, exdy0);
  Two_Two_Diff(dxey1, dxey0, exdy1, exdy0, de[3], de[2], de[1], de[0]);

  Two_Product(pe[0], pa[1], exay1, exay0);
  Two_Product(pa[0], pe[1], axey1, axey0);
  Two_Two_Diff(exay1, exay0, axey1, axey0, ea[3], ea[2], ea[1], ea[0]);

  Two_Product(pa[0], pc[1], axcy1, axcy0);
  Two_Product(pc[0], pa[1], cxay1, cxay0);
  Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]);

  Two_Product(pb[0], pd[1], bxdy1, bxdy0);
  Two_Product(pd[0], pb[1], dxby1, dxby0);
  Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]);

  Two_Product(pc[0], pe[1], cxey1, cxey0);
  Two_Product(pe[0], pc[1], excy1, excy0);
  Two_Two_Diff(cxey1, cxey0, excy1, excy0, ce[3], ce[2], ce[1], ce[0]);

  Two_Product(pd[0], pa[1], dxay1, dxay0);
  Two_Product(pa[0], pd[1], axdy1, axdy0);
  Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]);

  Two_Product(pe[0], pb[1], exby1, exby0);
  Two_Product(pb[0], pe[1], bxey1, bxey0);
  Two_Two_Diff(exby1, exby0, bxey1, bxey0, eb[3], eb[2], eb[1], eb[0]);

  temp8alen = scale_expansion_zeroelim(4, bc, pa[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, -pb[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ab, pc[2], temp8a);
  abclen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       abc);

  temp8alen = scale_expansion_zeroelim(4, cd, pb[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, -pc[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, bc, pd[2], temp8a);
  bcdlen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       bcd);

  temp8alen = scale_expansion_zeroelim(4, de, pc[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ce, -pd[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, cd, pe[2], temp8a);
  cdelen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       cde);

  temp8alen = scale_expansion_zeroelim(4, ea, pd[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, da, -pe[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, de, pa[2], temp8a);
  dealen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       dea);

  temp8alen = scale_expansion_zeroelim(4, ab, pe[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, eb, -pa[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ea, pb[2], temp8a);
  eablen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       eab);

  temp8alen = scale_expansion_zeroelim(4, bd, pa[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, da, pb[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ab, pd[2], temp8a);
  abdlen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       abd);

  temp8alen = scale_expansion_zeroelim(4, ce, pb[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, eb, pc[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, bc, pe[2], temp8a);
  bcelen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       bce);

  temp8alen = scale_expansion_zeroelim(4, da, pc[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, pd[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, cd, pa[2], temp8a);
  cdalen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       cda);

  temp8alen = scale_expansion_zeroelim(4, eb, pd[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, pe[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, de, pb[2], temp8a);
  deblen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       deb);

  temp8alen = scale_expansion_zeroelim(4, ac, pe[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ce, pa[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ea, pc[2], temp8a);
  eaclen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       eac);

  temp48alen = fast_expansion_sum_zeroelim(cdelen, cde, bcelen, bce, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(deblen, deb, bcdlen, bcd, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  bcdelen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, bcde);
  alen = scale_expansion_zeroelim(bcdelen, bcde, aheight, adet);

  temp48alen = fast_expansion_sum_zeroelim(dealen, dea, cdalen, cda, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(eaclen, eac, cdelen, cde, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  cdealen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, cdea);
  blen = scale_expansion_zeroelim(cdealen, cdea, bheight, bdet);

  temp48alen = fast_expansion_sum_zeroelim(eablen, eab, deblen, deb, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(abdlen, abd, dealen, dea, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  deablen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, deab);
  clen = scale_expansion_zeroelim(deablen, deab, cheight, cdet);

  temp48alen = fast_expansion_sum_zeroelim(abclen, abc, eaclen, eac, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(bcelen, bce, eablen, eab, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  eabclen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, eabc);
  dlen = scale_expansion_zeroelim(eabclen, eabc, dheight, ddet);

  temp48alen = fast_expansion_sum_zeroelim(bcdlen, bcd, abdlen, abd, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(cdalen, cda, abclen, abc, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  abcdlen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, abcd);
  elen = scale_expansion_zeroelim(abcdlen, abcd, eheight, edet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  cdelen = fast_expansion_sum_zeroelim(cdlen, cddet, elen, edet, cdedet);
  deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdelen, cdedet, deter);

  return deter[deterlen - 1];
}

REAL orient4dadapt(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe,
                   REAL aheight, REAL bheight, REAL cheight, REAL dheight, 
                   REAL eheight, REAL permanent)
{
  INEXACT REAL aex, bex, cex, dex, aey, bey, cey, dey, aez, bez, cez, dez;
  INEXACT REAL aeheight, beheight, ceheight, deheight;
  REAL det, errbound;

  INEXACT REAL aexbey1, bexaey1, bexcey1, cexbey1;
  INEXACT REAL cexdey1, dexcey1, dexaey1, aexdey1;
  INEXACT REAL aexcey1, cexaey1, bexdey1, dexbey1;
  REAL aexbey0, bexaey0, bexcey0, cexbey0;
  REAL cexdey0, dexcey0, dexaey0, aexdey0;
  REAL aexcey0, cexaey0, bexdey0, dexbey0;
  REAL ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
  INEXACT REAL ab3, bc3, cd3, da3, ac3, bd3;
  REAL abeps, bceps, cdeps, daeps, aceps, bdeps;
  REAL temp8a[8], temp8b[8], temp8c[8], temp16[16], temp24[24];
  int temp8alen, temp8blen, temp8clen, temp16len, temp24len;
  REAL adet[48], bdet[48], cdet[48], ddet[48];
  int alen, blen, clen, dlen;
  REAL abdet[96], cddet[96];
  int ablen, cdlen;
  REAL fin1[192];
  int finlength;

  REAL aextail, bextail, cextail, dextail;
  REAL aeytail, beytail, ceytail, deytail;
  REAL aeztail, beztail, ceztail, deztail;
  REAL aeheighttail, beheighttail, ceheighttail, deheighttail;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;


  aex = (REAL) (pa[0] - pe[0]);
  bex = (REAL) (pb[0] - pe[0]);
  cex = (REAL) (pc[0] - pe[0]);
  dex = (REAL) (pd[0] - pe[0]);
  aey = (REAL) (pa[1] - pe[1]);
  bey = (REAL) (pb[1] - pe[1]);
  cey = (REAL) (pc[1] - pe[1]);
  dey = (REAL) (pd[1] - pe[1]);
  aez = (REAL) (pa[2] - pe[2]);
  bez = (REAL) (pb[2] - pe[2]);
  cez = (REAL) (pc[2] - pe[2]);
  dez = (REAL) (pd[2] - pe[2]);
  aeheight = (REAL) (aheight - eheight);
  beheight = (REAL) (bheight - eheight);
  ceheight = (REAL) (cheight - eheight);
  deheight = (REAL) (dheight - eheight);

  Two_Product(aex, bey, aexbey1, aexbey0);
  Two_Product(bex, aey, bexaey1, bexaey0);
  Two_Two_Diff(aexbey1, aexbey0, bexaey1, bexaey0, ab3, ab[2], ab[1], ab[0]);
  ab[3] = ab3;

  Two_Product(bex, cey, bexcey1, bexcey0);
  Two_Product(cex, bey, cexbey1, cexbey0);
  Two_Two_Diff(bexcey1, bexcey0, cexbey1, cexbey0, bc3, bc[2], bc[1], bc[0]);
  bc[3] = bc3;

  Two_Product(cex, dey, cexdey1, cexdey0);
  Two_Product(dex, cey, dexcey1, dexcey0);
  Two_Two_Diff(cexdey1, cexdey0, dexcey1, dexcey0, cd3, cd[2], cd[1], cd[0]);
  cd[3] = cd3;

  Two_Product(dex, aey, dexaey1, dexaey0);
  Two_Product(aex, dey, aexdey1, aexdey0);
  Two_Two_Diff(dexaey1, dexaey0, aexdey1, aexdey0, da3, da[2], da[1], da[0]);
  da[3] = da3;

  Two_Product(aex, cey, aexcey1, aexcey0);
  Two_Product(cex, aey, cexaey1, cexaey0);
  Two_Two_Diff(aexcey1, aexcey0, cexaey1, cexaey0, ac3, ac[2], ac[1], ac[0]);
  ac[3] = ac3;

  Two_Product(bex, dey, bexdey1, bexdey0);
  Two_Product(dex, bey, dexbey1, dexbey0);
  Two_Two_Diff(bexdey1, bexdey0, dexbey1, dexbey0, bd3, bd[2], bd[1], bd[0]);
  bd[3] = bd3;

  temp8alen = scale_expansion_zeroelim(4, cd, bez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, -cez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, bc, dez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  alen = scale_expansion_zeroelim(temp24len, temp24, -aeheight, adet);

  temp8alen = scale_expansion_zeroelim(4, da, cez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, dez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, cd, aez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  blen = scale_expansion_zeroelim(temp24len, temp24, beheight, bdet);

  temp8alen = scale_expansion_zeroelim(4, ab, dez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, aez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, da, bez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  clen = scale_expansion_zeroelim(temp24len, temp24, -ceheight, cdet);

  temp8alen = scale_expansion_zeroelim(4, bc, aez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, -bez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, ab, cez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  dlen = scale_expansion_zeroelim(temp24len, temp24, deheight, ddet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  finlength = fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, fin1);

  det = estimate(finlength, fin1);
  errbound = isperrboundB * permanent;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(pa[0], pe[0], aex, aextail);
  Two_Diff_Tail(pa[1], pe[1], aey, aeytail);
  Two_Diff_Tail(pa[2], pe[2], aez, aeztail);
  Two_Diff_Tail(aheight, eheight, aeheight, aeheighttail);
  Two_Diff_Tail(pb[0], pe[0], bex, bextail);
  Two_Diff_Tail(pb[1], pe[1], bey, beytail);
  Two_Diff_Tail(pb[2], pe[2], bez, beztail);
  Two_Diff_Tail(bheight, eheight, beheight, beheighttail);
  Two_Diff_Tail(pc[0], pe[0], cex, cextail);
  Two_Diff_Tail(pc[1], pe[1], cey, ceytail);
  Two_Diff_Tail(pc[2], pe[2], cez, ceztail);
  Two_Diff_Tail(cheight, eheight, ceheight, ceheighttail);
  Two_Diff_Tail(pd[0], pe[0], dex, dextail);
  Two_Diff_Tail(pd[1], pe[1], dey, deytail);
  Two_Diff_Tail(pd[2], pe[2], dez, deztail);
  Two_Diff_Tail(dheight, eheight, deheight, deheighttail);
  if ((aextail == 0.0) && (aeytail == 0.0) && (aeztail == 0.0)
      && (bextail == 0.0) && (beytail == 0.0) && (beztail == 0.0)
      && (cextail == 0.0) && (ceytail == 0.0) && (ceztail == 0.0)
      && (dextail == 0.0) && (deytail == 0.0) && (deztail == 0.0)
      && (aeheighttail == 0.0) && (beheighttail == 0.0)
      && (ceheighttail == 0.0) && (deheighttail == 0.0)) {
    return det;
  }

  errbound = isperrboundC * permanent + resulterrbound * Absolute(det);
  abeps = (aex * beytail + bey * aextail)
        - (aey * bextail + bex * aeytail);
  bceps = (bex * ceytail + cey * bextail)
        - (bey * cextail + cex * beytail);
  cdeps = (cex * deytail + dey * cextail)
        - (cey * dextail + dex * ceytail);
  daeps = (dex * aeytail + aey * dextail)
        - (dey * aextail + aex * deytail);
  aceps = (aex * ceytail + cey * aextail)
        - (aey * cextail + cex * aeytail);
  bdeps = (bex * deytail + dey * bextail)
        - (bey * dextail + dex * beytail);
  det += ((beheight
           * ((cez * daeps + dez * aceps + aez * cdeps)
              + (ceztail * da3 + deztail * ac3 + aeztail * cd3))
           + deheight
           * ((aez * bceps - bez * aceps + cez * abeps)
              + (aeztail * bc3 - beztail * ac3 + ceztail * ab3)))
          - (aeheight
           * ((bez * cdeps - cez * bdeps + dez * bceps)
              + (beztail * cd3 - ceztail * bd3 + deztail * bc3))
           + ceheight
           * ((dez * abeps + aez * bdeps + bez * daeps)
              + (deztail * ab3 + aeztail * bd3 + beztail * da3))))
       + ((beheighttail * (cez * da3 + dez * ac3 + aez * cd3)
           + deheighttail * (aez * bc3 - bez * ac3 + cez * ab3))
          - (aeheighttail * (bez * cd3 - cez * bd3 + dez * bc3)
           + ceheighttail * (dez * ab3 + aez * bd3 + bez * da3)));
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  return orient4dexact(pa, pb, pc, pd, pe,
                       aheight, bheight, cheight, dheight, eheight);
}

REAL orient4d(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe)
{
 REAL aex, bex, cex, dex;
 REAL aey, bey, cey, dey;
 REAL aez, bez, cez, dez;
 REAL aexbey, bexaey, bexcey, cexbey, cexdey, dexcey, dexaey, aexdey;
 REAL aexcey, cexaey, bexdey, dexbey;
 REAL aeheight, beheight, ceheight, deheight;
 REAL ab, bc, cd, da, ac, bd;
 REAL abc, bcd, cda, dab;
 REAL aezplus, bezplus, cezplus, dezplus;
 REAL aexbeyplus, bexaeyplus, bexceyplus, cexbeyplus;
 REAL cexdeyplus, dexceyplus, dexaeyplus, aexdeyplus;
 REAL aexceyplus, cexaeyplus, bexdeyplus, dexbeyplus;
 REAL det;
 REAL permanent, errbound;


 aex = pa[0] - pe[0];
 bex = pb[0] - pe[0];
 cex = pc[0] - pe[0];
 dex = pd[0] - pe[0];
 aey = pa[1] - pe[1];
 bey = pb[1] - pe[1];
 cey = pc[1] - pe[1];
 dey = pd[1] - pe[1];
 aez = pa[2] - pe[2];
 bez = pb[2] - pe[2];
 cez = pc[2] - pe[2];
 dez = pd[2] - pe[2];
 aeheight = pa[3] - pe[3]; // aheight - eheight;
 beheight = pb[3] - pe[3]; // bheight - eheight;
 ceheight = pc[3] - pe[3]; // cheight - eheight;
 deheight = pd[3] - pe[3]; // dheight - eheight;

 aexbey = aex * bey;
 bexaey = bex * aey;
 ab = aexbey - bexaey;
 bexcey = bex * cey;
 cexbey = cex * bey;
 bc = bexcey - cexbey;
 cexdey = cex * dey;
 dexcey = dex * cey;
 cd = cexdey - dexcey;
 dexaey = dex * aey;
 aexdey = aex * dey;
 da = dexaey - aexdey;

 aexcey = aex * cey;
 cexaey = cex * aey;
 ac = aexcey - cexaey;
 bexdey = bex * dey;
 dexbey = dex * bey;
 bd = bexdey - dexbey;

 abc = aez * bc - bez * ac + cez * ab;
 bcd = bez * cd - cez * bd + dez * bc;
 cda = cez * da + dez * ac + aez * cd;
 dab = dez * ab + aez * bd + bez * da;

 det  = (deheight * abc - ceheight * dab) + (beheight * cda - aeheight * bcd);

  if (_use_inexact_arith) {
    return det;
  }

  if (_use_static_isp_filter) {
    if (fabs(det) > ispstaticfilter) return det;
    //if (det > ispstaticfilter) return det;
    //if (det < minus_ispstaticfilter) return det;

  }

 aezplus = Absolute(aez);
 bezplus = Absolute(bez);
 cezplus = Absolute(cez);
 dezplus = Absolute(dez);
 aexbeyplus = Absolute(aexbey);
 bexaeyplus = Absolute(bexaey);
 bexceyplus = Absolute(bexcey);
 cexbeyplus = Absolute(cexbey);
 cexdeyplus = Absolute(cexdey);
 dexceyplus = Absolute(dexcey);
 dexaeyplus = Absolute(dexaey);
 aexdeyplus = Absolute(aexdey);
 aexceyplus = Absolute(aexcey);
 cexaeyplus = Absolute(cexaey);
 bexdeyplus = Absolute(bexdey);
 dexbeyplus = Absolute(dexbey);
 permanent = ((cexdeyplus + dexceyplus) * bezplus
              + (dexbeyplus + bexdeyplus) * cezplus
              + (bexceyplus + cexbeyplus) * dezplus)
           * Absolute(aeheight)
           + ((dexaeyplus + aexdeyplus) * cezplus
              + (aexceyplus + cexaeyplus) * dezplus
              + (cexdeyplus + dexceyplus) * aezplus)
           * Absolute(beheight)
           + ((aexbeyplus + bexaeyplus) * dezplus
              + (bexdeyplus + dexbeyplus) * aezplus
              + (dexaeyplus + aexdeyplus) * bezplus)
           * Absolute(ceheight)
           + ((bexceyplus + cexbeyplus) * aezplus
              + (cexaeyplus + aexceyplus) * bezplus
              + (aexbeyplus + bexaeyplus) * cezplus)
           * Absolute(deheight);
 errbound = isperrboundA * permanent;
 if ((det > errbound) || (-det > errbound)) {
   return det;
 }

 return orient4dadapt(pa, pb, pc, pd, pe,
                      pa[3], pb[3], pc[3], pd[3], pe[3], permanent);
                      //aheight, bheight, cheight, dheight, eheight, permanent);
}

REAL detri2::Orient4d(Vertex *pa, Vertex *pb, Vertex *pc, Vertex *pd, Vertex *pe)
{
  return orient4d(pa->crd, pb->crd, pc->crd, pd->crd, pe->crd);
}

/*****************************************************************************/
/*                                                                           */
/*  inspherefast()   Approximate 3D insphere test.  Nonrobust.               */
/*  insphereexact()   Exact 3D insphere test.  Robust.                       */
/*  insphereslow()   Another exact 3D insphere test.  Robust.                */
/*  insphere()   Adaptive exact 3D insphere test.  Robust.                   */
/*                                                                           */
/*               Return a positive value if the point pe lies inside the     */
/*               sphere passing through pa, pb, pc, and pd; a negative value */
/*               if it lies outside; and zero if the five points are         */
/*               cospherical.  The points pa, pb, pc, and pd must be ordered */
/*               so that they have a positive orientation (as defined by     */
/*               orient3d()), or the sign of the result will be reversed.    */
/*                                                                           */
/*  Only the first and last routine should be used; the middle two are for   */
/*  timings.                                                                 */
/*                                                                           */
/*  The last three use exact arithmetic to ensure a correct answer.  The     */
/*  result returned is the determinant of a matrix.  In insphere() only,     */
/*  this determinant is computed adaptively, in the sense that exact         */
/*  arithmetic is used only to the degree it is needed to ensure that the    */
/*  returned value has the correct sign.  Hence, insphere() is usually quite */
/*  fast, but will run more slowly when the input points are cospherical or  */
/*  nearly so.                                                               */
/*                                                                           */
/*****************************************************************************/

REAL inspherefast(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL *pe)
{
  REAL aex, bex, cex, dex;
  REAL aey, bey, cey, dey;
  REAL aez, bez, cez, dez;
  REAL alift, blift, clift, dlift;
  REAL ab, bc, cd, da, ac, bd;
  REAL abc, bcd, cda, dab;

  aex = pa[0] - pe[0];
  bex = pb[0] - pe[0];
  cex = pc[0] - pe[0];
  dex = pd[0] - pe[0];
  aey = pa[1] - pe[1];
  bey = pb[1] - pe[1];
  cey = pc[1] - pe[1];
  dey = pd[1] - pe[1];
  aez = pa[2] - pe[2];
  bez = pb[2] - pe[2];
  cez = pc[2] - pe[2];
  dez = pd[2] - pe[2];

  ab = aex * bey - bex * aey;
  bc = bex * cey - cex * bey;
  cd = cex * dey - dex * cey;
  da = dex * aey - aex * dey;

  ac = aex * cey - cex * aey;
  bd = bex * dey - dex * bey;

  abc = aez * bc - bez * ac + cez * ab;
  bcd = bez * cd - cez * bd + dez * bc;
  cda = cez * da + dez * ac + aez * cd;
  dab = dez * ab + aez * bd + bez * da;

  alift = aex * aex + aey * aey + aez * aez;
  blift = bex * bex + bey * bey + bez * bez;
  clift = cex * cex + cey * cey + cez * cez;
  dlift = dex * dex + dey * dey + dez * dez;

  return (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
}

REAL insphereexact(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL *pe)
{
  INEXACT REAL axby1, bxcy1, cxdy1, dxey1, exay1;
  INEXACT REAL bxay1, cxby1, dxcy1, exdy1, axey1;
  INEXACT REAL axcy1, bxdy1, cxey1, dxay1, exby1;
  INEXACT REAL cxay1, dxby1, excy1, axdy1, bxey1;
  REAL axby0, bxcy0, cxdy0, dxey0, exay0;
  REAL bxay0, cxby0, dxcy0, exdy0, axey0;
  REAL axcy0, bxdy0, cxey0, dxay0, exby0;
  REAL cxay0, dxby0, excy0, axdy0, bxey0;
  REAL ab[4], bc[4], cd[4], de[4], ea[4];
  REAL ac[4], bd[4], ce[4], da[4], eb[4];
  REAL temp8a[8], temp8b[8], temp16[16];
  int temp8alen, temp8blen, temp16len;
  REAL abc[24], bcd[24], cde[24], dea[24], eab[24];
  REAL abd[24], bce[24], cda[24], deb[24], eac[24];
  int abclen, bcdlen, cdelen, dealen, eablen;
  int abdlen, bcelen, cdalen, deblen, eaclen;
  REAL temp48a[48], temp48b[48];
  int temp48alen, temp48blen;
  REAL abcd[96], bcde[96], cdea[96], deab[96], eabc[96];
  int abcdlen, bcdelen, cdealen, deablen, eabclen;
  REAL temp192[192];
  REAL det384x[384], det384y[384], det384z[384];
  int xlen, ylen, zlen;
  REAL detxy[768];
  int xylen;
  REAL adet[1152], bdet[1152], cdet[1152], ddet[1152], edet[1152];
  int alen, blen, clen, dlen, elen;
  REAL abdet[2304], cddet[2304], cdedet[3456];
  int ablen, cdlen;
  REAL deter[5760];
  int deterlen;
  int i;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  Two_Product(pa[0], pb[1], axby1, axby0);
  Two_Product(pb[0], pa[1], bxay1, bxay0);
  Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);

  Two_Product(pb[0], pc[1], bxcy1, bxcy0);
  Two_Product(pc[0], pb[1], cxby1, cxby0);
  Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]);

  Two_Product(pc[0], pd[1], cxdy1, cxdy0);
  Two_Product(pd[0], pc[1], dxcy1, dxcy0);
  Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]);

  Two_Product(pd[0], pe[1], dxey1, dxey0);
  Two_Product(pe[0], pd[1], exdy1, exdy0);
  Two_Two_Diff(dxey1, dxey0, exdy1, exdy0, de[3], de[2], de[1], de[0]);

  Two_Product(pe[0], pa[1], exay1, exay0);
  Two_Product(pa[0], pe[1], axey1, axey0);
  Two_Two_Diff(exay1, exay0, axey1, axey0, ea[3], ea[2], ea[1], ea[0]);

  Two_Product(pa[0], pc[1], axcy1, axcy0);
  Two_Product(pc[0], pa[1], cxay1, cxay0);
  Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]);

  Two_Product(pb[0], pd[1], bxdy1, bxdy0);
  Two_Product(pd[0], pb[1], dxby1, dxby0);
  Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]);

  Two_Product(pc[0], pe[1], cxey1, cxey0);
  Two_Product(pe[0], pc[1], excy1, excy0);
  Two_Two_Diff(cxey1, cxey0, excy1, excy0, ce[3], ce[2], ce[1], ce[0]);

  Two_Product(pd[0], pa[1], dxay1, dxay0);
  Two_Product(pa[0], pd[1], axdy1, axdy0);
  Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]);

  Two_Product(pe[0], pb[1], exby1, exby0);
  Two_Product(pb[0], pe[1], bxey1, bxey0);
  Two_Two_Diff(exby1, exby0, bxey1, bxey0, eb[3], eb[2], eb[1], eb[0]);

  temp8alen = scale_expansion_zeroelim(4, bc, pa[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, -pb[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ab, pc[2], temp8a);
  abclen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       abc);

  temp8alen = scale_expansion_zeroelim(4, cd, pb[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, -pc[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, bc, pd[2], temp8a);
  bcdlen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       bcd);

  temp8alen = scale_expansion_zeroelim(4, de, pc[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ce, -pd[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, cd, pe[2], temp8a);
  cdelen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       cde);

  temp8alen = scale_expansion_zeroelim(4, ea, pd[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, da, -pe[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, de, pa[2], temp8a);
  dealen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       dea);

  temp8alen = scale_expansion_zeroelim(4, ab, pe[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, eb, -pa[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ea, pb[2], temp8a);
  eablen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       eab);

  temp8alen = scale_expansion_zeroelim(4, bd, pa[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, da, pb[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ab, pd[2], temp8a);
  abdlen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       abd);

  temp8alen = scale_expansion_zeroelim(4, ce, pb[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, eb, pc[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, bc, pe[2], temp8a);
  bcelen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       bce);

  temp8alen = scale_expansion_zeroelim(4, da, pc[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, pd[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, cd, pa[2], temp8a);
  cdalen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       cda);

  temp8alen = scale_expansion_zeroelim(4, eb, pd[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, pe[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, de, pb[2], temp8a);
  deblen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       deb);

  temp8alen = scale_expansion_zeroelim(4, ac, pe[2], temp8a);
  temp8blen = scale_expansion_zeroelim(4, ce, pa[2], temp8b);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b,
                                          temp16);
  temp8alen = scale_expansion_zeroelim(4, ea, pc[2], temp8a);
  eaclen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                       eac);

  temp48alen = fast_expansion_sum_zeroelim(cdelen, cde, bcelen, bce, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(deblen, deb, bcdlen, bcd, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  bcdelen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, bcde);
  xlen = scale_expansion_zeroelim(bcdelen, bcde, pa[0], temp192);
  xlen = scale_expansion_zeroelim(xlen, temp192, pa[0], det384x);
  ylen = scale_expansion_zeroelim(bcdelen, bcde, pa[1], temp192);
  ylen = scale_expansion_zeroelim(ylen, temp192, pa[1], det384y);
  zlen = scale_expansion_zeroelim(bcdelen, bcde, pa[2], temp192);
  zlen = scale_expansion_zeroelim(zlen, temp192, pa[2], det384z);
  xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy);
  alen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, adet);

  temp48alen = fast_expansion_sum_zeroelim(dealen, dea, cdalen, cda, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(eaclen, eac, cdelen, cde, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  cdealen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, cdea);
  xlen = scale_expansion_zeroelim(cdealen, cdea, pb[0], temp192);
  xlen = scale_expansion_zeroelim(xlen, temp192, pb[0], det384x);
  ylen = scale_expansion_zeroelim(cdealen, cdea, pb[1], temp192);
  ylen = scale_expansion_zeroelim(ylen, temp192, pb[1], det384y);
  zlen = scale_expansion_zeroelim(cdealen, cdea, pb[2], temp192);
  zlen = scale_expansion_zeroelim(zlen, temp192, pb[2], det384z);
  xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy);
  blen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, bdet);

  temp48alen = fast_expansion_sum_zeroelim(eablen, eab, deblen, deb, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(abdlen, abd, dealen, dea, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  deablen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, deab);
  xlen = scale_expansion_zeroelim(deablen, deab, pc[0], temp192);
  xlen = scale_expansion_zeroelim(xlen, temp192, pc[0], det384x);
  ylen = scale_expansion_zeroelim(deablen, deab, pc[1], temp192);
  ylen = scale_expansion_zeroelim(ylen, temp192, pc[1], det384y);
  zlen = scale_expansion_zeroelim(deablen, deab, pc[2], temp192);
  zlen = scale_expansion_zeroelim(zlen, temp192, pc[2], det384z);
  xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy);
  clen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, cdet);

  temp48alen = fast_expansion_sum_zeroelim(abclen, abc, eaclen, eac, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(bcelen, bce, eablen, eab, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  eabclen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, eabc);
  xlen = scale_expansion_zeroelim(eabclen, eabc, pd[0], temp192);
  xlen = scale_expansion_zeroelim(xlen, temp192, pd[0], det384x);
  ylen = scale_expansion_zeroelim(eabclen, eabc, pd[1], temp192);
  ylen = scale_expansion_zeroelim(ylen, temp192, pd[1], det384y);
  zlen = scale_expansion_zeroelim(eabclen, eabc, pd[2], temp192);
  zlen = scale_expansion_zeroelim(zlen, temp192, pd[2], det384z);
  xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy);
  dlen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, ddet);

  temp48alen = fast_expansion_sum_zeroelim(bcdlen, bcd, abdlen, abd, temp48a);
  temp48blen = fast_expansion_sum_zeroelim(cdalen, cda, abclen, abc, temp48b);
  for (i = 0; i < temp48blen; i++) {
    temp48b[i] = -temp48b[i];
  }
  abcdlen = fast_expansion_sum_zeroelim(temp48alen, temp48a,
                                        temp48blen, temp48b, abcd);
  xlen = scale_expansion_zeroelim(abcdlen, abcd, pe[0], temp192);
  xlen = scale_expansion_zeroelim(xlen, temp192, pe[0], det384x);
  ylen = scale_expansion_zeroelim(abcdlen, abcd, pe[1], temp192);
  ylen = scale_expansion_zeroelim(ylen, temp192, pe[1], det384y);
  zlen = scale_expansion_zeroelim(abcdlen, abcd, pe[2], temp192);
  zlen = scale_expansion_zeroelim(zlen, temp192, pe[2], det384z);
  xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy);
  elen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, edet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  cdelen = fast_expansion_sum_zeroelim(cdlen, cddet, elen, edet, cdedet);
  deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdelen, cdedet, deter);

  return deter[deterlen - 1];
}

REAL insphereslow(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL *pe)
{
  INEXACT REAL aex, bex, cex, dex, aey, bey, cey, dey, aez, bez, cez, dez;
  REAL aextail, bextail, cextail, dextail;
  REAL aeytail, beytail, ceytail, deytail;
  REAL aeztail, beztail, ceztail, deztail;
  REAL negate, negatetail;
  INEXACT REAL axby7, bxcy7, cxdy7, dxay7, axcy7, bxdy7;
  INEXACT REAL bxay7, cxby7, dxcy7, axdy7, cxay7, dxby7;
  REAL axby[8], bxcy[8], cxdy[8], dxay[8], axcy[8], bxdy[8];
  REAL bxay[8], cxby[8], dxcy[8], axdy[8], cxay[8], dxby[8];
  REAL ab[16], bc[16], cd[16], da[16], ac[16], bd[16];
  int ablen, bclen, cdlen, dalen, aclen, bdlen;
  REAL temp32a[32], temp32b[32], temp64a[64], temp64b[64], temp64c[64];
  int temp32alen, temp32blen, temp64alen, temp64blen, temp64clen;
  REAL temp128[128], temp192[192];
  int temp128len, temp192len;
  REAL detx[384], detxx[768], detxt[384], detxxt[768], detxtxt[768];
  int xlen, xxlen, xtlen, xxtlen, xtxtlen;
  REAL x1[1536], x2[2304];
  int x1len, x2len;
  REAL dety[384], detyy[768], detyt[384], detyyt[768], detytyt[768];
  int ylen, yylen, ytlen, yytlen, ytytlen;
  REAL y1[1536], y2[2304];
  int y1len, y2len;
  REAL detz[384], detzz[768], detzt[384], detzzt[768], detztzt[768];
  int zlen, zzlen, ztlen, zztlen, ztztlen;
  REAL z1[1536], z2[2304];
  int z1len, z2len;
  REAL detxy[4608];
  int xylen;
  REAL adet[6912], bdet[6912], cdet[6912], ddet[6912];
  int alen, blen, clen, dlen;
  REAL abdet[13824], cddet[13824], deter[27648];
  int deterlen;
  int i;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL a0hi, a0lo, a1hi, a1lo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j, _k, _l, _m, _n;
  REAL _0, _1, _2;

  Two_Diff(pa[0], pe[0], aex, aextail);
  Two_Diff(pa[1], pe[1], aey, aeytail);
  Two_Diff(pa[2], pe[2], aez, aeztail);
  Two_Diff(pb[0], pe[0], bex, bextail);
  Two_Diff(pb[1], pe[1], bey, beytail);
  Two_Diff(pb[2], pe[2], bez, beztail);
  Two_Diff(pc[0], pe[0], cex, cextail);
  Two_Diff(pc[1], pe[1], cey, ceytail);
  Two_Diff(pc[2], pe[2], cez, ceztail);
  Two_Diff(pd[0], pe[0], dex, dextail);
  Two_Diff(pd[1], pe[1], dey, deytail);
  Two_Diff(pd[2], pe[2], dez, deztail);

  Two_Two_Product(aex, aextail, bey, beytail,
                  axby7, axby[6], axby[5], axby[4],
                  axby[3], axby[2], axby[1], axby[0]);
  axby[7] = axby7;
  negate = -aey;
  negatetail = -aeytail;
  Two_Two_Product(bex, bextail, negate, negatetail,
                  bxay7, bxay[6], bxay[5], bxay[4],
                  bxay[3], bxay[2], bxay[1], bxay[0]);
  bxay[7] = bxay7;
  ablen = fast_expansion_sum_zeroelim(8, axby, 8, bxay, ab);
  Two_Two_Product(bex, bextail, cey, ceytail,
                  bxcy7, bxcy[6], bxcy[5], bxcy[4],
                  bxcy[3], bxcy[2], bxcy[1], bxcy[0]);
  bxcy[7] = bxcy7;
  negate = -bey;
  negatetail = -beytail;
  Two_Two_Product(cex, cextail, negate, negatetail,
                  cxby7, cxby[6], cxby[5], cxby[4],
                  cxby[3], cxby[2], cxby[1], cxby[0]);
  cxby[7] = cxby7;
  bclen = fast_expansion_sum_zeroelim(8, bxcy, 8, cxby, bc);
  Two_Two_Product(cex, cextail, dey, deytail,
                  cxdy7, cxdy[6], cxdy[5], cxdy[4],
                  cxdy[3], cxdy[2], cxdy[1], cxdy[0]);
  cxdy[7] = cxdy7;
  negate = -cey;
  negatetail = -ceytail;
  Two_Two_Product(dex, dextail, negate, negatetail,
                  dxcy7, dxcy[6], dxcy[5], dxcy[4],
                  dxcy[3], dxcy[2], dxcy[1], dxcy[0]);
  dxcy[7] = dxcy7;
  cdlen = fast_expansion_sum_zeroelim(8, cxdy, 8, dxcy, cd);
  Two_Two_Product(dex, dextail, aey, aeytail,
                  dxay7, dxay[6], dxay[5], dxay[4],
                  dxay[3], dxay[2], dxay[1], dxay[0]);
  dxay[7] = dxay7;
  negate = -dey;
  negatetail = -deytail;
  Two_Two_Product(aex, aextail, negate, negatetail,
                  axdy7, axdy[6], axdy[5], axdy[4],
                  axdy[3], axdy[2], axdy[1], axdy[0]);
  axdy[7] = axdy7;
  dalen = fast_expansion_sum_zeroelim(8, dxay, 8, axdy, da);
  Two_Two_Product(aex, aextail, cey, ceytail,
                  axcy7, axcy[6], axcy[5], axcy[4],
                  axcy[3], axcy[2], axcy[1], axcy[0]);
  axcy[7] = axcy7;
  negate = -aey;
  negatetail = -aeytail;
  Two_Two_Product(cex, cextail, negate, negatetail,
                  cxay7, cxay[6], cxay[5], cxay[4],
                  cxay[3], cxay[2], cxay[1], cxay[0]);
  cxay[7] = cxay7;
  aclen = fast_expansion_sum_zeroelim(8, axcy, 8, cxay, ac);
  Two_Two_Product(bex, bextail, dey, deytail,
                  bxdy7, bxdy[6], bxdy[5], bxdy[4],
                  bxdy[3], bxdy[2], bxdy[1], bxdy[0]);
  bxdy[7] = bxdy7;
  negate = -bey;
  negatetail = -beytail;
  Two_Two_Product(dex, dextail, negate, negatetail,
                  dxby7, dxby[6], dxby[5], dxby[4],
                  dxby[3], dxby[2], dxby[1], dxby[0]);
  dxby[7] = dxby7;
  bdlen = fast_expansion_sum_zeroelim(8, bxdy, 8, dxby, bd);

  temp32alen = scale_expansion_zeroelim(cdlen, cd, -bez, temp32a);
  temp32blen = scale_expansion_zeroelim(cdlen, cd, -beztail, temp32b);
  temp64alen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64a);
  temp32alen = scale_expansion_zeroelim(bdlen, bd, cez, temp32a);
  temp32blen = scale_expansion_zeroelim(bdlen, bd, ceztail, temp32b);
  temp64blen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64b);
  temp32alen = scale_expansion_zeroelim(bclen, bc, -dez, temp32a);
  temp32blen = scale_expansion_zeroelim(bclen, bc, -deztail, temp32b);
  temp64clen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64c);
  temp128len = fast_expansion_sum_zeroelim(temp64alen, temp64a,
                                           temp64blen, temp64b, temp128);
  temp192len = fast_expansion_sum_zeroelim(temp64clen, temp64c,
                                           temp128len, temp128, temp192);
  xlen = scale_expansion_zeroelim(temp192len, temp192, aex, detx);
  xxlen = scale_expansion_zeroelim(xlen, detx, aex, detxx);
  xtlen = scale_expansion_zeroelim(temp192len, temp192, aextail, detxt);
  xxtlen = scale_expansion_zeroelim(xtlen, detxt, aex, detxxt);
  for (i = 0; i < xxtlen; i++) {
    detxxt[i] *= 2.0;
  }
  xtxtlen = scale_expansion_zeroelim(xtlen, detxt, aextail, detxtxt);
  x1len = fast_expansion_sum_zeroelim(xxlen, detxx, xxtlen, detxxt, x1);
  x2len = fast_expansion_sum_zeroelim(x1len, x1, xtxtlen, detxtxt, x2);
  ylen = scale_expansion_zeroelim(temp192len, temp192, aey, dety);
  yylen = scale_expansion_zeroelim(ylen, dety, aey, detyy);
  ytlen = scale_expansion_zeroelim(temp192len, temp192, aeytail, detyt);
  yytlen = scale_expansion_zeroelim(ytlen, detyt, aey, detyyt);
  for (i = 0; i < yytlen; i++) {
    detyyt[i] *= 2.0;
  }
  ytytlen = scale_expansion_zeroelim(ytlen, detyt, aeytail, detytyt);
  y1len = fast_expansion_sum_zeroelim(yylen, detyy, yytlen, detyyt, y1);
  y2len = fast_expansion_sum_zeroelim(y1len, y1, ytytlen, detytyt, y2);
  zlen = scale_expansion_zeroelim(temp192len, temp192, aez, detz);
  zzlen = scale_expansion_zeroelim(zlen, detz, aez, detzz);
  ztlen = scale_expansion_zeroelim(temp192len, temp192, aeztail, detzt);
  zztlen = scale_expansion_zeroelim(ztlen, detzt, aez, detzzt);
  for (i = 0; i < zztlen; i++) {
    detzzt[i] *= 2.0;
  }
  ztztlen = scale_expansion_zeroelim(ztlen, detzt, aeztail, detztzt);
  z1len = fast_expansion_sum_zeroelim(zzlen, detzz, zztlen, detzzt, z1);
  z2len = fast_expansion_sum_zeroelim(z1len, z1, ztztlen, detztzt, z2);
  xylen = fast_expansion_sum_zeroelim(x2len, x2, y2len, y2, detxy);
  alen = fast_expansion_sum_zeroelim(z2len, z2, xylen, detxy, adet);

  temp32alen = scale_expansion_zeroelim(dalen, da, cez, temp32a);
  temp32blen = scale_expansion_zeroelim(dalen, da, ceztail, temp32b);
  temp64alen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64a);
  temp32alen = scale_expansion_zeroelim(aclen, ac, dez, temp32a);
  temp32blen = scale_expansion_zeroelim(aclen, ac, deztail, temp32b);
  temp64blen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64b);
  temp32alen = scale_expansion_zeroelim(cdlen, cd, aez, temp32a);
  temp32blen = scale_expansion_zeroelim(cdlen, cd, aeztail, temp32b);
  temp64clen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64c);
  temp128len = fast_expansion_sum_zeroelim(temp64alen, temp64a,
                                           temp64blen, temp64b, temp128);
  temp192len = fast_expansion_sum_zeroelim(temp64clen, temp64c,
                                           temp128len, temp128, temp192);
  xlen = scale_expansion_zeroelim(temp192len, temp192, bex, detx);
  xxlen = scale_expansion_zeroelim(xlen, detx, bex, detxx);
  xtlen = scale_expansion_zeroelim(temp192len, temp192, bextail, detxt);
  xxtlen = scale_expansion_zeroelim(xtlen, detxt, bex, detxxt);
  for (i = 0; i < xxtlen; i++) {
    detxxt[i] *= 2.0;
  }
  xtxtlen = scale_expansion_zeroelim(xtlen, detxt, bextail, detxtxt);
  x1len = fast_expansion_sum_zeroelim(xxlen, detxx, xxtlen, detxxt, x1);
  x2len = fast_expansion_sum_zeroelim(x1len, x1, xtxtlen, detxtxt, x2);
  ylen = scale_expansion_zeroelim(temp192len, temp192, bey, dety);
  yylen = scale_expansion_zeroelim(ylen, dety, bey, detyy);
  ytlen = scale_expansion_zeroelim(temp192len, temp192, beytail, detyt);
  yytlen = scale_expansion_zeroelim(ytlen, detyt, bey, detyyt);
  for (i = 0; i < yytlen; i++) {
    detyyt[i] *= 2.0;
  }
  ytytlen = scale_expansion_zeroelim(ytlen, detyt, beytail, detytyt);
  y1len = fast_expansion_sum_zeroelim(yylen, detyy, yytlen, detyyt, y1);
  y2len = fast_expansion_sum_zeroelim(y1len, y1, ytytlen, detytyt, y2);
  zlen = scale_expansion_zeroelim(temp192len, temp192, bez, detz);
  zzlen = scale_expansion_zeroelim(zlen, detz, bez, detzz);
  ztlen = scale_expansion_zeroelim(temp192len, temp192, beztail, detzt);
  zztlen = scale_expansion_zeroelim(ztlen, detzt, bez, detzzt);
  for (i = 0; i < zztlen; i++) {
    detzzt[i] *= 2.0;
  }
  ztztlen = scale_expansion_zeroelim(ztlen, detzt, beztail, detztzt);
  z1len = fast_expansion_sum_zeroelim(zzlen, detzz, zztlen, detzzt, z1);
  z2len = fast_expansion_sum_zeroelim(z1len, z1, ztztlen, detztzt, z2);
  xylen = fast_expansion_sum_zeroelim(x2len, x2, y2len, y2, detxy);
  blen = fast_expansion_sum_zeroelim(z2len, z2, xylen, detxy, bdet);

  temp32alen = scale_expansion_zeroelim(ablen, ab, -dez, temp32a);
  temp32blen = scale_expansion_zeroelim(ablen, ab, -deztail, temp32b);
  temp64alen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64a);
  temp32alen = scale_expansion_zeroelim(bdlen, bd, -aez, temp32a);
  temp32blen = scale_expansion_zeroelim(bdlen, bd, -aeztail, temp32b);
  temp64blen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64b);
  temp32alen = scale_expansion_zeroelim(dalen, da, -bez, temp32a);
  temp32blen = scale_expansion_zeroelim(dalen, da, -beztail, temp32b);
  temp64clen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64c);
  temp128len = fast_expansion_sum_zeroelim(temp64alen, temp64a,
                                           temp64blen, temp64b, temp128);
  temp192len = fast_expansion_sum_zeroelim(temp64clen, temp64c,
                                           temp128len, temp128, temp192);
  xlen = scale_expansion_zeroelim(temp192len, temp192, cex, detx);
  xxlen = scale_expansion_zeroelim(xlen, detx, cex, detxx);
  xtlen = scale_expansion_zeroelim(temp192len, temp192, cextail, detxt);
  xxtlen = scale_expansion_zeroelim(xtlen, detxt, cex, detxxt);
  for (i = 0; i < xxtlen; i++) {
    detxxt[i] *= 2.0;
  }
  xtxtlen = scale_expansion_zeroelim(xtlen, detxt, cextail, detxtxt);
  x1len = fast_expansion_sum_zeroelim(xxlen, detxx, xxtlen, detxxt, x1);
  x2len = fast_expansion_sum_zeroelim(x1len, x1, xtxtlen, detxtxt, x2);
  ylen = scale_expansion_zeroelim(temp192len, temp192, cey, dety);
  yylen = scale_expansion_zeroelim(ylen, dety, cey, detyy);
  ytlen = scale_expansion_zeroelim(temp192len, temp192, ceytail, detyt);
  yytlen = scale_expansion_zeroelim(ytlen, detyt, cey, detyyt);
  for (i = 0; i < yytlen; i++) {
    detyyt[i] *= 2.0;
  }
  ytytlen = scale_expansion_zeroelim(ytlen, detyt, ceytail, detytyt);
  y1len = fast_expansion_sum_zeroelim(yylen, detyy, yytlen, detyyt, y1);
  y2len = fast_expansion_sum_zeroelim(y1len, y1, ytytlen, detytyt, y2);
  zlen = scale_expansion_zeroelim(temp192len, temp192, cez, detz);
  zzlen = scale_expansion_zeroelim(zlen, detz, cez, detzz);
  ztlen = scale_expansion_zeroelim(temp192len, temp192, ceztail, detzt);
  zztlen = scale_expansion_zeroelim(ztlen, detzt, cez, detzzt);
  for (i = 0; i < zztlen; i++) {
    detzzt[i] *= 2.0;
  }
  ztztlen = scale_expansion_zeroelim(ztlen, detzt, ceztail, detztzt);
  z1len = fast_expansion_sum_zeroelim(zzlen, detzz, zztlen, detzzt, z1);
  z2len = fast_expansion_sum_zeroelim(z1len, z1, ztztlen, detztzt, z2);
  xylen = fast_expansion_sum_zeroelim(x2len, x2, y2len, y2, detxy);
  clen = fast_expansion_sum_zeroelim(z2len, z2, xylen, detxy, cdet);

  temp32alen = scale_expansion_zeroelim(bclen, bc, aez, temp32a);
  temp32blen = scale_expansion_zeroelim(bclen, bc, aeztail, temp32b);
  temp64alen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64a);
  temp32alen = scale_expansion_zeroelim(aclen, ac, -bez, temp32a);
  temp32blen = scale_expansion_zeroelim(aclen, ac, -beztail, temp32b);
  temp64blen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64b);
  temp32alen = scale_expansion_zeroelim(ablen, ab, cez, temp32a);
  temp32blen = scale_expansion_zeroelim(ablen, ab, ceztail, temp32b);
  temp64clen = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                           temp32blen, temp32b, temp64c);
  temp128len = fast_expansion_sum_zeroelim(temp64alen, temp64a,
                                           temp64blen, temp64b, temp128);
  temp192len = fast_expansion_sum_zeroelim(temp64clen, temp64c,
                                           temp128len, temp128, temp192);
  xlen = scale_expansion_zeroelim(temp192len, temp192, dex, detx);
  xxlen = scale_expansion_zeroelim(xlen, detx, dex, detxx);
  xtlen = scale_expansion_zeroelim(temp192len, temp192, dextail, detxt);
  xxtlen = scale_expansion_zeroelim(xtlen, detxt, dex, detxxt);
  for (i = 0; i < xxtlen; i++) {
    detxxt[i] *= 2.0;
  }
  xtxtlen = scale_expansion_zeroelim(xtlen, detxt, dextail, detxtxt);
  x1len = fast_expansion_sum_zeroelim(xxlen, detxx, xxtlen, detxxt, x1);
  x2len = fast_expansion_sum_zeroelim(x1len, x1, xtxtlen, detxtxt, x2);
  ylen = scale_expansion_zeroelim(temp192len, temp192, dey, dety);
  yylen = scale_expansion_zeroelim(ylen, dety, dey, detyy);
  ytlen = scale_expansion_zeroelim(temp192len, temp192, deytail, detyt);
  yytlen = scale_expansion_zeroelim(ytlen, detyt, dey, detyyt);
  for (i = 0; i < yytlen; i++) {
    detyyt[i] *= 2.0;
  }
  ytytlen = scale_expansion_zeroelim(ytlen, detyt, deytail, detytyt);
  y1len = fast_expansion_sum_zeroelim(yylen, detyy, yytlen, detyyt, y1);
  y2len = fast_expansion_sum_zeroelim(y1len, y1, ytytlen, detytyt, y2);
  zlen = scale_expansion_zeroelim(temp192len, temp192, dez, detz);
  zzlen = scale_expansion_zeroelim(zlen, detz, dez, detzz);
  ztlen = scale_expansion_zeroelim(temp192len, temp192, deztail, detzt);
  zztlen = scale_expansion_zeroelim(ztlen, detzt, dez, detzzt);
  for (i = 0; i < zztlen; i++) {
    detzzt[i] *= 2.0;
  }
  ztztlen = scale_expansion_zeroelim(ztlen, detzt, deztail, detztzt);
  z1len = fast_expansion_sum_zeroelim(zzlen, detzz, zztlen, detzzt, z1);
  z2len = fast_expansion_sum_zeroelim(z1len, z1, ztztlen, detztzt, z2);
  xylen = fast_expansion_sum_zeroelim(x2len, x2, y2len, y2, detxy);
  dlen = fast_expansion_sum_zeroelim(z2len, z2, xylen, detxy, ddet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, deter);

  return deter[deterlen - 1];
}

REAL insphereadapt(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL *pe,
                   REAL permanent)
{
  INEXACT REAL aex, bex, cex, dex, aey, bey, cey, dey, aez, bez, cez, dez;
  REAL det, errbound;

  INEXACT REAL aexbey1, bexaey1, bexcey1, cexbey1;
  INEXACT REAL cexdey1, dexcey1, dexaey1, aexdey1;
  INEXACT REAL aexcey1, cexaey1, bexdey1, dexbey1;
  REAL aexbey0, bexaey0, bexcey0, cexbey0;
  REAL cexdey0, dexcey0, dexaey0, aexdey0;
  REAL aexcey0, cexaey0, bexdey0, dexbey0;
  REAL ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
  INEXACT REAL ab3, bc3, cd3, da3, ac3, bd3;
  REAL abeps, bceps, cdeps, daeps, aceps, bdeps;
  REAL temp8a[8], temp8b[8], temp8c[8], temp16[16], temp24[24], temp48[48];
  int temp8alen, temp8blen, temp8clen, temp16len, temp24len, temp48len;
  REAL xdet[96], ydet[96], zdet[96], xydet[192];
  int xlen, ylen, zlen, xylen;
  REAL adet[288], bdet[288], cdet[288], ddet[288];
  int alen, blen, clen, dlen;
  REAL abdet[576], cddet[576];
  int ablen, cdlen;
  REAL fin1[1152];
  int finlength;

  REAL aextail, bextail, cextail, dextail;
  REAL aeytail, beytail, ceytail, deytail;
  REAL aeztail, beztail, ceztail, deztail;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  aex = (REAL) (pa[0] - pe[0]);
  bex = (REAL) (pb[0] - pe[0]);
  cex = (REAL) (pc[0] - pe[0]);
  dex = (REAL) (pd[0] - pe[0]);
  aey = (REAL) (pa[1] - pe[1]);
  bey = (REAL) (pb[1] - pe[1]);
  cey = (REAL) (pc[1] - pe[1]);
  dey = (REAL) (pd[1] - pe[1]);
  aez = (REAL) (pa[2] - pe[2]);
  bez = (REAL) (pb[2] - pe[2]);
  cez = (REAL) (pc[2] - pe[2]);
  dez = (REAL) (pd[2] - pe[2]);

  Two_Product(aex, bey, aexbey1, aexbey0);
  Two_Product(bex, aey, bexaey1, bexaey0);
  Two_Two_Diff(aexbey1, aexbey0, bexaey1, bexaey0, ab3, ab[2], ab[1], ab[0]);
  ab[3] = ab3;

  Two_Product(bex, cey, bexcey1, bexcey0);
  Two_Product(cex, bey, cexbey1, cexbey0);
  Two_Two_Diff(bexcey1, bexcey0, cexbey1, cexbey0, bc3, bc[2], bc[1], bc[0]);
  bc[3] = bc3;

  Two_Product(cex, dey, cexdey1, cexdey0);
  Two_Product(dex, cey, dexcey1, dexcey0);
  Two_Two_Diff(cexdey1, cexdey0, dexcey1, dexcey0, cd3, cd[2], cd[1], cd[0]);
  cd[3] = cd3;

  Two_Product(dex, aey, dexaey1, dexaey0);
  Two_Product(aex, dey, aexdey1, aexdey0);
  Two_Two_Diff(dexaey1, dexaey0, aexdey1, aexdey0, da3, da[2], da[1], da[0]);
  da[3] = da3;

  Two_Product(aex, cey, aexcey1, aexcey0);
  Two_Product(cex, aey, cexaey1, cexaey0);
  Two_Two_Diff(aexcey1, aexcey0, cexaey1, cexaey0, ac3, ac[2], ac[1], ac[0]);
  ac[3] = ac3;

  Two_Product(bex, dey, bexdey1, bexdey0);
  Two_Product(dex, bey, dexbey1, dexbey0);
  Two_Two_Diff(bexdey1, bexdey0, dexbey1, dexbey0, bd3, bd[2], bd[1], bd[0]);
  bd[3] = bd3;

  temp8alen = scale_expansion_zeroelim(4, cd, bez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, -cez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, bc, dez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, aex, temp48);
  xlen = scale_expansion_zeroelim(temp48len, temp48, -aex, xdet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, aey, temp48);
  ylen = scale_expansion_zeroelim(temp48len, temp48, -aey, ydet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, aez, temp48);
  zlen = scale_expansion_zeroelim(temp48len, temp48, -aez, zdet);
  xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet);
  alen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, adet);

  temp8alen = scale_expansion_zeroelim(4, da, cez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, dez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, cd, aez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, bex, temp48);
  xlen = scale_expansion_zeroelim(temp48len, temp48, bex, xdet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, bey, temp48);
  ylen = scale_expansion_zeroelim(temp48len, temp48, bey, ydet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, bez, temp48);
  zlen = scale_expansion_zeroelim(temp48len, temp48, bez, zdet);
  xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet);
  blen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, bdet);

  temp8alen = scale_expansion_zeroelim(4, ab, dez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, bd, aez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, da, bez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, cex, temp48);
  xlen = scale_expansion_zeroelim(temp48len, temp48, -cex, xdet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, cey, temp48);
  ylen = scale_expansion_zeroelim(temp48len, temp48, -cey, ydet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, cez, temp48);
  zlen = scale_expansion_zeroelim(temp48len, temp48, -cez, zdet);
  xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet);
  clen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, cdet);

  temp8alen = scale_expansion_zeroelim(4, bc, aez, temp8a);
  temp8blen = scale_expansion_zeroelim(4, ac, -bez, temp8b);
  temp8clen = scale_expansion_zeroelim(4, ab, cez, temp8c);
  temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                          temp8blen, temp8b, temp16);
  temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                          temp16len, temp16, temp24);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, dex, temp48);
  xlen = scale_expansion_zeroelim(temp48len, temp48, dex, xdet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, dey, temp48);
  ylen = scale_expansion_zeroelim(temp48len, temp48, dey, ydet);
  temp48len = scale_expansion_zeroelim(temp24len, temp24, dez, temp48);
  zlen = scale_expansion_zeroelim(temp48len, temp48, dez, zdet);
  xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet);
  dlen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, ddet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  finlength = fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, fin1);

  det = estimate(finlength, fin1);
  errbound = isperrboundB * permanent;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(pa[0], pe[0], aex, aextail);
  Two_Diff_Tail(pa[1], pe[1], aey, aeytail);
  Two_Diff_Tail(pa[2], pe[2], aez, aeztail);
  Two_Diff_Tail(pb[0], pe[0], bex, bextail);
  Two_Diff_Tail(pb[1], pe[1], bey, beytail);
  Two_Diff_Tail(pb[2], pe[2], bez, beztail);
  Two_Diff_Tail(pc[0], pe[0], cex, cextail);
  Two_Diff_Tail(pc[1], pe[1], cey, ceytail);
  Two_Diff_Tail(pc[2], pe[2], cez, ceztail);
  Two_Diff_Tail(pd[0], pe[0], dex, dextail);
  Two_Diff_Tail(pd[1], pe[1], dey, deytail);
  Two_Diff_Tail(pd[2], pe[2], dez, deztail);
  if ((aextail == 0.0) && (aeytail == 0.0) && (aeztail == 0.0)
      && (bextail == 0.0) && (beytail == 0.0) && (beztail == 0.0)
      && (cextail == 0.0) && (ceytail == 0.0) && (ceztail == 0.0)
      && (dextail == 0.0) && (deytail == 0.0) && (deztail == 0.0)) {
    return det;
  }

  errbound = isperrboundC * permanent + resulterrbound * Absolute(det);
  abeps = (aex * beytail + bey * aextail)
        - (aey * bextail + bex * aeytail);
  bceps = (bex * ceytail + cey * bextail)
        - (bey * cextail + cex * beytail);
  cdeps = (cex * deytail + dey * cextail)
        - (cey * dextail + dex * ceytail);
  daeps = (dex * aeytail + aey * dextail)
        - (dey * aextail + aex * deytail);
  aceps = (aex * ceytail + cey * aextail)
        - (aey * cextail + cex * aeytail);
  bdeps = (bex * deytail + dey * bextail)
        - (bey * dextail + dex * beytail);
  det += (((bex * bex + bey * bey + bez * bez)
           * ((cez * daeps + dez * aceps + aez * cdeps)
              + (ceztail * da3 + deztail * ac3 + aeztail * cd3))
           + (dex * dex + dey * dey + dez * dez)
           * ((aez * bceps - bez * aceps + cez * abeps)
              + (aeztail * bc3 - beztail * ac3 + ceztail * ab3)))
          - ((aex * aex + aey * aey + aez * aez)
           * ((bez * cdeps - cez * bdeps + dez * bceps)
              + (beztail * cd3 - ceztail * bd3 + deztail * bc3))
           + (cex * cex + cey * cey + cez * cez)
           * ((dez * abeps + aez * bdeps + bez * daeps)
              + (deztail * ab3 + aeztail * bd3 + beztail * da3))))
       + 2.0 * (((bex * bextail + bey * beytail + bez * beztail)
                 * (cez * da3 + dez * ac3 + aez * cd3)
                 + (dex * dextail + dey * deytail + dez * deztail)
                 * (aez * bc3 - bez * ac3 + cez * ab3))
                - ((aex * aextail + aey * aeytail + aez * aeztail)
                 * (bez * cd3 - cez * bd3 + dez * bc3)
                 + (cex * cextail + cey * ceytail + cez * ceztail)
                 * (dez * ab3 + aez * bd3 + bez * da3)));
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  return insphereexact(pa, pb, pc, pd, pe);
}

REAL insphere(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL *pe)
{
  REAL aex, bex, cex, dex;
  REAL aey, bey, cey, dey;
  REAL aez, bez, cez, dez;
  REAL aexbey, bexaey, bexcey, cexbey, cexdey, dexcey, dexaey, aexdey;
  REAL aexcey, cexaey, bexdey, dexbey;
  REAL alift, blift, clift, dlift;
  REAL ab, bc, cd, da, ac, bd;
  REAL abc, bcd, cda, dab;
  REAL aezplus, bezplus, cezplus, dezplus;
  REAL aexbeyplus, bexaeyplus, bexceyplus, cexbeyplus;
  REAL cexdeyplus, dexceyplus, dexaeyplus, aexdeyplus;
  REAL aexceyplus, cexaeyplus, bexdeyplus, dexbeyplus;
  REAL det;
  REAL permanent, errbound;

  aex = pa[0] - pe[0];
  bex = pb[0] - pe[0];
  cex = pc[0] - pe[0];
  dex = pd[0] - pe[0];
  aey = pa[1] - pe[1];
  bey = pb[1] - pe[1];
  cey = pc[1] - pe[1];
  dey = pd[1] - pe[1];
  aez = pa[2] - pe[2];
  bez = pb[2] - pe[2];
  cez = pc[2] - pe[2];
  dez = pd[2] - pe[2];

  aexbey = aex * bey;
  bexaey = bex * aey;
  ab = aexbey - bexaey;
  bexcey = bex * cey;
  cexbey = cex * bey;
  bc = bexcey - cexbey;
  cexdey = cex * dey;
  dexcey = dex * cey;
  cd = cexdey - dexcey;
  dexaey = dex * aey;
  aexdey = aex * dey;
  da = dexaey - aexdey;

  aexcey = aex * cey;
  cexaey = cex * aey;
  ac = aexcey - cexaey;
  bexdey = bex * dey;
  dexbey = dex * bey;
  bd = bexdey - dexbey;

  abc = aez * bc - bez * ac + cez * ab;
  bcd = bez * cd - cez * bd + dez * bc;
  cda = cez * da + dez * ac + aez * cd;
  dab = dez * ab + aez * bd + bez * da;

  alift = aex * aex + aey * aey + aez * aez;
  blift = bex * bex + bey * bey + bez * bez;
  clift = cex * cex + cey * cey + cez * cez;
  dlift = dex * dex + dey * dey + dez * dez;

  det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

  aezplus = Absolute(aez);
  bezplus = Absolute(bez);
  cezplus = Absolute(cez);
  dezplus = Absolute(dez);
  aexbeyplus = Absolute(aexbey);
  bexaeyplus = Absolute(bexaey);
  bexceyplus = Absolute(bexcey);
  cexbeyplus = Absolute(cexbey);
  cexdeyplus = Absolute(cexdey);
  dexceyplus = Absolute(dexcey);
  dexaeyplus = Absolute(dexaey);
  aexdeyplus = Absolute(aexdey);
  aexceyplus = Absolute(aexcey);
  cexaeyplus = Absolute(cexaey);
  bexdeyplus = Absolute(bexdey);
  dexbeyplus = Absolute(dexbey);
  permanent = ((cexdeyplus + dexceyplus) * bezplus
               + (dexbeyplus + bexdeyplus) * cezplus
               + (bexceyplus + cexbeyplus) * dezplus)
            * alift
            + ((dexaeyplus + aexdeyplus) * cezplus
               + (aexceyplus + cexaeyplus) * dezplus
               + (cexdeyplus + dexceyplus) * aezplus)
            * blift
            + ((aexbeyplus + bexaeyplus) * dezplus
               + (bexdeyplus + dexbeyplus) * aezplus
               + (dexaeyplus + aexdeyplus) * bezplus)
            * clift
            + ((bexceyplus + cexbeyplus) * aezplus
               + (cexaeyplus + aexceyplus) * bezplus
               + (aexbeyplus + bexaeyplus) * cezplus)
            * dlift;
  errbound = isperrboundA * permanent;
  if ((det > errbound) || (-det > errbound)) {
    return det;
  }

  return insphereadapt(pa, pb, pc, pd, pe, permanent);
}
