#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

//==============================================================================

// On return, "E" is the TriEdge containing the point "pt".
// The returned value is one of the following:
//  - in triangle (may be outside, if this is a hull triangle).
//  - on edge, return by (E.org(), E.dest()).
//  - on vertex, returned by E.org().
//  - try to cross a segment (maybe interior) (used by delaunay refinement).
//
int Triangulation::locate_point(Vertex *pt, TriEdge &E, int encflag)
{
  if (op_db_verbose > 2) {
    printf("  Locating point [%d]\n", pt->idx);
  }

  if ((E.tri == NULL) || E.tri->is_deleted()) {
    // Try to search from the recent visited triangle.
    if ((tr_recnttri != NULL) && !tr_recnttri->is_deleted()) {
      E.tri = tr_recnttri; E.ver = 0;
    } else {
      E = tr_infvrt->adj;
    }
  }

  if (E.tri->is_hulltri()) {
    // Get a non-hull triangle.
    for (E.ver = 0; E.ver < 3; E.ver++) {
      if (E.apex() == tr_infvrt) break; 
    }
    assert(E.ver < 3);
    E = E.esym();
  }

  // Select an edge such that pt lies to CCW of it.
  for (E.ver = 0; E.ver < 3; E.ver++) {
    if (Orient2d(E.org(), E.dest(), pt) > 0) break;
  }

  // Let E = [a,b,c] and p lies to the CCW of [a->b].
  do {

    REAL ori1 = Orient2d(E.dest(), E.apex(), pt);
    REAL ori2 = Orient2d(E.apex(),  E.org(), pt);

    if (ori1 > 0) {
      if (ori2 > 0) {
        break; // Found.
      } else if (ori2 < 0) {
        E.ver = _eprev_tbl[E.ver]; 
      } else { // ori2 == 0
        E.ver = _eprev_tbl[E.ver]; 
        return LOC_ON_EDGE; // ONEDGE  p lies on edge [c,a]
      }
    } else if (ori1 < 0) {
      if (ori2 > 0) {
        E.ver = _enext_tbl[E.ver]; 
      } else if (ori2 < 0) {
        // Randomly choose one.
        if (rand() % 2) { // flipping a coin.
          E.ver = _enext_tbl[E.ver];
        } else {
          E.ver = _eprev_tbl[E.ver];
        }
      } else { // ori2 == 0
        E.ver = _enext_tbl[E.ver]; 
      }
    } else { // ori1 == 0
      if (ori2 > 0) {
        E.ver = _enext_tbl[E.ver]; // p lies on edge [b,c].
        return LOC_ON_EDGE; // ONEDGE
      } else if (ori2 < 0) {
        E.ver = _eprev_tbl[E.ver]; 
      } else { // ori2 == 0
        E.ver = _eprev_tbl[E.ver]; // p is coincident with apex.
        return LOC_ON_VERT; // ONVERTEX Org(E)
      }
    }

    E = E.esym(); // Go to the adjacent triangle.

    if (encflag) {
      if (E.is_segment()) {
        return LOC_ENC_SEG; // Do not cross a boundary edge.
      }
    }

    /*
    if (liftflag) {
      // Locating a point in a lifted triangulation in 3d.
      if (E.is_segment()) {
        return LOC_ENC_SEG; // Do not cross a boundary edge.
      }
      // Check the dihedral angle of this edge.
      TriEdge N = E.esym();
      REAL ang = get_dihedral(E.org(), E.dest(), E.apex(), N.apex());
      if (((ang > 5.) && (ang < 175.)) ||
          ((ang > 185.) && (ang < 355.))) {
        return LOC_ENC_SEG; // It might be not a segment.
      }
    }
    */
  } while (!E.tri->is_hulltri());

  tr_recnttri = E.tri; // Remember this triangle for next point location.

  if (E.tri->is_hulltri()) {
    return LOC_IN_OUTSIDE;
  }

  return LOC_IN_TRI; // P lies inside E = [a,b,c]

  /*
  if (E.tri->is_hulltri()) {
    //assert(encflag == 0);
    // pt lies at the exiterior of this hull edge [v1, v2].
    if (tr_nonconvex) {
      // This triangulation is non-convex, the search might get stuck
      //   at an interior hole or a non-convex segment.
      if (op_db_verbose) {
        printf("!! search non-convex triangulation (globally)\n");
      }
      int j;
      for (j = 0; j < tr_tris->used_items; j++) {
        Triang *t = (Triang *) tr_tris->get(j);
        if (t->is_deleted()) continue;
        if (t->is_hulltri()) continue;
        Vertex *pa = t->vrt[0];
        Vertex *pb = t->vrt[1];
        Vertex *pc = t->vrt[2];
        double o1 = Orient2d(pa, pb, pt);
        if (o1 >= 0) {
          double o2 = Orient2d(pb, pc, pt);
          if (o2 >= 0) {
            double o3 = Orient2d(pc, pa, pt);
            if (o3 >= 0) {
              E.tri = t; E.ver = 0;
              return locate_point(pt, E, encflag);
            }
          }
        }
      }
    }
    return LOC_IN_OUTSIDE;
  } // if (E.tri->is_hulltri())
  */

  // The point lies exactly inside the triangle (NOT in outside).
  //assert(!E.tri->is_hulltri());

  /*
  if (rndflag > 0) {
    // Approximating the location of the point w.r.t. this triangle.
    Vertex *v1 = E.org();
    Vertex *v2 = E.dest();
    Vertex *v3 = E.apex();
    double a1 = get_tri_area(pt, v1, v2);
    double a2 = get_tri_area(pt, v2, v3);
    double a3 = get_tri_area(pt, v3, v1);
    double triarea = a1 + a2 + a3;
    // Round the areas
    if (fabs(a1 / triarea - 1.0) < 1e-5) a1 = 0.0;
    if (fabs(a2 / triarea - 1.0) < 1e-5) a2 = 0.0;
    if (fabs(a3 / triarea - 1.0) < 1e-5) a3 = 0.0;
    if (a1 == 0) {
      if (a2 == 0) {
        //if (a3 > 0) {
          assert(a3 > 0);
          E = E.enext(); // E.enextself(); // On vertex E.dest().
          return LOC_ON_VERT; // return 3;
        //}
      } else if (a3 == 0) {
        return LOC_ON_VERT; // return 3; // On vertex E.org().
      } else {
        return LOC_ON_EDGE; // return 2; // On edge.
      }
    } else if (a2 == 0) {
      if (a3 == 0) {
        E = E.eprev(); // E.enext2self(); // On vertex E.apex().
        return LOC_ON_VERT; // return 3;
      } else {
        E = E.enext(); //E.enextself(); // On edge.
        return LOC_ON_EDGE; // return 2;
      }
    } else if (a3 == 0) {
      E = E.eprev(); // E.enext2self(); // On edge.
      return LOC_ON_EDGE; // return 2;
    }
  } // if (rndflag > 0)
  */

  //return LOC_IN_TRI; // P lies inside E = [a,b,c]
}

//==============================================================================
// Insert a vertex using Bowyer-Watson algorithm
// if `bwflag' is true, use Bowyer-Watson algorithm to enlarge cavity.

bool Triangulation::insert_point(Vertex *pt, TriEdge &E, int& loc, bool bwflag)
{
  if (loc == LOC_UNKNOWN) {
    loc = locate_point(pt, E, 0);
  }

  return true;
}

//==============================================================================
// Given two triangles [a,b,c] and [b,a,d], check if the edge [a,b]
//   is locally regular, i.e., locally Delaunay in Euclidean metric.
// Return true if it is non-regular, otherwise, return false which means that
//   it is regular or degenerate (co-hyperplane).

#define _set_height(pa) \
    x = (pa)->crd[0]; \
    y = (pa)->crd[1]; \
    (pa)->crd[2] = _a11*x*x + 2.*_a21*x*y + _a22*y*y - (pa)->wei\

bool Triangulation::regular_test(Vertex* pa, Vertex* pb, Vertex* pc, Vertex* pd)
{
  if (op_metric <= METRIC_Euclidean) {
    //printf("  O3d: (%d, %d, %d, %d)\n", tt[0].org()->idx, tt[0].dest()->idx, tt[0].apex()->idx, tt[1].apex()->idx);
    //return (Orient3d(pa, pb, pc, pd) * op_dt_nearest) > 0;
    // bakup heights
    double ha = pa->crd[2];
    double hb = pb->crd[2];
    double hc = pc->crd[2];
    double hd = pd->crd[2];
    // calculate heights
    double x, y;
    _set_height(pa);
    _set_height(pb);
    _set_height(pc);
    _set_height(pd);
    double s = (Orient3d(pa, pb, pc, pd) * op_dt_nearest);
    // Retstor heights
    pa->crd[2] = ha;
    pb->crd[2] = hb;
    pc->crd[2] = hc;
    pd->crd[2] = hd;
    return s > 0;
  } else if (op_metric == METRIC_Riemannian) {
    // bakup heights
    double ha = pa->crd[2];
    double hb = pb->crd[2];
    double hc = pc->crd[2];
    double hd = pd->crd[2];
    // calculate heights
    double x, y;
    _set_height(pa);
    _set_height(pb);
    _set_height(pc);
    _set_height(pd);
    double s = (Orient3d(pa, pb, pc, pd) * op_dt_nearest);
    // Retstor heights
    pa->crd[2] = ha;
    pb->crd[2] = hb;
    pc->crd[2] = hc;
    pd->crd[2] = hd;
    return s > 0;
  } else {
    assert(0); // not supported yet.
  }

  REAL l_ab = get_distance(pa, pb);
  REAL l_cd = get_distance(pc, pd);
  REAL l_bc = get_distance(pb, pc);
  REAL l_ca = get_distance(pc, pa);
  REAL l_ad = get_distance(pa, pd);
  REAL l_db = get_distance(pd, pb);

  REAL cosa = (l_ca*l_ca + l_ad*l_ad - l_cd*l_cd) / (2.0 * l_ca*l_ad);
  REAL cosb = (l_bc*l_bc + l_db*l_db - l_cd*l_cd) / (2.0 * l_bc*l_db);
  REAL cosc = (l_bc*l_bc + l_ca*l_ca - l_ab*l_ab) / (2.0 * l_bc*l_ca);
  REAL cosd = (l_ad*l_ad + l_db*l_db - l_ab*l_ab) / (2.0 * l_ad*l_db);

  // Rounding
  if (cosa >  1.0) cosa =  1.0;
  if (cosa < -1.0) cosa = -1.0;
  if (cosb >  1.0) cosb =  1.0;
  if (cosb < -1.0) cosb = -1.0;
  if (cosc >  1.0) cosc =  1.0;
  if (cosc < -1.0) cosc = -1.0;
  if (cosd >  1.0) cosd =  1.0;
  if (cosd < -1.0) cosd = -1.0;

  REAL angA = acos(cosa);
  REAL angB = acos(cosb);
  REAL angC = acos(cosc);
  REAL angD = acos(cosd);

  REAL sum_ab = angA + angB;
  REAL sum_cd = angC + angD;
  if ((fabs(sum_ab - PI) / PI) < 1.e-4) sum_ab = PI; // Roundoff
  if ((fabs(sum_cd - PI) / PI) < 1.e-4) sum_cd = PI;

  // Is regular (need flip)?
  if ((sum_cd > PI) && (sum_ab < PI)) {
    return true; // 1.0; //     ori = 1.0; // need to flip.
  } else {
    return false; // 0.0; // is regular
  }
}

//==============================================================================

int Triangulation::lawson_flip(Vertex *pt, int hullflag, arraypool *fqueue)
{
  arraypool *tmpfqueue = NULL;
  TriEdge E, N, tt[4];
  Vertex *delpt = NULL;
  bool ori;
  int ishullflip;
  int fcount = 0;

  if (fqueue == NULL) {
    tmpfqueue = new arraypool(sizeof(TriEdge), 10);
    fqueue = tmpfqueue;
  }

  if (fqueue->objects == 0) {
    if (pt != NULL) {
      // Collect the link edges of pt.
      E = pt->adj;
      do {
        N = E.enext();
        N.set_edge_infect();
        * (TriEdge *) fqueue->alloc() = N;
        E = E.eprev_esym(); // ccw rotate
      } while (E.tri != pt->adj.tri);
    } else {
      // Collect all edges of the triangulation.
      for (int i = 0; i < tr_tris->used_items; i++) {
        E.tri = (Triang *) tr_tris->get(i);
        if (E.tri->is_deleted()) continue;
        if (!E.tri->is_hulltri()) {
          for (E.ver = 0; E.ver < 3; E.ver++) {
            if (!E.esym().tri->is_infected()) {
              E.set_edge_infect();
              * (TriEdge *) fqueue->alloc() = E;
            }
          }
          E.tri->set_infect();
        }
      }
      // Uninfect all triangles.
      for (int i = 0; i < tr_tris->used_items; i++) {
        E.tri = (Triang *) tr_tris->get(i);
        if (E.tri->is_deleted()) continue;
        if (!E.tri->is_hulltri()) {
          E.tri->clear_infect();
        }
      }
    }
  }

  if (op_db_verbose > 2) {
    printf("    Lawson flipping %d edges.\n", fqueue->objects);
  }

  for (int i = 0; i < fqueue->used_items; i++) {
    TriEdge *pte = (TriEdge *) fqueue->get(i);
    if (pte->tri->is_deleted()) continue;
    if (!pte->is_edge_infected()) continue; // is it still in queue?
    pte->clear_edge_infect();
    if (pt != NULL) {
      if (pte->apex() != pt) continue;
    }
    // Check if this edge is locally regular.
    ori = false; ishullflip = 0;
    tt[0] = *pte;
    tt[1] = tt[0].esym();
    if (!tt[0].tri->is_hulltri()) {
      if (!tt[1].tri->is_hulltri()) {
        // An interior edge.
        if (op_db_verbose > 3) {
          printf("      O3d: (%d, %d, %d, %d)\n", tt[0].org()->idx, tt[0].dest()->idx, tt[0].apex()->idx, tt[1].apex()->idx);
        }
        /*
        if ((op_metric == METRIC_EUCLIDEAN) || (op_metric == METRIC_ISO)) {
          //printf("  O3d: (%d, %d, %d, %d)\n", tt[0].org()->idx, tt[0].dest()->idx, tt[0].apex()->idx, tt[1].apex()->idx);
          ori = Orient3d(tt[0].org(), tt[0].dest(), tt[0].apex(), tt[1].apex())
              * op_dt_nearest;
          //printf("  O3d = %g, op_dt_nearest=%d\n", ori, op_dt_nearest);
        } else {
          ori = regular_test(tt[0].org(), tt[0].dest(), tt[0].apex(), tt[1].apex());
        }
        */
        ori = regular_test(tt[0].org(), tt[0].dest(), tt[0].apex(), tt[1].apex());
      }
    } else {
      if (tt[1].tri->is_hulltri()) {
        // Only do flip if it is the Delaunay case.
        if (hullflag) {
          // Check if an exterior flip (for convexity) is needed.
          if (tt[0].org() != tr_infvrt) tt[0] = tt[0].esym();
          assert(tt[0].org() == tr_infvrt);
          tt[1] = tt[0].esym();
          if (op_db_verbose > 3) {
            printf("      O2d: (%d, %d, %d)\n", tt[1].apex()->idx, tt[1].org()->idx, tt[0].apex()->idx);
          }
          ori = (Orient2d(tt[1].apex(), tt[1].org(), tt[0].apex()) > 0);
          ishullflip = 1;
        }
      }
    }
    if (ori) { // if (ori > 0)
      // This edge is either locally non-regular or non-convex.
      if (op_no_incremental_flip && !ishullflip) {
        continue; // Ignore this flip.
      }
      int fflag = FLIP_UNKNOWN;
      if (flip(tt, &delpt, fflag, fqueue)) {
        fcount++;
      }
    }
  } // i

  if (op_db_verbose > 2) {
    printf("    Flipped %d edges.\n", fcount);
  }
  
  if (tmpfqueue != NULL) {
    delete tmpfqueue;
  } else {
    fqueue->clean(); // clean it.
  }
  return fcount;
}

//==============================================================================

int Triangulation::sort_vertices(Vertex* vrtarray, int arysize, Vertex**& permutarray)
{
  permutarray = new Vertex*[arysize];
  int randindex, i;

  if (op_db_verbose) {
    printf("Sorting vertices...\n");
  }

  if (so_nosort | so_norandom) { // -SN or -SR
    for (i = 0; i < arysize; i++) {
      permutarray[i] = &vrtarray[i];
    }
  } else {
    // Randomly permute the vertices.
    srand(arysize);
    for (i = 0; i < arysize; i++) {
      randindex = rand() % (i + 1); 
      permutarray[i] = permutarray[randindex];
      permutarray[randindex] = &vrtarray[i];
    }
  }

  if (!so_nosort && !so_nobrio) { // no -SN or -SB
    hilbert_init(2);
    brio_multiscale_sort2(permutarray, arysize, so_brio_threshold,
                          so_brio_ratio, so_hilbert_order, so_hilbert_limit,
                          io_xmin, io_xmax, io_ymin, io_ymax);
  }

  return 1;
}

//==============================================================================

int Triangulation::first_tri(Vertex **ptlist, int ptnum)
{
  int i, j, it = 0;
  REAL ori = 0.0;
  Vertex *swappt;

  if (so_nosort) {
    // Assume pt[0] and pt[1] are distinct.
    // Search the third non-collinear vertex.
    for (i = 2; i < ptnum; i++) {
      ori = Orient2d(ptlist[0], ptlist[1], ptlist[i]);
      if (ori != 0) break;
    }
    if (i == ptnum) return 0; // Failed
    // If i > 2, swap it to 2.
    if (i > 2) {
      swappt = ptlist[2];
      ptlist[2] = ptlist[i];
      ptlist[i] = swappt;
    }
  } else {
  // Randomly select 3 points.
    do {
      // Randomly select three vertices from the iuput list.
      for (i = 0; i < 3; i++) {
        // Swap ith and jth element.
        j = rand() % (ptnum - i);
        swappt = ptlist[i];
        ptlist[i] = ptlist[j];
        ptlist[j] = swappt;
      }
      ori = Orient2d(ptlist[0], ptlist[1], ptlist[2]);
      if (ori != 0) break;
      it++;
    } while (it < 10);

    if (it >= 10) return 0; // Failed.
  }

  if (ori < 0) {
    // Swap the first two vertices.
    swappt = ptlist[0];
    ptlist[0] = ptlist[1];
    ptlist[1] = swappt;
  }

  if (op_db_verbose > 1) {
    printf("  First triangle [%d,%d,%d]\n", ptlist[0]->idx, ptlist[1]->idx,
           ptlist[2]->idx);
  }

  if (tr_tris == NULL) {
    // Estimate the final mesh size.
    int est_size = 0;
    if (tr_segs == NULL) {
      est_size = ct_in_vrts * 4; // Delaunay triangulation case. 
    } else {
      est_size = 8192; // temp
    }
    int log2objperblk = 0;
    while (est_size >>= 1) log2objperblk++;
    tr_tris = new arraypool(sizeof(Triang), log2objperblk);
  }

  first_triangle(ptlist[0], ptlist[1], ptlist[2]);

  return 1;
}

//==============================================================================

int Triangulation::incremental_delaunay()
{
  if (op_db_verbose) {
    printf("Incremental Delaunay construction...\n");
  }

  Vertex** vrtarray = NULL;
  int arysize = ct_in_vrts;

  sort_vertices(in_vrts, ct_in_vrts, vrtarray);

  if (!first_tri(vrtarray, ct_in_vrts)) {
    return 0;
  }

  arraypool *fqueue = new arraypool(sizeof(TriEdge), 8);
  TriEdge E = tr_infvrt->adj, tt[4];
  int loc;

  for (int i = 3; i < arysize; i++) {
    if (op_db_verbose > 1) {
      printf("  Inserting vertex %d: %d\n", i+1, vrtarray[i]->idx);
    }
    //loc = locate_point(vrtarray[i], E, 0, 0); // encflag = 0
    loc = locate_point(vrtarray[i], E, 0); // encflag = 0
    if (loc != LOC_ON_VERT) { // ON_VERTEX
      REAL ori = 1.0;
      if (loc == LOC_IN_TRI) { //if ((loc == 1) && !E.tri->is_hulltri()) {
        // Make sure this vertex is regular w.r.t. E.
        // An interior edge.
        ori = Orient3d(E.org(), E.dest(), E.apex(), vrtarray[i])
            * op_dt_nearest;
      }
      if (ori > 0) {
        // Insert vertex
        tt[0] = E;
        if (loc == LOC_IN_OUTSIDE) { // IN outise
          int fflag = FLIP_13;
          flip(tt, &(vrtarray[i]), fflag, fqueue);
        } else if (loc == LOC_IN_TRI) { // IN_TRAING
          int fflag = FLIP_13;
          flip(tt, &(vrtarray[i]), fflag, fqueue);
        } else if (loc == LOC_ON_EDGE) { // ON_EDGE
          int fflag = FLIP_24;
          //tt[1] = tt[0].esym();
          flip(tt, &(vrtarray[i]), fflag, fqueue);
        } else {
          printf(" loc = %d\n", loc);
          assert(0); // not possible for encflag = 0
        }
        lawson_flip(vrtarray[i], 1, fqueue); // hullflag = 1
        E = vrtarray[i]->adj; // For next point location.
      } else {
        if (op_db_verbose > 1) {
          printf("  Skip a redundant vertex %d\n", vrtarray[i]->idx);
        }
      }
    } else {
      if (op_db_verbose) {
        printf("Warning:  Vertex %d is coincident with %d\n",
               vrtarray[i]->idx, E.org()->idx);
      }
      vrtarray[i]->Pair = E.org(); // Remember this vertex.
      assert(vrtarray[i]->typ == UNUSEDVERTEX);
      // the ct_unused_vrts is updated by flip13();
    }
  }

  delete fqueue;
  delete [] vrtarray;
  return 1;
}
