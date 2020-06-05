#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

//==============================================================================

REAL Triangulation::get_element_gradient(Triang* tri)
{// get fx, fy, saved in cct[0], [1]
  // The formulas are in Lu-femag2d-2008-09-05.pdf (page 6)
  // The piecewise function on this triangle is equation (8).
  REAL xi = tri->vrt[0]->crd[0];
  REAL yi = tri->vrt[0]->crd[1];
  REAL ui = tri->vrt[0]->fval;

  REAL xj = tri->vrt[1]->crd[0];
  REAL yj = tri->vrt[1]->crd[1];
  REAL uj = tri->vrt[1]->fval;

  REAL xm = tri->vrt[2]->crd[0];
  REAL ym = tri->vrt[2]->crd[1];
  REAL um = tri->vrt[2]->fval;

  // Validate only
  REAL ai = xj*ym - xm*yj;
  REAL aj = xm*yi - xi*ym;
  REAL am = xi*yj - xj*yi;
  REAL area1 = 0.5 * (ai + aj + am);
  if (area1 < 0.) area1 = -area1;

  REAL bi = yi - ym;
  REAL bj = ym - yi;
  REAL bm = yi - yj;

  REAL ci = xm - xj;
  REAL cj = xi - xm;
  REAL cm = xj - xi;

  REAL area = 0.5 * (bi*cj - bj*ci);
  if (area < 0.) area = -area;
  if ((fabs(area - area) / area) > 1e-6) {
    assert(0); // A bug.
  }

  tri->cct[0] = 0.5 * (bi*ui + bj*uj + bm*um) / area; // grad_x
  tri->cct[1] = 0.5 * (ci*ui + cj*uj + cm*um) / area; // grad_y

  // The Dirichlet energy (the dot product) (= square of H^1 norm) of this element.
  // From Marc Alexa's notation equ (13)
  //   1/2 f^T L_T f, where L_T is the discrete Laplace-Beltrami operator.
  //   It is a 3x3 matrix with components given in Lu's equ (30). (page 10).
  tri->cct[2] = (tri->cct[0] * tri->cct[0] + tri->cct[1] * tri->cct[1]) * area;
  if (fabs(tri->cct[2]) < 1.e-16) tri->cct[2] = 0.0;

  return tri->cct[2];
}

//==============================================================================

REAL Triangulation::get_Dirichlet_energy() // of the whole triangulation
{
  REAL sum = 0.;

  for (int i = 0; i < tr_tris->used_items; i++) {
    Triang *tri = (Triang *) tr_tris->get(i);
    if (!tri->is_deleted()) {
      if (!tri->is_hulltri() && !tri->is_exterior()) {
        sum += get_element_gradient(tri);
      }
    }
  }

  return sum;
}

REAL Triangulation::dump_Dirichlet_energy() // for visualisation
{
    REAL sum_energy = get_Dirichlet_energy();
    printf(" Energy = %g\n", sum_energy);

    // see file format: detri2/doc/UCD_Format.pdf
    char filename[256];
    sprintf(filename, "%s_energy.inp", io_outfilename);
    FILE *outfile = fopen(filename, "w");

    int ntri = (int) tr_tris->objects - ct_hullsize - ct_exteriors;
    printf("Writing %d triangles to file %s.\n", ntri, filename);
    int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
    //nv -= ct_unused_vrts;

    //fprintf(outfile, "%d %d %d 0 0\n", nv, ntri, save_val); // nodal data
    fprintf(outfile, "%d %d 1 1 0\n", nv, ntri); // with nodal and cell data

    int i, idx=1; // UCD index starts from 1.
    for (i = 0; i < ct_in_vrts; i++) {
      //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      fprintf(outfile, "%d %g %g 0\n", idx, in_vrts[i].crd[0], in_vrts[i].crd[1]);
      in_vrts[i].idx = idx;
      idx++;
    }
    if (tr_steiners != NULL) {
      for (i = 0; i < tr_steiners->used_items; i++) {
        Vertex *vrt = (Vertex *) tr_steiners->get(i);
        if (vrt->is_deleted()) continue;
        fprintf(outfile, "%d %g %g 0\n", idx, vrt->crd[0], vrt->crd[1]);
        vrt->idx = idx;
        idx++;
      }
    }

    // UCD assumes vertex index starts from 1.
    //int shift = (io_firstindex == 1 ? 0 : 1);
    idx = 1;
    for (i = 0; i < tr_tris->used_items; i++) {
      Triang* tri = (Triang *) tr_tris->get(i);
      if (!tri->is_deleted()) {
        // ignore a hull triangle.
        if (!tri->is_hulltri() && !tri->is_exterior()) {
          fprintf(outfile, "%d %d tri %d %d %d\n", idx, tri->tag,
                  tri->vrt[0]->idx, // + shift,
                  tri->vrt[1]->idx, // + shift,
                  tri->vrt[2]->idx); // + shift);
          tri->idx = idx;
          idx++;
        }
      }
    }

    // Output nodal data
    fprintf(outfile, "1 1\n");
    fprintf(outfile, "uh, adim\n");

    idx=1;
    for (i = 0; i < ct_in_vrts; i++) {
      //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      REAL val = in_vrts[i].fval; // in_vrts[i].val;
      if (fabs(val) < 1.e-15) val = 0.0;
      fprintf(outfile, "%d %g\n", idx, val);
      idx++;
    }
    if (tr_steiners != NULL) {
      for (i = 0; i < tr_steiners->used_items; i++) {
        Vertex *vrt = (Vertex *) tr_steiners->get(i);
        if (vrt->is_deleted()) continue;
        REAL val = vrt->fval; // vrt->val;
        if (fabs(val) < 1.e-15) val = 0.0;
        fprintf(outfile, "%d %g\n", idx, val);
        idx++;
      }
    }

    // Output cell data
    fprintf(outfile, "1 1\n");
    fprintf(outfile, "energy, unknown\n");

    idx = 1;
    for (i = 0; i < tr_tris->used_items; i++) {
      Triang* tri = (Triang *) tr_tris->get(i);
      if (!tri->is_deleted()) {
        // ignore a hull triangle.
        if (!tri->is_hulltri() && !tri->is_exterior()) {
          REAL val = tri->cct[2];
          fprintf(outfile, "%d %g\n", idx, val);
          idx++;
        }
      }
    }

    fclose(outfile);

    return sum_energy;
}

void Triangulation::dump_edge_energy_jump()
{
    REAL sum_energy = get_Dirichlet_energy();
    printf(" Energy = %g\n", sum_energy);

    // see file format: detri2/doc/UCD_Format.pdf
    char filename[256];
    sprintf(filename, "%s_jump.inp", io_outfilename);
    FILE *outfile = fopen(filename, "w");

    //int ntri = (int) tr_tris->objects - ct_hullsize - ct_exteriors;
    //printf("Writing %d triangles to file %s.\n", ntri, filename);
    int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
    //nv -= ct_unused_vrts;

    int ne = (3 * (tr_tris->objects - ct_hullsize) + ct_hullsize) / 2;
    printf("Writing %d segments to file %s.\n", ne, filename);

    //fprintf(outfile, "%d %d %d 0 0\n", nv, ntri, save_val); // nodal data
    fprintf(outfile, "%d %d 0 1 0\n", nv, ne); // cell data

    int i, idx=1; // UCD index starts from 1.
    for (i = 0; i < ct_in_vrts; i++) {
      //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      fprintf(outfile, "%d %g %g 0\n", idx, in_vrts[i].crd[0], in_vrts[i].crd[1]);
      in_vrts[i].idx = idx;
      idx++;
    }
    if (tr_steiners != NULL) {
      for (i = 0; i < tr_steiners->used_items; i++) {
        Vertex *vrt = (Vertex *) tr_steiners->get(i);
        if (vrt->is_deleted()) continue;
        fprintf(outfile, "%d %g %g 0\n", idx, vrt->crd[0], vrt->crd[1]);
        vrt->idx = idx;
        idx++;
      }
    }

    // UCD assumes vertex index starts from 1.
    //int shift = (io_firstindex == 1 ? 0 : 1);
    REAL *jump_list = new REAL[ne];
    TriEdge E, N;
    int tag;
    idx = 1;
    for (i = 0; i < tr_tris->used_items; i++) {
      E.tri = (Triang *) tr_tris->get(i);
      if (E.tri->is_deleted()) continue;
      if (!E.tri->is_hulltri()) {
        for (E.ver = 0; E.ver < 3; E.ver++) {
          N = E.esym(); // it neighbor
          if (E.apex()->idx > N.apex()->idx) {
            if(N.is_segment()) {
              Triang* seg = N.get_segment();
              tag = seg->tag;
              // A boundary edge has no jump.
              jump_list[idx-1] = 0.0;
            } else {
              assert(!N.tri->is_hulltri());
              tag = 0;
              jump_list[idx-1] = fabs(sqrt(E.tri->cct[2]) - sqrt(N.tri->cct[2]));
            }
            fprintf(outfile, "%d %d line %d %d\n", idx, tag,
                    E.org()->idx,
                    E.dest()->idx);
            idx++;
          }
        }
      }
    }
    assert(idx = ne + 1);

    // Output metric on nodes or elements
    fprintf(outfile, "1 1\n");
    fprintf(outfile, "jump, unknown\n");

    for (i = 0; i < ne; i++) {
      fprintf(outfile, "%d %g\n", i+1, jump_list[i]);
    }

    fclose(outfile);
}

//==============================================================================
/*
// Input:
//   E is a convex hull egde, v lies outside the convex hull and is visible by E.
// Output:
//   E returns a convex hull edge which contains the projection of v.

int  locate_hull_edge(Vertex *v, TriEdge &E) // for vertex interpolation
{
  REAL proj_v[2];

  

  return LOC_ON_EDGE;
}
*/
//==============================================================================
// P1 interpolation of vertex mesh size.

int Triangulation::set_vertex_metric(Vertex *v, TriEdge &E)
{
  if (E.tri == NULL) {
    // Locate the vertex first.
    //assert(0); // to do...
    return 0;
  }

  assert(!E.tri->is_deleted());
  assert(!E.tri->is_hulltri());

  Vertex *pa = E.org();
  Vertex *pb = E.dest();
  Vertex *pc = E.apex();
  double A  = get_tri_area(pa, pb, pc);
  double Wa = get_tri_area( v, pb, pc);
  double Wb = get_tri_area(pa,  v, pc);
  double Wc = get_tri_area(pa, pb,  v);
  if ((pa->val > 0) && (pb->val > 0) && (pc->val > 0)) {
    v->val = (pa->val * Wa + pb->val * Wb + pc->val * Wc) / A;
  }

  return 1;

  //======================================================
  // skipped
  // Set vertex metric.
  if (OMT_domain) {
    int loc = OMT_domain->locate_point(v, v->on_dm, 0);
    TriEdge E = v->on_dm;

    if (loc == LOC_IN_OUTSIDE) {
      return 0; // assert(0); // The background mesh does not cover the mesh domain.
    } else if (loc == LOC_ON_VERT) {
      v->val = E.org()->val;
      //for (int i = 0; i < 3; i++) {
      //  v->mtr[i] = E.org()->mtr[i];
      //}
    } else if (loc == LOC_ON_EDGE) {
      // Linear interpolation on edge E.
      Vertex *pa = E.org();
      Vertex *pb = E.dest();
      double L  = get_distance(pa, pb);
      double Wa = get_distance( v, pb);
      double Wb = get_distance(pa,  v);
      if ((pa->val > 0) && (pb->val > 0)) {
        v->val = (pa->val * Wa + pb->val * Wb) / L;
      }
      //for (int i = 0; i < 3; i++) {
      //  v->mtr[i] = (pa->mtr[i] * Wa + pb->mtr[i] * Wb) / L;
      //}
    } else if (loc == LOC_IN_TRI) {
      Vertex *pa = E.org();
      Vertex *pb = E.dest();
      Vertex *pc = E.apex();
      double A  = get_tri_area(pa, pb, pc);
      double Wa = get_tri_area( v, pb, pc);
      double Wb = get_tri_area(pa,  v, pc);
      double Wc = get_tri_area(pa, pb,  v);
      if ((pa->val > 0) && (pb->val > 0) && (pc->val > 0)) {
        v->val = (pa->val * Wa + pb->val * Wb + pc->val * Wc) / A;
      }
      //for (int i = 0; i < 3; i++) {
      //  v->mtr[i] = (pa->mtr[i] * Wa + pb->mtr[i] * Wb + pc->mtr[i] * Wc) / A;
      //}
    } else {
      assert(0); // Unknown case.
    }
  }
  /*
  else {
    if (SizeFunc) {
      v->val = SizeFunc(v);
      // if (op_size_is_density) v->val = 1. / (v->val * v->val);
    }
    if (Func) {
      // Function interpolation
      v->mtr[0] = Func(v);
      if (GradX) v->mtr[1] = GradX(v);
      if (GradY) v->mtr[2] = GradY(v);
    }
  }
  */

  /* [2019-08-01] Do not need this.
  // Regular test will use its z-coordinate.
  if (op_metric <= METRIC_Euclidean) {
    v->crd[2] = v->crd[0]*v->crd[0]+v->crd[1]*v->crd[1];
  } else if (op_metric == METRIC_HDE) {
    v->crd[2] = v->mtr[0] * op_hde_s1; // function value is the height
  }
  */

  return 1;
}

//==============================================================================

int Triangulation::set_vertex_metrics()
{
  if (op_db_verbose) {
    printf("  Set vertex metrics.\n");
  }

  /*
  // Calculate the diameter.
  REAL dx = io_xmax - io_xmin;
  REAL dy = io_ymax - io_ymin;
  geo_domain_diameter = sqrt(dx * dx + dy * dy);
  if (geo_domain_diameter == 0.0) {
    printf("!! Wrong domain diameter.\n");
    return 0;
  }
  if (op_db_verbose) {
    printf("  Diameter of the domain: %g\n", io_diagonal);
  }
  */

  /*
  // Get max-min function values.
  REAL maxFun, minFun;
  minFun =  1.e30;
  maxFun = -1.e30;
  for (int i = 0; i < ct_in_vrts; i++) {
    if (in_vrts[i].typ == UNUSEDVERTEX) continue;
    set_vertex_metric(&in_vrts[i]);
    REAL sol = in_vrts[i].mtr[0] * op_hde_s1; // .val
    minFun = (sol < minFun) ? sol : minFun;
    maxFun = (sol > maxFun) ? sol : maxFun;
  }
  if (tr_steiners != NULL) {
    for (int i = 0; i < tr_steiners->used_items; i++) {
      Vertex *v = (Vertex *) tr_steiners->get(i);
      if (v->is_deleted()) continue;
      set_vertex_metric(v);
      REAL sol = v->mtr[0] * op_hde_s1; // ->val
      minFun = (sol < minFun) ? sol : minFun;
      maxFun = (sol > maxFun) ? sol : maxFun;
    }
  }
  if (op_db_verbose) {
    printf("  Min-Max function values: %g, %g\n", minFun, maxFun);
  }

  // Use the maximum value.
  op_max_abs_Fun = fabs(maxFun);
  if (op_max_abs_Fun < fabs(minFun)) op_max_abs_Fun = fabs(minFun);
  if (op_db_verbose) {
    printf("  Maximum absolute value of the function: %g\n", op_max_abs_Fun);
  }
  */
  
  TriEdge E;

  for (int i = 0; i < ct_in_vrts; i++) {
    if (in_vrts[i].typ == UNUSEDVERTEX) continue;
    E.tri = NULL;
    set_vertex_metric(&in_vrts[i], E);
  }
  if (tr_steiners != NULL) {
    for (int i = 0; i < tr_steiners->used_items; i++) {
      Vertex *v = (Vertex *) tr_steiners->get(i);
      if (v->is_deleted()) continue;
      E.tri = NULL;
      set_vertex_metric(v, E);
    }
  }

  return 1;
}

//==============================================================================
// Return the shortest edge length at this vertex.
REAL Triangulation::get_vertex_min_edge_length(Vertex *pt)
{
  REAL lmin = 0., l;
  TriEdge E = pt->adj;
  do {
    if (E.dest() != tr_infvrt) { 
      l = get_distance(pt, E.dest());
      if (lmin == 0.) {
        lmin = l;
      } else {
        lmin = (l < lmin) ? l : lmin;
      }
    }
    E = E.eprev_esym();
  } while (E.tri != pt->adj.tri);
  assert(lmin > 0.);
  return lmin;
}

/*
REAL Triangulation::get_vertex_size_average(Vertex *pt)
{
  REAL lmax, lmin, lave, l;

  int ecount = 0;
  lmax = lmin = lave = 0.0;
  TriEdge E = pt->adj;
  do {
    // Skip the edge if it is a hull edge.
    if (E.dest() != tr_infvrt) { 
      l = get_distance(pt, E.dest());
      if (ecount == 0) {
        lmax = lmin = l;
      } else {
        lmax = (l > lmax) ? l : lmax;
        lmin = (l < lmin) ? l : lmin;
      }
      lave += l;
      ecount++;
    }
    E = E.eprev_esym();
  } while (E.tri != pt->adj.tri);

  lave -= (lmax + lmin);
  lave /= (ecount - 2);
  return lave;
}
*/
//==============================================================================
// [2018-11-07] This point might be on the (convex) hull.
// If this point lies in the interior of the mesh, it must be
//   removed, otherwise, a hull vertex might not be removed,
//   see doc-flip_check_hull_edge.pdf.
// [2019-01-29] Use flip_check(), only remove the given point,
//   do not remove adjacent points. This guarantees that
//   Ruppert's Delaunay refinement algorithm terminates. 

int Triangulation::remove_point(Vertex *pt, arraypool* fqueue)
{
  TriEdge E, tt[4];
  Vertex *ppt = NULL;
  int fflag;

  if (pt->is_fixed()) {
    return 0;
  }

  // Count the degree of this vertex.
  int deg = 0;
  E = pt->adj;
  do {
    deg++;
    E = E.eprev_esym(); // ccw
  } while (E.tri != pt->adj.tri);

  if (op_db_verbose > 3) {
    printf("    Removing vertex %d, degree(%d).\n", pt->idx, deg);
  }

  /* wrong due to constrained segment(s).
  E = pt->adj;
  do {
    //if (!E.tri->is_hulltri()) {
      tt[0] = E; fflag = FLIP_UNKNOWN;
      if (flip(tt, &ppt, fflag, fqueue)) {
        if (ppt == pt) {
          return 1; // point is removed.
        }
        E = pt->adj; // update the starting edge.
        continue;
      }
    //}
    E = E.eprev_esym(); // ccw
  } while (E.tri != pt->adj.tri);
  */

  /* [2019-01-29] Do not remove another vertex
  int count = 0;
  E = pt->adj;
  do {
    count++;
    //if (!E.tri->is_hulltri()) {
      tt[0] = E; fflag = FLIP_UNKNOWN;
      if (flip(tt, &ppt, fflag, fqueue)) {
        if (ppt == pt) {
          return 1; // point is removed.
        }
        E = pt->adj; // update the starting edge.
        count = 0;
        deg--;
        continue;
      }
    //}
    E = E.eprev_esym(); // ccw
  } while (count < deg);
  */

  int count = 0;
  E = pt->adj;
  do {
    count++;
    bool doflip = false;
    tt[0] = E;
    fflag = flip_check(&(tt[0]));
    if (fflag == FLIP_22) {
      doflip = true;
    } else if (fflag == FLIP_31) {
      // tt[0] is [a,b,c], a is to be removed.
      if (tt[0].org() == pt) {
        doflip = true;
      }
    } else if (fflag == FLIP_42) {
      // tt[0] is [a,b,c], remove [a] on edge [c,d].
      if (tt[0].org() == pt) {
        doflip = true;
      }
    }
    if (doflip) {
      //tt[0] = E; //fflag = FLIP_UNKNOWN;
      if (flip(tt, &ppt, fflag, fqueue)) {
        if (ppt == pt) {
          return 1; // point is removed.
        }
        if ((fflag == FLIP_22) && fqueue) {
          // Add the new edge into flip queue to check it.
          // This might be needed since a 3-1 flip might be rejected.
          TriEdge NN = tt[0].enext();
          NN.set_edge_infect();
          * (TriEdge *) fqueue->alloc() = NN;
        }
        E = pt->adj; // update the starting edge.
        count = 0;
        deg--;
        continue;
      }
    }
    E = E.eprev_esym(); // ccw
  } while (count < deg);

  // This vertex must be a hull vertex.
  return 0;
}

//==============================================================================

int Triangulation::coarsen_mesh()
{
  if (op_db_verbose) {
    printf("Mesh coarsening...\n");
  }

  arraypool *fqueue = new arraypool(sizeof(TriEdge), 4);
  double lmin, diff, ratio, tol;
  int rem_count = 0;
  bool rmflag;

  for (int i = 0; i < ct_in_vrts; i++) {
    Vertex *v = &(in_vrts[i]);
    if (v->typ == UNUSEDVERTEX) continue;
    if (v->is_fixed()) continue;
    if (op_no_bisect) { // -Y
      // Do not remove a boundary vertex.
      TriEdge E = v->adj;
      do {
        if (E.is_segment()) break;
        E = E.eprev_esym();
      } while (E.tri != v->adj.tri);
      if (E.is_segment()) continue;
    }
    rmflag = false;
    lmin = get_vertex_min_edge_length(v);
    REAL val = v->val; // mesh size at vertex. // To  (or density)
    if (val > 0.0) {
      diff = lmin - val;
      tol = fabs(diff) / val;
      rmflag = ((tol > 0.2) && (diff < 0));
    }
    if (op_target_length > 0.) {
      diff = lmin - op_target_length;
      ratio = lmin / op_target_length;
      rmflag = ((ratio < op_edge_collapse_factor) && (diff < 0));
    }
    if (rmflag) {
      if (op_db_verbose > 2) {
        printf("      Removing vertex [%d] lmin=%g.\n", v->idx, lmin);
      }
      if (remove_point(v, fqueue)) {
        rem_count++;
      }
      lawson_flip(NULL, 0, fqueue);
    }
  }

  if (tr_steiners != NULL) {
    for (int i = 0; i < tr_steiners->used_items; i++) {
      Vertex *v = (Vertex *) tr_steiners->get(i);
      if (v->is_deleted()) continue;
      if (v->is_fixed()) continue;
      if (op_no_bisect) { // -Y
        // Do not remove a boundary vertex.
        TriEdge E = v->adj;
        do {
          if (E.is_segment()) break;
          E = E.eprev_esym();
        } while (E.tri != v->adj.tri);
        if (E.is_segment()) continue;
      }
      rmflag = false;
      lmin = get_vertex_min_edge_length(v);
      REAL val = v->val;
      if (val > 0.0) {
        diff = lmin - val;
        tol = fabs(diff) / val;
        rmflag = ((tol > 0.2) && (diff < 0));
      }
      if (op_target_length > 0.) {
        diff = lmin - op_target_length;
        ratio = lmin / op_target_length;
        rmflag = ((ratio < op_edge_collapse_factor) && (diff < 0));
      }
      if (rmflag) {
        if (op_db_verbose > 2) {
          printf("      Removing vertex [%d] lmin=%g.\n", v->idx, lmin);
        }
        if (remove_point(v, fqueue)) {
          rem_count++;
        }
        lawson_flip(NULL, 0, fqueue);
      }
    }
  }

  delete fqueue;

  if (op_db_verbose > 1) {
    printf("  Removed %d vertices.\n", rem_count);
  }

  return rem_count;
}

//==============================================================================

int Triangulation::get_powercell_mass_centers(Vertex *massptlist)
{
  if (op_db_verbose > 1) {
    printf("  Calculating %d orthocenters\n", tr_tris->objects + ct_hullsize);
  }
  assert(OMT_domain != NULL);

  int i, idx;
  idx = io_firstindex;
  // Calculate circumcenters for hull triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    // Ignore exterior triangles.
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    get_tri_orthocenter(tri);
    tri->idx = idx; // hulltri is also indexed.
    idx++;
  }
  // Calculate bisectors for hull triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    // Ignore exterior triangles.
    if (tri->is_deleted()) continue;
    if (tri->is_hulltri()) { // A hull triangle.
      get_hulltri_orthocenter(tri);
    }
  }

  if (op_db_verbose > 1) {
    printf("  Calculating %d power cells and mass centers\n",
           ct_in_vrts + ((tr_steiners != NULL) ? tr_steiners->objects : 0));
  }
  int mcount = 0;

  // Calculate power cells and mass centers.
  idx = 0;
  for (i = 0; i < ct_in_vrts; i++) {
    Vertex *masspt = &(massptlist[idx]);
    masspt->init();
    idx++;

    Vertex *mesh_vert = &(in_vrts[i]);
    if (mesh_vert->typ == UNUSEDVERTEX) continue;
    if (mesh_vert->is_fixed()) continue;
    //printf("  idx = %d\n", i);
    Vertex *ptlist = NULL;
    int ptnum = 0;
    
    if (get_powercell(mesh_vert, &ptlist, &ptnum)) {
      // Calculate its mass center
      get_mass_center(ptlist, ptnum, masspt->crd);
      masspt->Pair = mesh_vert; // Remember it.
      mcount++;
      delete [] ptlist;
    } // if (Tr->get_powercell
  }

  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *masspt = &(massptlist[idx]);
      masspt->init();
      idx++;
    
      Vertex *mesh_vert = (Vertex *) tr_steiners->get(i);
      if (mesh_vert->is_deleted()) continue;
      if (mesh_vert->is_fixed()) continue;
      //printf("  idx = %d\n", i);
      Vertex *ptlist = NULL;
      int ptnum = 0;

      if (get_powercell(mesh_vert, &ptlist, &ptnum)) {
        get_mass_center(ptlist, ptnum, masspt->crd);
        masspt->Pair = mesh_vert; // Remember it.
        mcount++;
        delete [] ptlist;
      } // if (Tr->get_powercell
    }
  }

  return mcount;
}

int Triangulation::get_laplacian_center(Vertex *mesh_vertex, Vertex *movept)
{
  double mcx = 0., mcy = 0.;
  int deg = 0; // debgree
  bool is_seg_vertex = false;
  TriEdge E = mesh_vertex->adj;
  do {
    if (E.is_segment()) {
      is_seg_vertex = true;
      break;
    }
    if (!E.tri->is_hulltri()) {
      deg++;
      mcx += E.dest()->crd[0];
      mcy += E.dest()->crd[1];
    }
    E = E.eprev_esym(); // CCW
  } while (E.tri != mesh_vertex->adj.tri);
  
  if (!is_seg_vertex) {
    //if (deg <= 2) continue;
    mcx /= deg;
    mcy /= deg;
    // Save this point.
    movept->crd[0] = mcx;
    movept->crd[1] = mcy;
    //masspt->Pair = mesh_vertex; // Remember it.
    //mcount++;
    return 1;
  }
  return 0;
}

int Triangulation::get_laplacian_centers(Vertex *massptlist)
{
  if (op_db_verbose > 1) {
    printf("  Calculating %d Laplacian centers\n",
           ct_in_vrts + ((tr_steiners != NULL) ? tr_steiners->objects : 0));
  }
  int mcount = 0;

  int idx = 0;
  for (int i = 0; i < ct_in_vrts; i++) {
    Vertex *masspt = &(massptlist[idx]);
    masspt->init();
    idx++;

    Vertex *mesh_vertex = &(in_vrts[i]);
    if (mesh_vertex->typ == UNUSEDVERTEX) continue;
    //printf("  idx = %d fixed(%d)\n", i, mesh_vertex->is_fixed());
    if (mesh_vertex->is_fixed()) continue;

    if (get_laplacian_center(mesh_vertex, masspt)) {
      masspt->Pair = mesh_vertex; // Remember it.
      mcount++;
    }
  }

  if (tr_steiners != NULL) {
    for (int i = 0; i < tr_steiners->used_items; i++) {
      Vertex *masspt = &(massptlist[idx]);
      masspt->init();
      idx++;
    
      Vertex *mesh_vertex = (Vertex *) tr_steiners->get(i);
      if (mesh_vertex->is_deleted()) continue;
      //printf("  idx = %d fixed(%d)\n", i, mesh_vertex->is_fixed());
      if (mesh_vertex->is_fixed()) continue;

      if (get_laplacian_center(mesh_vertex, masspt)) {
        masspt->Pair = mesh_vertex; // Remember it.
        mcount++;
      }
    }
  }

  return mcount;
}

// distmesh.m
// ...
// % 6. Move mesh points based on bar lengths L and forces F
//  barvec=p(bars(:,1),:)-p(bars(:,2),:);              % List of bar vectors
//  L=sqrt(sum(barvec.^2,2));                          % L = Bar lengths
//  hbars=feval(fh,(p(bars(:,1),:)+p(bars(:,2),:))/2,varargin{:});
//  L0=hbars*Fscale*sqrt(sum(L.^2)/sum(hbars.^2));     % L0 = Desired lengths
// ...

REAL Triangulation::get_distmesh_target_length()
{
  //assert(op_metric == METRIC_Euclidean);
  op_metric = METRIC_Euclidean;

  // The number of edges (also used to validate if we have iterate all edges).
  int ne = (3 * (tr_tris->objects - ct_hullsize) + ct_hullsize) / 2;
  REAL sumL2 = 0.0;
  REAL Fscale = 1.2;
  TriEdge E;

  int idx = io_firstindex;
  for (int i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    if (tri->is_deleted()) continue;
    E.tri = tri;
    if (!E.tri->is_hulltri()) {
      for (E.ver = 0; E.ver < 3; E.ver++) {
        if (E.esym().tri->is_hulltri() || !E.esym().tri->is_infected()) {
          sumL2 += get_innerproduct(E.org(), E.dest());
          idx++;
        }
      } // E.ver
      E.tri->set_infect();
    }
  }

  // Uninfect all triangles.
  for (int i = 0; i < tr_tris->used_items; i++) {
	Triang* tri = (Triang *) tr_tris->get(i);
	if (tri->is_deleted()) continue;
	if (!tri->is_hulltri()) {
      tri->clear_infect();
	} // if (!tri->is_hulltri()) {
  }

  assert(idx == (ne + 1));
  
  //  L0=hbars*Fscale*sqrt(sum(L.^2)/sum(hbars.^2));     % L0 = Desired lengths
  
  return Fscale * sqrt(sumL2 / ne);
}

int Triangulation::get_distmesh_point(Vertex *mesh_vertex, Vertex *movept)
{
  // Variables cooresponding to the paper.
  // L, bar length, barvec[2], bar (edge) vector (unique).
  // F, force (> 0) on each bar, the direction of the force is given by
  // its barvec[2]. Ftot[2], the sum of force vectors at node.
  REAL L, F, barvec[2], Ftot[2];

  Ftot[0] = Ftot[1] = 0.;
  bool is_seg_vertex = false;
  TriEdge E = mesh_vertex->adj;
  do {
    if (E.is_segment()) {
      is_seg_vertex = true;
      break;
    }
    Vertex *pd = E.dest();
    if (pd != tr_infvrt) {
      //if (mesh_vertex->idx < pd->idx) {
        barvec[0] = pd->crd[0] - mesh_vertex->crd[0];
        barvec[1] = pd->crd[1] - mesh_vertex->crd[1];
      //} else {
      //  barvec[0] = mesh_vertex->crd[0] - pd->crd[0]; // test
      //  barvec[1] = mesh_vertex->crd[1] - pd->crd[1]; // test
      //}
      L = get_distance(mesh_vertex, pd);
      F = op_target_length - L;
      if (F < 0.) F = 0.;
      // Ftot += Fvec
      Ftot[0] += (-barvec[0] * F / L);
      Ftot[1] += (-barvec[1] * F / L);
    }
    E = E.eprev_esym(); // CCW
  } while (E.tri != mesh_vertex->adj.tri);

  if (!is_seg_vertex) {
    // Save this point.
    movept->crd[0] = mesh_vertex->crd[0] + op_smooth_deltat * Ftot[0];
    movept->crd[1] = mesh_vertex->crd[1] + op_smooth_deltat * Ftot[1];
    //movept->Pair = mesh_vertex; // Remember it.
    return 1;
  }

  return 0;
}

int Triangulation::get_distmesh_points(Vertex *moveptlist)
{
  // Only calculate it once.
  //op_target_length = get_distmesh_target_length();

  if (op_db_verbose > 1) {
    printf("  Calculating locations for %d points by distmesh, L0 = %g\n",
      ct_in_vrts + ((tr_steiners != NULL) ? tr_steiners->objects : 0),
      op_target_length);
  }
  int mcount = 0;
  int idx = 0;
  
  for (int i = 0; i < ct_in_vrts; i++) {
    Vertex *movept = &(moveptlist[idx]);
    movept->init();
    idx++;

    Vertex *mesh_vertex = &(in_vrts[i]);
    if (mesh_vertex->typ == UNUSEDVERTEX) continue;
    //printf("  idx = %d fixed(%d)\n", i, mesh_vertex->is_fixed());
    if (mesh_vertex->is_fixed()) continue;

    if (get_distmesh_point(mesh_vertex, movept)) {
      movept->Pair = mesh_vertex; // Remember it.
      mcount++;
    }
  }

  if (tr_steiners != NULL) {
    for (int i = 0; i < tr_steiners->used_items; i++) {
      Vertex *movept = &(moveptlist[idx]);
      movept->init();
      idx++;
    
      Vertex *mesh_vertex = (Vertex *) tr_steiners->get(i);
      if (mesh_vertex->is_deleted()) continue;

      if (get_distmesh_point(mesh_vertex, movept)) {
        movept->Pair = mesh_vertex; // Remember it.
        mcount++;
      }
    }
  }

  return mcount;
}

//==============================================================================

int Triangulation::smooth_vertices()
{
  int ptcount = ct_in_vrts;
  if (tr_steiners) {
    ptcount += tr_steiners->objects;
  }
  Vertex *massptlist = new Vertex[ptcount];
  int mcount = 0; // count the number of moved vertices.

  if (op_smooth_criterion == SMOOTH_LAPLACIAN) {
    printf("\nUsing Lapacian smoother\n");
    mcount = get_laplacian_centers(massptlist);
  } else if (op_smooth_criterion == SMOOTH_CVT) {
    printf("\nUsing CVT smoother\n");
    mcount = get_powercell_mass_centers(massptlist);
  } else if (op_smooth_criterion == SMOOTH_DISTMESH) {
    printf("\nUsing DISTMESH smoother\n");
    mcount = get_distmesh_points(massptlist);
  } else {
    printf("Smooth option not available yet.\n");
    delete [] massptlist;
    return 0;
  }

  if (op_db_verbose > 1) {
    printf("  Relocating %d vertices.\n", mcount);
  }
  arraypool *fqueue = new arraypool(sizeof(TriEdge), 4);

  for (int i = 0; i < ptcount; i++) {
    Vertex *mcpt = &(massptlist[i]);
    if (mcpt->Pair == NULL) continue; 

    Vertex *mesh_vert = mcpt->Pair;
    if (op_db_verbose > 2) {
      printf("    Relocating vertex %d.\n", mesh_vert->idx);
    }

    // [2019-09-11] Do not move it, if it belongs to a boundary segment.
    int bdryflag = 0; // Check if it is a boundary segment vertex (fixed).
    TriEdge E = mesh_vert->adj;
    do {
      if (E.is_segment()) {
        bdryflag = 1; break;
      }
      E = E.eprev_esym(); // ccw
    } while (E.tri != mesh_vert->adj.tri);
    if (bdryflag) continue;

    // Check if we can directly move it.    
    bool bflag = true;
    int hullflag = 0; // Check if it is a hull vertex.
    E = mesh_vert->adj;
    do {
      if (!E.tri->is_hulltri()) {
        if (Orient2d(E.dest(), E.apex(), mcpt) <= 0.) {
          bflag = false; break;
        } else {
          hullflag = 1;
        }
      }
      E = E.eprev_esym(); // ccw
    } while (E.tri != mesh_vert->adj.tri);   

    if (bflag) {
      REAL x = mcpt->crd[0];
      REAL y = mcpt->crd[1];
      if (op_db_verbose > 2) {
        printf("    Directly move to its new position - hull(%d).\n", hullflag);
      }
      mesh_vert->crd[0] = x;
      mesh_vert->crd[1] = y;
      mesh_vert->crd[2] = x*x + y*y - mesh_vert->wei;

      if (op_metric > 0) {
        //set_vertex_metric(mesh_vert);
      }

      // Do flips to recover Delaunayness.
      TriEdge N;
      E = mesh_vert->adj;
      do {
        E.set_edge_infect(); // A star edge
        * (TriEdge *) fqueue->alloc() = E;
        N = E.enext();       // A link edge
        N.set_edge_infect();
        * (TriEdge *) fqueue->alloc() = N;
        E = E.eprev_esym(); // ccw
      } while (E.tri != mesh_vert->adj.tri);

      lawson_flip(NULL, hullflag, fqueue);
      continue;
    }

    // Check if the new location still inside the domain.
    int chkloc = locate_point(mcpt, E, 1); // Stop at segment.
    if (chkloc == LOC_ENC_SEG) {
      // The new location is outside the domain.
      // Project it back to the segment.
      // to do...
      printf("!!! Warning: ignore an exterior location.\n");
      continue;
    }

    if (op_db_verbose > 2) {
      printf("    Delete-and-Insert to its new position.\n");
    }
    // Bakup
    mcpt->idx = mesh_vert->idx;
    mcpt->tag = mesh_vert->tag;
    mcpt->wei = mesh_vert->wei;
    mcpt->val = mesh_vert->val;
    mcpt->typ = mesh_vert->typ;
    mcpt->on_dm = mesh_vert->on_dm; // for set_vertex_metric()

    // Remove this vertex from the triangulation.
    if (!remove_point(mesh_vert, fqueue)) {
      //assert(0); // This should always succeed.
      if (op_db_verbose > 2) {
        printf("    Failed to remove vertex %d\n", mesh_vert->idx);
      }
      // Recover Delaunay at this vertex.
      // push the link edges of mesh_vert.
      TriEdge N;
      E = mesh_vert->adj;
      do {
        E.set_edge_infect(); // A star edge
        * (TriEdge *) fqueue->alloc() = E;
        N = E.enext();       // A link edge
        N.set_edge_infect();
        * (TriEdge *) fqueue->alloc() = N;
        E = E.eprev_esym(); // ccw
      } while (E.tri != mesh_vert->adj.tri);

      lawson_flip(NULL, hullflag, fqueue);
      continue;
    }

    lawson_flip(NULL, hullflag, fqueue); // hullflag = 0

    // Insert a new vertex at (x,y).
    Vertex *newpt;
    if (mcpt->typ == STEINERVERTEX) {
      newpt = (Vertex *) tr_steiners->alloc();
      newpt->init();
    } else {
      newpt = mesh_vert;
      assert(newpt->typ == UNUSEDVERTEX);
      ct_unused_vrts++;
    }
    REAL x = mcpt->crd[0];
    REAL y = mcpt->crd[1];
    newpt->crd[0] = x;
    newpt->crd[1] = y;
    newpt->crd[2] = x*x + y*y - mcpt->wei;
    newpt->idx = mcpt->idx;
    newpt->wei = mcpt->wei; // the weight
    newpt->val = mcpt->val; // mesh size
    newpt->typ = mcpt->typ;
    newpt->on_dm = mcpt->on_dm;
    //newpt->Pair = mcpt->Pair;

    if (op_metric > 0) {
      //set_vertex_metric(newpt);
    }

    TriEdge tt[4];
    E.tri = NULL;
    int loc = locate_point(newpt, E, 1); // Stop at segment.
    tt[0] = E;
    if (loc == LOC_IN_OUTSIDE) {
      int fflag = FLIP_13;
      flip(tt, &newpt, fflag, fqueue);
    } else if (loc == LOC_IN_TRI) {
      int fflag = FLIP_13;
      flip(tt, &newpt, fflag, fqueue);
    } else if (loc == LOC_ON_EDGE) {
      int fflag = FLIP_24;
      flip(tt, &newpt, fflag, fqueue);
    } else {
      assert(0); // Not possible.
    }
    lawson_flip(newpt, hullflag, fqueue);
  } // i

  //if (check_mesh(0,0) > 0) {
  //  printf("!!! Mesh invalid after relocating vertices.\n");
  //  assert(0);
  //}

  delete fqueue;
  delete [] massptlist;
  return 1;
}

//==============================================================================

int Triangulation::mesh_adapt()
{
  // We will need Steiner points.
  if (tr_steiners == NULL) {
    // Estimate the final mesh size.
    int est_size = 8192;
    int log2objperblk = 0;
    while (est_size >>= 1) log2objperblk++;
    tr_steiners = new arraypool(sizeof(Vertex), log2objperblk);
  }

  printf("Mesh adaptation using ");
  if (op_metric == METRIC_Euclidean) {
    printf("Euclidean metric\n");
  } else if (op_metric == METRIC_HDE) {
    printf("HDE metric\n");
    //printf("  - test function = %d\n", op_test_fun);
    printf("  - target length = %g\n", op_target_length);
    //printf("  - s1 = %g, s2 = %g\n", op_hde_s1, op_hde_s2);
  } else if (op_metric == METRIC_Riemannian) {
    printf("Riemannian metric\n");
    assert(0); // to do ...
  } else {
    printf("No metric, use default METRIC_Euclidean\n");
    op_metric = METRIC_Euclidean;
  }

  if (!io_with_metric) {
    assert((OMT_domain != NULL) && OMT_domain->io_with_metric);
    set_vertex_metrics();
  }

  //int meshidx = 0;
  //if (op_save_inter_meshes) {
  //  save_to_ucd(meshidx, 1); meshidx++;
  //}

  op_db_verbose--; // Disable quality statistic

  //if (op_db_verbose) {
    mesh_statistics();
  //}

  if (op_use_coarsening) {
    coarsen_mesh();
  }

  if (op_use_splitting) {
    delaunay_refinement();
  }

  /*
  if (op_use_smoothing) {
    int iter = 0;
    do {
      if (op_db_verbose) {
        printf("Iter = %d\n", iter + 1);
      }
      smooth_vertices();
      iter++;
    } while (iter < op_max_iter);
  }
  */

  //if (op_db_verbose) {
    mesh_statistics(); //quality_statistic();
  //}

  return 1;
}
