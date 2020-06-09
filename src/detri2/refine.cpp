#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

//==============================================================================
// S is a segment. S.org() is a SEGMENTVERTEX.
// Return true if S.org() is an acute vertex.

bool Triangulation::is_acute_vertex(TriEdge& S)
{
  assert(S.is_segment());
  double theta, sum_angle, minang = 2.*PI;
  //double min_radius = 1.e+30;
  //Vertex *chkpt = S.org();

  if (!S.tri->is_hulltri()) {
    // Search for the next segment at its CCW direction of S.
    sum_angle = 0.;
    TriEdge E = S;
    do {
      theta = get_angle(E.org(), E.dest(), E.apex());
      sum_angle += theta;
      E = E.eprev_esym();
      if (E.is_segment()) {
        // Calculate the min_radius.
        //REAL len = get_distance(E.org(), E.dest());
        //if (len < min_radius) min_radius = len;
        break;
      }
    } while (E.tri != S.tri);
    
    if (sum_angle < minang) {
      minang = sum_angle;
    }
  }
  
  TriEdge Sn = S.esym();
  if (!Sn.tri->is_hulltri()) {
    // Search for the next segment at its CW direction of S.
    sum_angle = 0.;
    TriEdge E = Sn;
    do {
      theta = get_angle(E.dest(), E.apex(), E.org());
      sum_angle += theta;
      E = E.enext_esym();
      if (E.is_segment()) {
        // Calculate the min_radius.
        //REAL len = get_distance(E.org(), E.dest());
        //if (len < min_radius) min_radius = len;
        break;
      }
    } while (E.tri != Sn.tri);
    
    if (sum_angle < minang) {
      minang = sum_angle;
    }
  }

  REAL ang = minang / PI * 180.;

  if (ang < 60.) {
    // This is an acute vertex. Set its mesh size (min_radius).
    
    return true;
  }

  return false;
}

void Triangulation::mark_acute_vertices()
{
  TriEdge E, N;
  int i, j;
  int count = 0;

  if (op_db_verbose) {
    printf("  Mark all acute vertices.\n");
  }

  for (i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted() ||
        E.tri->is_hulltri()) continue;
    for (E.ver = 0; E.ver < 3; E.ver++) {
      N = E.esym(); // N.tri might be a hulltri.
      if (E.apex()->idx > N.apex()->idx) {
        if (E.is_segment()) {
          TriEdge S = E;
          for (j = 0; j < 2; j++) {
            if (S.org()->typ == SEGMENTVERTEX) {
              // It is an input vertex.
              if (!S.org()->is_acute()) {
                //if (S.org()->idx == 131) {
                //  printf("debug\n");
                //}
                if (is_acute_vertex(S)) {
                  S.org()->set_acute();
                  count++;
                }
              }
            }
            S = S.esym();
          } // j
        }
      }
    } // E.ver
  } // i
  
  if (op_db_verbose) {
    printf("  Marked %d acute vertices.\n", count);
  }
}

//==============================================================================

// If 'pt' is given, only check if S is encroached by pt.
// Otherwise, check the apex of S.
// In Delaunay refinement case, 'pt' must be given.
int Triangulation::check_segment(Vertex *pt, TriEdge &S, arraypool *encsegs)
{
  assert(S.is_segment());
  Vertex *e1 = S.org();
  Vertex *e2 = S.dest();

  // debug only.
  //if ((e1 == NULL) || (e2 == NULL)) {
  //  printf("debug.\n");
  //  return 0;
  //}

  if (pt != NULL) {
    // A mesh vertex is given.
    assert(pt != tr_infvrt);
    /*
    double v1[2], v2[2];
    double costh = 0;
    v1[0] = e1->crd[0] - pt->crd[0];
    v1[1] = e1->crd[1] - pt->crd[1];
    v2[0] = e2->crd[0] - pt->crd[0];
    v2[1] = e2->crd[1] - pt->crd[1];
    costh = v1[0] * v2[0] + v1[1] * v2[1];
    */
    double ang = get_angle(pt, e1, e2) * 2.;
    if (ang > PI) { // if (costh < 0) {
      Triang *pE = (Triang *) encsegs->alloc();
      pE->init();
      pE->vrt[0] = e1;
      pE->vrt[1] = e2;
      pE->vrt[2] = pt;
      return 1;
    }
    return 0;
  }

  for (int i = 0; i < 2; i++) {
    Vertex *chkpt = S.apex();
    if (chkpt != tr_infvrt) {
      /*
      double v1[2], v2[2];
      double costh = 0;
      v1[0] = e1->crd[0] - chkpt->crd[0];
      v1[1] = e1->crd[1] - chkpt->crd[1];
      v2[0] = e2->crd[0] - chkpt->crd[0];
      v2[1] = e2->crd[1] - chkpt->crd[1];
      costh = v1[0] * v2[0] + v1[1] * v2[1];
      if (costh < 0) {
      */
      double ang = get_angle(chkpt, e1, e2) * 2.;
      if (ang > PI) { // if (costh < 0) {
        Triang *pE = (Triang *) encsegs->alloc();
        pE->init();
        pE->vrt[0] = e1;
        pE->vrt[1] = e2;
        pE->vrt[2] = chkpt; // Save the encroached point.
        return 1;
      }
    }
    S = S.esym();
  }

  TriEdge E;
  E.tri = S.get_segment();
  double len = get_distance(e1, e2);
  if (E.tri->val > 0) {
    //printf("Check segment %d, %d -- val=%g\n", e1->idx, e2->idx, E.tri->val);
    if (len > E.tri->val) {
      //printf(" len (%g) > E->val (%g)\n", len, E->val);
      Triang *pE = (Triang *) encsegs->alloc();
      pE->init();
      pE->vrt[0] = e1;
      pE->vrt[1] = e2;
      pE->vrt[2] = NULL; // Don't save a rejected vertex.
      pE->set_infect();  // split due to mesh size
      return 1; // split this segment.
    }
  }

  if (op_target_length > 0) {
    if (len > op_target_length) {
      //printf(" len (%g) > target_len (%g)\n", len, op_target_length);
      Triang *pE = (Triang *) encsegs->alloc();
      pE->init();
      pE->vrt[0] = e1;
      pE->vrt[1] = e2;
      pE->vrt[2] = NULL; // Don't save a rejected vertex.
      pE->set_infect();  // split due to mesh size
      return 1; // split this segment.
    }
  }

  /*
  if (op_metric > 0) { // > 0 Euclidean, Riemannian, HDE
    // Check if this segment satisifies mesh size.
    //double len = get_distance(e1, e2);
    //TriEdge E; E.tri = S.get_segment();
    // Check mesh size at the middle of the segment.
    Vertex mid; mid.init();
    mid.crd[0] = (e1->crd[0] + e2->crd[0]) / 2.;
    mid.crd[1] = (e1->crd[1] + e2->crd[1]) / 2.;
    get_vertex_metric(&mid);
  }
  */

  return 0;
}

//==============================================================================

int Triangulation::split_enc_segment(TriEdge &S, Vertex *encpt, bool mtrflag,
  arraypool *fqueue, arraypool *encsegs, arraypool *enctris)
{
  if (op_db_verbose > 2) {
    printf("    Split encroached segment [%d, %d]\n", S.org()->idx, S.dest()->idx);
  }

  if ((encpt == NULL) && !mtrflag) {
    // Do not split this segment if one of its endpoints is acute.
    if (S.org()->is_acute() ||
        S.dest()->is_acute()) {
      return 0;
    }
  }

  bool isseg = true;
  int stag = 0; REAL val = 0.;
  Triang *seg = S.get_segment();
  if (seg != NULL) {
    stag = seg->tag;
    val = seg->val;
    //printf("    val = %g\n", val);
  } else {
    isseg = false; // This is possible in lifted triangulation case.
  }

  Vertex *newvrt = (Vertex *) tr_steiners->alloc();
  newvrt->init();

  // For debugging only
  newvrt->idx = io_firstindex + (ct_in_vrts + tr_steiners->objects - 1);

  // Default split it at the middle.
  Vertex *e1 = S.org();
  Vertex *e2 = S.dest();
  
  if (encpt != NULL) {
    REAL L  = get_distance(e1, e2);
    REAL L1 = get_distance(e1, encpt);
    REAL L2 = get_distance(e2, encpt);
    REAL A  = get_tri_area(e1, e2, encpt);
    // The projection distance from encpt onto {e1, e2}.
    REAL h  = 2.0 * A / L;
    REAL split;
    if (e1->is_acute()) {
      if (e2->is_acute()) {
        // Choose the base point bteween e1 or e2.
        if (L1 < L2) {
          // Cut {e1, e2} using the length {e1, encpt} (L1)
          // Make sure the rest of edge is not too short.
          if ((L - L1) < h) {
            split = 0.5;
          } else {
            split = L1 / L;
          }
          newvrt->crd[0] = e1->crd[0] + split * (e2->crd[0] - e1->crd[0]);
          newvrt->crd[1] = e1->crd[1] + split * (e2->crd[1] - e1->crd[1]);
        } else {
          // Cut {e1, e2} using the length {e2, encpt} (L2)
          // Make sure the rest of edge is not too short.
          if ((L - L2) < h) {
            split = 0.5;
          } else {
            split = L2 / L;
          }
          newvrt->crd[0] = e2->crd[0] + split * (e1->crd[0] - e2->crd[0]);
          newvrt->crd[1] = e2->crd[1] + split * (e1->crd[1] - e2->crd[1]);
        }
      } else {
        // Cut {e1, e2} using the length {e1, encpt} (L1)
        // Make sure the rest of edge is not too short.
        if ((L - L1) < h) {
          split = 0.5;
        } else {
          split = L1 / L;
        }
        newvrt->crd[0] = e1->crd[0] + split * (e2->crd[0] - e1->crd[0]);
        newvrt->crd[1] = e1->crd[1] + split * (e2->crd[1] - e1->crd[1]);
      }
    } else if (e2->is_acute()) {
      // Cut {e1, e2} using the length {e2, encpt} (L2)
      // Make sure the rest of edge is not too short.
      if ((L - L2) < h) {
        split = 0.5;
      } else {
        split = L2 / L;
      }
      newvrt->crd[0] = e2->crd[0] + split * (e1->crd[0] - e2->crd[0]);
      newvrt->crd[1] = e2->crd[1] + split * (e1->crd[1] - e2->crd[1]);
    } else {
      // neither e1 nor e2 is acute. Project encpt onto {e1, e2}.
      REAL L3 = sqrt(L1*L1 - h*h); // Using Pythagorean theorem
      split = L3 / L;
      newvrt->crd[0] = e1->crd[0] + split * (e2->crd[0] - e1->crd[0]);
      newvrt->crd[1] = e1->crd[1] + split * (e2->crd[1] - e1->crd[1]);
      // Shift to middle if it is too close to e2.
      REAL res = get_distance(newvrt, e2);
      if (res < h) {
        split = 0.5;
        newvrt->crd[0] = e1->crd[0] + split * (e2->crd[0] - e1->crd[0]);
        newvrt->crd[1] = e1->crd[1] + split * (e2->crd[1] - e1->crd[1]);
      }
    }
  } else {
    newvrt->crd[0] = 0.5 * (e1->crd[0] + e2->crd[0]);
    newvrt->crd[1] = 0.5 * (e1->crd[1] + e2->crd[1]);
  }

  // set metric for the new vertex.
  newvrt->on_dm = e1->on_dm; // for searching in OMT_domain (may not be used).
  TriEdge G = S;
  if (G.tri->is_hulltri()) G = G.esym();
  int iloc = LOC_ON_EDGE;
  set_vertex_metric(newvrt, G, iloc);

  newvrt->idx = io_firstindex + (ct_in_vrts + tr_steiners->objects - 1);
  newvrt->tag = stag; //seg->tag;
  newvrt->typ = STEINERVERTEX;

  //if (newvrt->idx == 324) {
  //  printf("debug... idx==324\n");
  //  save_triangulation();
  //}

  //int stag = seg->tag; // done above
  if (isseg) {
    remove_segment(seg); // Remove this segment.
    //assert(seg->is_deleted());
    //printf("   nnnnnnnnnnn\n");
    //if (check_mesh(0,0) > 0) {
    //  assert(0);
    //}
  }

  // Insert the new point (split the segment).
  int fflag = FLIP_24;
  TriEdge tt[4];
  tt[0] = S; // [e1, e2, c]
  tt[2] = tt[0].esym(); // [e2, e1, d]
  flip(tt, &newvrt, fflag, fqueue);

  if (isseg) {
    S = tt[1].enext(); // [e1, newpt, c]
    assert(S.org() == e1);
    assert(S.dest() == newvrt);
    insert_segment(S, stag, val, NULL);

    S = tt[0].eprev(); // [newpt, e2, d]
    assert(S.org() == newvrt);
    assert(S.dest() == e2);
    insert_segment(S, stag, val, NULL);
  }

  // For Delaunay refinement, we fix the segmment Steiner vertex
  //   to prevent to remove them, this will cause endless loop.
  assert(!newvrt->is_fixed());
  //newvrt->set_fix();

  lawson_flip(newvrt, 0, fqueue); // hullflag = 0

  if (encsegs != NULL) {
    // Add encraoched segments into the list.
    TriEdge E = newvrt->adj, N;
    do {
      N = E.enext();
      if (N.is_segment()) {
        check_segment(newvrt, N, encsegs);
      }
      E = E.eprev_esym(); // CCW.
    } while (E.tri != newvrt->adj.tri);
    
    // The two subsegments need to be checked.
    if (get_edge(newvrt, e1, E)) {
      check_segment(NULL, E, encsegs);
    } else {
      assert(0); // a subsegment is missing.
    }
    
    if (get_edge(newvrt, e2, E)) {
      check_segment(NULL, E, encsegs);
    } else {
      assert(0); // a subsegment is missing.
    }
  } // if (encsegs != NULL)

  if (enctris != NULL) {
    // Add the triangles in the star of newvrt into list.
    enq_vertex_star(newvrt, enctris);
  }
  
  //if (newvrt->idx == 237) {
  //  //===== debug only ================
  //  save_triangulation();
  //  //===== debug only ================
  //}

  return 1;
}

//==============================================================================

int Triangulation::repair_encsegments(arraypool *encsegs, arraypool *enctris)
{
  arraypool *fqueue = new arraypool(sizeof(TriEdge), 8);
  bool smtrflag;
  TriEdge E;

  while (encsegs->objects > 0) {
    for (int i = 0; i < encsegs->used_items; i++) {
      Triang *pE = (Triang *) encsegs->get(i);
      if (pE->is_deleted()) continue;
      if (!op_no_bisect) { // no -Y option
        if (get_edge(pE->vrt[0], pE->vrt[1], E)) {
          smtrflag = pE->is_infected(); // mesh size
          split_enc_segment(E, pE->vrt[2], smtrflag, fqueue, encsegs, enctris);
        }
      }
      pE->set_deleted();
      encsegs->dealloc(pE);
    }
  }

  encsegs->clean();

  delete fqueue;
  return 1;
}

//==============================================================================
// Queue a triangle to refine.
// If "mtrflag" is true, the reason to refine this triangle is due to mesh size
//   requirement. Otherwise, it is a bad quality triangle.

void Triangulation::enq_triangle(Triang* tri, bool mtrflag, arraypool* enctris)
{
  Triang *paryEle = (Triang *) enctris->alloc();
  paryEle->init(); // Initialize
  //paryEle->nei[0].tri = tri;
  for (int j = 0; j < 3; j++) {
    paryEle->vrt[j] = tri->vrt[j];
  }
  //for (int j = 0; j < 3; j++) {
  //  paryEle->cct[j] = cct[j];
  //}
  if (mtrflag) {
    paryEle->set_infect();
  }
}

void Triangulation::enq_vertex_star(Vertex *pt, arraypool* enctris)
{
  TriEdge E = pt->adj;
  do {
    check_triangle(E.tri, enctris);
    E = E.eprev_esym();
  } while (E.tri != pt->adj.tri);
}

int Triangulation::check_triangle(Triang *tri, arraypool* enctris)
{
  // Skip a hull and exterior triangle.
  if (tri->is_hulltri() || 
      tri->is_exterior()) return 0;

  //                apex
  //                 /\
  //              c /  \ b
  //               /    \
  //              /______\
  //            org   a   dest

  // The three edge lengths of this triangle.
  //REAL a = get_distance(tri->vrt[1], tri->vrt[2]); // [B,C]
  //REAL b = get_distance(tri->vrt[2], tri->vrt[0]); // [C,A]
  //REAL c = get_distance(tri->vrt[0], tri->vrt[1]); // [A,B]
  TriEdge E = TriEdge(tri, 0);
  REAL a = get_distance(E.org(), E.dest());  // [B,C]
  REAL b = get_distance(E.dest(), E.apex()); // [C,A]
  REAL c = get_distance(E.apex(), E.org());  // [A,B]

  // (1) validate the triangle.
  REAL s = (a + b + c) / 2.;
  REAL delta = s * (s - a) * (s - b) * (s - c);
  REAL area = 0.;
  if (delta > 0) {
    area = sqrt(delta);
  } else {
    assert(0); // a non-valide triangle
  }

  // (2) metric (mesh size) has priority than mesh quality
  if (op_maxarea > 0) {
    if (area > op_maxarea) {
      enq_triangle(tri, true, enctris);
      return 1;
    }
  }

  if (tri->val > 0) {
    if (area > tri->val) {
      enq_triangle(tri, true, enctris);
      return 1;
    }
  }

  if (op_target_length > 0) {
    if ((a > op_target_length) || (b > op_target_length) || (c > op_target_length)) {
      enq_triangle(tri, true, enctris);
      return 1;
    }
  }

  // Check mesh size at vertex.
  if (E.org()->val > 0) { // A
    if ((a > tri->vrt[0]->val) || (c > E.org()->val)) {
      enq_triangle(tri, true, enctris);
      return 1;
    }
  }
  if (E.dest()->val > 0) { // B
    if ((a > E.dest()->val) || (b > E.dest()->val)) {
      enq_triangle(tri, true, enctris);
      return 1;
    }
  }
  if (E.apex()->val > 0) { // C
    if ((c > E.apex()->val) || (b > E.apex()->val)) {
      enq_triangle(tri, true, enctris);
      return 1;
    }
  }

  // Check mesh quality
  if (op_minangle > 0) {
    // Get the smallest angle of this triangle.
    // The smallest angle is opposite to the shortest edge.
    //TriEdge E = TriEdge(tri, 0);
    REAL aa, bb, cc; // cc is the shortest edge length.
    if (a < b) {
      if (a < c) {
        // a is the shortest, apex -> org
        cc = a; aa = b; bb = c;
        E = E.eprev();
      } else {
        // c is the shortest, dest -> org
        cc = c; aa = a; bb = b;
        E = E.enext();
      }
    } else {
      if (b < c) {
        // b is the shortest, org -> org
        cc = b; aa = c; bb = a;
      } else {
        // c is the shortest, dest -> org
        cc = c; aa = a; bb = b;
        E = E.enext();
      }
    }
    // Using cosine law to find the angle.
    REAL cosC = (aa*aa + bb*bb - cc*cc) / (2.0*aa*bb);
    if (cosC > 1.0) cosC = 1.0;
    if (cosC < -1.0) cosC = -1.0;

    if (cosC > op_cosminangle) {
      // Do not split this triangle if it contains a small input angle.
      // In this case, it is no hope to improve the mesh quality.
      TriEdge N1 = E;
      TriEdge N2 = E.eprev();
      if (!(N1.is_segment() && N2.is_segment())) {
        enq_triangle(tri, false, enctris);
        return 1;
      }
    }
  } // if (op_minangle > 0)

  return 0;
}

int Triangulation::get_triangle(Vertex *e1, Vertex *e2, Vertex *e3, TriEdge& E)
{
  if (e1->is_deleted() || (e1->typ == UNUSEDVERTEX) ||
      e2->is_deleted() || (e2->typ == UNUSEDVERTEX) ||
      e3->is_deleted() || (e3->typ == UNUSEDVERTEX)) {
    return 0;
  }
  if (get_edge(e1, e2, E)) {
    if (E.apex() == e3) return 1;
    E = E.esym();
    return E.apex() == e3;
  }
  return 0;
}

int Triangulation::get_tri_circumcenter(Triang *tri, REAL ccent[3])
{
  if (op_metric <= METRIC_Euclidean) {
    // torg  -> tri->vrt[0]->crd
    // tdest -> tri->vrt[1]->crd
    // tapex -> tri->vrt[2]->crd
    REAL xdo = tri->vrt[1]->crd[0] - tri->vrt[0]->crd[0];
    REAL ydo = tri->vrt[1]->crd[1] - tri->vrt[0]->crd[1];
    REAL xao = tri->vrt[2]->crd[0] - tri->vrt[0]->crd[0];
    REAL yao = tri->vrt[2]->crd[1] - tri->vrt[0]->crd[1];
    REAL dodist = xdo * xdo + ydo * ydo;
    REAL aodist = xao * xao + yao * yao;
    //REAL dadist = (tri->vrt[1]->crd[0] - tri->vrt[2]->crd[0]) *
    //              (tri->vrt[1]->crd[0] - tri->vrt[2]->crd[0]) +
    //              (tri->vrt[1]->crd[1] - tri->vrt[2]->crd[1]) *
    //              (tri->vrt[1]->crd[1] - tri->vrt[2]->crd[1]);
    REAL denominator = 0.5 / (xdo * yao - xao * ydo); 
    REAL dx = (yao * dodist - ydo * aodist) * denominator;
    REAL dy = (xdo * aodist - xao * dodist) * denominator;
    //REAL dz = 0.0;
    ccent[0] = tri->vrt[0]->crd[0] + dx;
    ccent[1] = tri->vrt[0]->crd[1] + dy;
    ccent[2] = 0.; //ccent[0]*ccent[0] + ccent[1]*ccent[1];
  } else if (op_metric == METRIC_Riemannian) {
    // to do...
  } else if (op_metric == METRIC_HDE) {
    // Find the circumcenter of a triangle in 3d.
    // Use the following formula from J. Shewchuk (Geometry Junkyard).
    //         |c-a|^2 [(b-a)x(c-a)]x(b-a) + |b-a|^2 (c-a)x[(b-a)x(c-a)]
    // m = a + ---------------------------------------------------------.
    //                        2 | (b-a)x(c-a) |^2
    // https://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
    double xba, yba, zba, xca, yca, zca;
    double balength, calength;
    double xcrossbc, ycrossbc, zcrossbc;
    xba = tri->vrt[1]->crd[0] - tri->vrt[0]->crd[0]; // pb-pa
    yba = tri->vrt[1]->crd[1] - tri->vrt[0]->crd[1];
    zba = tri->vrt[1]->crd[2] - tri->vrt[0]->crd[2];
    xca = tri->vrt[2]->crd[0] - tri->vrt[0]->crd[0]; // pc-pa
    yca = tri->vrt[2]->crd[1] - tri->vrt[0]->crd[1];
    zca = tri->vrt[2]->crd[2] - tri->vrt[0]->crd[2];
    balength = xba * xba + yba * yba + zba * zba;
    calength = xca * xca + yca * yca + zca * zca;
    xcrossbc = yba * zca - yca * zba;
    ycrossbc = zba * xca - zca * xba;
    zcrossbc = xba * yca - xca * yba;
    REAL denominator = 0.5 / (xcrossbc * xcrossbc + ycrossbc * ycrossbc +
                       zcrossbc * zcrossbc);
    REAL dx = ((balength * yca - calength * yba) * zcrossbc -
          (balength * zca - calength * zba) * ycrossbc) * denominator;
    REAL dy = ((balength * zca - calength * zba) * xcrossbc -
          (balength * xca - calength * xba) * zcrossbc) * denominator;
    REAL dz = ((balength * xca - calength * xba) * ycrossbc -
          (balength * yca - calength * yba) * xcrossbc) * denominator;
    ccent[0] = tri->vrt[0]->crd[0] + dx;
    ccent[1] = tri->vrt[0]->crd[1] + dy;
    ccent[2] = tri->vrt[0]->crd[2] + dz;
  } else {
    assert(0); // an unknown case  
  }

  return 1;
}

//==============================================================================
// Assume all triangles to be checked are in chktris, and are infected.

int Triangulation::repair_triangles(arraypool *enctris)
{
  arraypool *encsegs = new arraypool(sizeof(Triang), 8);
  arraypool *fqueue = new arraypool(sizeof(TriEdge), 8);
  TriEdge E, S, tt[4];
  double v1[2], v2[2];
  double costh = 0;
  //REAL x, y;

  while (enctris->objects > 0) {
    for (int i = 0; i < enctris->used_items; i++) {
      Triang *qtri = (Triang *) enctris->get(i);
      if (qtri->is_deleted()) continue;
      if (get_triangle(qtri->vrt[0], qtri->vrt[1], qtri->vrt[2], E)) {
        // Try to split it.
        bool mtrflag = qtri->is_infected();
        get_tri_circumcenter(qtri, qtri->cct);
        Vertex *newvrt = (Vertex *) tr_steiners->alloc();
        newvrt->init();
        newvrt->crd[0] = qtri->cct[0];
        newvrt->crd[1] = qtri->cct[1];
        //newvrt->crd[2] = 0.; // x*x + y*y;
        //newvrt->crd[2] = op_lambda1 * x*x + op_lambda2 * y*y;
        //if (op_metric <= 1) {
        //  newvrt->crd[2] = qtri->cct[0]*qtri->cct[0]+qtri->cct[1]*qtri->cct[1];
        //} else {
          //newvrt->on_dm = qtri->vrt[0]->on_dm; // For locating the vertex.
          //set_vertex_metric(newvrt);
        //}
        newvrt->idx = io_firstindex + (ct_in_vrts + tr_steiners->objects - 1);
        newvrt->typ = STEINERVERTEX;
        //int liftflag = 0;
        //if (op_metric == METRIC_HDE) {
        //  liftflag = 1;
        //}
        //loc = locate_point(newvrt, E, 0, 1); // encflag = 1
        //int loc = locate_point(newvrt, E, 1, liftflag); // encflag = 1
        int loc = locate_point(newvrt, E, 1); // encflag = 1
        if (loc == LOC_ENC_SEG) {
          // Try to cross a segment (or a non-planar edge). Split it.
          Triang *pE = (Triang *) encsegs->alloc();
          pE->init();
          pE->vrt[0] = E.org();
          pE->vrt[1] = E.dest();
          pE->vrt[2] = NULL;
          if (mtrflag) pE->set_infect(); // due to mesh size.
          newvrt->set_deleted();
          tr_steiners->dealloc(newvrt);
        } else if (loc != LOC_IN_OUTSIDE) {          
          // Insert the vertex.
          assert(!E.tri->is_hulltri());
          newvrt->on_dm = E.tri->vrt[0]->on_dm; // For locating the vertex.
          int iloc = loc;
          set_vertex_metric(newvrt, E, iloc); // interpolate mesh size for the new vertex.
          tt[0] = E;
          if (loc == LOC_IN_TRI) {
            int fflag = FLIP_13;
            flip(tt, &newvrt, fflag, fqueue);
          } else if (loc == LOC_ON_EDGE) {
            int fflag = FLIP_24;
            flip(tt, &newvrt, fflag, fqueue);
          } else {
            assert(0); // on vertex? Must be a bug
          }
          lawson_flip(newvrt, 0, fqueue);
          // Check if this vertex encroaches upon other segments.
          E = newvrt->adj;
          do {
            S = E.enext(); // its oppsite edge.
            if (S.is_segment()) {
              //check_segment(newvrt, S, encsegs);
              v1[0] =  S.org()->crd[0] - newvrt->crd[0];
              v1[1] =  S.org()->crd[1] - newvrt->crd[1];
              v2[0] = S.dest()->crd[0] - newvrt->crd[0];
              v2[1] = S.dest()->crd[1] - newvrt->crd[1];
              costh = v1[0] * v2[0] + v1[1] * v2[1];
              if (costh < 0.) {
                Triang *pE = (Triang *) encsegs->alloc();
                pE->init();
                pE->vrt[0] = S.org();
                pE->vrt[1] = S.dest();
                pE->vrt[2] = NULL;
                if (mtrflag) pE->set_infect(); // due to mesh size.
              }
            }
            E = E.eprev_esym();
          } while (E.tri != newvrt->adj.tri);
          if (encsegs->objects > 0) {
            // Reject this Steiner vertex.
            remove_point(newvrt, fqueue);
            lawson_flip(NULL, 0, fqueue);
          }
        }
        if (encsegs->objects > 0) {
          repair_encsegments(encsegs, enctris);
        } else {
          // A Steiner vertex is inserted.
          enq_vertex_star(newvrt, enctris);
        }
      } // if (get_triangle)
      qtri->set_deleted();
      enctris->dealloc(qtri);
    } // i
  } // while

  delete fqueue;
  delete encsegs;
  return 1;
}

//==============================================================================

// This function implements Ruppert's Delaunay refinement algorithm [1995].
// It uses Si and Gartner [2005]'s CDT edge splitting algorithm to protect
// encroached segments with sharp angles. Miller, Pav, and Walkington [2003]
// showed that Rupper's algorithm terminates as long as the sharp angle is
// no less than 30 degree (theoretically 26.53 degree).

int Triangulation::delaunay_refinement()
{
  //clock_t tv[4]; // Timing informations (defined in time.h)
  //REAL cps = (REAL) CLOCKS_PER_SEC;

  //tv[0] = clock();

  if (tr_steiners == NULL) {
    tr_steiners = new arraypool(sizeof(Vertex), 13); // 2^13 = 8192
  }

  mark_acute_vertices();

  //mesh_phase = 4;

  if ((op_no_gabriel == 0) && (op_no_bisect == 0)) { // no -G and no -Y.
    // Put all encroached segments into queue.
    arraypool *encsegs = new arraypool(sizeof(Triang), 10);

    TriEdge E;
    for (int i = 0; i < tr_tris->objects; i++) {
      E.tri = (Triang *) tr_tris->get(i);
      if (!E.tri->is_deleted()) {
        if (!E.tri->is_hulltri()) { 
          for (E.ver = 0; E.ver < 3; E.ver++) {
            if (E.is_segment()) {
              check_segment(NULL, E, encsegs);
            }
          }
        }
      }
    }

    repair_encsegments(encsegs, NULL);

    delete encsegs;
  }

  if (op_db_verbose) {
    printf("Adding Steiner points to enforce quality.\n");
  }

  if (op_minangle > 0) {
    op_cosminangle = cos(op_minangle / 180.0 * PI);
  }

  // Calculate the square of the maximum radius-edge ratio.
  // r/d = 1/(2*sin(theta_min));
  //op_maxratio2 = 0.5 / sin(PI * op_minangle / 180.0);
  //op_maxratio2 *= op_maxratio2;

  int est_size = tr_tris->objects * 2;
  int log2objperblk = 0;
  while (est_size >>= 1) log2objperblk++;
  if (log2objperblk < 10) log2objperblk = 10; // At least 1024.
  arraypool *enctris = new arraypool(sizeof(Triang), log2objperblk);

  // Put all triangles into list
  for (int i = 0; i < tr_tris->used_items; i++) {
    Triang *tri = (Triang *) tr_tris->get(i);
    if (tri->is_deleted()) continue;
    check_triangle(tri, enctris);
  }

  repair_triangles(enctris);

  delete enctris;

  //tv[1] = clock();
  //if (db_verbose) {
  //  printf("  Delaunay refinement seconds:  %g\n", ((REAL)(tv[1]-tv[0])) / cps);
  //}

  //mesh_phase = 0;

  return 1;
}
