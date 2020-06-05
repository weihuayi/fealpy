#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

//==============================================================================
// see doc/voronoi-edge-boundary-cut-2018-11-1.pdf

int Triangulation::get_boundary_cut_dualedge(Vertex& In_pt, Vertex& Out_pt, TriEdge& S)
{
  if (op_db_verbose > 3) {
    printf("  S.org(%d) -> S.dest(%d)\n", S.org()->idx, S.dest()->idx);
    printf("  In (%g,%g) - Out (%g,%g)\n", In_pt.crd[0], In_pt.crd[1], Out_pt.crd[0], Out_pt.crd[1]);
  }

  REAL ss = Orient2d(S.org(), S.dest(), &In_pt);  
  // assert(ss < 0); // CW orientation
  if (ss >= 0) { // debug this case
    printf("Warning: debugging\n");
    printf("  S.org()=%d, S.dest()=%d isseg(%d)\n", S.org()->idx, S.dest()->idx, S.is_segment());
    printf("  S.org [%g,%g], S.dest [%g,%g]\n",
           S.org()->crd[0], S.org()->crd[1], S.dest()->crd[0], S.dest()->crd[1]);
    printf("  Inpt [%g,%g], Outpt [%g,%g]\n", In_pt.crd[0], In_pt.crd[1], Out_pt.crd[0], Out_pt.crd[1]);
    return 0; //assert(0);
  }

  REAL s1 = Orient2d(&In_pt, &Out_pt, S.org());
  REAL s2 = Orient2d(&In_pt, &Out_pt, S.dest());
  if (op_db_verbose > 3) {
    printf("  [In, out, S.org(%d)]  s1 = %g\n", S.org()->idx, s1);
    printf("  [In, out, S.dest(%d)] s2 = %g\n", S.dest()->idx, s2);
  }

  if (s1 < 0) {
    do {
      // Must go left of S.org();
      TriEdge searchEdge = S.esym();
      //printf("DBG: adjust S to the left of S.org().\n");
      TriEdge workedge = searchEdge.enext();
      while (true) {
        if (op_db_verbose > 3) {
          printf("  workE: (%d,%d,%d)\n", workedge.org()->idx,
                 workedge.dest()->idx, workedge.apex()->idx);
        }
        assert(workedge.org() == S.org());
        searchEdge = workedge.esym_enext(); // CW
        if (searchEdge.is_segment()) break;
        if ((searchEdge.esym()).tri->is_hulltri()) break;
        workedge = searchEdge;
      }
      S = searchEdge.esym(); // Update S.
      s1 = Orient2d(&In_pt, &Out_pt, S.org());
    } while (s1 < 0);
  } else if (s2 > 0) {
    // Muts go right of S.dest();
    //printf("DBG: adjust S to the right of S.dest().\n");
    do {
      TriEdge searchEdge = S.esym();
      TriEdge workedge = searchEdge.eprev();
      while (true) {
        if (op_db_verbose > 3) {
          printf("  workE: (%d,%d,%d)\n", workedge.org()->idx,
                 workedge.dest()->idx, workedge.apex()->idx);
        }
        assert(workedge.dest() == S.dest());
        searchEdge = workedge;
        if (searchEdge.is_segment()) break;
        if ((searchEdge.esym()).tri->is_hulltri()) break;
        workedge = searchEdge.esym_eprev();
      }
      S = searchEdge.esym(); // Update S
      s2 = Orient2d(&In_pt, &Out_pt, S.dest());
    } while (s2 > 0);
  }

  if (op_db_verbose > 3) {
    printf("  Found background boundary S: [%d,%d,%d]\n\n",
           S.org()->idx, S.dest()->idx, S.apex()->idx);
  }

  return 1;
}

//==============================================================================

int Triangulation::get_hulltri_orthocenter(Triang* hulltri)
{
  TriEdge E; E.tri = hulltri;
  for (E.ver = 0; E.ver < 3; E.ver++) {
    if (E.apex() == tr_infvrt) break;
  }
  assert(E.ver < 3);

  REAL Ux, Uy, U_weight;
  REAL Vx, Vy, V_weight;

  Vertex *v1 = E.org();
  Vertex *v2 = E.dest();

  Ux = v1->crd[0];
  Uy = v1->crd[1];
  Vx = v2->crd[0];
  Vy = v2->crd[1];

  // Translate heights to weights.
  U_weight = v1->wei; //Ux*Ux + Uy*Uy - v1->crd[2];
  V_weight = v2->wei; //Vx*Vx + Vy*Vy - v2->crd[2];

  if (op_db_verbose > 2) {
    printf("A Hull edge:\n");
    printf(" [%d]: %g,%g,%g, r(%g)\n", v1->idx, Ux, Uy, U_weight, sqrt(fabs(U_weight)));
    printf(" [%d]: %g,%g,%g, r(%g)\n", v2->idx, Vx, Vy, V_weight, sqrt(fabs(V_weight)));
  }

  get_bissector(Ux, Uy, U_weight,
                Vx, Vy, V_weight,
                &E.tri->cct[0], &E.tri->cct[1], &E.tri->cct[2],
                _a11, _a21, _a22);

  if (OMT_domain) {
    // The bisector lies exactly in the edge [v1, v2].
    // We move it along the edge normal towards exterior of the triangulaiton.
    TriEdge N = E.esym();
    // The dual vertex of N.tri was already calcualted.
    assert(N.tri->on.tri != NULL); 
    if (op_db_verbose > 2) {
      printf("  Bisector on the hull edge:  [%g,%g]\n", E.tri->cct[0], E.tri->cct[1]);
      printf("  Ccenter of hull triangle N: [%g,%g]\n", N.tri->cct[0], N.tri->cct[1]);
      //printf("p:show_vector(%g,%g,0,%g,%g,0)\n",
      //       N.tri->cct[0], N.tri->cct[1], E.tri->cct[0], E.tri->cct[1]);
    }

    // Get the edge normal towards exterior (rotated 90 deg).    
    double outerN[2];
    //outerN[0] =  (E.org()->crd[1] - E.dest()->crd[1]);
    //outerN[1] = -(E.org()->crd[0] - E.dest()->crd[0]);
    //double len = sqrt(outerN[0]*outerN[0] + outerN[1]*outerN[1]);
    //outerN[0] /= len;
    //outerN[1] /= len;

    // Calculate the bisector line (may not be orthogonal if metric is not Euclidean).
    // The line direction vector.
    outerN[0] =   2.*_a22*(Uy-Vy) + 2.*_a21*(Ux-Vx);
    outerN[1] = -(2.*_a11*(Ux-Vx) + 2.*_a21*(Uy-Vy));
    double len = sqrt(outerN[0]*outerN[0] + outerN[1]*outerN[1]);
    outerN[0] /= len;
    outerN[1] /= len;
    // Choose the line direction towards exterior.
    Vertex tmp_pt; // Construct a point on the bisector line.
    tmp_pt.init();
    tmp_pt.crd[0] = E.tri->cct[0] + 2 * len * outerN[0];
    tmp_pt.crd[1] = E.tri->cct[1] + 2 * len * outerN[1];
    REAL check_sign = Orient2d(E.org(), E.dest(), &tmp_pt);
    if (check_sign < 0) {
      // not ccw orientation. reverse the direction.
      outerN[0] = -outerN[0];
      outerN[1] = -outerN[1];
    } else if (check_sign == 0) {
      assert(0); // this is a bug.
    }

    if (op_dt_nearest < 0) { // The farthest-point Voronoi diagram
      // change direction of the outer normal.
      outerN[0] = -outerN[0];
      outerN[1] = -outerN[1];
    }

    len = get_distance(E.org(), E.dest());

    if (!N.tri->is_dual_in_exterior() && !N.tri->is_dual_on_bdry()) {
      // This dual vertex lies inside the domain.
      if (!N.is_segment()) {
        // Move the dual vertex to the boundary of the OMT_domain.
        // For this we need a far point outside the OMT_domain.
        double scale = len;
        /*
        Vertex Mass_pt; // pt;
        REAL Wx = N.apex()->crd[0];
        REAL Wy = N.apex()->crd[1];
        Mass_pt.crd[0] = (Ux + Vx + Wx) / 3.;
        Mass_pt.crd[1] = (Uy + Vy + Wy) / 3.;
        TriEdge S = N.tri->on; // Must be an interior triangle of OMT_domain.
        int loc = OMT_domain->locate_point(&Mass_pt, S, 0, 0);
        */
        Vertex In_pt; // The bisector of this hull edge.
        In_pt.init();
        In_pt.crd[0] = E.tri->cct[0];
        In_pt.crd[1] = E.tri->cct[1];
        TriEdge S = N.tri->on; // Must be an interior triangle of OMT_domain.
        if (op_db_verbose > 2) {
          printf("  N.tri->on (cct) background tri: [%d,%d,%d]\n", S.org()->idx, S.dest()->idx, S.apex()->idx);
        }
        //int loc = OMT_domain->locate_point(&In_pt, S, 0, 0);
        int loc = OMT_domain->locate_point(&In_pt, S, 0);
        if (op_db_verbose > 2) {
          printf("  In_pt on background tri: [%d,%d,%d]\n", S.org()->idx, S.dest()->idx, S.apex()->idx);
        }
        assert(loc != LOC_IN_OUTSIDE);
        Vertex Out_pt;
        Out_pt.init();
        while (1) { // we must find a point.
          scale = scale * 2.0;
          Out_pt.crd[0] = E.tri->cct[0] + scale * outerN[0];
          Out_pt.crd[1] = E.tri->cct[1] + scale * outerN[1];
          // Search the Out_pt, stop at the first segment.
          //loc = OMT_domain->locate_point(&Out_pt, S, 0, 1);
          loc = OMT_domain->locate_point(&Out_pt, S, 1);
          if (loc == LOC_IN_OUTSIDE) {
            assert(S.apex() == OMT_domain->tr_infvrt);
            break;
          } else if (loc == LOC_IN_TRI) {
            if (S.tri->is_exterior()) break;
          } else if (loc == LOC_ENC_SEG) {
            assert(S.is_segment());
            break; // hit a segment.
          }
        } // while (1)
        if (op_db_verbose > 3) {
          printf("  Background boundary: [%d,%d,%d]\n", S.org()->idx, S.dest()->idx, S.apex()->idx);
        }
        //assert(S.is_segment()); // Debug
        get_boundary_cut_dualedge(In_pt, Out_pt, S);
        /*
        // Check if S [org, dest] cuts the line [In_pt, Out_pt].
        //===================== Subroutine start =================================
        // see doc/voronoi-edge-boundary-cut-2018-11-1.pdf
        REAL ss = Orient2d(S.org(), S.dest(), &In_pt);
        assert(ss < 0); // Bisect_pt must be CW oriented (inside of OMT_domain).
        REAL s1 = Orient2d(&In_pt, &Out_pt, S.org());
        REAL s2 = Orient2d(&In_pt, &Out_pt, S.dest());
        if (op_db_verbose > 2) {
          printf("  [In, out, S.org(%d)]  s1 = %g\n", S.org()->idx, s1);
          printf("  [In, out, S.dest(%d)] s2 = %g\n", S.dest()->idx, s2);
        }
        if (s1 < 0) {
          do {
            // Must go left of S.org();
            TriEdge searchEdge = S.esym();
            //printf("DBG: adjust S to the left of S.org().\n");
            TriEdge workedge = searchEdge.enext();
            while (true) {
              if (op_db_verbose > 3) {
                printf("  workE: (%d,%d,%d)\n", workedge.org()->idx,
                       workedge.dest()->idx, workedge.apex()->idx);
              }
              assert(workedge.org() == S.org());
              searchEdge = workedge.esym_enext(); // CW
              if (searchEdge.is_segment()) break;
              if ((searchEdge.esym()).tri->is_hulltri()) break;
              workedge = searchEdge;
            }
            S = searchEdge.esym(); // Update S.
            s1 = Orient2d(&In_pt, &Out_pt, S.org());
          } while (s1 < 0);
        } else if (s2 > 0) {
          // Muts go right of S.dest();
          //printf("DBG: adjust S to the right of S.dest().\n");
          do {
            TriEdge searchEdge = S.esym();
            TriEdge workedge = searchEdge.eprev();
            while (true) {
              if (op_db_verbose > 3) {
                printf("  workE: (%d,%d,%d)\n", workedge.org()->idx,
                       workedge.dest()->idx, workedge.apex()->idx);
              }
              assert(workedge.dest() == S.dest());
              searchEdge = workedge;
              if (searchEdge.is_segment()) break;
              if ((searchEdge.esym()).tri->is_hulltri()) break;
              workedge = searchEdge.esym_eprev();
            }
            S = searchEdge.esym(); // Update S
            s2 = Orient2d(&In_pt, &Out_pt, S.dest());
          } while (s2 > 0);
        }
        if (op_db_verbose > 3) {
          printf("  Found background boundary S: [%d,%d,%d]\n\n",
                 S.org()->idx, S.dest()->idx, S.apex()->idx);
        }
        //===================== Subroutine end =================================
        */
        E.tri->on = S; // Remember this triangle in OMT_domain.

        // Calculate the cut point.
        Vertex *e1 = E.tri->on.org();
        Vertex *e2 = E.tri->on.dest();
        double X0 = N.tri->cct[0];
        double Y0 = N.tri->cct[1];
        double X1 = Out_pt.crd[0];
        double Y1 = Out_pt.crd[1];
        double X2 = e1->crd[0];
        double Y2 = e1->crd[1];
        double X3 = e2->crd[0];
        double Y3 = e2->crd[1];
        double t1, t2;
        if (line_line_intersection(X0, Y0, X1, Y1, X2, Y2, X3, Y3, &t1, &t2)) {
          E.tri->cct[0] = X0 + t1 * (X1 - X0);
          E.tri->cct[1] = Y0 + t1 * (Y1 - Y0);
          // Calulcate rr2. (the weight)
          // The weighted point of this cutpoint must lie on the polar plane
          //   of the weighted point V1 (or V2).
          // The weight of this cut-point can be found as following:
          //   - project this point (E.tri->cct[0], E.tri->cct[1]) vertically to
          //     the tangent plane passing through the lifted point of V1 (or V2),
          //   - then caculate the vertical distance from projection point to the
          //     lifted point of the paraboloid.
          // Let a = e1->crd[0], b = e1->crd[1], wei=e1->wei
          // The tangent plane at e1' is:
          //    z = 2ax + 2by - (a^2+b^2) + wei
          // Let \alpha = E.tri->cct[0], \beta= E.tri->cct[1]
          //    z_proj = 2a(\alpha)+2b(\beta)-(a^2+b^2) + wei
          // Then the weight of the cut-point is:
          //    (\alpha * \alpha) + (\beta * \beta) - z_proj
          double a = Ux; // e1->crd[0];
          double b = Uy; // e1->crd[1];
          double alpha = E.tri->cct[0];
          double beta  = E.tri->cct[1];
          double z_proj = 2.*a*alpha+2.*b*beta - (a*a+b*b) + U_weight;
          E.tri->cct[2] = alpha*alpha+beta*beta - z_proj; // its weight
        } else {
          printf("!! Warning: Failed at calculating line-line intersection.\n");
        }
      } // if (!N.is_segment()
      else {
        // [2019-07-28] Remember this boundary segment.
        E.tri->on = N.esym();
      }
      // It is either on a segment or a hull edge of OMT_domain.
      E.tri->set_dual_on_bdry();
    } else {
      // The dual vertex of its neighbor lies outside.
      // Move this dual vertex further than the neighbor dual vertex.
      // This is only for visualization useful.
      E.tri->cct[0] = N.tri->cct[0] + len * outerN[0];
      E.tri->cct[1] = N.tri->cct[1] + len * outerN[1];
      E.tri->cct[2] = 0; // will be calulcated later.
      E.tri->on = N.tri->on;
      E.tri->set_dual_in_exterior();
    }
    if (op_db_verbose > 10) {
      printf("p:show_vector(%g,%g,0,%g,%g,0)\n", N.tri->cct[0], N.tri->cct[1],
             E.tri->cct[0], E.tri->cct[1]);
    }
  } // if (OMT_domain)

  return 1;
}

// "tri" must be not a hulltri and exterior
int Triangulation::get_tri_orthocenter(Triang* tri)
{
  assert(!tri->is_hulltri());

  REAL Ux, Uy, U_weight;
  REAL Vx, Vy, V_weight;
  REAL Wx, Wy, W_weight;

  Vertex *v1 = tri->vrt[0];
  Vertex *v2 = tri->vrt[1];
  Vertex *v3 = tri->vrt[2];

  Ux = v1->crd[0];
  Uy = v1->crd[1];
  Vx = v2->crd[0];
  Vy = v2->crd[1];
  Wx = v3->crd[0];
  Wy = v3->crd[1];

  // Translate heights to weights.
  U_weight = v1->wei; //Ux*Ux + Uy*Uy - v1->crd[2];
  V_weight = v2->wei; //Vx*Vx + Vy*Vy - v2->crd[2];
  W_weight = v3->wei; //Wx*Wx + Wy*Wy - v3->crd[2];

  if (op_db_verbose > 2) {
    printf(" [%d]: %g,%g,%g\n", v1->idx, Ux, Uy, U_weight);
    printf(" [%d]: %g,%g,%g\n", v2->idx, Vx, Vy, V_weight);
    printf(" [%d]: %g,%g,%g\n", v3->idx, Wx, Wy, W_weight);
  }

  get_orthocenter(Ux, Uy, U_weight,
                  Vx, Vy, V_weight,
                  Wx, Wy, W_weight,
                  &tri->cct[0], &tri->cct[1], &tri->cct[2],
                  _a11, _a21, _a22);

  if (OMT_domain != NULL) {
    // Determine if this vertex is inside or outside of the OMT domain (or the subdomain
    //   which contains this triangle -- check segments).
    Vertex Mass_pt, pt;
    Mass_pt.crd[0] = (Ux + Vx + Wx) / 3.;
    Mass_pt.crd[1] = (Uy + Vy + Wy) / 3.;
    pt.crd[0] = tri->cct[0];
    pt.crd[1] = tri->cct[1];
    // First search the mass center of "tri" in OMT_domain.
    // It must be inside the domain.
    //int loc = OMT_domain->locate_point(&Mass_pt, tri->on, 0, 0); // encflag = 0
    int loc = OMT_domain->locate_point(&Mass_pt, tri->on, 0); // encflag = 0
    assert(!tri->on.tri->is_hulltri()); 
    // Locate the orthocenter of "tri" from this triangle (containing its mass center). 
    //loc = OMT_domain->locate_point(&pt, tri->on, 0, 1); // encflag = 1 (stop at first segment).
    loc = OMT_domain->locate_point(&pt, tri->on, 1); // encflag = 1 (stop at first segment).
    // Set the interior /exterior flag of the dual vertex.
    // Default, the dual (orothocenter) is in the interior.
    if (loc == LOC_IN_OUTSIDE) {
      tri->set_dual_in_exterior();
    } else if (loc == LOC_IN_TRI) {
      if (tri->on.tri->is_exterior()) {
        tri->set_dual_in_exterior();
      }
    } else if (loc == LOC_ON_EDGE) {
      if (tri->on.is_segment()) {
        tri->set_dual_on_bdry();
      }
    } else if (loc == LOC_ON_VERT) {
      // On a vertex of the background mesh, to be done.
      TriEdge E = tri->on;
      do {
        if (tri->on.tri->is_hulltri() || tri->on.tri->is_exterior()) {
          tri->set_dual_in_exterior(); break;
        } else if (tri->on.is_segment()) {
          tri->set_dual_on_bdry(); break;
        }
        tri->on = tri->on.eprev_esym(); // CCW rotate
        assert(tri->on.org() == E.org());
      } while (E.tri != tri->on.tri);
    } else if (loc == LOC_ENC_SEG) {
      // This dual vertex is behind a segment, which means it lies outside of the
      //   subdomain contains the triangle. We treat it as lying in outside.
      //   This way, a cut point on the segment will be calculated for Voronoi cells.
      assert(tri->on.is_segment());
      tri->set_dual_in_exterior();
    } 
  } // if (OMT_domain != NULL)

  return 1;
}

//==============================================================================
// The power cell of an interior mesh vertex is given by the convex hull of a
//   set of corners, which are Voronoi vertices and cutting vertices (of Voronoi
//   edges with domain boundary edges). The set of corners are returned by
//   an array ``pptlist" ordered in counterclockwise direction.
//   It assumes that a ``OMT_domain" must be provided.
//
// see doc/doc-get_powercell.key

int Triangulation::get_powercell(Vertex *mesh_vertex, Vertex** pptlist, int* ptnum)
{
  // A background mesh must be provided (maybe itself).
  assert(OMT_domain != NULL);
  *pptlist = NULL;
  *ptnum = 0;

  TriEdge E = mesh_vertex->adj;
  if (E.tri == NULL) {
    // This vertex does not has an adjacent triangle.
    // It is either an unused vertex or this triangulation is empty.
    return 0;
  }
  if (E.tri->is_deleted()) {
    // This should be a bug.
    assert(0);
  }

  int vcount = 0; // Count the number of Voronoi vertices.
  do {
    vcount++;
    //if (E.is_segment()) { //if (E.tri->is_hulltri()) {
    //  if (op_db_verbose > 3) {
    //    printf("  Vertex %d is a boundary (fixed) vertex. Skipped.\n", mesh_vertex->idx);
    //  }
    //  return 0;
    //}
    E = E.eprev_esym(); // ccw
  } while (E.tri != mesh_vertex->adj.tri);

  if (op_db_verbose > 1) {
    printf("  Found %d adjacent triangles at vertex %d.\n", vcount, mesh_vertex->idx);
  }

  /*  We assume circumcenters are already calculated.
  if (calcflag) {
    // Calculate the circumcenters of triangles at this vertex.
    TriEdge E = mesh_vertex->adj;
    do {
      if (!E.tri->is_hulltri()) {
        get_tri_orthocenter(E.tri);
      }
      E = E.eprev_esym(); // ccw
    } while (E.tri != mesh_vertex->adj.tri);

    E = mesh_vertex->adj;
    do {
      if (E.tri->is_hulltri()) {
        get_hulltri_orthocenter(E.tri);
      }
      E = E.eprev_esym(); // ccw
    } while (E.tri != mesh_vertex->adj.tri);
  }
  */

  // N_Last_Exit is a boundary edge or a hull edge ([a,b,-1]) of the OMT_domain.
  //   If it is a boundary edge which is not a hull edge, i.e., [a,b,c], c != -1,
  //   it must be an interior segment which separates two subdomains. Then the
  //   vertex 'c' must be in the other subdomain different to this power cell.
  TriEdge N, N_Last_Exit; // [a,b,-1] or [a,b,c]
  N_Last_Exit.tri = NULL;

  // We need to start from an interior dual edge -- need one interior Voronoi vertex.
  // So the initial condition (N_Last_Exit.tri == NULL) is valid.
  // [2019-07-28] We must also check segment.
  //printf("\n DBG start: \n");
  E = mesh_vertex->adj;
  N = E.esym();
  bool is_interor_dual_edge = (!N.tri->is_dual_in_exterior() && !N.tri->is_dual_on_bdry());
  if (!is_interor_dual_edge) {
    do {
      //printf("  E: [%d,%d,%d] %d\n", E.org()->idx, E.dest()->idx, E.apex()->idx, E.triis_dual_in_exterior());
      //printf("  N: [%d,%d,%d] %d\n", N.org()->idx, N.dest()->idx, N.apex()->idx, N.triis_dual_in_exterior());
      E = E.eprev_esym(); // ccw
      N = E.esym();
      is_interor_dual_edge = (!N.tri->is_dual_in_exterior() && !N.tri->is_dual_on_bdry());
      if (is_interor_dual_edge) break;
    } while (E.tri != mesh_vertex->adj.tri);
    mesh_vertex->adj = E; // Must update this.
  }
  //printf(" DBG end: \n");
  if (!is_interor_dual_edge) {    
    if (op_db_verbose) {
      printf("  All dual edges of this cell lie ouside of the OMT_domain.\n");
    }
    return 0;
  }

  // The weight of this cut-point can be found as following:
  //   - project this point (E.tri->cct[0], E.tri->cct[1]) vertically to
  //     the tangent plane passing through the lifted point of e1 (or e2),
  //   - then caculate the vertical distance from projection point to the
  //     lifted point of the paraboloid.
  // Let a = mesh_vertex->crd[0], b = mesh_vertex->crd[1], wei=mesh_vertex->wei
  // The polar (tangent) plane at of the lifted mesh_vertex' is:
  //    z = 2ax + 2by - (a^2+b^2) + wei
  // Let \alpha = cutpoint->crd[0], \beta= cutpoint->crd[1]
  //    z_proj = 2a(\alpha)+2b(\beta)-(a^2+b^2) + wei
  // Then the weight of the cut-point is:
  //    (\alpha * \alpha) + (\beta * \beta) - z_proj
  double a = mesh_vertex->crd[0];
  double b = mesh_vertex->crd[1];
  double wei = mesh_vertex->wei;
  //double alpha, beta, z_proj;

  // Use an array to save the list of Voronoi or cut vertices (in ccw order).
  // [Remark] Due to the possible cut vertcies, the total number of vertcies
  //   of a power cell may be more than vcount. However, it must be lower than
  //   vcount + hullsize (of OMT)
  Vertex *ptlist = new Vertex[vcount + OMT_domain->ct_hullsize];
  int ptcount = 0;

  do {
    if (op_db_verbose > 3) {
      printf("  Get tri: [%d,%d,%d].\n", E.org()->idx, E.dest()->idx, E.apex()->idx);
    }
    N = E.esym();

    if (E.tri->is_dual_in_exterior()) {
      if (N.tri->is_dual_in_exterior() || N.tri->is_dual_on_bdry()) {
        // We're still walking in OUTSIDE. (see Page 3, v3->v4)
        assert(N_Last_Exit.tri != NULL);
      } else {
        // N's dual vertex lies inside the domain. (see Page 3, v2->v3)
        // We're walking from INSIDE to OUTSIDE.
        assert(N_Last_Exit.tri == NULL);
        // Calculating a cut vertex between this dual edge (from N.tri->cct to E.tri->cct)
        //    and a boundary edge (or a segment) S of the background mesh.
        //    We must ensure that the dual edge and S intersect in S's interior.
        TriEdge S = E.tri->on;
        Vertex In_pt, Out_pt;
        In_pt.init();
        Out_pt.init();
        In_pt.crd[0] = N.tri->cct[0]; // see Page 3, v2
        In_pt.crd[1] = N.tri->cct[1];
        Out_pt.crd[0] = E.tri->cct[0]; // see Page 3, v3
        Out_pt.crd[1] = E.tri->cct[1];

        if (!get_boundary_cut_dualedge(In_pt, Out_pt, S)) { // see Page 3, S = {b1, b2}.
          // Failed to get this boundary edge which cuts the dual edge.
          //assert(0); // should be a bug.
          delete [] ptlist;          
          return 0;
        }

        // Calculate the cut vertex, see Page 3, cut1.
        E.tri->on = S; // Update E.tri->on
        //printf("  Last Exit [%d,%d,%d]\n", S.org()->idx, S.dest()->idx, S.apex()->idx);
        
        Vertex *e1 = E.tri->on.org();
        Vertex *e2 = E.tri->on.dest();
        //assert((e1 != NULL) && (e2 != NULL));
        double X0 = E.tri->cct[0];
        double Y0 = E.tri->cct[1];
        double X1 = N.tri->cct[0];
        double Y1 = N.tri->cct[1];
        double X2 = e1->crd[0];
        double Y2 = e1->crd[1];
        double X3 = e2->crd[0];
        double Y3 = e2->crd[1];
        double t1 = 0, t2 = 0;
        if (line_line_intersection(X0, Y0, X1, Y1, X2, Y2, X3, Y3, &t1, &t2)) {
          double c0 = X0 + t1 * (X1 - X0);
          double c1 = Y0 + t1 * (Y1 - Y0);
          // Calculate the weight
          double z_proj = 2.*a*c0+2.*b*c1 - (a*a+b*b) + wei;
          double c2 = c0*c0+c1*c1 - z_proj;
          /*
          x1 = (int) ( Sxy * N.tri->cct[0] + Cx);
          y1 = (int) (-Sxy * N.tri->cct[1] + Cy);
          x2 = (int) ( Sxy * c0 + Cx);
          y2 = (int) (-Sxy * c1 + Cy);
          painter->drawLine(QPoint(x1, y1), QPoint(x2,y2));
          */
          Vertex *pt = &(ptlist[ptcount]);
          pt->init();
          pt->crd[0] = c0;
          pt->crd[1] = c1;
          pt->crd[2] = c2; // its weight
          pt->tag = -1; // A cut vertex.
          pt->idx = ptcount+1;
          ptcount++;
        }

        // Remember this (exit) boundary edge.
        N_Last_Exit = E.tri->on;
        //printf("  2 Last Exit [%d,%d,%d]\n", N_Last_Exit.org()->idx,
        //       N_Last_Exit.dest()->idx, N_Last_Exit.apex()->idx);
      }
    } else if (E.tri->is_dual_on_bdry()) {
      if (N.tri->is_dual_in_exterior() || N.tri->is_dual_on_bdry()) {
        // We're walking from OUTSIDE to INSIDE.
        // (see Page 2, E.tri (p,-1,p2)'s cct is v3, N.tri (p,p1,-1)'s cct is v2.)
        assert(N_Last_Exit.tri != NULL);
        // Since E's dual vertex lies exactly on boundary, then E.tri->on must contain this dual vertex.
        //   This is guaranteed by the function get_hulltri_orthocenter().
        //   No need to call get_boundary_cut_dualedge().
        //printf("\n DBG start: \n");
        //printf("  N_Last_Exit: [%d,%d,%d]\n", N_Last_Exit.org()->idx, N_Last_Exit.dest()->idx, N_Last_Exit.apex()->idx);
        //printf("  E.tri->on:   [%d,%d,%d]\n", E.tri->on.org()->idx, E.tri->on.dest()->idx, E.tri->on.apex()->idx);
        //printf("\n DBG end: \n");
        if ((E.tri->on.tri != N_Last_Exit.tri) ||
            ((E.tri->on.tri == N_Last_Exit.tri) && (E.tri->on.ver != N_Last_Exit.ver))) {
          // They are on different boundary edges.
          // (see Page 2, the bdry edges containing v2 and v3 are different.)
          // There might be corner vertices to be the vertices of this cell.
          //printf("  !!! Searching background corners (2) DEBUG!!!\n");
          // [Important] The search is by rotating the triangles in the inside of the
          //   power cell. From startEdge to endEdge.
          TriEdge startEdge, endEdge, searchEdge;
          startEdge = N_Last_Exit.esym();
          endEdge = E.tri->on.esym();
          //===================== Subroutine start =================================
          // Start searching corner vertices from startEdge towards endEdge.
          // (see Page 2 for an example.)
          searchEdge = startEdge;
          while (searchEdge.tri != endEdge.tri) {
            // Found a corner vertex.
            Vertex *pt = &(ptlist[ptcount]);
            pt->init();
            pt->crd[0] = searchEdge.dest()->crd[0];
            pt->crd[1] = searchEdge.dest()->crd[1];
            // Calculate its weight
            double alpha = pt->crd[0];
            double beta  = pt->crd[1];
            double z_proj = 2.*a*alpha+2.*b*beta - (a*a+b*b) + wei;
            pt->crd[2] = alpha*alpha+beta*beta - z_proj; // its weight
            pt->tag = -2; // A background vertex.
            pt->idx = ptcount+1;
            ptcount++;
            // Go to the next boundary edge (around dest()).
            TriEdge workedge = searchEdge.enext();
            while (true) {
              searchEdge = workedge.esym_enext(); // CW
              if (searchEdge.is_segment()) break;
              if ((searchEdge.esym()).tri->is_hulltri()) break;
              workedge = searchEdge;
            }
            //if (searchEdge.is_segment()) {
            //  // hit a segment, we must stop.
            //  // It is possible that this segment is not the endEdge.
            //  break;
            //}
          }
          //printf("Done!\n");
          //===================== Subroutine end =================================
        }
        // Clear it. We're walking from OUTSIDE to INSIDE already.
        N_Last_Exit.tri = NULL;
      } else {
        // N's dual vertex lies inside the domain.
        // We're walking to the direction from INSIDE to OUTSIDE.
        // (see Page 2, E.tri (p,p1,-1)'s cct is v2, N.tri (p,p3,p1)'s cct is v1.)
        // Remeber this boundary edge.
        assert(N_Last_Exit.tri == NULL);
        N_Last_Exit = E.tri->on; // (see Page 2, N_Last_Exit = startEdge.)
        //printf("  2 Last Exit [%d,%d,%d]\n", N_Last_Exit.org()->idx,
        //       N_Last_Exit.dest()->idx, N_Last_Exit.apex()->idx);
        /*
        // They are in the same subdomain.
        x1 = (int) ( Sxy * E.tri->cct[0] + Cx);
        y1 = (int) (-Sxy * E.tri->cct[1] + Cy);
        x2 = (int) ( Sxy * N.tri->cct[0] + Cx);
        y2 = (int) (-Sxy * N.tri->cct[1] + Cy);
        painter->drawLine(QPoint(x1, y1), QPoint(x2,y2));
        */
      }

      Vertex *pt = &(ptlist[ptcount]);
      pt->init();
      pt->crd[0] = E.tri->cct[0]; // see Page 2, v2
      pt->crd[1] = E.tri->cct[1];
      // Calculate its weight
      double alpha = pt->crd[0];
      double beta  = pt->crd[1];
      double z_proj = 2.*a*alpha+2.*b*beta - (a*a+b*b) + wei;
      pt->crd[2] = alpha*alpha+beta*beta - z_proj; // its weight
      //pt->crd[2] = E.tri->cct[2]; // its weight
      pt->tag = E.tri->idx; // A Voronoi vertex.
      pt->idx = ptcount+1;
      ptcount++;
    } else {
      // E's dual vertex lies inside the domain (or subdomain).
      // There must exist a (or part of) dual edge.
      if (!N.tri->is_dual_in_exterior() || N.tri->is_dual_on_bdry()) {
        // We're walking inside of the OMT_domain.
        assert(N_Last_Exit.tri == NULL);
        /*
        // An interior voronoi edge.
        x1 = (int) ( Sxy * E.tri->cct[0] + Cx);
        y1 = (int) (-Sxy * E.tri->cct[1] + Cy);
        x2 = (int) ( Sxy * N.tri->cct[0] + Cx);
        y2 = (int) (-Sxy * N.tri->cct[1] + Cy);
        painter->drawLine(QPoint(x1, y1), QPoint(x2,y2));
        */
      } else {
        // The adjacent voronoi vertex is in exterior of the OMT.
        // We're walking from OUTSIDE to INSIDE.
        // (see Page 3, E,tri (p,p5,p6)'cct is v5, N.tri (p,p4,p5)'cct is v4,
        //    the walk is from v4 to v5.)
        assert(N_Last_Exit.tri != NULL);
        // Calculating a cut vertex between this dual edge (from N.tri->cct to E.tri->cct)
        //    and a boundary edge (or a segment) S of the background mesh.
        //    We must ensure that the dual edge and S intersect in S's interior.
        TriEdge S = N.tri->on;
        Vertex In_pt, Out_pt;
        In_pt.init();
        Out_pt.init();
        In_pt.crd[0] = E.tri->cct[0]; // see Page 3, v5
        In_pt.crd[1] = E.tri->cct[1];
        Out_pt.crd[0] = N.tri->cct[0]; // see Page 3, v4
        Out_pt.crd[1] = N.tri->cct[1];

        if (!get_boundary_cut_dualedge(In_pt, Out_pt, S)) {
          delete [] ptlist;
          return 0;
        }

        N.tri->on = S; // Update N.tri->on

        //printf("\n DBG start: \n");
        //printf("  N_Last_Exit: [%d,%d,%d]\n", N_Last_Exit.org()->idx, N_Last_Exit.dest()->idx, N_Last_Exit.apex()->idx);
        //printf("  N.tri->on:   [%d,%d,%d]\n", N.tri->on.org()->idx, N.tri->on.dest()->idx, N.tri->on.apex()->idx);
        //printf("\n DBG end: \n");
        if ((N.tri->on.tri != N_Last_Exit.tri) ||
            ((N.tri->on.tri == N_Last_Exit.tri) && (N.tri->on.ver != N_Last_Exit.ver))) {
          // They are on different boundary.
          // There might be corner vertices to be the vertices of this cell.
          //printf("  !!! Searching background corners (2) DEBUG!!!\n");
          // (see Page 2 for an example.)
          TriEdge startEdge, endEdge, searchEdge;
          startEdge = N_Last_Exit.esym();
          endEdge = N.tri->on.esym();
          //===================== Subroutine start =================================
          // Start searching corner vertices from startEdge towards endEdge.
          searchEdge = startEdge;
          while (searchEdge.tri != endEdge.tri) {
            // Found a corner vertex.
            Vertex *pt = &(ptlist[ptcount]);
            pt->init();
            pt->crd[0] = searchEdge.dest()->crd[0];
            pt->crd[1] = searchEdge.dest()->crd[1];
            // Calculate its weight
            double alpha = pt->crd[0];
            double beta  = pt->crd[1];
            double z_proj = 2.*a*alpha+2.*b*beta - (a*a+b*b) + wei;
            pt->crd[2] = alpha*alpha+beta*beta - z_proj; // its weight
            pt->tag = -2; // A background vertex.
            pt->idx = ptcount+1;
            ptcount++;
            // Go to the next boundary edge (around dest()).
            TriEdge workedge = searchEdge.enext();
            while (true) {
              searchEdge = workedge.esym_enext(); // CW
              if (searchEdge.is_segment()) break;
              if ((searchEdge.esym()).tri->is_hulltri()) break;
              workedge = searchEdge;
            }
            //if (searchEdge.is_segment()) {
            //  // to do ...
            //}
          }
          //printf("Done!\n");
          //===================== Subroutine end =================================
        }

        // Calculate the cut2 vertex (see Page 3, cut2)
        Vertex *e1 = N.tri->on.org();
        Vertex *e2 = N.tri->on.dest();
        assert((e1 != NULL) && (e2 != NULL));
        double X0 = E.tri->cct[0];
        double Y0 = E.tri->cct[1];
        double X1 = N.tri->cct[0];
        double Y1 = N.tri->cct[1];
        double X2 = e1->crd[0];
        double Y2 = e1->crd[1];
        double X3 = e2->crd[0];
        double Y3 = e2->crd[1];
        double t1 = 0, t2 = 0;
        if (line_line_intersection(X0, Y0, X1, Y1, X2, Y2, X3, Y3, &t1, &t2)) {
          double c0 = X0 + t1 * (X1 - X0);
          double c1 = Y0 + t1 * (Y1 - Y0);
          // Calculate weight
          double z_proj = 2.*a*c0+2.*b*c1 - (a*a+b*b) + wei;
          double c2 = c0*c0+c1*c1 - z_proj;
          /*
          x1 = (int) ( Sxy * E.tri->cct[0] + Cx);
          y1 = (int) (-Sxy * E.tri->cct[1] + Cy);
          x2 = (int) ( Sxy * c0 + Cx);
          y2 = (int) (-Sxy * c1 + Cy);
          painter->drawLine(QPoint(x1, y1), QPoint(x2,y2));
          */
          Vertex *pt = &(ptlist[ptcount]);
          pt->init();
          pt->crd[0] = c0;
          pt->crd[1] = c1;
          pt->crd[2] = c2; // its weight
          pt->tag = -1; // A cut vertex.
          pt->idx = ptcount+1;
          ptcount++;
        }
      }

      Vertex *pt = &(ptlist[ptcount]);
      pt->init();
      pt->crd[0] = E.tri->cct[0]; // see Page 3, v5
      pt->crd[1] = E.tri->cct[1];
      // Calculate its weight
      double alpha = pt->crd[0];
      double beta  = pt->crd[1];
      double z_proj = 2.*a*alpha+2.*b*beta - (a*a+b*b) + wei;
      pt->crd[2] = alpha*alpha+beta*beta - z_proj;
      //pt->crd[2] = E.tri->cct[1]; // its weight
      pt->tag = E.tri->idx; // A Voronoi vertex.
      pt->idx = ptcount+1;
      ptcount++;

      // Clear it, we're inside the OMT_domain again.
      N_Last_Exit.tri = NULL;
    }
  
    // Go to the next edge.
    E = E.eprev_esym(); // ccw
  } while (E.tri != mesh_vertex->adj.tri);

  *pptlist = ptlist;
  *ptnum = ptcount;

  return 1;
}

//==============================================================================

int Triangulation::get_mass_center(Vertex* ptlist, int ptnum, REAL mc[2])
{
  // Calculate cell mass center
  mc[0] = mc[1] = 0.;
  
  // The formula from Wikipedia.
  double Area = 0.; // signed area.
  for (int i = 0; i < ptnum; i++) {
    Vertex *v1 = &(ptlist[i]);
    Vertex *v2 = &(ptlist[(i+1)%ptnum]);
    double x0 = v1->crd[0];
    double y0 = v1->crd[1];
    double x1 = v2->crd[0];
    double y1 = v2->crd[1];
    Area += 0.5 * (x0 * y1 - x1 * y0);
  }

  double mcx = 0., mcy = 0.;
  for (int i = 0; i < ptnum; i++) {
    Vertex *v1 = &(ptlist[i]);
    Vertex *v2 = &(ptlist[(i+1)%ptnum]);
    double x0 = v1->crd[0];
    double y0 = v1->crd[1];
    double x1 = v2->crd[0];
    double y1 = v2->crd[1];
    mcx += (x0 + x1)*(x0 * y1 - x1 * y0);
    mcy += (y0 + y1)*(x0 * y1 - x1 * y0);
  }
  mcx /= (6 * Area);
  mcy /= (6 * Area);

  mc[0] = mcx;
  mc[1] = mcy;
  return 1;
}

//==============================================================================

void Triangulation::save_voronoi()
{
  // We need a background grid to cut the exterior Voronoi cells.
  bool clean_omt_domain = false;
  if (OMT_domain == NULL) {
    OMT_domain = this; // Use the current triangulation.
    clean_omt_domain = true;
  }

  // Calculate Voronoi vertices for triangles.
  int i, idx;
  idx = io_firstindex;
  // Calculate circumcenters for hull triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    // Ignore exterior triangles.
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    get_tri_orthocenter(tri);
    tri->idx = idx;
    idx++;
  }
  //Tr->op_db_verbose=10; // debug
  // Calculate bisectors for hull triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    // Ignore exterior triangles.
    if (tri->is_deleted()) continue;
    if (tri->is_hulltri()) { // A hull triangle.
      get_hulltri_orthocenter(tri);
      tri->idx = idx; // hulltri is also indexed.
      idx++;
    }
  }

  // Store the list of Voronoi vertices and Voronoi edges.
  arraypool *vert_list = new arraypool(sizeof(Vertex), 10);
  arraypool *edge_list = new arraypool(sizeof(Vertex *) * 2, 10);
  arraypool *cell_list = new arraypool(sizeof(Vertex **), 10);
  arraypool *cell_size_list = new arraypool(sizeof(int), 10);

  for (i = 0; i < ct_in_vrts; i++) {
    Vertex *mesh_vertex = &(in_vrts[i]);
    if (mesh_vertex->typ == UNUSEDVERTEX) continue;
    Vertex *ptlist = NULL;
    int ptnum = 0;
    if (get_powercell(mesh_vertex, &ptlist, &ptnum)) {
      // Store the cell vertices (pointers).
      int *cell_size = (int *) cell_size_list->alloc();
      *cell_size = ptnum;
      Vertex ***cell = (Vertex ***) cell_list->alloc();
      *cell = new Vertex*[ptnum];
      
      for (int j = 0; j < ptnum; j++) {
        Vertex *pt = (Vertex *) vert_list->alloc();
        pt->init();
        pt->crd[0] = ptlist[j].crd[0];
        pt->crd[1] = ptlist[j].crd[1];
        pt->crd[2] = ptlist[j].crd[2];
        (*cell)[j] = pt;
      }
      for (int j = 0; j < ptnum; j++) {
        Vertex **edge = (Vertex **) edge_list->alloc();
        edge[0] = (*cell)[j];
        edge[1] = (*cell)[(j+1)%ptnum];
      }
    }
    delete [] ptlist;
  }

  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *mesh_vertex = (Vertex *) tr_steiners->get(i);
      if (mesh_vertex->is_deleted()) continue;
      Vertex *ptlist = NULL;
      int ptnum = 0;
      if (get_powercell(mesh_vertex, &ptlist, &ptnum)) {
        // Store the cell vertices (pointers).
        int *cell_size = (int *) cell_size_list->alloc();
        *cell_size = ptnum;
        Vertex ***cell = (Vertex ***) cell_list->alloc();
        *cell = new Vertex*[ptnum];
      
        for (int j = 0; j < ptnum; j++) {
          Vertex *pt = (Vertex *) vert_list->alloc();
          pt->init();
          pt->crd[0] = ptlist[j].crd[0];
          pt->crd[1] = ptlist[j].crd[1];
          pt->crd[2] = ptlist[j].crd[2];
          (*cell)[j] = pt;
        }
        for (int j = 0; j < ptnum; j++) {
          Vertex **edge = (Vertex **) edge_list->alloc();
          edge[0] = (*cell)[j];
          edge[1] = (*cell)[(j+1)%ptnum];
        }
      }
      delete [] ptlist;
    }
  } // if (tr_steiners != NULL)

  // Unifying vertices and counting edges.
  Triangulation *Voro = new Triangulation();

  // debug only.
  strcpy(Voro->io_outfilename, "/Users/si/tmp/voro");

  Voro->ct_in_vrts = vert_list->objects;
  Voro->in_vrts = new Vertex[vert_list->objects];
  Voro->io_xmin = Voro->io_ymin =  1.e+30;
  Voro->io_xmax = Voro->io_ymin = -1.e+30;
  for (i = 0; i < vert_list->objects; i++) {
    Vertex *p = (Vertex *) vert_list->get(i);
    Vertex *v = & (Voro->in_vrts[i]);
    v->init();
    // Default vertex type is UNUSEDVERTEX (0)
    v->typ = UNUSEDVERTEX;
    Voro->ct_unused_vrts++;
    v->crd[0] = p->crd[0];
    v->crd[1] = p->crd[1];
    v->crd[2] = p->crd[0]*p->crd[0] + p->crd[1]*p->crd[1];
    p->Pair = v; // remember this vertex.
    Voro->io_xmin = p->crd[0] < Voro->io_xmin ? p->crd[0] : Voro->io_xmin;
    Voro->io_xmax = p->crd[0] > Voro->io_xmax ? p->crd[0] : Voro->io_xmax;
    Voro->io_ymin = p->crd[1] < Voro->io_ymin ? p->crd[1] : Voro->io_ymin;
    Voro->io_ymax = p->crd[1] > Voro->io_ymax ? p->crd[1] : Voro->io_ymax;
  }
  assert(Voro->ct_unused_vrts == Voro->ct_in_vrts);

  Voro->tr_segs = new arraypool(sizeof(Triang), 10);
  for (i = 0; i < edge_list->objects; i++) {
    Vertex **edge = (Vertex **) edge_list->get(i);
    Triang *seg = (Triang *) Voro->tr_segs->alloc();
    seg->init();
    seg->vrt[0] = edge[0]->Pair;
    seg->vrt[1] = edge[1]->Pair;
    seg->tag = 1; // A non-zero marker
  }
  assert(Voro->ct_segments == 0);

  Voro->incremental_delaunay();
  Voro->save_triangulation(); // debug only.

  Voro->recover_segments();
  Voro->save_triangulation(); // debug only.

  if (clean_omt_domain) {
    OMT_domain = NULL;
  }

  char filename[256];
  sprintf(filename, "%s.voro", io_outfilename);
  
  printf("Writing Voronoi diagram to %s\n", filename);
  
  FILE *outfile = fopen(filename, "w");

  int nv = Voro->ct_in_vrts - Voro->ct_unused_vrts;
  int ne = Voro->ct_segments;
  int nc = cell_list->objects;

  fprintf(outfile, "%d %d %d\n", nv, ne, nc);

  printf("  Writing %d vertices.\n", nv);
  idx = 1;  // indexing all vertices.
  for (i = 0; i < Voro->ct_in_vrts; i++) {
    Vertex *v = &(Voro->in_vrts[i]);
    if (v->typ == UNUSEDVERTEX) continue;
    fprintf(outfile, "%g %g %g\n", v->crd[0], v->crd[1], v->crd[2]);
    v->idx = idx;
    idx++;
  }
  //assert(idx == nv);

  printf("  Writing %d edges.\n", ne);
  idx = 1;
  for (i = 0; i < Voro->tr_segs->used_items; i++) {
    Triang *seg = (Triang *) Voro->tr_segs->get(i);
    if (seg->tag == 0) continue;
    Vertex *v1 = seg->vrt[0];
    Vertex *v2 = seg->vrt[1];
    if (seg->vrt[0]->typ == UNUSEDVERTEX) {
      v1 = seg->vrt[0]->Pair;
    } 
    if (seg->vrt[1]->typ == UNUSEDVERTEX) {
      v2 = seg->vrt[1]->Pair;
    }
    fprintf(outfile, "%d %d  %d\n", v1->idx, v2->idx, seg->tag);    
    idx++;
  }

  printf("  Writing %d cells.\n", nc);
  for (i = 0; i < cell_list->objects; i++) {
    int size = * (int *) cell_size_list->get(i);
    fprintf(outfile, "%d ", size);
    Vertex ***cell = (Vertex ***) cell_list->get(i);
    for (int j = 0; j < size; j++) {
      Vertex *p = (*cell)[j];
      Vertex *v = p->Pair; // This is the mesh vertex in Voro.
      if (v->typ == UNUSEDVERTEX) {
        v = v->Pair;
      }
      fprintf(outfile, " %d", v->idx);
    }
    fprintf(outfile, "\n");
    // release memory.
    delete [] *cell;
  }

  fclose(outfile);

  delete vert_list;
  delete edge_list;
  delete cell_list;
  delete cell_size_list;
  delete Voro;
}
