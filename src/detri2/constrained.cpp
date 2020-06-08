#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

//==============================================================================

bool Triangulation::is_ridge_vertex(Vertex *v)
{
  // Count how many segments at this vertex (from the link list).
  // Debug: check the correctness of the link list.
  int j = 0;
  TriEdge seg = v->on_bd;
  while (seg.tri != NULL) {
    j++;
    assert(seg.tri->vrt[seg.ver] == v);
    seg = seg.tri->nei[seg.ver];
  }

  if (j == 0) {
    return false; // It belongs to no segment.
  } else if (j != 2) {
    return true;  // It is a ridge vertex.
  }

  // j = 2, there are two segments at this vertex.
  TriEdge seg1 = v->on_bd;
  TriEdge seg2 = seg1.tri->nei[seg1.ver];
  if (seg1.tri->tag != seg2.tri->tag) {
    return true; // It is a ridge vertex.
  } else {
    // Check if the three vertices are (nearly) collinear.
    Vertex *e1 = seg1.tri->vrt[1 - seg1.ver];
    Vertex *e2 = seg2.tri->vrt[1 - seg2.ver];
    /*
    REAL l1, l2, L;
    //ori = Orient2d(e1, e2, v);
    L  = get_distance(e1, e2);
    l1 = get_distance(e1, v);
    l2 = get_distance(e2, v);
    //if ((fabs(ori) / (l1 * l2)) > 1e-4) {
    if (fabs(1.0 - (l1 + l2) / L) > 1.e-6) {
      // They are not (nearly) collinear.
      v->typ = RIDGEVERTEX;
      rcount++;
    }
    */
    // Check the angle at v (between edges [v,e1] and [v,e2])
    REAL ang1 = get_angle(e1, e2, v);
    REAL ang2 = get_angle(e2, e1, v);
    // Debug only
    //printf(" debug: angle at %d: %g degree\n", e1->idx, ang1 / PI * 180.0);
    //printf(" debug: angle at %d: %g degree\n", e2->idx, ang2 / PI * 180.0);
    if ((ang1 > io_tol_minangle) && (ang2 > io_tol_minangle)) {
      // They are not (nearly) collinear.
      //printf(" debug: is fixed.\n");
      return true; //v->set_fix(); // v->typ = RIDGEVERTEX;
    }
    else { // debug only
      //printf(" debug: is collinear.\n");
    }
  }

  return false; // not a ridge vertex.
}

//==============================================================================
/*
// Return the segment that this TriEdge lies on.

TriEdge Triangulation::get_segment(TriEdge &E)
{
  assert(E.is_segment()); // It must be marked as a segment.
  Vertex *e1 = E.org();
  Vertex *e2 = E.dest();
  
  //if ((e1->typ != RIDGEVERTEX) ||
  //    !((e1->typ == STEINERVERTEX) && !e1->is_fixed())) {
  //  return e1->on_bd;
  //}
  //if ((e2->typ != RIDGEVERTEX) ||
  //    !((e2->typ == STEINERVERTEX) && !e2->is_fixed())) {
  //  return e2->on_bd;
  //}
  
  // Both endpoints must be RIDGEVERTEX or fixed Steiner vertices.
  // Search the segment from the vertex-to-segment map.
  TriEdge N = e1->on_bd;
  do {
    assert(N.tri->vrt[N.ver] == e1);
    if (N.tri->vrt[1 - N.ver] == e2) break; // Found
    N = N.tri->nei[N.ver]; // Get the next segment.
  } while (N.tri != NULL);

  return N;
}
*/

int Triangulation::get_edge(Vertex *e1, Vertex *e2, TriEdge &E)
{
  // It assumes e1 and e2 must be mesh vertices.
  if (e1->is_deleted() || (e1->typ == UNUSEDVERTEX) ||
      e2->is_deleted() || (e2->typ == UNUSEDVERTEX)) {
    return 0;
  }
  E = e1->adj;
  do {
    if (E.dest() == e2) return 1;
    E = E.eprev_esym(); // ccw
  } while (E.tri != e1->adj.tri);
  return 0;
}

//==============================================================================
/*
// Removing a segment at vertex, update the vertex-to-segment ring.
//   'v' must be an endpoint of a segment, i.e., v->typ is either RIDGEVERTEX or
//   SEGMENTVERTEX. In detri2, v->typ might be STEINERPOINT and v->is_fixed().

int Triangulation::remove_segment_at_vertex(Vertex *v, Triang *seg)
{
    TriEdge nextseg;
    if (seg->vrt[0] == v) {
      nextseg = seg->nei[0];
    } else if (seg->vrt[1] == v) {
      nextseg = seg->nei[1];
    } else {
      assert(0); // seg does not contain v.
    }
    if (nextseg.tri != NULL) {
      assert(nextseg.tri->vrt[nextseg.ver] == v);
    }

    TriEdge prevseg = v->on_bd;
    assert(prevseg.tri->vrt[prevseg.ver] == v);
    if (prevseg.tri == seg) {
      // remove the seg by restting the vertex-to-seg map.
      v->on_bd = nextseg;
      if (nextseg.tri == NULL) {
        // There is no segment at this vertex.
        // v is not a restricted vertex anymore.
        if (v->typ == RIDGEVERTEX) {
          v->typ = FACETVERTEX; // a free input vertex.
        } else if (v->typ == SEGMENTVERTEX) {
          v->typ = FACETVERTEX; // a free input vertex.
        } else if (v->typ == STEINERVERTEX) {
          v->clear_fix();
        } else {
          assert(0); // should not be possible.
        }
      }
      return 1;
    }

    // Search the seg in the vertex-to-segment map.
    TriEdge searchseg = prevseg.tri->nei[prevseg.ver];
    while (searchseg.tri != NULL) {
      assert(searchseg.tri->vrt[searchseg.ver] == v);
      if (searchseg.tri == seg) {
        // Found. Remove seg from the list.
        prevseg.tri->nei[prevseg.ver] = nextseg;
        return 1;
      }
      // Go the next of the next segment.
      TriEdge tmp = searchseg.tri->nei[searchseg.ver];
      searchseg = tmp;
    }

    return 0; // not found.
}
*/

//==============================================================================
// Remove the given segment by updating the vertex-to-segment map.

int Triangulation::remove_segment(Triang* seg)
{
  //printf("  debug: removing segment [%d,%d]\n", seg->vrt[0]->idx, seg->vrt[1]->idx);
  for (int i = 0; i < 2; i++) {
    Vertex *v = seg->vrt[i];
    //printf("  !debug: i=%d remove-seg-at-vertex %d\n", i, v->idx);
    TriEdge nextseg = seg->nei[i];
    if (nextseg.tri != NULL) {
      assert(nextseg.tri->vrt[nextseg.ver] == v);
      //printf("  !debug: seg->nei[%d] (nextseg) is [%d,%d]\n", i,
      //       nextseg.tri->vrt[nextseg.ver]->idx, nextseg.tri->vrt[1-nextseg.ver]->idx);
    }

    TriEdge prevseg = v->on_bd;
    assert(prevseg.tri->vrt[prevseg.ver] == v);
    //printf("  !debug: [%d]->on_bd (prevseg) is [%d,%d] is_fixed(%d) before\n", v->idx,
    //       prevseg.tri->vrt[prevseg.ver]->idx, prevseg.tri->vrt[1-prevseg.ver]->idx,
    //       v->is_fixed());
    if (prevseg.tri == seg) {
      //printf("  !debug: found prevseg.tri == seg\n");
      // remove the seg by restting the vertex-to-seg map.
      v->on_bd = nextseg;
      if (nextseg.tri == NULL) {
        //printf("  !debug: nextseg.tri == NULL, no segment at vertex [%d]\n", v->idx);
        // There is no segment at this vertex.
        // v is not a restricted vertex anymore.
        if (v->typ == SEGMENTVERTEX) {
          v->typ = FREEVERTEX; // a free input vertex.
        }
        v->clear_fix();
      } else {
        //printf("  !debug: updated [%d]->on_bd by [%d,%d]\n", v->idx,
        //       nextseg.tri->vrt[nextseg.ver]->idx, nextseg.tri->vrt[1-nextseg.ver]->idx);
      }
      //continue; //return 1;
    } else {
      // Search the seg in the vertex-to-segment map.
      TriEdge searchseg = prevseg.tri->nei[prevseg.ver];
      while (searchseg.tri != NULL) {
        //printf("  !debug: searcheg is [%d,%d]\n",
        //       searchseg.tri->vrt[searchseg.ver]->idx, searchseg.tri->vrt[1-searchseg.ver]->idx);
        assert(searchseg.tri->vrt[searchseg.ver] == v);
        if (searchseg.tri == seg) {
          // Found. Remove seg from the list.
          //printf("  !debug: found searchseg.tri == seg\n");
          prevseg.tri->nei[prevseg.ver] = nextseg;
          break; //return 1;
        }
        // Go the next of the next segment.
        prevseg = searchseg; // Update the prevseg first ****
        TriEdge tmp = searchseg.tri->nei[searchseg.ver];
        searchseg = tmp;
      }
      assert(searchseg.tri != NULL);
    }
    
    if (v->is_fixed()) {
      if (!is_ridge_vertex(v)) {
        v->clear_fix();
      }
    }
    //printf("  !debug: [%d] is_fixed(%d) after\n", v->idx, v->is_fixed());
  } // i

  //printf("  !debug: remove seg from tr_segs\n");
  TriEdge S;
  get_edge(seg->vrt[0], seg->vrt[1], S);
  assert(S.tri != NULL);
  S.clear_segment();
  (S.esym()).clear_segment();
  ct_segments--;

  seg->set_deleted();
  tr_segs->dealloc(seg);

  return 1;
}

//==============================================================================
// Insert the segment at the edge E.

int Triangulation::insert_segment(TriEdge E, int stag, REAL val, Triang** pseg)
{
  //printf("    debugs: insert segment [%d, %d]\n", E.org()->idx, E.dest()->idx);
  assert(!E.is_segment());

  Triang *seg = (Triang *) tr_segs->alloc();
  seg->init();
  seg->vrt[0] = E.org();
  seg->vrt[1] = E.dest();
  seg->vrt[2] = NULL;
  seg->tag = stag;
  seg->val = val;

  // Link this segment into the linklists of its endpoints.
  for (int j = 0; j < 2; j++) {
    // Update the vertex type
    if (seg->vrt[j]->typ == FREEVERTEX) {
      seg->vrt[j]->typ = SEGMENTVERTEX;
    }
    // Link the segment to its endpoints.
    TriEdge nextseg = seg->vrt[j]->on_bd;
    seg->vrt[j]->on_bd = TriEdge(seg, j);
    seg->vrt[j]->on_bd.tri->nei[j] = nextseg;
    //printf("  debug: check vertex_type %d is_fixed(%d) bdefore\n",
    //       seg->vrt[j]->idx, seg->vrt[j]->is_fixed());
    // If the vertex is fixed, it might become not fixed (e.g., split_segment case).
    // If the vertex is not fixed, it might become fixed (e.g., insert_segment case).
    //if (!seg->vrt[j]->is_fixed()) {
      // Check if it becomes a ridge vertex (un-removable).
      if (is_ridge_vertex(seg->vrt[j])) {
        seg->vrt[j]->set_fix();
      } else {
        seg->vrt[j]->clear_fix();
      }
    //}
    //printf("  debug: check vertex_type %d is_fixed(%d) after\n",
    //       seg->vrt[j]->idx, seg->vrt[j]->is_fixed());
  }

  E.set_segment();
  (E.esym()).set_segment();
  ct_segments++;

  if (pseg != NULL) {
    *pseg = seg;
  }
  return 1;
}

//==============================================================================
/* [2018-11-19] implemented directly in flip24().
// 'pt' lies on the edge tt[0], which is a segment.

int Triangulation::split_segment(Vertex *newpt, TriEdge tt[4], arraypool* fqueue)
{
  Vertex *v1 = tt[0].org();
  Vertex *v2 = tt[0].dest();

  if (op_db_verbose > 2) {
    printf("    Splitting segment [%d,%d] by %d.\n", v1->idx, v2->idx, newpt->idx);
  }

  // Remove the segment S from Triangulation.
  Triang *remove_seg = tt[0].get_segment();
  assert(remove_seg != NULL);
  int stag = remove_seg->tag;
  remove_segment(remove_seg);

  int fflag = FLIP_24;
  // tt[0] is [v1, v2, c]
  //tt[1] = tt[0].esym(); // [v2, v1, d]
  flip(tt, &newpt, fflag, fqueue);

  TriEdge S = tt[1].enext(); // [v1, newpt, c]
  assert(S.org() == v1);
  assert(S.dest() == newpt);
  insert_segment(S, stag, NULL);

  S = tt[0].eprev(); // [newpt, v2, d]
  assert(S.org() == newpt);
  assert(S.dest() == v2);
  insert_segment(S, stag, NULL);

  //lawson_flip(newpt, 1, fqueue);
  //delete fqueue;
  return 1;
}
*/
//==============================================================================
// Let E be the triangle [a,b,c], search the edge [a, pt].
// Return 1 if the edge [a,pt] crosses the edge E = [b,c]-a.
// Return 2 if the edge [a,pt] is collinear with the vertex E.org(), and
//   E.org() lies exactly inside [a,pt].
// It should not return 0.

int Triangulation::find_direction(TriEdge& E, Vertex *pt)
{
  if (E.tri->is_hulltri()) {
    // Search a non-hull tri at vertex a.
    do {
      E = E.eprev_esym(); // ccw
    } while (E.tri->is_hulltri());
  }

  int loc = 0;
  Vertex *pa = E.org();
  REAL s1 = Orient2d(pa, E.dest(), pt); // [a,b]
  REAL s2 = Orient2d(pa, E.apex(), pt); // [a,c]

  do {
    if (s1 > 0) {
      if (s2 > 0) {
        E = E.eprev_esym(); // ccw.
        s2 = Orient2d(pa, E.apex(), pt); // [a,c]
      } else if (s2 < 0) {
        // Cross edge [b,c]
        E.ver = _enext_tbl[E.ver];
        loc = 1;
      } else { // s2 == 0
        // Collinear with [a,c]
        E = E.eprev_esym(); // ccw.
        loc = 2;
      }
    } else if (s1 < 0) {
      if (s2 > 0) {
        // Rotate randomly
        if (rand() % 2) {
          E = E.eprev_esym(); // ccw
          s1 = s2;
          s2 = Orient2d(pa, E.apex(), pt);
        } else {
          E = E.esym_enext(); // cw
          s2 = s1;
          s1 = Orient2d(pa, E.dest(), pt);
        }
      } else if (s2 < 0) {
        E = E.esym_enext(); // cw
        s2 = s1;
        s1 = Orient2d(pa, E.dest(), pt);
      } else { // s2 == 0
        E = E.esym_enext(); // cw
        s2 = s1;
        s1 = Orient2d(pa, E.dest(), pt);
      }
    } else { // s1 == 0
      if (s2 > 0) {
        E = E.eprev_esym(); // ccw.
        s1 = s2;
        s2 = Orient2d(pa, E.apex(), pt); // [a,c]
      } else if (s2 < 0) {
        // Collinear with [a,b]
        loc = 2;
      } else { // s2 == 0
        assert(0); // not possible
      }
    }
  } while (loc == 0);

  return loc;
}

//==============================================================================
// Check if the edge [e1,e2] is intersected by any segment or vertex.
// The assumption is that both e1 and e2 must be vertices of this triangulation.
// The return value is one of the following:
//   - INTERSECT_NONE, it is not intersected by any segment.
//   - INTERSECT_SHARE_EDGE, it coincidents with the edge [S.org(), S.dest()].
//   - INTERSECT_SEGMENT, it intersects with the segment [S.org(), S.dest()].
//   - INTERSECT_VERTEX, is intersected with a vertex, returned by S.org().

int Triangulation::detect_intersection(Vertex *e1, Vertex *e2, TriEdge &S)
{
  // Perform a line walk from e1 to e2.
  TriEdge E = e1->adj;
  int loc = find_direction(E, e2);

  assert((loc == 1) || (loc == 2));
  // loc == 1, means that [e1,e2] crosses the edge [E.org(), E.dest()].
  // loc == 2, means that e1, E.dest(), e2 are collinear.
  //           It may be E.dest == e2, if so, the [e1, e2] = [E.org(), E.dest()].
  if (loc == 1) {
    assert(E.apex() == e1);
    // Detect whether if there is a nearly collinear endpoint.
    double ang = get_angle(e1, e2, E.org());
    if (ang < io_tol_minangle) {
      // Round to collinear with E.org();
      //qDebug()<<" rounding find_direction: collinear to E.org(), ang="<<ang/PI*180<<" degree";
      E = E.eprev();
      assert(E.org() == e1);
      loc = 2;
    } else {
      ang = get_angle(e1, e2, E.dest());
      if (ang < io_tol_minangle) {
        // Round to collinear with E.dest();
        //qDebug()<<" rounding find_direction: collinear to E.dest() ang="<<ang/PI*180<<" degree";
        E = E.enext_esym();
        assert(E.org() == e1);
        loc = 2;
      }
    }
  }

  if (loc == 2) {
    assert(E.org() == e1);
    if (E.dest() == e2) {
      // The edge exists.
      S = E;
      if (E.is_segment()) {
        return INTERSECT_CONFLICT; // A segment already exists
      } else {
        return INTERSECT_SHARE_EDGE;
      }
    } else {
      // Found a vertex E.dest() intersect this edge.
      S = E.enext();
      return INTERSECT_VERTEX;
    }
  } else if (loc == 1) {
    // crossing an edge.
    if (E.is_segment()) {
      S = E;
      return INTERSECT_SEGMENT;
    }
  } else {
    assert(0); // not possible.
  }

  // Continue the search for the next intersected edge of [e1, e2].
  while (true) {
    E = E.esym();
    // E is [a,b,c], where e2 lies in the halfspace of [a,b] which contains c.
    Vertex *pc = E.apex();
    if (pc == e2) {
      // End of the walk. No intersection is found.
      break; //return INTERSECT_NONE;
    }

    double s3 = Orient2d(e1, e2, pc);
    if (s3 != 0) {
      double ang = get_angle(pc, e1, e2);
      if (ang < io_tol_minangle) {
        //qDebug()<<"  ang="<<ang/detri2::PI*180<<" degree";
        s3 = 0; // Round to collinear case.
      }
    }

    if (s3 > 0) {
      E = E.enext(); // intersect with [b,c]
    } else if (s3 < 0) {
      E = E.eprev(); // intersect with [c,a]
    } else { // if (s3 == 0) {
      // collinear with pc.
      E = E.enext(); // [c,a]
      return INTERSECT_VERTEX;
    }

    if (E.is_segment()) {
      S = E;
      return INTERSECT_SEGMENT;
    }
  } // while (true)

  return INTERSECT_NONE;
}

//==============================================================================
// Insert an edge with endpoints [e1, e2].
// Assume that e1 and e2 must be existing vertices.
// Return values indicates one of the following cases:
//  - INTERSECT_SHARE_EDGE, found the edge [E.org(),E.dest()].
//  - INTERSECT_SEGMENT, this edge intersects the segment [E.org(),E.dest()].
//  - INTERSECT_VERTEX, this edge intersects the vertex E.org().

int Triangulation::recover_edge(Vertex *e1, Vertex *e2, TriEdge &E, arraypool* fqueue)
{
  E = e1->adj;
  int loc = find_direction(E, e2);

  if (loc == 1) {
    // [e1,e2] intersects the edge [E.org(),E.dest()].
    if (E.is_segment()) {
      return INTERSECT_SEGMENT;
    }
  } else if (loc == 2) {
    // [e1,e2] is collinear with the vertex E.dest().
    if (E.dest() == e2) {
      return INTERSECT_SHARE_EDGE; // Found the edge.
    } else {
      E = E.enext();
      return INTERSECT_VERTEX;
    }
  } else {
    assert(0); // not possible.
  }

  // [e1,e2] intersects the edge [E.org(),E.dest()].
  // Try to flip it.
  while (true) {
    // Check if the edge [E.org(),E.dest()] admits a 2-2 flip.
    Vertex *pc = E.apex();
    Vertex *pd = (E.esym()).apex();
    REAL s1 = Orient2d(pc, pd, E.dest());
    REAL s2 = Orient2d(pd, pc, E.org());
    if ((s1 > 0) && (s2 > 0)) {
      break; // Found a 2-2 flip on edge [E.org(),E.dest()].
    }
    // Search a flippable crossing edge by line walk towards e2.
    E = E.esym();
    assert(!E.tri->is_hulltri());
    printf("  E [%d, %d %d]\n", E.org()->idx, E.dest()->idx, E.apex()->idx);
    // E is [v1, v2, v3], where [v1, v2] intersects [e1, e2].
    // Check [e1, e2] intersects which edge [v2,v3] or [v3,v1].
    // Since we already know that [e1,e2] intersects [v1,v2],
    // we only need to test: v3 lies on which side of [e1, e2].
    REAL ss = Orient2d(e1, e2, E.apex());
    printf("  ss[e1(%d), e2(%d) %d] = %g\n", e1->idx, e2->idx, E.apex()->idx, ss);
    if (ss > 0) {
      E = E.enext(); // [v2,v3]
    } else if (ss < 0) {
      E = E.eprev(); // [v3,v1]
    } else { // ss == 0
      assert(E.apex() != e2);
      E = E.eprev(); // v3 is collinear with [e1,e2].
      return INTERSECT_VERTEX;
    }
    if (E.is_segment()) {
      return INTERSECT_SEGMENT; // cross a segment.
    }
  }

  // Flip the crossing edge.
  TriEdge tt[4];
  tt[0] = E;
  int fflag = FLIP_22;
  flip(tt, NULL, fflag, fqueue);

  // Continue to recover the edge (switch e1<->e2).
  return recover_edge(e2, e1, E, fqueue);
}

//==============================================================================
// Insert segments from e1 to e2, may be intersect with existing segments.

int Triangulation::insert_segment_intersect(Vertex *e1, Vertex *e2, int stag, REAL val)
{
  //assert(0); // Debug this function.
  Vertex *startpt = e1;

  while (startpt != NULL) {
    if (op_db_verbose > 2) {
      printf("    Insert segment [%d,%d]\n", startpt->idx, e2->idx);
    }

    TriEdge S;
    int chk = detect_intersection(startpt, e2, S);
    if (op_db_verbose > 2) {
      printf("    detect_intersection result = %d\n", chk);
    }

    if (chk == INTERSECT_SHARE_EDGE) {
      insert_segment(S, stag, val, NULL);
      return 1;
    }

    if (chk == INTERSECT_NONE) {
      arraypool *fqueue = new arraypool(sizeof(TriEdge), 4);
      int result = recover_edge(startpt, e2, S, fqueue);
      assert(result == INTERSECT_SHARE_EDGE);
      insert_segment(S, stag, val, NULL);
      if (fqueue->objects > 0) {
        lawson_flip(NULL, 0, fqueue);
      }
      delete fqueue;
      return 1;
    }
    
    if (chk == INTERSECT_CONFLICT) {
      assert(0); // may be a bug.
      return 0; // This segment alraedy exists.
    }

    if (chk == INTERSECT_VERTEX) {
      if (op_db_verbose > 2) {
        printf("    Intersecting an existing vertex %d\n", S.org()->idx);
      }
      Vertex *cutpt = S.org();
      arraypool *fqueue = new arraypool(sizeof(TriEdge), 4);
      int result = recover_edge(startpt, cutpt, S, fqueue);
      assert(result == INTERSECT_SHARE_EDGE);
      insert_segment(S, stag, val, NULL);
      if (fqueue->objects > 0) {
        lawson_flip(NULL, 0, fqueue);
      }
      delete fqueue;
      // Continue to insert the rest part of this segment.
      startpt = cutpt;
      continue;
    }

    if (chk == INTERSECT_SEGMENT) {
      if (op_db_verbose > 2) {
        printf("    Intersecting an existing segment [%d,%d]\n", S.org()->idx, S.dest()->idx);
      }
      // S must be an existing segment of Tr, [e1,e2] intersects with S.
      //   (1) Calculate the intersection point.
      //   (2) Remove the segment S from the triangulation (Tr).
      //   (3) Insert the intersection point into Tr.
      //   (4) Insert four line segments at this intersection point to Tr.
      assert(S.is_segment());
      // (1) Calculate the intersecting point.
      REAL xx, yy;
      Vertex *v1 = S.org();
      Vertex *v2 = S.dest();
      double X0 = v1->crd[0];
      double Y0 = v1->crd[1];
      double X1 = v2->crd[0];
      double Y1 = v2->crd[1];
      double X2 = e1->crd[0];
      double Y2 = e1->crd[1];
      double X3 = e2->crd[0];
      double Y3 = e2->crd[1];
      double t1, t2;
      if (line_line_intersection(X0, Y0, X1, Y1, X2, Y2, X3, Y3, &t1, &t2)) {
        xx = X0 + t1 * (X1 - X0);
        yy = Y0 + t1 * (Y1 - Y0);
        //qDebug()<<"  - Intersection point ("<<xx<<","<<yy<<")";
        if (op_db_verbose > 2) {
          printf("    Intersecting point (%g,%g).\n", xx, yy);
        }
      } else {
        printf("!! Warning: Failed at calculating line-line intersection.\n");
        return 0;
      }

      if (op_db_verbose > 2) {
        printf("    Inserting the intersection point.\n");
      }

      // (3) Insert a new vertex on edge S.
      Vertex *newpt = (Vertex *) tr_steiners->alloc();
      newpt->init();
      newpt->crd[0] = xx;
      newpt->crd[1] = yy;
      newpt->crd[2] = xx*xx + yy*yy; // - w;
      //newpt->crd[2] = op_lambda1 * xx*xx + op_lambda2 * yy*yy;
      newpt->typ = STEINERVERTEX;
      newpt->idx = io_firstindex + (ct_in_vrts + tr_steiners->objects - 1);

      arraypool *fqueue = new arraypool(sizeof(TriEdge), 4);

      int fflag = FLIP_24;
      TriEdge tt[4];
      tt[0] = S;            // [v1, v2, c]
      //tt[1] = tt[0].esym(); // [v2, v1, d]
      //split_segment(newpt, tt, fqueue);
      flip(tt, &newpt, fflag, fqueue);
      lawson_flip(newpt, 0, fqueue); // hullflag = 0
      //delete fqueue;

      //qDebug()<<"  - New point: idx="<<newpt->idx<<" ("<<newpt->crd[0]<<","<<newpt->crd[1]<<")";
      int result = recover_edge(startpt, newpt, S, fqueue);
      assert(result == INTERSECT_SHARE_EDGE);
      insert_segment(S, stag, val, NULL);
      lawson_flip(NULL, 0, fqueue);

      delete fqueue;

      // Continue to insert subsegment [startpt, e2]
      //assert(newpt->is_fixed());
      startpt = newpt;
    } // if (chk == INTERSECT_SEGMENT)
  } // while (startpt != NULL)

  return 0; // Should not be here
}

//==============================================================================
// [Comment 2018-07-05] This function assume all vertices of the line segments
//   must already exist. However, some endpoints of segment might be
//   UNUSEDVERTEX, in this case, there must exist a mesh vertex which is
//   coincident with this vertex.

int Triangulation::recover_segments()
{
  if (op_db_verbose) {
    printf("Incrementally recovering segments.\n");
  }
  //mesh_phase = 2;
  arraypool *fqueue = new arraypool(sizeof(TriEdge), 8);

  for (int i = 0; i < tr_segs->objects; i++) {
    Triang *seg = (Triang *) tr_segs->get(i);
    // Check if both endpoints of this edge are already inserted.
    bool bflag = true;
    for (int j = 0; j < 2; j++) {
      if (seg->vrt[j]->typ == UNUSEDVERTEX) {
        // Insert this vertex first.
        //TriEdge E;
        //int loc = locate_point(seg->vrt[j], E, 0);
        //if (loc == LOC_ON_VERT) { // On vertex
          //if (op_db_verbose) {
          //  printf("  Vertex %d is replaced with %d\n",
          //         seg->vrt[j]->idx, E.org()->idx);
          //}
          //seg->vrt[j] = E.org();
        if (seg->vrt[j]->Pair != NULL) {
          if (op_db_verbose) {
            printf("  Vertex %d is replaced with %d\n",
                   seg->vrt[j]->idx, seg->vrt[j]->Pair->idx);
          }
          seg->vrt[j] = seg->vrt[j]->Pair;
        } else {
          if (op_db_verbose) {
            printf("Warning:  Vertex %d does not exist in mesh.\n",
                   seg->vrt[j]->idx);
          }
          bflag = false;
          break; // assert(0);
        }
      }
    }
    if (!bflag) continue; // Skip this segment.
    
    if (op_db_verbose > 1) {
      printf("  Recovering segment [%d, %d]\n", seg->vrt[0]->idx, seg->vrt[1]->idx);
    }
    TriEdge E;
    int res = recover_edge(seg->vrt[0], seg->vrt[1], E, fqueue);
    if (res == INTERSECT_SHARE_EDGE) {
      if (!E.is_segment()) {
        E.set_segment();
        E.esym().set_segment();
        ct_segments++;
        // Link this segment into its vertices.
        for (int j = 0; j < 2; j++) {
          seg->vrt[j]->typ = SEGMENTVERTEX; // set an initial type.
          TriEdge pseg = seg->vrt[j]->on_bd;
          seg->vrt[j]->on_bd = TriEdge(seg, j);
          seg->vrt[j]->on_bd.tri->nei[j] = pseg;
          // Check if it becomes a ridge vertex (un-removable).
          if (is_ridge_vertex(seg->vrt[j])) {
            seg->vrt[j]->set_fix();
          } else {
            seg->vrt[j]->clear_fix();
          }
        }
      } else {
        if (op_db_verbose) {
          printf("  Segment: (%d,%d) already exists.\n",
                 seg->vrt[0]->idx, seg->vrt[1]->idx);
        }
        seg->tag = 0;
      }
      if (fqueue->objects > 0) {
        lawson_flip(NULL, 0, fqueue);
      }
    } else {
      if ((res == INTERSECT_VERTEX) || (res == INTERSECT_SEGMENT)) {
        // Try to insert this segment by intersecting other segments.
        if (insert_segment_intersect(seg->vrt[0], seg->vrt[1], seg->tag, seg->val)) {
          // Delete this segment.
          seg->set_deleted();
          tr_segs->dealloc(seg);
        } else {
          // Should not be here. Debug it.
          printf("!! Failed to recover segment [%d,%d] tga(%d).\n",
                 seg->vrt[0]->idx, seg->vrt[1]->idx, seg->tag);
          assert(0);
        }
      } else {
        // Unknown case, Debug it.
        assert(0);
      }
    }
  }

  delete fqueue;

  //if (check_mesh(0, 0)) {
  //  printf("Failed.\n");
  //  assert(0);
  //}

  set_ridgevertices();

  //mesh_phase = 0;
  return 1;
}

//==============================================================================
// Mark RIDGEVERTEX (endpoints) of segments.

int Triangulation::set_ridgevertices()
{
  int rcount = 0;

  for (int i = 0; i < ct_in_vrts; i++) {
    Vertex *v = &(in_vrts[i]);
    if (v->typ == UNUSEDVERTEX) continue;
    //if (!v->is_fixed()) {
      if (is_ridge_vertex(v)) {
        v->set_fix();
        rcount++;
      }
    //}
    /*
    if (v->typ == SEGMENTVERTEX) {
      // Count how many segments at this vertex (from the link list).
      // Debug: check the correctness of the link list.
      int j = 0;
      TriEdge seg = v->on_bd;
      do {
        j++;
        assert(seg.tri->vrt[seg.ver] == v);
        seg = seg.tri->nei[seg.ver];
      } while (seg.tri != NULL);
      if (j == 2) {
        // Two segments share at this vertex.
        TriEdge seg1 = v->on_bd;
        TriEdge seg2 = seg1.tri->nei[seg1.ver];
        if (seg1.tri->tag != seg2.tri->tag) {
          v->set_fix(); //v->typ = RIDGEVERTEX;
          rcount++;
        } else {
          // Check if the three vertices are (nearly) collinear.
          Vertex *e1 = seg1.tri->vrt[1 - seg1.ver];
          Vertex *e2 = seg2.tri->vrt[1 - seg2.ver];

          // REAL l1, l2, L;
          // //ori = Orient2d(e1, e2, v);
          // L  = get_distance(e1, e2);
          // l1 = get_distance(e1, v);
          // l2 = get_distance(e2, v);
          // //if ((fabs(ori) / (l1 * l2)) > 1e-4) {
          // if (fabs(1.0 - (l1 + l2) / L) > 1.e-6) {
          //   // They are not (nearly) collinear.
          //   v->typ = RIDGEVERTEX;
          //   rcount++;
          // }

          // Check the angle at v (between edges [v,e1] and [v,e2])
          REAL ang1 = get_angle(e1, e2, v);
          REAL ang2 = get_angle(e2, e1, v);
          if ((ang1 > io_tol_minangle) && (ang2 > io_tol_minangle)) {
            // They are not (nearly) collinear.
            v->set_fix(); // v->typ = RIDGEVERTEX;
            rcount++;
          }
        }
      } else {
        assert((j == 1) || (j > 2));
        v->set_fix(); //v->typ = RIDGEVERTEX;
        rcount++;
      }
    } // if (v->typ == SEGMENTVERTEX)
    */
  }

  if (tr_steiners != NULL) {
    for (int i = 0; i < tr_steiners->used_items; i++) {
      Vertex *v = (Vertex *) tr_steiners->get(i);
      if (v->is_deleted()) continue;
      //if (!v->is_fixed()) {
        if (is_ridge_vertex(v)) {
          v->set_fix();
          rcount++;
        }
      //}
    }
  }

  return rcount;
}

//==============================================================================
// Set subdomains and mark exterior triangles
// This function does not remove exterior triangles, the domain remain convex.

int Triangulation::set_subdomains()
{
  if (op_db_verbose) {
    printf("Setting exterior and interior elements.\n");
  }
  // [2019-01-31] This function may be called multiple times (in detri2qt).
  //assert(ct_exteriors == 0);
  //assert(tr_nonconvex == false);

  arraypool *exttris = new arraypool(sizeof(Triang *), 8);
  TriEdge E, N;
  int i, j;

  // Loop through hull triangles, Mark adjacent triangles which are not
  //   protected by segments.
  for (i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted()) continue;
    if (E.tri->is_hulltri()) {
      for (E.ver = 0; E.ver < 3; E.ver++) {
        if (E.apex() == tr_infvrt) break;
      }
      if (!E.is_segment()) {
        N = E.esym();
        if (!N.tri->is_exterior()) {
          N.tri->set_exterior();
          * (Triang **) exttris->alloc() = N.tri;
        }
      }
    }
  }

  // Loop through regions, find and infect a tet in holes.
  int hcount = 0;
  for (i = 0; i < ct_in_sdms; i++) {
    // If a vertex has marker 0, it means a hole.
    if (in_sdms[i].tag == 0) {
      //int loc = locate_point(&(in_sdms[i]), E, 0, 0); // encflag = 0
      int loc = locate_point(&(in_sdms[i]), E, 0); // encflag = 0
      // Skip any un-dermined cases.
      if (loc == LOC_ON_VERT) {
        Vertex *pt = E.org();
        if (pt->typ != FREEVERTEX) continue;
      }
      if (loc == LOC_ON_EDGE) { // on edge
        if (E.is_segment()) continue;
      }
      if (loc == LOC_IN_TRI) { // in triangle
        if (E.tri->is_hulltri()) continue;
      }
      if (!E.tri->is_exterior()) {
        E.tri->set_exterior();
        * (Triang **) exttris->alloc() = E.tri;
        hcount++;
      }
    }
  }

  // Find all exterior triangles by depth searching.
  for (i = 0; i < exttris->used_items; i++) {
    E.tri = * (Triang **) exttris->get(i);
    for (E.ver = 0; E.ver < 3; E.ver++) {
      if (!E.is_segment()) {
        N = E.esym();
        if (!N.tri->is_hulltri() && !N.tri->is_exterior()) {
          N.tri->set_exterior();
          * (Triang **) exttris->alloc() = N.tri;
        }
      }
    }
  }

  ct_exteriors += exttris->objects;

  // Set regions (subdomains) markers.
  exttris->clean(); // will be re-sued
  int max_rtag = 0;
  int rcount = 0;

  for (i = 0; i < ct_in_sdms; i++) {
    // If a vertex has marker != 0, it means a subdomain.
    if (in_sdms[i].tag != 0) {
      //int loc = locate_point(&(in_sdms[i]), E, 0, 0); // encflag = 0
      int loc = locate_point(&(in_sdms[i]), E, 0); // encflag = 0
      // Skip any ambiguous cases.
      if (loc == LOC_ON_VERT) {
        Vertex *pt = E.org();
        if (pt->typ != FREEVERTEX) continue;
      }
      if (loc == LOC_ON_EDGE) { // on edge
        if (E.is_segment()) continue;
      }
      if (loc == LOC_IN_TRI) { // in triangle
        if (E.tri->is_hulltri()) continue;
      }
      if (max_rtag < in_sdms[i].tag) max_rtag = in_sdms[i].tag;
      // Create a new subdomain.
      E.tri->tag = in_sdms[i].tag;
      E.tri->val = in_sdms[i].val;
      * (Triang **) exttris->alloc() = E.tri;
      // Mark all triangles in this subdomain.
      for (j = 0; j < exttris->used_items; j++) {
        E.tri = * (Triang **) exttris->get(j);
        for (E.ver = 0; E.ver < 3; E.ver++) {
          if (!E.is_segment()) {
            N = E.esym();
            if (N.tri->tag == 0) {
              N.tri->tag = in_sdms[i].tag;
              N.tri->val = in_sdms[i].val;
              * (Triang **) exttris->alloc() = N.tri;
            }
          }
        }
      } // j
      exttris->clean();
      rcount++;
    }
  } // i

  // Set other un-marked regions.
  for (i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted()) continue;
    if (E.tri->is_hulltri() || E.tri->is_exterior()) continue;
    if (E.tri->tag == 0) {
      // Found a new region. Mark all triangles in it.
      max_rtag++;
      E.tri->tag = max_rtag;
      E.tri->val = 0.0;
      * (Triang **) exttris->alloc() = E.tri;
      // Mark all triangles in this subdomain.
      for (j = 0; j < exttris->used_items; j++) {
        E.tri = * (Triang **) exttris->get(j);
        for (E.ver = 0; E.ver < 3; E.ver++) {
          if (!E.is_segment()) {
            N = E.esym();
            if (N.tri->tag == 0) {
              N.tri->tag = max_rtag;
              N.tri->val = 0.0;
              * (Triang **) exttris->alloc() = N.tri;
            }
          }
        }
      } // j
      exttris->clean();
      rcount++;
    } // if (E.tri->tag == 0)
  }

  if (op_db_verbose) {
    printf("  Found %d holes, %d subdomains.\n", hcount, rcount);
  }

  delete exttris;
  return 1;
}

//==============================================================================
// It assumes that the vertcies and triangles have been read.
//   Segments may be read (and will be created automatically).
//   Triangles are not connected to each other yet.

// The re-constructed mesh is treated as non-convex, i.e., tr_nonconvex = true.

int Triangulation::reconstruct_mesh(int check_delaunay)
{
  // Allocate an array for creating the vertex-to-triangles map.
  // For each triangle, we allocate additional 3+1 pointers to store the
  //   three link lists (stacks) of triangle sharing at its three vertices.
  //   The extra pointer is not used. It is only needed to make the calulation
  //   faster (use bit operation << 2 instead of i*3).
  TriEdge *tristacks = new TriEdge[tr_tris->objects * 4]; 
  // We will use the tags as index pointing to tristacks.
  int *bak_tags = new int[tr_tris->objects];
  TriEdge E, N;
  int i;

  for (i = 0; i < tr_tris->objects; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    bak_tags[i] = E.tri->tag;
    E.tri->tag = i; // Set the index-to-tristacks.
    for (E.ver = 0; E.ver < 3; E.ver++) {
      N = E.org()->adj;
      if (N.tri == NULL) {
        // This vertex is used.
        E.org()->typ = FREEVERTEX;
        ct_unused_vrts--;
      }
      // Link the current triangle to the next one in the stack.
      tristacks[(i << 2) + E.ver] = N; // i*4+E.ver
      // Push the current triangle onto the stack.
      E.org()->adj = E;
      if (!E.is_connected() || !(E.eprev()).is_connected()) {
        // There are unconnected edges at E.org().
        while (N.tri != NULL) {
          // E and N must have the same origin.
          assert(N.org() == E.org());
          if (E.dest() == N.apex()) {
            if (!E.is_connected()) {
              assert(!(N.eprev()).is_connected());
              E.connect(N.eprev());
            }
          } else if (E.apex() == N.dest()) {
            if (!N.is_connected()) {
              assert(!(E.eprev()).is_connected());
              (E.eprev()).connect(N);
            }
          }
          // Find the next triangle in the stack.
          N = tristacks[(N.tri->tag << 2) + N.ver];
        }
      }
    } // E.ver
  } // i

  delete [] tristacks;

  // Create hull triangles, and connect them together.
  arraypool *hulltris = new arraypool(sizeof(TriEdge), 8);

  for (i = 0; i < tr_tris->objects; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    E.tri->tag = bak_tags[i]; // Restore the tags.
    for (E.ver = 0; E.ver < 3; E.ver++) {
      if (E.esym().tri == NULL) {
        // Save the hull triangle.
        * (TriEdge *) hulltris->alloc() = E;
      }
    }
  }

  delete [] bak_tags;

  assert(ct_hullsize == 0);
  ct_hullsize = hulltris->objects;

  // Create the hull triangles.
  for (i = 0; i < hulltris->objects; i++) {
    TriEdge *parytri = (TriEdge *) hulltris->get(i);
    E = *parytri;
    N.tri = (Triang *) tr_tris->alloc();
    N.tri->init();
    N.set_vertices(E.dest(), E.org(), tr_infvrt);
    N.tri->set_hullflag();
    N.connect(E);
    *parytri = N; // Save the hull triangle.
  }
  tr_infvrt->adj = N.eprev(); // vertex-to-tri

  // Connect hull triangles to each other.
  for (i = 0; i < hulltris->objects; i++) {
    TriEdge *parytri = (TriEdge *) hulltris->get(i);
    E = parytri->enext();
    if (!E.is_connected()) {
      N = *parytri;
      do {
        N = N.esym();
        N.ver = _eprev_tbl[N.ver]; // N.enext2self();
      } while (N.is_connected());
      E.connect(N);
    }
  }

  // Segments will be introduced.
  int min_stag = 99999;
  if (tr_segs != NULL) {
    // First recover all constrained segments.
    for (i = 0; i < tr_segs->objects; i++) {
      Triang *seg = (Triang *) tr_segs->get(i); 
      // Record the maximum segment tag (for new segments).
      assert(seg->tag != 0); // must have a non-zero marker.
      if (min_stag > seg->tag) min_stag = seg->tag;
      if (get_edge(seg->vrt[0], seg->vrt[1], N)) {
        N.set_segment();
        N.esym().set_segment();
        ct_segments++;
        // Link this segment to list in vertex.
        for (int j = 0; j < 2; j++) {
          seg->vrt[j]->typ = SEGMENTVERTEX;
          TriEdge pseg = seg->vrt[j]->on_bd;
          seg->vrt[j]->on_bd = TriEdge(seg, j);
          seg->vrt[j]->on_bd.tri->nei[j] = pseg;
        }
      } else {
        printf("!! Input segment [%d,%d] tag(%d) does not match mesh edge. Skipped.\n",
               E.tri->vrt[0]->idx, E.tri->vrt[1]->idx, E.tri->tag);
      }
    }
  } else {
    min_stag = 0;
    int est_size = ct_in_vrts * 2;
    int log2objperblk = 0;
    while (est_size >>= 1) log2objperblk++;
    tr_segs = new arraypool(sizeof(Triang), log2objperblk);
  }

  // All other exterior/interior segments will have marker new_stag. 
  int new_stag = min_stag - 1; 
  if (new_stag == 0) new_stag = -1; // Avoid zero.

  // Create segments at exterior and interior boundary.
  for (i = 0; i < tr_tris->objects; i++) {
    E.tri = (Triang *) tr_tris->get(i); 
    if (!E.tri->is_hulltri()) {
      for (E.ver = 0; E.ver < 3; E.ver++) {
        if (!E.is_segment()) {
          N = E.esym();
          if (N.tri->is_hulltri() || (E.tri->tag != N.tri->tag)) {
            assert(!N.is_segment());
            E.set_segment();
            N.set_segment();
            // DEBUG ******
            //printf(" i=%d, org=%d, dest=%d\n", ct_segments, E.org()->idx, E.dest()->idx);
            ct_segments++;
            // Create a new segment.
            Triang *seg = (Triang *) tr_segs->alloc();
            seg->vrt[0] = E.org();
            seg->vrt[1] = E.dest();
            //seg->tag = max_stag+1;
            //max_stag++;
            seg->tag = new_stag;
            // Link this segment to list in vertex.
            for (int j = 0; j < 2; j++) {
              seg->vrt[j]->typ = SEGMENTVERTEX;
              TriEdge pseg = seg->vrt[j]->on_bd;
              seg->vrt[j]->on_bd = TriEdge(seg, j);
              seg->vrt[j]->on_bd.tri->nei[j] = pseg;
            }
          }
        }
      }
    } // not hulltri and not exterior tri
  }

  set_ridgevertices();

  assert(ct_exteriors == 0);
  assert(tr_nonconvex == false);

  //arraypool *fqueue = new arraypool(sizeof(TriEdge), 8);
  arraypool *fqueue = NULL;

  if (op_convex) { // -c
    // The mesh may be non-convex. Add triangles to make it convex.
    // [Comment: 2018-06-29] The following code does not work in general.
    //   *** Must re-do it.***
    TriEdge tt[4];
    int fflag;

    for (i = 0; i < hulltris->used_items; i++) {
      E = * (TriEdge *) hulltris->get(i);
      // Skip it if it is deleted or not a hulltri anymore.
      if (E.tri->is_deleted()) continue;
      if (!E.tri->is_hulltri()) continue;
      // E is [a,b,-1].
      if (E.apex() != tr_infvrt) {
        for (int j = 0; j < 3; j++) {
          E.ver = _enext_tbl[E.ver];
          if (E.apex() == tr_infvrt) break;
        }
      }
      assert(E.apex() == tr_infvrt);
      N = E.enext();
      N = N.esym();
      N.ver = _enext_tbl[N.ver]; // N is [b,c,-1]
      assert(N.apex() == tr_infvrt);
      REAL ori = Orient2d(E.org(), E.dest(), N.dest());
      if (ori > 0) {
        // [a,b,-1] and [b,c,-1] are non-convex.
        // Do f22 or f31, to check if [c,a,-1] exists.
        tt[2] = N.enext();
        tt[2] = tt[2].esym();
        tt[2].ver = _enext_tbl[tt[2].ver]; // [c,#,-1]
        assert(tt[2].apex() == tr_infvrt);
        if (tt[2].dest() == E.org()) {
          // Do a f31 to remove tr_infvrt.
          tt[0] = E.eprev(); // [-1,a,b] delete -1.
          fflag = FLIP_31;
          flip(tt, NULL, fflag, fqueue);
          assert(!tt[0].tri->is_hulltri());
          tt[0].tri->set_exterior();
          ct_exteriors++;
          // tr_infvrt->adj might be deleted.
          tr_infvrt->adj = tt[0]; // Its not true.
        } else {
          // Do a f22 to remove edge [b,-1]
          tt[0] = E.enext(); // [b,-1,a]
          tt[1] = N.eprev(); // [-1,b,c]
          //flip22(tt);
          fflag = FLIP_22;
          flip(tt, NULL, fflag, fqueue);
          assert(tt[0].tri->is_hulltri()); // [a,c,-1]
          assert(!tt[1].tri->is_hulltri()); // [c,a, b]
          // Add a new hulltri into list.
          * (TriEdge *) hulltris->alloc() = tt[0];
          // Mark an exterior triangle.
          tt[1].tri->set_exterior();
          ct_exteriors++;
        }
      } // if (ori > 0)
    } // i
    tr_nonconvex = false;
  } //if (op_convex) { // -c
  else {
    // We treat the re-constructed non-convex (for point location).
    tr_nonconvex = true;
  }

  if (check_delaunay) {
    // The mesh may be non-Delaunay.
    fqueue = new arraypool(sizeof(TriEdge), 8);
    lawson_flip(NULL, 0, fqueue);
    delete fqueue;
  }

  delete hulltris;
  return 1;
}
