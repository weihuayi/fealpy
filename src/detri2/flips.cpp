#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

//#define WITH_QT_LIB

#ifdef WITH_QT_LIB
#include "../detri2qt.h"
#include "../scribblearea.h"
#include <QPainter>
#endif

using namespace detri2;

//==============================================================================

int Triangulation::flip13(Vertex *pt, TriEdge *tt)
{
  TriEdge nn[3];
  bool is_ext = false;
  double val = 0.0;
  int tag, i;

  // On input
  //   tt[0] = [a,b,c]
  // On ouput
  //   tt[0] = [a,b,p]
  //   tt[1] = [b,c,p]
  //   tt[2] = [c,a,p]
  Vertex *pa = tt[0].org();
  Vertex *pb = tt[0].dest();
  Vertex *pc = tt[0].apex();

  if (op_db_verbose > 3) {
    printf("      flip 1-3: [%d] - [%d,%d,%d]\n", pt->idx, pa->idx, pb->idx, pc->idx);
  }

  for (i = 0; i < 3; i++) {
    nn[i] = tt[0].esym();
    tt[0].ver = _enext_tbl[tt[0].ver];
  }
  tag = tt[0].tri->tag;
  val = tt[0].tri->val;

  if (op_db_verbose > 3) {
    printf("  tag(%d), val(%g)\n", tag, val);
  }

  if (tt[0].tri->is_hulltri()) ct_hullsize--;
  if (tt[0].tri->is_exterior()) {ct_exteriors--; is_ext = true;}

  tt[1].tri = (Triang *) tr_tris->alloc();
  tt[2].tri = (Triang *) tr_tris->alloc();
  for (i = 0; i < 3; i++) tt[i].tri->init();

  tt[0].set_vertices(pa, pb, pt);
  tt[1].set_vertices(pb, pc, pt);
  tt[2].set_vertices(pc, pa, pt);

  // Set hullflag, exterior, tags
  for (i = 0; i < 3; i++) {
    if (nn[i].tri->is_hulltri() && (nn[i].apex() != tr_infvrt)) {
      tt[i].tri->set_hullflag(); ct_hullsize++;
    }
    if (is_ext) {
      tt[i].tri->set_exterior(); ct_exteriors++;
    }
    tt[i].tri->tag = tag;
    tt[i].tri->val = val;
  }

  // Connect adjacent triangles.
  for (i = 0; i < 3; i++) {
    nn[i].connect(tt[i]);
    if (nn[i].is_segment()) tt[i].set_segment();
  }
  for (i = 0; i < 3; i++) {
    (tt[i].enext()).connect(tt[(i+1)%3].eprev());
  }

  if (op_db_verbose > 3) {
    for (i = 0; i < 3; i++) {
      printf("  tt[%d]: %d,%d,%d, tag(%d), val(%g)\n", i,
             tt[i].org()->idx, tt[i].dest()->idx, tt[i].apex()->idx,
             tt[i].tri->tag, tt[i].tri->val);
    }
  }

  // Update the vrt-to-tri map.
  pa->adj = tt[0];
  pb->adj = tt[1];
  pc->adj = tt[2];
  pt->adj = tt[2].eprev();

  if (pt->typ == UNUSEDVERTEX) {
    // This is needed by incremental_delaunay().
    pt->typ = FREEVERTEX;
    ct_unused_vrts--;
  } else {
    assert((pt->typ == FREEVERTEX) ||
           (pt->typ == STEINERVERTEX));
  }

  return 1;
}

int Triangulation::flip31(TriEdge *tt, Vertex **ppt)
{
  TriEdge nn[3];
  bool is_ext = false;
  double val = 0.0;
  int tag, i;

  // On input
  //   tt[0] = [a,b,p]
  //   tt[1] = [b,c,p]
  //   tt[2[ = [c,a,p]
  // On ouput
  //   tt[0] = [a,b,c]
  //   tt[1] = [b,c,a]
  //   tt[2] = [c,a,b]
  Vertex *pa = tt[0].org();
  Vertex *pb = tt[0].dest();
  Vertex *pt = tt[0].apex();
  Vertex *pc = tt[2].org();

  if (op_db_verbose > 3) {
    printf("      flip 3-1: [%d,%d,%d] - [%d]\n", pa->idx, pb->idx, pc->idx, pt->idx);
  }

  for (i = 0; i < 3; i++) {
    if (tt[i].tri->is_hulltri()) ct_hullsize--;
    nn[i] = tt[i].esym();
  }
  if (tt[0].tri->is_exterior()) {
    assert(tt[1].tri->is_exterior() && tt[2].tri->is_exterior());
    ct_exteriors-=3; is_ext = true;
  }
  tag = tt[0].tri->tag;
  val = tt[0].tri->val;

  if (op_db_verbose > 3) {
    printf("  tag(%d), val(%g)\n", tag, val);
  }

  tt[0].tri->init();
  for (i = 1; i < 3; i++) {
    tt[i].tri->set_deleted();
    tr_tris->dealloc(tt[i].tri);
  }

  // The new tri [a,b,c]
  tt[0].set_vertices(pa, pb, pc);

  // Set hullflag (if it contains tr_infvrt)
  for (i = 0; i < 3; i++) {
    if (tt[0].tri->vrt[i] == tr_infvrt) {
      tt[0].tri->set_hullflag(); ct_hullsize++;
      break;
    }
  }
  if (is_ext) {
    tt[0].tri->set_exterior(); ct_exteriors++;
  }
  tt[0].tri->tag = tag;
  tt[0].tri->val = val;

  // Connect neighbors
  for (i = 0; i < 3; i++) {
    tt[i].connect(nn[i]);
    if (nn[i].is_segment()) tt[i].set_segment();
    tt[i+1] = tt[i].enext();
  }

  if (op_db_verbose > 3) {
    printf("  tt[0] %d,%d,%d, tag(%d), val(%g)\n",
           tt[0].org()->idx, tt[0].dest()->idx, tt[0].apex()->idx,
           tt[0].tri->tag, tt[0].tri->val);
  }

  // Update the vrt-to-tri map.
  pa->adj = tt[0]; // [a,b,c]
  pb->adj = tt[1]; // [b,c,a]
  pc->adj = tt[2]; // [c,a,b]

  assert((pt->typ != UNUSEDVERTEX) &&
         (pt->typ != DEADVERTEX));
  if (pt->typ == STEINERVERTEX) {
    pt->set_deleted();
    tr_steiners->dealloc(pt);
  } else if (pt != tr_infvrt) {
    pt->typ = UNUSEDVERTEX;
    ct_unused_vrts++;
  }

  if (ppt != NULL) *ppt = pt;

  return 1;
}

int Triangulation::flip22(TriEdge *tt)
{
  TriEdge nn[4];
  int is_ext = false;
  double val = 0.0;
  int tag, i;

  // On input,
  //   tt[0] is [a,b,c], where [a,b] to be flipped.
  //   tt[1] is [b,a,d]
  // On output:
  //   tt[0] is [b,c,d],
  //   tt[1] is [c,a,d],
  //   tt[2] is [a,d,c].
  //   tt[3] is [d,b,c].
  Vertex *pa = tt[0].org();
  Vertex *pb = tt[0].dest();
  Vertex *pc = tt[0].apex();
  Vertex *pd = tt[1].apex();

  if (op_db_verbose > 3) {
    printf("      flip 2-2: [%d,%d] - [%d,%d]\n", pa->idx, pb->idx, pc->idx, pd->idx);
  }

  nn[0] = (tt[0].enext()).esym(); // [b,c]
  nn[1] = (tt[0].eprev()).esym(); // [c,a]
  nn[2] = (tt[1].enext()).esym(); // [a,d]
  nn[3] = (tt[1].eprev()).esym(); // [d,b]

  if (tt[0].tri->is_hulltri()) {
    assert(tt[1].tri->is_hulltri());
    ct_hullsize-=2;
  }
  if (tt[0].tri->is_exterior()) {
    assert(tt[1].tri->is_exterior());
    is_ext = true; 
    // ct_exteriors-=2; // ct_exterior does not change
  }
  tag = tt[0].tri->tag;
  val = tt[0].tri->val;

  if (op_db_verbose > 3) {
    printf("  tag(%d), val(%g)\n", tag, val);
  }

  tt[0].tri->init();
  tt[1].tri->init();

  tt[0].set_vertices(pc, pd, pb);
  tt[1].set_vertices(pd, pc, pa);

  if ((pc == tr_infvrt) || (pd == tr_infvrt)) {
    tt[0].tri->set_hullflag();
    tt[1].tri->set_hullflag();
    ct_hullsize+=2;
  } else if (pb == tr_infvrt) {
    tt[0].tri->set_hullflag();
    ct_hullsize++;
  } else if (pa == tr_infvrt) {
    tt[1].tri->set_hullflag();
    ct_hullsize++;
  }
  if (is_ext) {
    tt[0].tri->set_exterior();
    tt[1].tri->set_exterior();
    // ct_exteriors+=2;
  }
  tt[0].tri->tag = tt[1].tri->tag = tag;
  tt[0].tri->val = tt[1].tri->val = val;

  tt[0].connect(tt[1]);

  tt[0] = tt[0].eprev(); // [b,c,d]
  tt[1] = tt[1].enext(); // [c,a,d]
  tt[2] = tt[1].enext(); // [a,d,c]
  tt[3] = tt[0].eprev(); // [d,b,c]

  for (i = 0; i < 4; i++) {
    tt[i].connect(nn[i]);
    if (nn[i].is_segment()) tt[i].set_segment();
  }
  
  pa->adj = tt[2];
  pb->adj = tt[0];
  pc->adj = tt[1];
  pd->adj = tt[3];

  return 1;
}

int Triangulation::flip24(Vertex *pt, TriEdge *tt)
{
  TriEdge nn[4];
  bool is_ext_c = false, is_ext_d = false;
  double val_c, val_d;
  int tag_c, tag_d;
  int i;

  // On input,
  //   tt[0] is [a,b,c], where [a,b] to be split.
  //   tt[1] is [b,a,d]
  // On output:
  //   tt[0] is [b,c,p],
  //   tt[1] is [c,a,p],
  //   tt[2] is [a,d,p],
  //   tt[3] is [d,b,p],
  Vertex *pa = tt[0].org();
  Vertex *pb = tt[0].dest();
  Vertex *pc = tt[0].apex();
  Vertex *pd = tt[1].apex();

  if (op_db_verbose > 3) {
    printf("      flip 2-4: [%d] - [%d,%d,%d,%d]\n", pt->idx, pa->idx, pb->idx, pc->idx, pd->idx);
  }

  nn[0] = (tt[0].enext()).esym(); // [b,c]
  nn[1] = (tt[0].eprev()).esym(); // [c,a]
  nn[2] = (tt[1].enext()).esym(); // [a,d]
  nn[3] = (tt[1].eprev()).esym(); // [d,b]

  if (tt[0].tri->is_hulltri()) ct_hullsize--;
  if (tt[1].tri->is_hulltri()) ct_hullsize--;
  if (tt[0].tri->is_exterior()) {ct_exteriors--; is_ext_c = true;}
  if (tt[1].tri->is_exterior()) {ct_exteriors--; is_ext_d = true;}
  tag_c = tt[0].tri->tag; val_c = tt[0].tri->val;
  tag_d = tt[1].tri->tag; val_d = tt[1].tri->val;

  if (op_db_verbose > 3) {
    printf("  tag_c(%d), val_c(%g)\n", tag_c, val_c);
    printf("  tag_d(%d), val_d(%g)\n", tag_d, val_d);
  }

  int stag = 0; REAL val = 0.;
  bool is_seg = tt[0].is_segment(); // split a segment?
  if (is_seg) {
    Triang *seg = tt[0].get_segment();
    stag = seg->tag;
    val = seg->val;
    remove_segment(seg);
  }

  tt[2].tri = (Triang *) tr_tris->alloc();
  tt[3].tri = (Triang *) tr_tris->alloc();
  for (i = 0; i < 4; i++) tt[i].tri->init();

  tt[0].set_vertices(pb, pc, pt);
  tt[1].set_vertices(pc, pa, pt);
  tt[2].set_vertices(pa, pd, pt);
  tt[3].set_vertices(pd, pb, pt);

  for (i = 0; i < 4; i++) {
    if (nn[i].tri->is_hulltri() && (nn[i].apex() != tr_infvrt)) {
      tt[i].tri->set_hullflag(); ct_hullsize++;
    }
  }
  if (is_ext_c) {
    tt[0].tri->set_exterior();
    tt[1].tri->set_exterior();
    ct_exteriors+=2;
  }
  if (is_ext_d) {
    tt[2].tri->set_exterior();
    tt[3].tri->set_exterior();
    ct_exteriors+=2;
  }
  tt[0].tri->tag = tt[1].tri->tag = tag_c;
  tt[2].tri->tag = tt[3].tri->tag = tag_d;

  tt[0].tri->val = tt[1].tri->val = val_c;
  tt[2].tri->val = tt[3].tri->val = val_d;

  for (i = 0; i < 4; i++) {
    nn[i].connect(tt[i]);
    if (nn[i].is_segment()) tt[i].set_segment();
  }
  for (i = 0; i < 4; i++) {
    (tt[i].enext()).connect(tt[(i+1)%4].eprev());
  }

  /*
  if (is_seg) { // [a,b] is a segment
    (tt[0].eprev()).set_segment(); // [p,b]
    (tt[1].enext()).set_segment(); // [a,p]
    (tt[2].eprev()).set_segment(); // [p,a]
    (tt[3].enext()).set_segment(); // [b,p]
    ct_segments++;
    pt->on_bd = sseg; // pt is a Steiner point.
  }
  */

  if (is_seg) { // [a,b] is a segment
    TriEdge S = tt[1].enext();
    assert(S.org() == pa);
    assert(S.dest() == pt);
    insert_segment(S, stag, val, NULL);
    S = tt[0].eprev();
    assert(S.org() == pt);
    assert(S.dest() == pb);
    insert_segment(S, stag, val, NULL);
  }

  pa->adj = tt[2];
  pb->adj = tt[0];
  pc->adj = tt[1];
  pd->adj = tt[3];
  pt->adj = tt[3].eprev();

  // This is needed by incremental_delaunay().
  if (pt->typ == UNUSEDVERTEX) {
    pt->typ = FREEVERTEX;
    ct_unused_vrts--;
  } /*else {
    assert(pt->typ == STEINERVERTEX);
  }*/
  return 1;
}

// The four vertices around the removing vertex (p) are
// a,b,c,d, where a,b,p are collinear, i.e., p will be
// removed and the edge [a,b] will be created.
int Triangulation::flip42(TriEdge *tt, Vertex **ppt)
{
  TriEdge nn[4];
  bool is_ext_c = false, is_ext_d = false;
  double val_c, val_d;
  int tag_c, tag_d;
  int i;

  // On input,
  //   tt[0] is [b,c,p],
  //   tt[1] is [c,a,p],
  //   tt[2] is [a,d,p],
  //   tt[3] is [d,b,p],
  // On output:
  //   tt[0] is [b,c,a],
  //   tt[1] is [c,a,b]
  //   tt[2] is [a,d,b],
  //   tt[3] is [d,b,a]
  Vertex *pb = tt[0].org();
  Vertex *pc = tt[0].dest();
  Vertex *pt = tt[0].apex();
  Vertex *pa = tt[2].org();
  Vertex *pd = tt[2].dest();

  if (op_db_verbose > 3) {
    printf("      flip 4-2: [%d,%d,%d,%d] - [%d]\n", pa->idx, pb->idx, pc->idx, pd->idx, pt->idx);
  }

  for (i = 0; i < 4; i++) {
    nn[i] = tt[i].esym();
    if (tt[i].tri->is_hulltri()) ct_hullsize--;
  }
  if (tt[0].tri->is_exterior()) {
    assert(tt[1].tri->is_exterior());
    ct_exteriors-=2; is_ext_c = true;
  }
  if (tt[2].tri->is_exterior()) {
    assert(tt[3].tri->is_exterior());
    ct_exteriors-=2; is_ext_d = true;
  }
  tag_c = tt[0].tri->tag; val_c = tt[0].tri->val;
  tag_d = tt[2].tri->tag; val_d = tt[2].tri->val;

  if (op_db_verbose > 3) {
    printf("  tag_c(%d), val_c(%g)\n", tag_c, val_c);
    printf("  tag_d(%d), val_d(%g)\n", tag_d, val_d);
  }

  int stag = 0; REAL val = 0.;
  bool is_seg = (tt[0].eprev()).is_segment();
  if (is_seg) {
    assert(!pt->is_fixed());
    TriEdge N = tt[0].eprev();
    Triang *seg = N.get_segment(); // [p,b]-c
    stag = seg->tag;
    val = seg->val;
    remove_segment(seg);
    N = tt[1].enext();
    seg = N.get_segment(); // [a,p]-c
    assert(seg != NULL);
    remove_segment(seg);
  }

  tt[0].tri->init();
  tt[1].tri->init();
  for (i = 2; i < 4; i++) {
    tt[i].tri->set_deleted();
    tr_tris->dealloc(tt[i].tri);
  }

  tt[0].set_vertices(pa, pb, pc);
  tt[1].set_vertices(pb, pa, pd);

  if ((pa == tr_infvrt) || (pb == tr_infvrt)) {
    tt[0].tri->set_hullflag();
    tt[1].tri->set_hullflag();
    ct_hullsize+=2;
  } else if (pc == tr_infvrt) {
    tt[0].tri->set_hullflag();
    ct_hullsize++;
  } else if (pd == tr_infvrt) {
    tt[1].tri->set_hullflag();
    ct_hullsize++;
  }
  if (is_ext_c) {
    tt[0].tri->set_exterior();
    ct_exteriors++;
  }
  if (is_ext_d) {
    tt[1].tri->set_exterior();
    ct_exteriors++;
  }
  tt[0].tri->tag = tag_c; tt[0].tri->val = val_c;
  tt[1].tri->tag = tag_d; tt[1].tri->val = val_d;

  tt[0].connect(tt[1]);

  if (is_seg) {
    // remove a vertex from a segment.
    // tt[0] is [a,b,c], where [a,b] is the new edge (segment).
    // Insert a new segment here.
    TriEdge E = tt[0]; // [a,b]
    assert((E.org() == pa) && (E.dest() == pb));
    insert_segment(E, stag, val, NULL); // [a,b]
  }

  tt[0] = tt[0].enext(); // [b,c,a]
  tt[2] = tt[1].enext(); // [a,d,b]
  tt[3] = tt[2].enext(); // [d,b,a]
  tt[1] = tt[0].enext(); // [c,a,b]

  for (i = 0; i < 4; i++) {
    tt[i].connect(nn[i]);
    if (nn[i].is_segment()) tt[i].set_segment();
  }

  pa->adj = tt[2];
  pb->adj = tt[0];
  pc->adj = tt[1];
  pd->adj = tt[3];

  assert((pt->typ != UNUSEDVERTEX) &&
         (pt->typ != DEADVERTEX));
  if (pt->typ == STEINERVERTEX) {
    pt->set_deleted();
    tr_steiners->dealloc(pt);
  } else {
    assert(pt != tr_infvrt);
    pt->typ = UNUSEDVERTEX;
    ct_unused_vrts++;
  }

  if (ppt != NULL) *ppt = pt;

  return 1;
}

//==============================================================================
// te is the edge to be flipped.
//
// Possible flip types: f22 (1), f31 (2), f42 (4)
// Unflippable types: f_hull (0), f_seg (-1), f_vrt (-2),
// Need further check: f31_nm (3), f42_nm (5)

int Triangulation::flip_check(TriEdge *te)
{
  int fflag = FLIP_UNKNOWN;

  if (op_db_verbose > 3) {
    printf("      Check edge: [%d,%d]-[%d]\n",
           te->org()->idx, te->dest()->idx, te->apex()->idx);
  }

  if (te->is_segment()) {
    fflag = FLIP_CONST; // f_seg
    return fflag;
  }

  int roundflag = 0;

  if (te->tri->is_hulltri()) {
    if (te->esym().tri->is_hulltri()) {
      if (te->org() == tr_infvrt) *te = te->esym();
      assert(te->org() != tr_infvrt);
      assert(te->dest() == tr_infvrt);
      // Check if this hull edge can be flipped. Three cases to be checked.
      // see doc/doc-flip_check_hull_edge.pdf
      TriEdge E = te->esym();
      TriEdge N = te->eprev_esym(); // ccw rotation
      if (N.apex() == E.apex()) {
        // Case (1) a 3-1 flip to remove a hull vertex.
        *te = N; // te is [p,a,b], p is the removing vertex.
        assert(!te->tri->is_hulltri());
        // Check if we can remove this vertex.
        if (!te->is_segment()) {
          fflag = FLIP_31;
        } else {
          // p is a boundary vertex connected by segments.
          fflag = FLIP_CONST;
        }
      } else {
        // Check if it is a 4-2 or 2-2 flip.
        // Let te be [p,d,b], then E is [d,p,a] (d=tr_infvert).
        Vertex *pa = E.apex();
        Vertex *pb = te->apex();
        Vertex *pt = te->org();
        REAL ss = Orient2d(pb, pa, pt);
        if (ss != 0.) {
          // Check if they are collinear by rounding.
          //if ((pt->typ != RIDGEVERTEX) &&
          //  te->eprev().is_segment() && E.enext().is_segment()) {
          if (!pt->is_fixed() &&
              te->eprev().is_segment() && E.enext().is_segment()) {
            ss = 0.0; roundflag = 1;
          }
        }
        if (ss < 0) {
          // Case (2) a 2-2 flip to add a hull edge [a,b].
          fflag = FLIP_22;
        } else if (ss == 0) {
          // Check case (2) a 4-2 flip.
          // N is [p,b,c]
          TriEdge NN = E.enext_esym(); // [a,p,#]
          if (N.apex() == NN.apex()) {
            // Case (2) a 4-2 flip to remove a hull vertex.
            // [Comment 2019-01-07] Wrong, do not return this edge.
            //*te = N; // te is [p,b,c]
            assert(te->org() != tr_infvrt);
            assert(te->dest() == tr_infvrt);
            assert(te->tri->is_hulltri());
            // By this setting, flip42() will remove p by creating the
            //   hull edge [a,b].
            if (roundflag) {
              // Check if the new mesh is valid.
              assert(0); // to be debugged.
            }
            fflag = FLIP_42;
          } else {
            fflag = FLIP_HULL; // unflippable.
          }
        }
      }
    } else {
      fflag = FLIP_HULL;
    }
    return fflag;
  } else {
    if (te->esym().tri->is_hulltri()) {
      fflag = FLIP_HULL; // unflippable
      return fflag;
    }
  }

  // Check if an interior edge is flippable.
  TriEdge tn = te->esym();
  Vertex* pa = te->org();
  Vertex* pb = te->dest();
  Vertex* pc = te->apex();
  Vertex* pd = tn.apex();

  // Check if edge [c, d] intersects [a,b].
  REAL s1 = Orient2d(pc, pd, pb);
  REAL s2 = Orient2d(pd, pc, pa);

  if ((s1 != 0.0) && te->enext().is_segment() && tn.eprev().is_segment()) {
    // Check if pb is a RIDGEVERTEX or not.
    //assert(pb->typ != FACETVERTEX);
    //if ((pb->typ == SEGMENTVERTEX) || (pb->typ == STEINERVERTEX)) {
    //  s1 = 0.0; // Rounding.
    //  roundflag = 1;
    //}
    /*REAL l1, l2, L;
    L  = get_distance(pc, pd);
    l1 = get_distance(pc, pb);
    l2 = get_distance(pd, pb);
    //if ((fabs(s1) / (l1 * l2)) < 1e-5) {
    if (fabs((l1+l2) / L - 1.0) < 1e-5) {
      s1 = 0.0;
      roundflag = 1;
    }
    */
    if (!pb->is_fixed()) {
      //assert(!is_ridge_vertex(pb)); // Only for debug
      if (is_ridge_vertex(pb)) {
        printf("Failed:  vertex %d is a ridge vertex, but it is not fixed.\n", pb->idx);
        assert(0);
      }
      s1 = 0.0;
      roundflag = 1;
    }
  }

  if ((s2 != 0.0) && te->eprev().is_segment() && tn.enext().is_segment()) {
    // Check if pb is a RIDGEVERTEX or not.
    //assert(pa->typ != FACETVERTEX);
    //if ((pa->typ == SEGMENTVERTEX) || (pa->typ == STEINERVERTEX)) {
    //  s2 = 0.0; // Rounding.
    //  roundflag = 1;
    //}
    /*REAL l1, l2, L;
    L  = get_distance(pc, pd);
    l1 = get_distance(pd, pa);
    l2 = get_distance(pc, pa);
    //if ((fabs(s2) / (l1 * l2)) < 1e-5) {
    if (fabs((l1+l2) / L - 1.0) < 1e-5) {
      s2 = 0.0;
    }
    */
    if (!pa->is_fixed()) {
      //assert(!is_ridge_vertex(pa)); // only for debug
      if (is_ridge_vertex(pa)) {
        printf("Failed:  vertex %d is a ridge vertex, but it is not fixed.\n", pa->idx);
        assert(0);
      }
      s2 = 0.0;
      roundflag = 1;
    }
  }

  if (op_round_flip != 0) {
    if (s1 != 0) {
      REAL cosang_at_pb = get_cosangle(pb, pc, pd);
      REAL tol = 1.0 - fabs(cosang_at_pb);
      /*
      tetview>a=0.5/180.*math.pi
      tetview>print(a)
      0.0087266462599716
      tetview>print(math.cos(a))
      0.99996192306417
      tetview>a=179.5/180.*math.pi
      tetview>print(a)
      3.1328660073298
      tetview>print(math.cos(a))
      -0.99996192306417
      tetview>print(1-math.abs(math.cos(a)))
      3.807693582869e-05
      */
      if (fabs(tol) < 1e-5) { // approx < 0.5 degree or > 179.5
        s1 = 0.;
        roundflag = 1;
      }
    }
    if (s2 != 0) {
      REAL cosang_at_pa = get_cosangle(pa, pc, pd);
      REAL tol = 1.0 - fabs(cosang_at_pa);
      if (fabs(tol) < 1e-5) { // // approx < 0.5 degree or > 179.5
        s2 = 0.;
        roundflag = 1;
      }
    }
  }

  if (s1 > 0) {
    if (s2 > 0) { // ++
      fflag = FLIP_22; // f22 [a,b,c] remove [a,b]
    } else if (s2 < 0) { // +-
      fflag = FLIP_31_NM; // f31_nm [a,b,c] remove [a]
    } else { // +0
      fflag = FLIP_42_NM; // f42_nm [a,b,c] remove [a] on edge [c,d]
    }
  } else if (s1 < 0) {
    if (s2 > 0) { // -+
      *te = te->esym(); // f31_nm [b,a,d] remove [b]
      fflag = FLIP_31_NM;
    } else if (s2 < 0) { // --
      //assert(0); // not possible
      // both [a] and [b] are at the same side of [c,d].
      // One of the triangles is inverted.
      ct_flipcount_inverted++;
      fflag = FLIP_INVERTED;
    } else { // -0
      //assert(0); // not possible
      ct_flipcount_inverted++;
      fflag = FLIP_INVERTED;
    }
  } else { // s1 == 0
    if (s2 > 0) { // 0+
      *te = te->esym(); // f42_nm [b,a,d] remove [b] on edge [c,d]
      fflag = FLIP_42_NM;
    } else if (s2 < 0) { // 0-
      //assert(0); // not possible [2018-02-28] igore this case for HDE
      ct_flipcount_inverted++;
      fflag = FLIP_INVERTED;
    } else { // 00
      //assert(0); // not possible [2018-02-28] igore this case for HDE
      ct_flipcount_inverted++;
      fflag = FLIP_INVERTED;
    }
  }

  if (fflag == FLIP_31_NM) { // f31_nm
    /*
    if (te->org()->typ == RIDGEVERTEX) {
      fflag = FLIP_CONST; // f_vrt
    }
    else if ((te->org()->typ == STEINERVERTEX) &&
             (te->org()->on_bd.tri != NULL)) { 
      fflag = FLIP_CONST; // f_vrt A Steiner point on segment.
    }
    else if ((te->org()->typ == STEINERVERTEX) &&
             (te->org()->is_fixed())) {
      fflag = FLIP_CONST; // A fixed Steiner point (Delaunay refinement).
    } */
    if (te->org()->is_fixed()) {
      fflag = FLIP_CONST;
    } else {
      TriEdge E = te->esym();
      TriEdge N = te->eprev_esym(); // ccw rotation
      if (N.apex() == E.apex()) {
        fflag = FLIP_31; // flippable. f31
      }
    }
  } else if (fflag == FLIP_42_NM) { // f42_nm
    /*
    if (te->org()->typ == RIDGEVERTEX) { 
      fflag = FLIP_CONST; // f_vrt
    } // [2018-11-10] A segment vertex can be removed, but needs a special function.
    else if ((te->org()->typ == STEINERVERTEX) &&
               (te->org()->on_bd.tri != NULL)) { 
      fflag = FLIP_CONST; // f_vrt A Steiner point on segment.
    } //
    else if ((te->org()->typ == STEINERVERTEX) &&
             (te->org()->is_fixed())) {
      fflag = FLIP_CONST; // A fixed Steiner point (Delaunay refinement).
    } */
    if (te->org()->is_fixed() ||
        te->is_segment()) { // If [a,b] is a segment, we could not remove a by
                            //   creating edge [c,d].
      fflag = FLIP_CONST; // f_vrt
    } else {
      // pa is possible to be removed by a 4-2 flip which creates edge [c,d].
      TriEdge E = te->esym();
      TriEdge N = te->eprev_esym(); // 1st ccw rotation
      //assert(N.apex() != E.apex());
      // [2019-11-14] Due to rounding, the above case can happen. Skip the flip.
      if (N.apex() != E.apex()) {
        N = N.eprev_esym(); // 2nd ccw rotation
        if (N.apex() == E.apex()) {
          // Found a 4-to-2 flip to remove a from edge [c,d].
          if (roundflag) {
            // Check if the new triangles are correct.
            assert(N.org() == te->org());
            //Vertex *pt = te->org(); // the removing vertex.
            Vertex *p1 = te->dest();
            Vertex *p2 = te->apex();
            Vertex *p3 = N.dest();
            Vertex *p4 = E.apex();
            // p1, p2, p3, p4 are ccw around pt.
            // [p2,p4] is the (almost) collinear edge containing [pt].
            // [p1] and [p3] are the two apexes of [p2,p4].
            // The two new triangles are [p2,p4,p1] and [p4,p2,p3]
            if ((p1 == tr_infvrt) || (p3 == tr_infvrt)) {
              fflag = FLIP_42; // flippable.
            } else {
              REAL o1, o2;
              o1 = Orient2d(p2, p4, p1);
              o2 = Orient2d(p4, p2, p3);
              if ((o1 > 0) && (o2 > 0)) {
                fflag = FLIP_42; // flippable. f42
              }
            }
          } // if (roundflag)
          else {
            fflag = FLIP_42; // flippable. f42
          }
        }
      } // if (N.apex() != E.apex())
    }
  }

  return fflag;
}

//==============================================================================

int Triangulation::flip(TriEdge tt[4], Vertex** ppt, int& fflag, arraypool* fqueue)
{
  // tt[0] is the edge to be flipped.
  if (fflag == FLIP_UNKNOWN) {
    fflag = flip_check(&(tt[0]));
  }

  int c = 0;
  if (fflag == FLIP_22) { // f22
    tt[1] = tt[0].esym();
    if (op_check_min_angle) {
      // On input,
      //   tt[0] is [a,b,c], where [a,b] to be flipped.
      //   tt[1] is [b,a,d]
      // The two new tris are:
      //   tt[0].set_vertices(pc, pd, pb);
      //   tt[1].set_vertices(pd, pc, pa);
      Vertex *pa = tt[0].org();
      Vertex *pb = tt[0].dest();
      Vertex *pc = tt[0].apex();
      Vertex *pd = tt[1].apex();
      REAL cosang = get_min_cosangle(pc, pd, pb);
      if (cosang > op_target_cos_min_angle) {
        return 0;
      }
      cosang = get_min_cosangle(pd, pc, pa);
      if (cosang > op_target_cos_min_angle) {
        return 0;
      }
    }
    flip22(tt); c = 4;
  } else if (fflag == FLIP_31) { // f31
    // tt[0] is [a,b,c], a is to be removed.
    tt[1] = tt[0].eprev_esym(); // ccw
    tt[2] = tt[1].eprev_esym(); // ccw
    for (int i = 0; i < 3; i++) {
      tt[i].ver = _enext_tbl[tt[i].ver];
    }
    if (op_check_min_angle) {
      // On input
      //   tt[0] = [a,b,p]
      //   tt[1] = [b,c,p]
      //   tt[2[ = [c,a,p]
      // The new triangle is:
      //   tt[0].set_vertices(pa, pb, pc);
      Vertex *pa = tt[0].org();
      Vertex *pb = tt[0].dest();
      //Vertex *pt = tt[0].apex();
      Vertex *pc = tt[2].org();
      REAL cosang = get_min_cosangle(pa, pb, pc);
      if (cosang > op_target_cos_min_angle) {
        return 0;
      }
    }
    flip31(tt, ppt); c = 3;
  } else if (fflag == FLIP_42) { // f42
    // tt[0] is [a,b,c], remove [a] on edge [c,d].
    // Check for flip42() for the detail of the order of vertices, so that
    //   the correct edge [c,d] will be created.
    tt[0] = tt[0].eprev_esym();
    for (int i = 1; i < 4; i++) {
      tt[i] = tt[i-1].eprev_esym(); // ccw  
    }
    for (int i = 0; i < 4; i++) {
      tt[i].ver = _enext_tbl[tt[i].ver];
    }
    if (op_check_min_angle) {
      // On input,
      //   tt[0] is [b,c,p],
      //   tt[1] is [c,a,p],
      //   tt[2] is [a,d,p],
      //   tt[3] is [d,b,p],
      // The two new tris are:
      //   tt[0].set_vertices(pa, pb, pc);
      //   tt[1].set_vertices(pb, pa, pd);
      Vertex *pb = tt[0].org();
      Vertex *pc = tt[0].dest();
      //Vertex *pt = tt[0].apex();
      Vertex *pa = tt[2].org();
      Vertex *pd = tt[2].dest();
      REAL cosang = get_min_cosangle(pa, pb, pc);
      if (cosang > op_target_cos_min_angle) {
        return 0;
      }
      cosang = get_min_cosangle(pb, pa, pd);
      if (cosang > op_target_cos_min_angle) {
        return 0;
      }
    }
    flip42(tt, ppt); c = 4;
  } else if (fflag == FLIP_13) {
    flip13(*ppt, tt); c = 3;
  } else if (fflag == FLIP_24) {
    tt[1] = tt[0].esym();
    flip24(*ppt, tt); c = 4;
  } else {
    // it is not flipped.
    return 0;
  }

  if (fqueue != NULL) {
    // The edge is flipped. Add exposed edges into fqueue.
    for (int i = 0; i < c; i++) {
      tt[i].set_edge_infect();
      * (TriEdge *) fqueue->alloc() = tt[i];
    }
  }

  tr_recnttri = tt[0].tri; // Remember a recent triangle.

  ct_flipcount++;
  if (io_dump_to_ucd) { // -Id
    save_to_ucd(ct_flipcount, 0);
  }

#ifdef WITH_QT_LIB
  if (vw_win != NULL) {
    ScribbleArea *_win = (ScribbleArea *) vw_win;
    QImage *canvas = new QImage(_win->width(), _win->height(), QImage::Format_RGB32);
    canvas->fill(qRgb(255,255,255));
    QPainter painter(canvas);
    _win->draw_scene(painter);
    if ((fflag == FLIP_13) || (fflag == FLIP_24)) {
      // Highlight the new vertex.
      QPen pen;
      pen.setColor(qRgb(255, 0, 0)); // red
      pen.setWidth(2);
      painter.setPen(pen);
      detri2_draw_point(*ppt, _win->_draw_vertex_size+2, &painter, _win->_cx, _win->_cy, _win->_scale_xy);
    } else if (fflag == FLIP_22) {
      // Highlight the new edge
      QPen pen;
      pen.setColor(qRgb(255, 0, 0)); // red
      pen.setWidth(_win->_draw_edge_width+2);
      painter.setPen(pen);
      // flip22
      // On input,
      //   tt[0] is [a,b,c], where [a,b] to be flipped.
      //   tt[1] is [b,a,d]
      // On output:
      //   tt[0] is [b,c,d],
      //   tt[1] is [c,a,d],
      //   tt[2] is [a,d,c].
      //   tt[3] is [d,b,c].
      detri2_draw_line(tt[0].dest(), tt[0].apex(), &painter, _win->_cx, _win->_cy, _win->_scale_xy);
    }
    QImage **pimg = (QImage **) vw_imgs->alloc();
    *pimg = canvas;
  }
  // https://stackoverflow.com/questions/27958716/save-a-qimage-in-qt
  // bool QImage::save ( const QString & fileName, const char * format = 0, int quality = -1 ) const
  // Saves the image to the file with the given fileName, using the given image
  //   file format and quality factor. If format is 0, QImage will attempt to guess
  //   the format by looking at fileName's suffix.
  // The quality factor must be in the range 0 to 100 or -1. Specify 0 to obtain
  //   small compressed files, 100 for large uncompressed files, and -1 (the default)
  //   to use the default settings.
  // Returns true if the image was successfully saved; otherwise returns false.
#endif

  return 1;
}

//==============================================================================

int Triangulation::first_triangle(Vertex *pa, Vertex *pb, Vertex *pc)
{
  TriEdge tt[4];
  int i;

  if (op_db_verbose > 3) {
    printf("      frist triangle: [%d,%d,%d]\n", pa->idx, pb->idx, pc->idx);
  }

  // Create 4 new triangles.
  for (i = 0; i < 4; i++) {
    tt[i].tri = (Triang *) tr_tris->alloc();
    tt[i].tri->init();
  }

  Vertex *ptlist[3];
  ptlist[0] = pa;
  ptlist[1] = pb;
  ptlist[2] = pc;
  tt[0].set_vertices(ptlist[0], ptlist[1], ptlist[2]);
  tt[1].set_vertices(ptlist[1], ptlist[0], tr_infvrt);
  tt[2].set_vertices(ptlist[2], ptlist[1], tr_infvrt);
  tt[3].set_vertices(ptlist[0], ptlist[2], tr_infvrt);

  for (i = 1; i < 4; i++) {
    tt[i].tri->set_hullflag();
  }
  ct_hullsize += 3;

  for (i = 0; i < 3; i++) {
    ptlist[i]->typ = FREEVERTEX;
  }
  ct_unused_vrts -= 3;

  // Connect the 4 triangles.
  tt[1].connect(tt[0]);
  tt[2].connect(tt[0].enext());
  tt[3].connect(tt[0].eprev());

  (tt[1].eprev()).connect(tt[2].enext());
  (tt[2].eprev()).connect(tt[3].enext());
  (tt[3].eprev()).connect(tt[1].enext());

  // Setup the vertex-to-triangle map.
  for (i = 0; i < 3; i++) {
    ptlist[i]->adj = tt[i];
  }
  tr_infvrt->adj = tt[1].eprev();

  ct_flipcount++;
  if (io_dump_to_ucd) { // -Id
    save_to_ucd(ct_flipcount, 0);
  }

#ifdef WITH_QT_LIB
  if (vw_win != NULL) {
    ScribbleArea *_win = (ScribbleArea *) vw_win;
    QImage *canvas = new QImage(_win->width(), _win->height(), QImage::Format_RGB32);
    canvas->fill(qRgb(255,255,255));
    QPainter painter(canvas);
    _win->draw_scene(painter);
    // Highlight the 3 vertices.
    QPen pen;
    pen.setColor(qRgb(255, 0, 0)); // red
    pen.setWidth(2);
    painter.setPen(pen);
    detri2_draw_point(pa, 6, &painter, _win->_cx, _win->_cy, _win->_scale_xy);
    detri2_draw_point(pb, 6, &painter, _win->_cx, _win->_cy, _win->_scale_xy);
    detri2_draw_point(pc, 6, &painter, _win->_cx, _win->_cy, _win->_scale_xy);
    QImage **pimg = (QImage **) vw_imgs->alloc();
    *pimg = canvas;
  }
#endif

  return 1;
}
