#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

double detri2::PI = 3.14159265358979323846264338327950288419716939937510582;

extern int metric_use_gmp; // an option set by op_use_gmp (must initialise it)

//==============================================================================
// Lookup tables for a speed-up in operations

// Pre-calcuated bit masks
unsigned int detri2::_test_bit_masks[32];     // = 2^i, i = 0, ..., 31.
unsigned int detri2::_clear_bit_masks[32];    // = ~(2^i), i = 0, ..., 31.
unsigned int detri2::_extract_bits_masks[32]; // = 2^i - 1, i = 0, ..., 31.

//                 v2
//                 /\
//             E1 /  \ E0
//               /    \
//              /______\
//            v0   E2   v1

unsigned char detri2::_vo[3] = {1, 2, 0};
unsigned char detri2::_vd[3] = {2, 0, 1};
unsigned char detri2::_va[3] = {0, 1, 2};

unsigned char detri2::_enext_tbl[3] = {1, 2, 0}; // = _vo 
unsigned char detri2::_eprev_tbl[3] = {2, 0, 1}; // = _vd

void detri2::initialize_lookup_tables()
{
  for (int i = 0; i < 32; i++) {
    _test_bit_masks[i] = (1 << i);
    _clear_bit_masks[i] = ~(1 << i);
    _extract_bits_masks[i] = (1 << i) - 1;
  }
}

//==============================================================================
// TriEdge

bool TriEdge::is_connected() {return tri->nei[ver].tri != NULL;}
void TriEdge::connect(const TriEdge& te)
{
  tri->nei[ver] = te;
  te.tri->nei[te.ver] = *this;
}

TriEdge TriEdge::enext() {return TriEdge(tri, _enext_tbl[ver]);}
TriEdge TriEdge::eprev() {return TriEdge(tri, _eprev_tbl[ver]);}
TriEdge TriEdge::esym()  {return tri->nei[ver];}
TriEdge TriEdge::enext_esym()  {return tri->nei[_enext_tbl[ver]];}
TriEdge TriEdge::eprev_esym()  {return tri->nei[_eprev_tbl[ver]];}
TriEdge TriEdge::esym_enext()
  {return TriEdge(tri->nei[ver].tri, _enext_tbl[tri->nei[ver].ver]);}
TriEdge TriEdge::esym_eprev()
  {return TriEdge(tri->nei[ver].tri, _eprev_tbl[tri->nei[ver].ver]);}

bool TriEdge::is_edge_infected() {return (tri->flags & _test_bit_masks[ver+22]);}
void TriEdge::set_edge_infect() {tri->flags |= _test_bit_masks[ver+22];}
void TriEdge::clear_edge_infect() {tri->flags &= _clear_bit_masks[ver+22];}

// Triang->edges(segments)
bool TriEdge::is_segment() {return (tri->flags & _test_bit_masks[ver+16]);}
void TriEdge::set_segment() {tri->flags |= _test_bit_masks[ver+16];}
void TriEdge::clear_segment() {tri->flags &= _clear_bit_masks[ver+16];}

Triang* TriEdge::get_segment()
{
  if (is_segment()) {
    Vertex *e1 = org();
    Vertex *e2 = dest();
    // Search the segment from the vertex-to-segment map.
    TriEdge N = e1->on_bd;
    while (N.tri != NULL) {
      assert(N.tri->vrt[N.ver] == e1);
      if (N.tri->vrt[1 - N.ver] == e2) break; // Found
      N = N.tri->nei[N.ver]; // Get the next segment.
    }
    return N.tri; // It might be NULL (not found).
  } else {
    return NULL;
  }
}

Vertex* TriEdge:: org() {return tri->vrt[_vo[ver]];}
Vertex* TriEdge::dest() {return tri->vrt[_vd[ver]];}
Vertex* TriEdge::apex() {return tri->vrt[_va[ver]];}

void TriEdge::set_vertices(Vertex *pa, Vertex *pb, Vertex *pc)
{
  tri->vrt[0] = pa;
  tri->vrt[1] = pb;
  tri->vrt[2] = pc;
  ver = 2; // Edge [a,b]
}

//==============================================================================
// Debug functions

void TriEdge::print()
{
  printf("x%lx v(%d) [%d,%d,%d] ", (unsigned long) tri, ver,
         org()->idx, dest()->idx, apex()->idx);
  if (tri->is_hulltri()) printf(" (hull tri)");
  if (tri->is_infected()) printf(" (tri infected)");
  if (tri->is_exterior()) printf(" (tri exterior)");
  if (is_segment()) printf(" (segment)");
  if (is_edge_infected()) printf(" (edge infected)");
  printf("\n");
}

void Triang::print(int detail)
{
  printf("Tri: x%lx [%d,%d,%d]", (unsigned long) this,
         vrt[0]->idx,vrt[1]->idx,vrt[2]->idx);
  if (is_hulltri()) printf(" (hull tri)");
  if (is_infected()) printf(" (tri infected)");
  if (is_exterior()) printf(" (tri exterior)");
  if (is_deleted()) printf(" (tri deleted)");
  printf("\n");

  if (detail) {
    for (int i = 0; i < 3; i++) {
      printf("  N[%d]: ", i);
      if (nei[i].tri != NULL) {
        if (!nei[i].tri->is_deleted()) {
          nei[i].print();
        } else {
          printf("(tri deleted)\n");
        }
      } else {
        printf("NULL\n");
      }
    }
  }
}

const char *vertextypename[] = {
  "UNUSEDVERTEX",  // 0
  "INFVERTEX",     // 1
  "SEGMENTVERTEX", // 2
  "FREEVERTEX",    // 3
  "STEINERVERTEX", // 4
  "DEADVERTEX",    // 5
};

void Vertex::print()
{
  printf("Vertex %d: typ(%s)", idx, vertextypename[(int) typ]);
  if (is_infected()) printf(" (infected)");
  if (is_fixed()) printf(" (fixed)");
  if (is_deleted()) printf(" (deleted)");
  printf("\n");
  printf("  (%g,%g) h(%g) w(%g)\n", crd[0], crd[1], crd[2],
         crd[0]*crd[0]+crd[1]*crd[1]-crd[2]);
         // w + h = |x|^2 ==> w = |x|^2 - h
  if (adj.tri != NULL) {
    printf("  adj: x%lx v(%d) [%d,%d,%d]\n", (unsigned long) adj.tri, adj.ver,
           adj.org()->idx, adj.dest()->idx, adj.apex()->idx);
  } else {
    printf("  adj: NULL\n");
  }
  if (on_bd.tri != NULL) {
    printf("  on_bd: x%lx v(%d) [%d,%d]\n", (unsigned long) on_bd.tri, on_bd.ver,
           on_bd.tri->vrt[0]->idx, on_bd.tri->vrt[1]->idx);
  }
  if (on_dm.tri != NULL) {
    printf("  on_dm: x%lx v(%d) [%d,%d,%d]\n", (unsigned long) on_dm.tri, on_dm.ver,
           on_dm.org()->idx, on_dm.dest()->idx, on_dm.apex()->idx);
  }
}

Vertex* Triangulation::db_vrt(int v)
{
  for (int i = 0; i < ct_in_vrts; i++) {
    if (in_vrts[i].idx == v) {
      in_vrts[i].print();
      return &(in_vrts[i]);
    }
  }
  if (tr_steiners != NULL) {
    for (int i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      if (vrt->idx == v) {
        vrt->print();
        return vrt;
      }
    }
  }
  printf("!! Not exist.\n");
  return NULL;
}

TriEdge Triangulation::db_seg(int v1, int v2)
{
  TriEdge E;
  for (int i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted()) continue;
    for (E.ver = 0; E.ver < 3; E.ver++) {
      if (((E.org()->idx == v1) && (E.dest()->idx == v2)) ||
          ((E.dest()->idx == v1) && (E.org()->idx == v2))) {
        E.print();
        E.esym().print();
        return E;
      }
    }
  }
  printf("!! Not exist.\n");
  E.tri = NULL;
  return E;
}

TriEdge Triangulation::db_tri(int v1, int v2, int v3)
{
  TriEdge E;
  for (int i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted()) continue;
    for (E.ver = 0; E.ver < 3; E.ver++) {
      if (((E.org()->idx == v1) && (E.dest()->idx == v2)) ||
          ((E.dest()->idx == v1) && (E.org()->idx == v2))) {
        if (E.apex()->idx == v3) {
          E.tri->print(1);
          return E;
        }
      }
    }
  }
  printf("!! Not exist.\n");
  E.tri = NULL;
  return E;
}

void Triangulation::dump_vertex_star(int v)
{
  Vertex *pt = db_vrt(v);
  if (pt == NULL) return;

  if (pt->typ == UNUSEDVERTEX) {
    return;
  }

  TriEdge E = pt->adj;
  do {
    if (E.dest() != tr_infvrt) {
      printf("p:draw_subseg(%d,%d) -- %d\n", E.org()->idx, E.dest()->idx, E.apex()->idx);
    }
    E = E.eprev_esym();
  } while (E.tri != pt->adj.tri);
}

//==============================================================================
// arraypool

void arraypool::poolinit(int sizeofobject, int log2objperblk)
{
  // Each object must be at least one byte long.
  objectbytes = sizeofobject > 1 ? sizeofobject : 1;

  log2objectsperblock = log2objperblk;
  // Compute the number of objects in each block.
  objectsperblock = ((int) 1) << log2objectsperblock;
  objectsperblockmask = objectsperblock - 1;

  // Allocate the top array, and NULL out its contents.
  int initsize = 128;
  toparray = (char **) malloc((size_t) (initsize * sizeof(char *)));
  toparraylen = initsize;
  for (int i = 0; i < initsize; i++) {
    toparray[i] = (char *) NULL;
  }
  // Account for the memory.
  totalmemory = initsize * (unsigned long) sizeof(char *);

  // Ready all indices to be allocated.
  clean();
}

arraypool::arraypool(int sizeofobject, int log2objperblk)
{
  poolinit(sizeofobject, log2objperblk);
}

arraypool::~arraypool()
{
  // Walk through the top array.
  for (int i = 0; i < toparraylen; i++) {
    // Check every pointer; NULLs may be scattered randomly.
    if (toparray[i] != (char *) NULL) {
      // Free an allocated block.
      free((void *) toparray[i]);
    }
  }
  // Free the top array.
  free((void *) toparray);
}

// Ready all indices to be allocated.
void arraypool::clean()
{
  objects = 0;
  used_items = 0;
  deaditemstack = NULL;
}

// Used by alloc()
char* arraypool::getblock(int objectindex)
{
  // Compute the index in the top array (upper bits).
  int topindex = (objectindex >> log2objectsperblock);

  // Does the top array need to be resized?
  if (topindex >= toparraylen) {
    // Resize the top array, making sure it holds 'topindex'.
    int newsize = topindex + 128;
    // Allocate the new array, copy the contents, NULL out the rest, and
    //   free the old array.
    char** newarray = (char **) malloc((size_t) (newsize * sizeof(char *)));
    for (int i = 0; i < toparraylen; i++) {
      newarray[i] = toparray[i];
    }
    for (int i = toparraylen; i < newsize; i++) {
      newarray[i] = (char *) NULL;
    }
    free(toparray);
    // Account for the memory.
    totalmemory += (newsize - toparraylen) * sizeof(char *);
    toparray = newarray;
    toparraylen = newsize;
  }

  // Find the block, or learn that it hasn't been allocated yet.
  char* block = toparray[topindex];
  if (block == (char *) NULL) {
    // Allocate a block at this index.
    block = (char *) malloc((size_t) (objectsperblock * objectbytes));
    toparray[topindex] = block;
    // Account for the memory.
    totalmemory += objectsperblock * objectbytes;
  }

  // Return a pointer to the block.
  return block;
}

char* arraypool::alloc()
{
  char *newptr;
  if (deaditemstack != (void *) NULL) {
    newptr = (char *) deaditemstack; // Take first item in list.
    deaditemstack = * (void **) deaditemstack;
  } else {
    // Allocate an object at index 'firstvirgin'.
    newptr =(getblock(objects) + (objects & objectsperblockmask) * objectbytes);
    used_items++;
  }
  objects++;
  return newptr;
}

void arraypool::dealloc(void *dyingitem)
{
  // Push freshly killed item onto stack.
  *((void **) dyingitem) = deaditemstack;
  deaditemstack = dyingitem;
  objects--;
}

// Return the pointer to the object with a given index, or NULL
//   if the object's block doesn't exist yet.
char* arraypool::lookup(int objectindex)
{
  // Has the top array been allocated yet?
  if (toparray == (char **) NULL) {
    return NULL;
  }

  // Compute the index in the top array (upper bits).
  int topindex = objectindex >> log2objectsperblock;
  // Does the top index fit in the top array?
  if (topindex >= toparraylen) {
    return NULL;
  }

  // Find the block, or learn that it hasn't been allocated yet.
  char* block = toparray[topindex];
  if (block == (char *) NULL) {
    return NULL;
  }

  // Compute a pointer to the object with the given index.  Note that
  //   'objectsperblock' is a power of two, so the & operation is a bit mask
  //   that preserves the lower bits.
  return (block + (objectindex & objectsperblockmask) * objectbytes);
}

// Fast lookup, but unsave.
char* arraypool::get(int index)
{
  return (toparray[index >> log2objectsperblock] + 
            ((index) & objectsperblockmask) * objectbytes);
}

void arraypool::traversalinit()
{
  blks = used_items / objectsperblock + 1;
  rsdu = used_items % objectsperblock;
}

int arraypool::get_block_items(int hi)
{
  return (hi < (blks - 1)) ? objectsperblock : rsdu;
}

// Report the current status and usage of arraypool
void arraypool::print()
{
  printf("arraypool: used (%d) allocated (%d)\n", objects, used_items);
  printf("  objectbytes: %d\n", objectbytes);
  printf("  objectsperblock: %d (2^%d)\n", objectsperblock, log2objectsperblock);
  printf("  toparraylen: %d\n", toparraylen);
  printf("  totalmemory (bytes): %ld\n", totalmemory);
}

//==============================================================================

// Check the validity of the mesh.
// Default topo_only = 0, check_level = 0;
int Triangulation::check_mesh(int topo_only, int check_level)
{
  TriEdge E, N, S;
  int horror = 0;

  for (int i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (!E.tri->is_deleted()) {
      if (!topo_only) {
        if (!E.tri->is_hulltri()) {
          // Check if this triangle is inverted.
          REAL ori = Orient2d(E.tri->vrt[0], E.tri->vrt[1], E.tri->vrt[2]);
          if (ori <= 0) {
            printf("  !!!! Inverted Triangle [%d,%d,%d]\n", E.tri->vrt[0]->idx,
                   E.tri->vrt[1]->idx, E.tri->vrt[2]->idx);
            horror++; continue;
          }
        }
      }
      // Check if this tri is correctly connected.
      for (E.ver = 0; E.ver < 3; E.ver++) {
        if (E.is_connected()) {
          N = E.tri->nei[E.ver];
          if (!((E.org() == N.dest()) && (E.dest() == N.org()))) {
            printf("  !!!! WRONG CONNECTION AT [%d,%d,%d] - [%d,%d,%d]\n",
                   E.org()->idx, E.dest()->idx, E.apex()->idx,
                   N.org()->idx, N.dest()->idx, N.apex()->idx);
            horror++; continue;
          }
        } else {
          printf("  !! NO CONNECTION at [%d,%d]-%d\n", E.org()->idx,
                 E.dest()->idx, E.apex()->idx);
          horror++; continue;
        }
        if (E.is_segment()) {
          N = E.esym();
          if (!N.is_segment()) {
            printf("  !!!! WRONG TRI->SEG AT [%d,%d] - %d\n",
                     E.org()->idx, E.dest()->idx, E.apex()->idx);
            horror++; continue;
          } 
          // Check vertex type
          if ((E.org()->typ != SEGMENTVERTEX) &&
              (E.org()->typ != STEINERVERTEX)) {
            printf("  !! WRONG VERTEX (%d) TYPE AT SEGMENT.\n", E.org()->idx);
            horror++; continue;
          }
          if ((E.dest()->typ != SEGMENTVERTEX) &&
              (E.dest()->typ != STEINERVERTEX)) {
            printf("  !! WRONG VERTEX (%d) TYPE AT SEGMENT.\n", E.dest()->idx);
            horror++; continue;
          }
        }
        // Check vertex-to-triang map
        if (E.org()->adj.tri == NULL) {
          printf("  !! Vertex %d has no adjacent triangle.\n", E.org()->idx);
          horror++; continue;
        } else {
          N = E.org()->adj;
          if (E.org() != N.org()) {
            printf("  !! Vertex %d has wrong adjacent tri [%d,%d,%d] x%lx %d.\n", 
                   E.org()->idx, N.org()->idx, N.dest()->idx, N.apex()->idx,
                   (unsigned long) N.tri, N.ver);
            horror++; continue;
          }
        }
        // Check infect flag of this edge (this may not be a bug).
        if (check_level > 0) {
          if (E.is_edge_infected()) {
            printf("  TriEdge: [%d,%d] %d is infected.\n", E.org()->idx, 
                   E.dest()->idx, E.apex()->idx);
          }
        }
      } // E.ver
      // Check infect flag of this triangle (this may not be a bug).
      if (check_level > 0) {
        if (E.tri->is_infected()) {
          printf("  Tri: [%d,%d,%d] is infected.\n", E.tri->vrt[0]->idx, 
                 E.tri->vrt[1]->idx, E.tri->vrt[2]->idx);
        }
      }
    } // if (!tris->is_deleted
  } // i

  if ((tr_segs != NULL) && (tr_segs->objects > 0)) {
    // check vertex-to-segment and segment-to-vertxex map.
    for (int i = 0; i < ct_in_vrts; i++) {
      Vertex *v = &(in_vrts[i]);
      if (v->typ == SEGMENTVERTEX) {
        TriEdge sseg = v->on_bd;
        if (sseg.tri == NULL) {
          printf("  !! RIDGEVERTEX (%d) DOES NOT POINT TO SEGMENT.\n", v->idx);
          horror++; continue;
        }
        if (sseg.tri->vrt[sseg.ver] != v) {
          printf("  !! WRONG SEGMENT [%d,%d] %d TO VERTEX (%d).\n",
                 sseg.tri->vrt[0]->idx, sseg.tri->vrt[1]->idx, sseg.ver, v->idx);
          horror++; continue;
        }
        TriEdge nextseg = sseg.tri->nei[sseg.ver];
        while (nextseg.tri != NULL) {
          if (nextseg.tri->vrt[nextseg.ver] != v) {
            printf("  !! WRONG SEGMENT [%d,%d] %d TO VERTEX (%d).\n",
                   nextseg.tri->vrt[0]->idx, nextseg.tri->vrt[1]->idx, nextseg.ver, v->idx);
            horror++; break;
          }
          // Go to the next segment.
          TriEdge tmp = nextseg.tri->nei[nextseg.ver];
          nextseg = tmp;
        }
      } // if (v->typ == RIDGEVERTEX)
    } // in_verts

    if (tr_steiners && (tr_steiners->objects > 0)) {
      for (int i = 0; i < tr_steiners->used_items; i++) {
        Vertex *v = (Vertex *) tr_steiners->get(i);
        if (v->is_deleted()) continue;
        if (v->on_bd.tri != NULL) { // v->is_fixed()
          TriEdge sseg = v->on_bd;
          if (sseg.tri == NULL) {
            printf("  !! STEINER (%d) DOES NOT POINT TO SEGMENT.\n", v->idx);
            horror++; continue;
          }
          printf("  [%d] [%d,%d] %d\n", v->idx,
                 sseg.tri->vrt[0]->idx, sseg.tri->vrt[1]->idx, sseg.ver);
          if (sseg.tri->vrt[sseg.ver] != v) {
            printf("  !! WRONG SEGMENT [%d,%d] %d TO VERTEX (%d).\n",
                   sseg.tri->vrt[0]->idx, sseg.tri->vrt[1]->idx, sseg.ver, v->idx);
            horror++; continue;
          }
          TriEdge nextseg = sseg.tri->nei[sseg.ver];
          while ((nextseg.tri != NULL)) {
            printf("  [%d] [%d,%d] %d\n", v->idx,
                   nextseg.tri->vrt[0]->idx, nextseg.tri->vrt[1]->idx, nextseg.ver);
            if (nextseg.tri->vrt[nextseg.ver] != v) {
              printf("  !! WRONG SEGMENT [%d,%d] %d TO VERTEX (%d).\n",
                     nextseg.tri->vrt[0]->idx, nextseg.tri->vrt[1]->idx, nextseg.ver, v->idx);
              horror++; break;
            }
            // Go to the next segment.
            TriEdge tmp = nextseg.tri->nei[nextseg.ver];
            nextseg = tmp;
          }
        } // if (v->is_fixed())
      } // tr_steiners
    } // if (tr_steiners->objects > 0)

    // Check if every segment is properly attached to edges.
    for (int i = 0; i < tr_segs->used_items; i++) {
      Triang *seg = (Triang *) tr_segs->get(i);
      if (seg->is_deleted()) continue;
      Vertex *e1 = seg->vrt[0];
      Vertex *e2 = seg->vrt[1];
      if ((e1 == NULL) || (e2 == NULL)) {
        printf("  !! WRONG SEGMENT %d ENDPOINTS (NULL POINTERs).\n", i);
        horror++;
      } else {
        TriEdge chkE; chkE.tri = NULL;
        if (!get_edge(e1, e2, chkE)) {
          printf("  !! SEGMENT %d [%d,%d] DOES NOT EXIST.\n", i, e1->idx, e2->idx);
          //horror++;
        } else {
          if (!chkE.is_segment() || !(chkE.esym()).is_segment()) {
            printf("  !! SEGMENT %d [%d,%d] IS MISS MARKED.\n", i, e1->idx, e2->idx);
            horror++;
          }
        }
      }
    }
  } // if (tr_segs->objects > 0)
  // End of checking vertex-to-segment map and segments-at-vertex rings.

  if (1) {
    if (horror == 0) {
      printf("The mesh is consistent.\n");
    } else {
      printf("  !! !! !! !! %d %s witnessed.\n", horror, 
             horror > 1 ? "abnormity" : "abnormities");
    }
  }

  return horror;
}

//==============================================================================

void Triangulation::quality_statistics()
{
  REAL cossquaretable[8];
  REAL ratiotable[16];
  REAL dx[3], dy[3];
  REAL edgelength[3];
  REAL dotproduct;
  REAL cossquare;
  REAL triarea;
  REAL shortest, longest;
  REAL trilongest2;
  REAL smallestarea, biggestarea;
  REAL triminaltitude2;
  REAL minaltitude;
  REAL triaspect2;
  REAL worstaspect;
  REAL smallestangle, biggestangle;
  REAL radconst, degconst;
  int angletable[18];
  int aspecttable[16];
  int aspectindex;
  int tendegree;
  int acutebiggest;
  int i, ii, j, k;

  printf("Mesh quality statistics:\n\n");
  radconst = PI / 18.0;
  degconst = 180.0 / PI;
  for (i = 0; i < 8; i++) {
    cossquaretable[i] = cos(radconst * (REAL) (i + 1));
    cossquaretable[i] = cossquaretable[i] * cossquaretable[i];
  }
  for (i = 0; i < 18; i++) {
    angletable[i] = 0;
  }

  ratiotable[0]  =      1.5;      ratiotable[1]  =     2.0;
  ratiotable[2]  =      2.5;      ratiotable[3]  =     3.0;
  ratiotable[4]  =      4.0;      ratiotable[5]  =     6.0;
  ratiotable[6]  =     10.0;      ratiotable[7]  =    15.0;
  ratiotable[8]  =     25.0;      ratiotable[9]  =    50.0;
  ratiotable[10] =    100.0;      ratiotable[11] =   300.0;
  ratiotable[12] =   1000.0;      ratiotable[13] = 10000.0;
  ratiotable[14] = 100000.0;      ratiotable[15] =     0.0;
  for (i = 0; i < 16; i++) {
    aspecttable[i] = 0;
  }

  worstaspect = 0.0;
  minaltitude = io_xmax - io_xmin + io_ymax - io_ymin;
  minaltitude = minaltitude * minaltitude;
  shortest = minaltitude;
  longest = 0.0;
  smallestarea = minaltitude;
  biggestarea = 0.0;
  worstaspect = 0.0;
  smallestangle = 0.0;
  biggestangle = 2.0;
  acutebiggest = 1;

  for (int I = 0; I < tr_tris->used_items; I++) {
    Triang *tri = (Triang *) tr_tris->get(I);
    if (tri->is_deleted()) continue;
    if (tri->is_hulltri()) continue;

    Vertex **p = tri->vrt;
    trilongest2 = 0.0;

    for (i = 0; i < 3; i++) {
      j = (i+1)%3; //plus1mod3[i];
      k = (i+2)%3; // minus1mod3[i];
      dx[i] = p[j]->crd[0] - p[k]->crd[0];
      dy[i] = p[j]->crd[1] - p[k]->crd[1];
      edgelength[i] = dx[i] * dx[i] + dy[i] * dy[i];
      if (edgelength[i] > trilongest2) {
        trilongest2 = edgelength[i];
      }
      if (edgelength[i] > longest) {
        longest = edgelength[i];
      }
      if (edgelength[i] < shortest) {
        shortest = edgelength[i];
      }
    }

    triarea = Orient2d(p[0], p[1], p[2]);
    triarea = fabs(triarea);
    if (triarea < smallestarea) {
      smallestarea = triarea;
    }
    if (triarea > biggestarea) {
      biggestarea = triarea;
    }
    triminaltitude2 = triarea * triarea / trilongest2;
    if (triminaltitude2 < minaltitude) {
      minaltitude = triminaltitude2;
    }
    triaspect2 = trilongest2 / triminaltitude2;
    if (triaspect2 > worstaspect) {
      worstaspect = triaspect2;
    }
    aspectindex = 0;
    while ((triaspect2 > ratiotable[aspectindex] * ratiotable[aspectindex])
           && (aspectindex < 15)) {
      aspectindex++;
    }
    aspecttable[aspectindex]++;

    for (i = 0; i < 3; i++) {
      j = (i+1)%3; //plus1mod3[i];
      k = (i+2)%3; // minus1mod3[i];
      dotproduct = dx[j] * dx[k] + dy[j] * dy[k];
      cossquare = dotproduct * dotproduct / (edgelength[j] * edgelength[k]);
      tendegree = 8;
      for (ii = 7; ii >= 0; ii--) {
        if (cossquare > cossquaretable[ii]) {
          tendegree = ii;
        }
      }
      if (dotproduct <= 0.0) {
        angletable[tendegree]++;
        if (cossquare > smallestangle) {
          smallestangle = cossquare;
        }
        if (acutebiggest && (cossquare < biggestangle)) {
          biggestangle = cossquare;
        }
      } else {
        angletable[17 - tendegree]++;
        if (acutebiggest || (cossquare > biggestangle)) {
          biggestangle = cossquare;
          acutebiggest = 0;
        }
      }
    }
  } // I

  shortest = sqrt(shortest);
  longest = sqrt(longest);
  minaltitude = sqrt(minaltitude);
  worstaspect = sqrt(worstaspect);
  smallestarea *= 0.5;
  biggestarea *= 0.5;
  if (smallestangle >= 1.0) {
    smallestangle = 0.0;
  } else {
    smallestangle = degconst * acos(sqrt(smallestangle));
  }
  if (biggestangle >= 1.0) {
    biggestangle = 180.0;
  } else {
    if (acutebiggest) {
      biggestangle = degconst * acos(sqrt(biggestangle));
    } else {
      biggestangle = 180.0 - degconst * acos(sqrt(biggestangle));
    }
  }

  printf("  Smallest area: %16.5g   |  Largest area: %16.5g\n",
         smallestarea, biggestarea);
  printf("  Shortest edge: %16.5g   |  Longest edge: %16.5g\n",
         shortest, longest);
  printf("  Shortest altitude: %12.5g   |  Largest aspect ratio: %8.5g\n\n",
         minaltitude, worstaspect);

  printf("  Triangle aspect ratio histogram:\n");
  printf("  1.1547 - %-6.6g    :  %8d    | %6.6g - %-6.6g     :  %8d\n",
         ratiotable[0], aspecttable[0], ratiotable[7], ratiotable[8],
         aspecttable[8]);
  for (i = 1; i < 7; i++) {
    printf("  %6.6g - %-6.6g    :  %8d    | %6.6g - %-6.6g     :  %8d\n",
           ratiotable[i - 1], ratiotable[i], aspecttable[i],
           ratiotable[i + 7], ratiotable[i + 8], aspecttable[i + 8]);
  }
  printf("  %6.6g - %-6.6g    :  %8d    | %6.6g -            :  %8d\n",
         ratiotable[6], ratiotable[7], aspecttable[7], ratiotable[14],
         aspecttable[15]);
  printf("  (Aspect ratio is longest edge divided by shortest altitude)\n\n");

  printf("  Smallest angle: %15.5g   |  Largest angle: %15.5g\n\n",
         smallestangle, biggestangle);

  printf("  Angle histogram:\n");
  for (i = 0; i < 9; i++) {
    printf("    %3d - %3d degrees:  %8d    |    %3d - %3d degrees:  %8d\n",
           i * 10, i * 10 + 10, angletable[i],
           i * 10 + 90, i * 10 + 100, angletable[i + 9]);
  }
  printf("\n");
}

void Triangulation::memory_statistics()
{
  printf("Memory allocation statistics:\n\n");
  printf("  Size of a Vertex (%d bytes), a Traingle (%d bytes)\n",
         (int) sizeof(Vertex), (int) sizeof(Triang));
  printf("  Maximum number of vertices: %d\n", ct_in_tris +
         (tr_steiners != NULL ? tr_steiners->used_items : 0));
  printf("  Maximum number of triangles: %d\n", tr_tris->used_items);
  if (tr_segs != NULL) {
    printf("  Maximum number of segments: %d\n", tr_segs->used_items);
  }
  //if (quality) {
  //  printf("  Maximum number of queued triangles: %ld\n", max_badtri_queue_length / sizeof(Triangle));
  //  printf("  Maximum number of queued segments: %ld\n", max_encseg_queue_length / sizeof(Triangle));
  //}
  //printf("  Maximum length of flip-queue: %ld\n", max_flipqueue_length / sizeof(TriEdge));

  unsigned long totalkbytes = 0l;
  // Stroage of the mesh.
  totalkbytes  = (ct_in_vrts * sizeof(Vertex)) / 1024;
  if (tr_steiners != NULL) {
    totalkbytes += (tr_steiners->totalmemory / 1024);
  }
  totalkbytes += (tr_tris->totalmemory / 1024);
  if (tr_segs != NULL) {
    totalkbytes += (tr_segs->totalmemory / 1024);
  }
  // used by point sorting.
  //totalkbytes += (ct_in_vrts * sizeof(Vertex) / 1024);
  // used by Lawson's flips.
  //totalkbytes += (max_flipqueue_length / 1024);
  // used by Delaunay refinement.
  //totalkbytes += (max_badtri_queue_length / 1024);
  //totalkbytes += (max_encseg_queue_length / 1024);
  printf("  Approximate heap memory use (kbyte): %ld\n\n", totalkbytes);
}

void Triangulation::mesh_statistics()
{
  printf("\nStatistics:\n\n");
  printf("  Input vertices: %d\n", ct_in_vrts);
  if (tr_segs != NULL) {
    printf("  Input segments: %d\n", tr_segs->objects);
  }
  if (ct_in_tris > 0) {
    printf("  Input triangles: %d\n", ct_in_tris);
  }
  if (ct_in_sdms > 0l) {
    printf("  Input region (holes): %d\n", ct_in_sdms);
  }

  int nvrt = ct_in_vrts + - ct_unused_vrts;
  if (tr_steiners != NULL) {
    nvrt += tr_steiners->objects;
  }
  printf("\n  Mesh vertices: %d\n", nvrt);

  int nedg = (3 * ((int) tr_tris->objects - ct_hullsize) + ct_hullsize) / 2;
  printf("  Mesh edges: %d\n", nedg);

  int ntri = (int) tr_tris->objects - ct_hullsize - ct_exteriors;
  printf("  Mesh triangles: %d\n", ntri);

  if (tr_segs != NULL) {
    printf("  Exterior boundary edges: %d\n", ct_hullsize);
    if (ct_segments > ct_hullsize) {
      printf("  Interior boundary edges: %d\n", ct_segments - ct_hullsize);
    }
  } else {
    printf("  Convex hull edges: %d\n", ct_hullsize);
  }

  if (tr_steiners != NULL) {
    printf("  Number of Steiner points: %d\n", tr_steiners->objects);
  }
  if (ct_unused_vrts > 0) {
    printf("  Number of unused input vertices: %d\n", ct_unused_vrts);
  }
  printf("\n");

  if (op_db_verbose) {
    quality_statistics();
    memory_statistics();
  }
}

//==============================================================================
// Initialization and clean

//#define USING_GMP

#ifdef USING_GMP
  #include <gmpxx.h>
  #include <mpfr.h>
#endif

Triangulation::Triangulation()
{
  initialize();
}

Triangulation::~Triangulation()
{
  clean();
  initialize();
}

void Triangulation::reset_options()
{
  OMT_domain = NULL;
  tr_recnttri = NULL;
  tr_nonconvex = false;

  op_db_verbose = 0;
  op_dt_nearest = 1; // comput nearest DT
  op_no_incremental_flip = 0;
  op_no_gabriel = 0;
  op_no_bisect = 0;
  op_poly = 0;
  op_convex = 0;
  op_quality = 0;
  op_use_gmp = 1;
  metric_use_gmp = 1;
  op_mpfr_precision = 1024;
#ifdef USING_GMP
  mpf_set_default_prec(op_mpfr_precision);
#endif

  _a11 = 1.0;
  _a21 = 0.0;
  _a22 = 1.0;

  op_metric = 0;
  op_round_flip = 0;
  op_use_coarsening = 1;
  op_use_splitting = 1;
  op_use_smoothing = 0;
  op_max_iter = 3;
  op_smooth_criterion = 1;
  op_ada_use_intpoints = 2;
  op_save_inter_meshes = 0;
  op_minlen = 0.0;
  op_maxarea = 0.0;
  op_minangle = 20.0;
  op_cosminangle = cos(op_minangle/180.*PI);
  //op_maxratio2 = 0.0;
  op_target_length = 0.0;
  // Angle bounds for avoiding very streched triangles.
  op_check_min_angle = 0;
  op_target_min_angle = 2.0;
  //op_target_max_angle = 176.0;
  op_target_cos_min_angle = cos(op_target_min_angle/180.*PI);
  //op_target_cos_max_angle = cos(op_target_max_angle/180.*PI);
  //op_hde_s1 = op_hde_s2 = 1.0;
  op_smooth_deltat = 0.1;
  op_edge_collapse_factor = 0.45;
  op_edge_split_factor = 1.45;

  _a11 = 1.0;
  _a21 = 0.;
  _a22 = 1.0;

  SizeFunc = NULL;
  Func = GradX = GradY = NULL;
  //op_test_fun = 0;

  so_nosort = so_norandom = so_nobrio = 0;
  so_hilbert_order = 24;
  so_hilbert_limit = 8;
  so_brio_threshold = 64;
  so_brio_ratio = 0.125;

  io_noindices = 0;
  io_firstindex = 0;
  io_poly = 0;
  io_inria_mesh = 0;
  io_with_metric = io_with_sol = io_with_grd = 0;
  io_voronoi = 0;
  io_point_array = 0;
  io_no_unused = 0;
  io_outedges = 0;
  io_out_voronoi = 0;
  io_dump_to_ucd = 0;
  io_dump_lift_map = 0;
  io_commandline[0] = '\0';
  io_infilename[0] = '\0';
  io_outfilename[0] = '\0';
  io_omtfilename[0] = '\0';
  io_xmax = io_xmin = io_ymax = io_ymin = 0.0;
  io_metric_min = io_metric_max = 0.;
  io_diagonal = io_diagonal2 = 0.0;
  io_tol_rel_gap = 1.e-3;
  io_tol_minangle = 0.1 * PI / 180.0; // Radian

  ct_in_vrts = ct_in_tris = ct_in_sdms = 0;
  ct_hullsize = ct_exteriors = 0;
  ct_unused_vrts = ct_segments = 0;
  ct_flipcount = 0; ct_flipcount_inverted = 0;

  vw_win = NULL;
  vw_imgs = NULL;
}

void Triangulation::initialize()
{
  in_vrts = in_sdms = NULL;
  tr_steiners = tr_segs = tr_tris = NULL;

  //tr_infvrt = NULL;
  tr_infvrt = new Vertex;
  tr_infvrt->init();
  tr_infvrt->idx = -1;
  tr_infvrt->typ = INFVERTEX;

  initialize_lookup_tables();
  reset_options();
}

void Triangulation::clean()
{
  if (in_vrts) delete [] in_vrts;
  if (in_sdms) delete [] in_sdms;
  if (tr_steiners) delete tr_steiners;
  if (tr_segs) delete tr_segs;
  if (tr_tris) delete tr_tris;
  if (tr_infvrt)  delete tr_infvrt;
  //if ((OMT_domain != NULL) && (OMT_domain != this)) delete OMT_domain;
}
