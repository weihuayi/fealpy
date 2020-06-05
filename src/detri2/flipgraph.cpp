#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using namespace detri2;

#define MAX_EDGE_NUM 100

class FlipData {
 public:
  int type;      // FT_13, FT_22, FT_31
  Vertex *vt[4]; // the four vertices

  void init() {
    type = FLIP_UNKNOWN;
    vt[0] = vt[1] = vt[2] = vt[3] = NULL;
  }
  FlipData() {init();}
};

// Every node in this flip graph is a lifted triangulation
//   We only store the set of triangles (using vertex indices)
//   In addition, we stroe connections to all other nodes 
//   (lifted triangulations) which are connected to it.
 
class FlipGraphNode {
 public:
  int level; // level=0 is the root.
 
  // The lifted triangulation
  int num_tris;
  Triang *tri_list; 

  // Connections to nodes which are below to it.
  int           num_in;
  FlipGraphNode *in_nodes;
  FlipData      *in_flips;

  // Connections to nodes which are higher than it.
  int           num_out;
  FlipGraphNode *out_nodes;
  FlipData      *out_flips;

  void init() {
    level = 0;
    num_tris = 0;
    tri_list = NULL;
    num_in = 0;
    in_nodes = NULL;
    in_flips = NULL;
    num_out = 0;
    out_nodes = NULL;
    out_flips = NULL;
  }

  void clean() {
    if (num_tris > 0) {
      delete [] tri_list;
    }
    if (num_in > 0) {
      delete [] in_nodes;
      delete [] in_flips;
    }
    if (num_out > 0) {
      delete [] out_nodes;
      delete [] out_flips;
    }
  }

  FlipGraphNode() {init();}
  ~FlipGraphNode() {clean();}

  //void generate_out_nodes();
};

void FlipGraphNode::generate_out_nodes()
{
  TriEdge E, tt[2];
  double ori;
  int i;

  Vertex* V1[MAX_EDGE_NUM], V2[MAX_EDGE_NUM];
  int edge_count = 0;

  // First get all locally non-regular edges.
  for (i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted()) continue;
    if (!E.tri->is_hulltri()) {
      for (E.ver = 0; E.ver < 3; E.ver++) {
        if (!E.esym().tri->is_infected()) {
          //E.set_edge_infect();
          //* (TriEdge *) fqueue->alloc() = E;
          // Check if this edge is locally regular
          ori = 0.0;
          tt[0] = E;
          tt[1] = tt[0].esym();
          if (!tt[0].tri->is_hulltri()) {
            if (!tt[1].tri->is_hulltri()) { 
              // An interior edge.
              //printf("  O3d: (%d, %d, %d, %d)\n", tt[0].org()->idx, tt[0].dest()->idx, tt[0].apex()->idx, tt[1].apex()->idx);
              ori = Orient3d(tt[0].org(), tt[0].dest(), tt[0].apex(), tt[1].apex()) 
                  * op_dt_nearest;
              //printf("  O3d = %g, op_dt_nearest=%d\n", ori, op_dt_nearest);
            }
          }
          if (ori > 0.0) {
            // Found a locally non-regular edge. Save it.
            V1[edge_count] = E.org();
            V2[edge_count] = E.dest();
            edge_count++;
            if (edge_count >= MAX_EDGE_NUM) {
              assert(0); // Increase the number.
            }
          }
        }
      } // E.ver
      E.tri->set_infect();
    } // if (!E.tri->is_hulltri()) 
  } // i
  // Uninfect all triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted()) continue;
    if (!E.tri->is_hulltri()) {
      E.tri->clear_infect();
    }
  }

  for (i = 0; i < edge_count; i++) {
    
  }
}

