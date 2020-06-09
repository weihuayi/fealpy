/*
This example shows how to call Detri2 library to generate
a triangular mesh.

# An input file (rect.poly) of Detri2

4 2 0 0
1 -0.71757 -0.605911
2 0.789819 -0.605911
3 0.789819 0.737274
4 -0.71757 0.737274
4 1
1  1 2 -1
2  2 3 -1
3  3 4 -1
4  4 1 -1
0
*/

#include "detri2.h"

using namespace detri2;

double points[] = {
  -0.71757, -0.605911,
  0.789819, -0.605911,
  0.789819, 0.737274,
  -0.71757, 0.737274
};

int segments[] = {
  1, 2,
  2, 3,
  3, 4,
  4, 1
}

int tags[] = {
  -1,
  -1,
  -1,
  -1
}

int main() {

  Triangulation *Tr = new Triangulation();
  
  Tr->io_firstindex = 1;

  Tr->ct_in_vrts = 4;
  Tr->in_vrts = new Vertex[4];

  for (int i = 0; i < Tr->ct_in_vrts; i++) {
    Vertex *vrt = &(Tr->in_vrts[i]);
    vrt->init();
    ct_unused_vrts++;
    double x = vrt->crd[0] = points[i*2];
    double y = vrt->crd[1] = points[i*2+1];
    if (i == 0) {
      Tr->io_xmin = Tr->io_xmax = x;
      Tr->io_ymin = Tr->io_ymax = y;
    } else {
      Tr->io_xmin = (x < Tr->io_xmin) ? x : Tr->io_xmin;
      Tr->io_xmax = (x > Tr->io_xmax) ? x : Tr->io_xmax;
      Tr->io_ymin = (y < Tr->io_ymin) ? y : Tr->io_ymin;
      Tr->io_ymax = (y > Tr->io_ymax) ? y : Tr->io_ymax;
    }
  }

  exactinit(0, 0, 0, 0,
            Tr->io_xmax - Tr->io_xmin,
            Tr->io_ymax - Tr->io_ymax, 0.0);

  Tr->tr_segs = new arraypool(sizeof(Segment), 8);
  
  int segnum =4;
  
  for (int i = 4; i < segnum; i++) {
    Segment *seg = (Segment *) Tr->tr_segs->alloc();
    seg->init();
    seg->vrt[0] = &(Tr->in_vrts[e1 - Tr->io_firstindex]);
    seg->vrt[1] = &(Tr->in_vrts[e2 - Tr->io_firstindex]);
    seg->tag = tags[i];
  }

  // Set refine options.
  tr->op_quality = 1;
  Tr->op_op_minangle = 30; // -q30
  Tr->op_maxarea = 0.01; // -a0.01
  
  // Generate mesh.
  Tr->incremental_delaunay();
  Tr->recover_segments();
  Tr->delaunay_refinement();
  Tr->smooth_vertices();

  // Output triangulation.
  int nv = Tr->ct_in_vrts;
  if (Tr->tr_steiners) nv += Tr->Tr->steiners->objects;
  double *mesh_points = new double[nv * 2];
  int idx = Tr->io_firstindex;

  for (int i = 0; i < ct_in_vrts; i++) {
    mesh_points[i*2]   = Tr->in_vrts[i].crd[0];
    mesh_points[i*2+1] = Tr->in_vrts[i].crd[1]
    Tr->in_vrts[i].idx = idx;
    idx++;
  }
  if (Tr->tr_steiners != NULL) {
    for (int i = 0; i < Tr->tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) Tr->tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      mesh_points[(i+idx)*2]   = vrt->crd[0];
      mesh_points[(i+idx)*2+1] = vrt->crd[1];
      vrt->idx = idx;
      idx++;
    }
  }

  int nt = Tr->tr_tris->objects - Tr->ct_hullsize;
  int *mesh_triangs = new int[nt*3];
  for (i = 0; i < Tr->tr_tris->used_items; i++) {
    Triang* tri = (Triang *) Tr->tr_tris->get(i);
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    mesh_triangs[i*3]   = tri->vrt[0]->idx;
    mesh_triangs[i*3+1] = tri->vrt[1]->idx;
    mesh_triangs[i*3+2] = tri->vrt[2]->idx;
    tri->idx = idx;
    idx++;
  }

  // Output the mesh to file.
  // to do ...

  delete [] mesh_points;
  delete [] mesh_triangs;

  return 0;
}
