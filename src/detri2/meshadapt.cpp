#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

int mesh_adapt(
	// Inputs
	int nv,            // Number of input nodes (.node file)
	double* nodes,
	double* nvals,     // Nodal mesh size (.val)
	int *ntags,        // Nodal tags
	int ne,            // Number of input triangles (.ele file)
	int* eles,
	int* eletags,      // Element markers
	int ns,            // NUmber of input segments (.edge file)
	int* segs,
	int* stags,        // Segment markers
					   // Outputs
	int* outnv,        // Number of output nodes
	double** outnodes,
    double** outnvals, // *****
	int** outntags,    // Nodal tags
	int* outne,        // Number of output triangles
	int** outeles,
	int** outeletags,
	int* outns,            // NUmber of input segments (.edge file)
	int** outsegs,
	int** outstags,        // Segment markers
	int** out_p2t,     // output_points-to-input_cells map (default = NULL)
					   // Options,
	int iter,          // iterations (for Paraview) (default = 0)
    double minlen      // minimum edge length (default = 0)
)
{
	// Initialize the (empty) triangulaiton./Users/si/Programs/acoustic/adapt/meshadapt.cpp
	Triangulation *triang = new Triangulation();

	if (iter > 0) {
	  strcpy(triang->io_outfilename, "out");
	}

	// Mesh adaptation options.
	triang->op_metric = 1;
	triang->op_quality = 1;
    // if (1) { // for fu example.
    //   triang->op_minangle = 26.5;
    //   triang->op_minlen = 0.05; // for size r = 0.05
    // }
    triang->op_minlen = minlen; 

	//////////////////////////////////////////////////////////////////////////////
	// Read input nodes, eles,
	int firstindex = 1; // Set it to be 0 or 1.
	triang->io_firstindex = firstindex;
	double x, y;
    int i, idx;

	// Read the input points and weights.
	triang->ct_in_vrts = nv;
	triang->in_vrts = new Vertex[nv];
	for (i = 0; i < nv; i++) {
		Vertex *vrt = &(triang->in_vrts[i]);
		triang->ct_unused_vrts++;
		vrt->init();
		x = vrt->crd[0] = nodes[i * 2];
		y = vrt->crd[1] = nodes[i * 2 + 1];
		vrt->crd[2] = x*x + y*y; // height
		vrt->idx = i + firstindex;
		vrt->val = nvals[i]; // Read nodal value
		vrt->tag = ntags[i];
	}

	// Read the input triangles.
	triang->ct_in_tris = ne;
	int log2objperblk = 0;
	int trinum = ne;
	while (trinum >>= 1) log2objperblk++;
	triang->tr_tris = new arraypool(sizeof(Triang), log2objperblk);

	for (i = 0; i < ne; i++) {
		Vertex *p1 = &(triang->in_vrts[eles[i * 3] - firstindex]);
		Vertex *p2 = &(triang->in_vrts[eles[i * 3 + 1] - firstindex]);
		Vertex *p3 = &(triang->in_vrts[eles[i * 3 + 2] - firstindex]);
		// Make sure all triangles are all CCW oriented.
		int ori = Orient2d(p1, p2, p3);
		if (ori < 0) {
			// Swap the points.
			Vertex *swap = p1;
			p1 = p2;
			p2 = swap;
		}
		Triang *tri = (Triang *)triang->tr_tris->alloc();
		tri->init();
		tri->vrt[0] = p1;
		tri->vrt[1] = p2;
		tri->vrt[2] = p3;
		tri->tag = eletags[i];
	}

	// Read input segments.
	int est_size = ns;
	log2objperblk = 0;
	while (est_size >>= 1) log2objperblk++;
	triang->tr_segs = new arraypool(sizeof(Triang), log2objperblk);

	for (i = 0; i < ns; i++) {
		if (stags[i] > 0) {
			Triang *seg = (Triang *)triang->tr_segs->alloc();
			seg->init();
			seg->vrt[0] = &(triang->in_vrts[segs[i * 2] - firstindex]);
			seg->vrt[1] = &(triang->in_vrts[segs[i * 2 + 1] - firstindex]);
			seg->tag = stags[i];
		}
	}

	triang->reconstruct_mesh(1); // Check_delaunay

    if (iter > 0) {
      triang->save_to_ucd(iter, 1); // save_val = 1
    }

	//////////////////////////////////////////////////////////////////////////////
	// For background mesh, read them again.
	triang->OMT_domain = new Triangulation();
	triang->OMT_domain->io_firstindex = firstindex;

	// Read the input points and weights.
	triang->OMT_domain->ct_in_vrts = nv;
	triang->OMT_domain->in_vrts = new Vertex[nv];
	for (i = 0; i < nv; i++) {
		Vertex *vrt = &(triang->OMT_domain->in_vrts[i]);
		triang->OMT_domain->ct_unused_vrts++;
		vrt->init();
	    x = vrt->crd[0] = nodes[i * 2];
		y = vrt->crd[1] = nodes[i * 2 + 1];
		vrt->crd[2] = x*x + y*y; // height
		vrt->idx = i + firstindex;
		vrt->val = nvals[i]; // Read nodal value
		vrt->tag = ntags[i];
	}

    if (out_p2t != NULL) {
      triang->OMT_domain->op_convex = 1;
    }

    if (triang->OMT_domain->op_convex) {
      triang->OMT_domain->tr_tris = new arraypool(sizeof(Triang), 10);
      idx = firstindex;
      for (i = 0; i < ne; i++) {
	    Vertex *p1 = &(triang->OMT_domain->in_vrts[eles[i*3] - firstindex]);
	    Vertex *p2 = &(triang->OMT_domain->in_vrts[eles[i*3+1] - firstindex]);
	    Vertex *p3 = &(triang->OMT_domain->in_vrts[eles[i*3+2] - firstindex]);
	    // Make sure all triangles are CCW oriented.
	    int ori = Orient2d(p1, p2, p3);
	    if (ori < 0) {
	      // Swap the points.
	      Vertex *swap = p1;
	      p1 = p2;
	      p2 = swap;
	    }
	    Triang *tri = (Triang *) triang->OMT_domain->tr_tris->alloc();
	    tri->init();
	    tri->vrt[0] = p1;
	    tri->vrt[1] = p2;
	    tri->vrt[2] = p3;
	    //tri->tag = idx; // eletags[i];
	    idx++;
	  }
	  triang->OMT_domain->reconstruct_mesh(0); // check_delaunay = 0
    } else {
      triang->OMT_domain->incremental_delaunay();
    }

	//////////////////////////////////////////////////////////////////////////////
	// Do mesh adapation.
	//triang->delaunay_refinement();

	if (triang->op_metric || (triang->op_minlen > 0.0)) {
		triang->coarsen_mesh();
	}

	if (triang->op_quality) { // -q
		if (triang->op_db_verbose) {
			printf("Adding Steiner points to enforce quality.\n");
		}
		// Calculate the squre of the maximum radius-edge ratio.
		// r/d = 1/(2*sin(theta_min));
		triang->op_maxratio2 = 0.5 / sin(PI * triang->op_minangle / 180.0);
		triang->op_maxratio2 *= triang->op_maxratio2;

		int est_size = triang->tr_tris->objects * 2;
		int log2objperblk = 0;
		while (est_size >>= 1) log2objperblk++;
		arraypool *enctris = new arraypool(sizeof(Triang), log2objperblk);

		if (triang->tr_steiners == NULL) {
			triang->tr_steiners = new arraypool(sizeof(Vertex), log2objperblk);
		}

		// Put all triangles into list
		for (int i = 0; i < triang->tr_tris->used_items; i++) {
			Triang *tri = (Triang *)triang->tr_tris->get(i);
			if (tri->is_deleted()) continue;
			triang->check_triangle(tri, enctris);
		}

		triang->repair_triangles(enctris);

		delete enctris;
	}

	//if (iter > 0) {
	//  triang->save_to_ucd(iter * 2 + 1); // Save mesh (for Paraview)
	//}

	//////////////////////////////////////////////////////////////////////////////
	// Export mesh.
	triang->io_no_unused = 1; // -IJ
	*outnv = triang->ct_in_vrts + triang->tr_steiners->objects;
	if (triang->io_no_unused) (*outnv) -= triang->ct_unused_vrts;
	*outnodes = new double[*outnv * 2];
    *outnvals = new double[*outnv];
	*outntags = new int[*outnv];

	idx = firstindex;
	for (i = 0; i < triang->ct_in_vrts; i++) {
		if (triang->io_no_unused) { // -IJ
			if (triang->in_vrts[i].typ == UNUSEDVERTEX) continue;
		}
		Vertex *v = &(triang->in_vrts[i]);
		int ii = idx - firstindex;
		(*outnodes)[ii * 2] = v->crd[0];
		(*outnodes)[ii * 2 + 1] = v->crd[1];
        (*outnvals)[ii] = v->val;
		(*outntags)[ii] = v->tag;
		v->idx = idx;
		idx++;
	}
	if (triang->tr_steiners != NULL) {
		for (i = 0; i < triang->tr_steiners->used_items; i++) {
			Vertex *v = (Vertex *)triang->tr_steiners->get(i);
			if (v->is_deleted()) continue;
			int ii = idx - firstindex;
			(*outnodes)[ii * 2] = v->crd[0];
			(*outnodes)[ii * 2 + 1] = v->crd[1];
            (*outnvals)[ii] = v->val;
			(*outntags)[ii] = v->tag;
			v->idx = idx;
			idx++;
		}
	}

    if (triang->ct_exteriors > 0) {
      triang->remove_exteriors();
    }

	*outne = triang->tr_tris->objects - triang->ct_hullsize;
	*outeles = new int[*outne * 3];
	*outeletags = new int[*outne];

	int eleidx = 0;
	for (i = 0; i < triang->tr_tris->used_items; i++) {
		Triang* t = (Triang *)triang->tr_tris->get(i);
		if (t->is_deleted() || t->is_hulltri()) continue;
		(*outeles)[eleidx * 3] = t->vrt[0]->idx;
		(*outeles)[eleidx * 3 + 1] = t->vrt[1]->idx;
		(*outeles)[eleidx * 3 + 2] = t->vrt[2]->idx;
		(*outeletags)[eleidx] = t->tag;
		eleidx++;
	}

	// Output all edges.
	triang->io_outedges = 1;
    *outns = (3 * (triang->tr_tris->objects - triang->ct_hullsize) + triang->ct_hullsize) / 2;
	*outsegs = new int[(*outns) * 2];
	*outstags = new int[*outns];
	idx = 0;

	TriEdge E, S;
	//int uvflag = triang->tr_infvrt->adj.tri->get_unvisited();

	for (i = 0; i < triang->tr_tris->used_items; i++) {
		Triang* tri = (Triang *)triang->tr_tris->get(i);
		if (tri->is_deleted()) continue;
		E.tri = tri;
		if (!E.tri->is_hulltri()) {
		  for (E.ver = 0; E.ver < 3; E.ver++) {
			if (E.esym().tri->is_hulltri() || !E.esym().tri->is_infected()) {
				// A segment will always be output.
				if (E.is_segment()) {
					S = triang->get_segment(E);
					if (S.tri->tag == 0) S.tri->tag = -1;
					(*outsegs)[idx * 2] = E.org()->idx;
					(*outsegs)[idx * 2 + 1] = E.dest()->idx;
					(*outstags)[idx] = S.tri->tag;
					idx++;
				}
				else {
					if (triang->io_outedges) { // -Ie
						//if (!E.tri->is_hulltri()) {
							// Output an interior edge.
							(*outsegs)[idx * 2] = E.org()->idx;
							(*outsegs)[idx * 2 + 1] = E.dest()->idx;
							(*outstags)[idx] = 0;
							idx++;
						//}
					}
				}
			}
		  } // for (E.ver = 0, ...)
          E.tri->set_infect();
		} // if (!E.tri->is_hulltri()) {
	}

    // Uninfect all triangles.
    for (i = 0; i < triang->tr_tris->used_items; i++) {
		Triang* tri = (Triang *)triang->tr_tris->get(i);
		if (tri->is_deleted()) continue;
		E.tri = tri;
		if (!E.tri->is_hulltri()) {
          E.tri->clear_infect();
		} // if (!E.tri->is_hulltri()) {
	}

    if (out_p2t != NULL) {
	  // Output point-to-cell map.
      // Index all background triangles.
      idx = firstindex;
      for (i = 0; i < triang->OMT_domain->tr_tris->used_items; i++) {
        Triang *tri = (Triang *) triang->OMT_domain->tr_tris->get(i);
        assert(!tri->is_deleted());
        tri->tag = idx;
        idx++;
      }

      int *p2t = new int[(*outnv)];
      TriEdge E;
      int loc;
      idx = 0; // firstindex;
	  for (i = 0; i < triang->ct_in_vrts; i++) {
	  	//if (triang->io_no_unused) { // -IJ
        if (triang->in_vrts[i].typ == UNUSEDVERTEX) continue;
	  	//}
	  	Vertex *v = &(triang->in_vrts[i]);
        loc = triang->OMT_domain->locate_point(v, E, 1, 0); // encflag = 0
        if (E.tri->is_hulltri()) {
          E = E.esym();
        }
        if (E.tri->is_hulltri()) {
          assert(0);
        } else if (E.tri->is_exterior()) { // Outside
          assert(0); p2t[idx] = 0;
        } else if (loc == 3) {
          p2t[idx] = - E.org()->idx;
        } else {
          p2t[idx] = E.tri->tag;
        }
        idx++;
	  }
	  if (triang->tr_steiners != NULL) {
	  	for (i = 0; i < triang->tr_steiners->used_items; i++) {
          Vertex *v = (Vertex *) triang->tr_steiners->get(i);
          if (v->is_deleted()) continue;
          loc = triang->OMT_domain->locate_point(v, E, 1, 0); // encflag = 0
          if (E.tri->is_hulltri()) {
            E = E.esym();
          }
          if (E.tri->is_hulltri()) {
            assert(0);
          } else if (E.tri->is_exterior()) { // Outside
            assert(0); p2t[idx] = 0;
          } else if (loc == 3) {
            p2t[idx] = - E.org()->idx;
          } else {
            p2t[idx] = E.tri->tag;
          }
          idx++;
	  	}
	  }
      *out_p2t = p2t;
    } //if (*out_p2t != NULL)

    delete triang->OMT_domain;
	delete triang;

	return 0;
}

//==============================================================================

int main(int argc, char *argv[])
{
  Triangulation *Tr = new Triangulation();

  return 1;
}
