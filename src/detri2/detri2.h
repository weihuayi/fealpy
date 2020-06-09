#ifndef __DETRI2_H__  // Include this file only once!
#define __DETRI2_H__

namespace detri2 {

// In case NULL is not defined (e.g., on Windows).
#ifndef NULL
  #define NULL 0
#endif

#define REAL double

#define UNDEFINED       127

// Vertex types
#define UNUSEDVERTEX  0
#define INFVERTEX     1
#define SEGMENTVERTEX 2
#define FREEVERTEX    3
#define STEINERVERTEX 4
#define DEADVERTEX    5

// Flip types
#define FLIP_INVERTED  -2
#define FLIP_UNKNOWN   -1
#define FLIP_HULL   0
#define FLIP_CONST  1
#define FLIP_31_NM  2
#define FLIP_42_NM  3
#define FLIP_13     4
#define FLIP_24     5
#define FLIP_22     6
#define FLIP_31     7
#define FLIP_42     8

// Point location results
#define LOC_UNKNOWN    -1
#define LOC_IN_OUTSIDE  0
#define LOC_IN_TRI      1
#define LOC_ON_EDGE     2
#define LOC_ON_VERT     3
#define LOC_ENC_SEG     4

// Check if the edge [e1,e2] is intersected by any segment or vertex.
#define INTERSECT_NONE           0   // initial
#define INTERSECT_SHARE_EDGE     1   // A edge exists
#define INTERSECT_CONFLICT       2   // A segment already exists
#define INTERSECT_VERTEX         3   // cutting by a vertex
#define INTERSECT_SEGMENT        4   // intersecting another segment

// Type of metrics
#define METRIC_Euclidean_no_weight     0
#define METRIC_Euclidean    1
#define METRIC_Riemannian   2
#define METRIC_HDE          3

// Vertex smoothing criteria
#define SMOOTH_LAPLACIAN    1
#define SMOOTH_CVT          2
#define SMOOTH_DISTMESH     3

extern double PI; // = 3.14159265358979323846264338327950288419716939937510582;

//==============================================================================
// Mesh data structure

// Lookup tables for a speed-up in operations
// They are declared and initialised in detri2.cpp

// Pre-calcuated bit masks
extern unsigned int _test_bit_masks[32];     // = 2^i, i = 0, ..., 31.
extern unsigned int _clear_bit_masks[32];    // = ~(2^i), i = 0, ..., 31.
extern unsigned int _extract_bits_masks[32]; // = 2^i - 1, i = 0, ..., 31.

extern unsigned char _enext_tbl[3], _eprev_tbl[3];
extern unsigned char _vo[3], _vd[3], _va[3];

void initialize_lookup_tables();

//==============================================================================

class Vertex;
class Triang;

class TriEdge 
{
public:
  Triang*       tri;
  unsigned char ver; // = 0,1,2

  TriEdge() {tri = NULL; ver = 0;}
  TriEdge(Triang* _t, int _v) {tri = _t; ver = _v;}

  bool is_connected();
  void connect(const TriEdge& te);

  TriEdge enext();
  TriEdge eprev();
  TriEdge esym();
  TriEdge enext_esym();
  TriEdge eprev_esym(); // ccw rotate
  TriEdge esym_enext(); //  cw rotate
  TriEdge esym_eprev();

  bool is_edge_infected();
  void set_edge_infect();
  void clear_edge_infect();

  bool is_segment();
  void set_segment();
  void clear_segment();
  Triang *get_segment();

  Vertex* org();
  Vertex* dest();
  Vertex* apex();
  void set_vertices(Vertex *pa, Vertex *pb, Vertex *pc);

  void print(); // debug
}; // class TriEdge

class VertexData; 

class Vertex
{
 public:
  REAL       crd[3], wei; // x, y, height, weight (height = x^2+y^2 - wei).
  REAL       val;    // mesh size (or density)
  REAL       fval;   // function value (or nodal FEM solution)
  REAL       grd[2]; // gradient of a function (not normalised)
  REAL       mtr[3]; // metric tensor (a11,a21,a22)
  int        idx;    // vertex index (0 or 1-based).
  int        tag;    // Boundary tag.
  char       typ;    // Vertex type.
  char       flags;  // flags of infected, etc.
  TriEdge    adj;    // Adjacent (or dual) triangle.
  TriEdge    on_bd;  // A boundary segment containing this vertex.
  TriEdge    on_dm;  // A OMT_domain triangle containing this vertex.
  Vertex     *Pair;  // A paired vertex (e.g., on periodic boundary).
  Vertex     *Next, *Prev;  // next vertex in a link list.
  VertexData *Data;  // User provided vertex data

  void init() {
    crd[0] = crd[1] = crd[2] = wei = val = 0.0;
    mtr[0] = mtr[1] = mtr[2] = grd[0] = grd[1] = fval = 0.0;
    idx = tag = typ = flags = 0;
    adj.tri = on_bd.tri = on_dm.tri = NULL;
    adj.ver = on_bd.ver = on_dm.ver = 0;
    Pair = Next = Prev = NULL;
    Data = NULL;
  }

  Vertex() {init();}

  // Primitive functions
  bool is_infected() {return flags & 1;} // bit 0 (1 bit)
  void set_infect() {flags |= 1;}
  void clear_infect() {flags &= _clear_bit_masks[0];}

  bool is_fixed() {return flags & 2;} // bit 1 (1 bit)
  void set_fix() {flags |= 2;}
  void clear_fix() {flags &= _clear_bit_masks[1];}

  // for mesh refinement.
  bool is_acute() {return flags & 4;} // bit 1 (1 bit)
  void set_acute() {flags |= 4;}
  void clear_acute() {flags &= _clear_bit_masks[2];}

  bool is_deleted() {return typ == DEADVERTEX;}
  void set_deleted() {typ = DEADVERTEX;}

  void print(); // debug
}; // class Vertex

class Triang
{
 public:
  Vertex*  vrt[3];
  TriEdge  nei[3];
  REAL     cct[3]; // circumcenter and radius (dual and weight)
  REAL     val;    // a scalar value (also used as mesh size)
  int      flags;  // Encode, hullflag, infect, etc.
  int      tag;    // Boundary marker.
  int      idx;    // Index of its dual (Voronoi) vertex
  TriEdge  on;     // A domain triangle containing cct (Voronoi.cpp).
  Triang   *Next;  // next triangle in a link list.

  void init() {
    vrt[0] = vrt[1] = vrt[2] = NULL;
    nei[0].tri = nei[1].tri = nei[2].tri = NULL;
    nei[0].ver = nei[1].ver = nei[2].ver = 0;
    cct[0] = cct[1] = cct[2] = 0.0;
    val = 0.0;
    on.tri = NULL; on.ver = 0;
    flags = tag = idx = 0;
    Next = NULL;
  }

  Triang() {init();}

  bool is_hulltri() {return (flags & 1);} // bit 0 (1 bit)
  void set_hullflag() {flags |= 1;}
  void clear_hullflag() {flags &= _clear_bit_masks[0];}

  bool is_infected() {return (flags & 2);} // bit 1 (1 bit)
  void set_infect() {flags |= 2;}
  void clear_infect() {flags &= _clear_bit_masks[1];}

  bool is_exterior() {return (flags & _test_bit_masks[7]);} // bit 7 (1 bit)
  void set_exterior() {flags |= _test_bit_masks[7];}
  void clear_exterior() {flags &= _clear_bit_masks[7];}

  bool is_dual_in_exterior() {return (flags & _test_bit_masks[6]);} // bit 6 (1 bit)
  void set_dual_in_exterior() {flags |= _test_bit_masks[6];}
  void clear_dual_in_exterior() {flags &= _clear_bit_masks[6];}

  bool is_dual_on_bdry() {return (flags & _test_bit_masks[5]);} // bit 5 (1 bit)
  void set_dual_on_bdry() {flags |= _test_bit_masks[5];}
  void clear_dual_on_bdry() {flags &= _clear_bit_masks[5];}

  bool is_deleted() {return nei[0].ver == UNDEFINED;}
  void set_deleted() {nei[0].ver = UNDEFINED;}

  void print(int deatil = 0); // debug
};

// A Segment is just a degenerated Triang (vrt[2] == NULL)

typedef Triang Segment;

//==============================================================================
// Resizeable arrays

class arraypool
{
 public:
  int  objectbytes;           // size of an object of the array
  int  objectsperblock;       // bottom-array length (must be 2^k)
  int  log2objectsperblock;   // k
  int  objectsperblockmask;   // 2^k - 1
  int  toparraylen;           // top-array length (> 0)
  int  objects, used_items;   // actual and allocated objects
  int  blks, rsdu;            // for fast access entries.
  unsigned long totalmemory;  // memory used
  char **toparray;            // the top-array
  void *deaditemstack;        // a stack of dead elements (can be re-used)

  arraypool(int sizeofobject, int log2objperblk);
  ~arraypool();

  void  poolinit(int sizeofobject, int log2objperblk);
  void  clean();
  char* getblock(int objectindex);
  char* alloc();
  void  dealloc(void *dyingitem);
  char* lookup(int index); // save
  char* get(int index); // fast-lookup (unsave)
  void  traversalinit();
  int   get_block_items(int hi);
  char* get(int hi, int li) // (fast) fast-lookup (unsave) 
          {return (toparray[hi] + li * objectbytes);}

  void print(); // debug
}; // class arraypool

//==============================================================================
// Robust adaptive predicates (by J. R. Shewchuk, pred3d.cpp)

void exactinit(int, int, int, int, REAL, REAL, REAL);
REAL Orient2d(Vertex *pa, Vertex *pb, Vertex *pc);
REAL Orient3d(Vertex *pa, Vertex *pb, Vertex *pc, Vertex *pd);
REAL Orient4d(Vertex *pa, Vertex *pb, Vertex *pc, Vertex *pd, Vertex *pe);

//==============================================================================
// Metric functions (metric.cpp)

//extern int metric_use_gmp; // an option set by op_use_gmp (must initialise it)

REAL get_MatVVP2x2(double a11, double a21, double a22,
                   double *lambda1, double *lambda2,
                   double vec1[2], double vec2[2]);
REAL get_rotation_angle(double vec1[2]);

int  line_line_intersection(double X0, double Y0, double X1, double Y1,
                            double X2, double Y2, double X3, double Y3,
                            double *t1, double *t2);

int  get_orthocenter(REAL, REAL, REAL,     // Ux, Uy, U_weight
                     REAL, REAL, REAL,     // Vx, Vy, V_weight
                     REAL, REAL, REAL,     // Wx, Wy, W_weight
                     REAL*, REAL*, REAL*,  // Cx, Cy, C_weight (= radius^2)
                     double a11, double a21, double a22);

int  get_bissector(REAL, REAL, REAL,       // Ux, Uy, U_weight
                   REAL, REAL, REAL,       // Vx, Vy, V_weight
                   REAL*, REAL*, REAL*,    // Cx, Cy, C_weight (= radius^2)
                   double a11, double a21, double a22);

//==============================================================================
// Sorting vertices (sort.cpp)

void hilbert_init(int n);
int  hilbert_split2(Vertex** vertexarray, int arraysize, int gc0, int gc1,
                    REAL, REAL, REAL, REAL);
void hilbert_sort2(Vertex** vertexarray, int arraysize, int e, int d,
                   int hilbert_order, int hilbert_limit,
                   REAL, REAL, REAL, REAL, int depth);
void brio_multiscale_sort2(Vertex**, int, int threshold, REAL ratio,
                           int hilbert_order, int hilbert_limit,
                           REAL, REAL, REAL, REAL);

void dump_vertexarray(Vertex **vrtarray, int arysize);

// Scalar function (return a value for a given vertex)
typedef REAL (scalarFunType)(const Vertex * pt);

//==============================================================================

class Triangulation
{
 public:
  // Input (in_) points, triangles, segments, subdomains.
  Vertex* in_vrts;
  Vertex* in_sdms;

  // Triangulation (tr_) elements
  arraypool* tr_steiners; // Steiner vertices
  arraypool* tr_segs;
  arraypool* tr_tris;
  Vertex*    tr_infvrt;   // The infinite vertex

  Triangulation *OMT_domain; // A background mesh (default is itself)
  Triang *tr_recnttri; // A recent triangle for searching.
  bool tr_nonconvex;  // Is triangulation non-convex (and with holes)?

  // Input and output (io_)
  int  io_noindices;   // 0, -IN no input index column
  int  io_firstindex;  // 0, -I0 or -I1
  int  io_poly;        // read .poly or .smesh
  int  io_inria_mesh;
  int  io_voronoi;
  int  io_point_array;
  int  io_with_metric, io_with_sol, io_with_grd, io_with_wei;
  int  io_keep_unused;   // 0, -IJ output unused input vertices
  int  io_outedges;      // 0, -Ie output all edges of the mesh
  int  io_out_voronoi;   // 0, -Iv output the power diagram
  int  io_out_ucd;       // 0, -Iu save a .inp file for Paraview
  int  io_dump_to_ucd;   // 0, -Id save intermediate meshes
  int  io_dump_lift_map; // 0, -Il save the lifting map
  char io_commandline[1024];
  char io_infilename[1024];
  char io_outfilename[1024];
  char io_omtfilename[1024]; // -m filename
  REAL io_xmax, io_xmin, io_ymax, io_ymin;
  REAL io_metric_min, io_metric_max;
  REAL io_diagonal, io_diagonal2;
  REAL io_tol_rel_gap;  // -T#, relative tolerance (io_diagonal) for distinct vertices (1e-3).
  REAL io_tol_minangle; // -T,# minimum collinear angle (default < 0.01 degree)

  // Algorithm options and parameters (op_).
  int  op_dt_nearest; // -u, 1 or -1 (farthest dt)
  int  op_db_verbose; // -V
  int  op_no_gabriel; // -G
  int  op_no_bisect;  // -Y preserve boundary
  int  op_no_incremental_flip; // -F
  int  op_poly; // -p
  int  op_convex; // -c
  int  op_quality; // -q
  int  op_use_gmp; // -gmp
  int  op_mpfr_precision; // -gmp 1024

  // Mesh adaptation options/parameters.
  int  op_metric; // 1 (Euclidean), 2 (Riemannian constant), 3 (HDE)
  int  op_round_flip; // default = 0, needed by aniso and HDE mesh adaptation
  int  op_max_iter; // 3
  int  op_use_coarsening; // default = 1
  int  op_use_splitting;  // default = 1
  int  op_use_smoothing;  // default = 0
  int  op_smooth_criterion; // 1 (Laplacian)
  int  op_smooth_iter;  // 3
  int  op_ada_use_intpoints; // default = 2
  int  op_save_inter_meshes;
  REAL op_minlen; // -L#
  REAL op_maxarea; // -a#
  REAL op_minangle; // -q# default (20 degree)
  REAL op_cosminangle; // from op_minangle
  //REAL op_maxratio2; // from op_minangle
  //REAL op_max_abs_Fun;
  REAL op_target_length; // The desired unit edge length
  int  op_check_min_angle; //
  REAL op_target_min_angle;  // default = 2 degree
  REAL op_target_cos_min_angle;
  //REAL op_hde_s1, op_hde_s2; // HDE parameters
  REAL op_edge_collapse_factor; // = 0.45;
  REAL op_edge_split_factor; // default = 1.45;
  REAL op_smooth_deltat; // default 0.1

  // A user-provided function, its gradients, and its Hessian matrix.
  scalarFunType *SizeFunc, *Func, *GradX, *GradY;
  //int  op_test_fun; // test only

  // Sort (so_)
  int  so_nosort; // -SN
  int  so_norandom; // -SR
  int  so_nobrio; // -SB
  int  so_hilbert_order;  // =24, -Sh#,#
  int  so_hilbert_limit;  // =8,
  int  so_brio_threshold; // =64, -Sb#,#
  REAL so_brio_ratio;     // =0.125,

  // Counters (ct_)
  int ct_in_vrts, ct_in_tris, ct_in_sdms;
  int ct_hullsize, ct_exteriors;
  int ct_unused_vrts, ct_segments;
  int ct_flipcount, ct_flipcount_inverted;

  // Quadratic form parameters: // a11 * x^2 + 2*a21 xy + a22 * y^2,
  //   default: a11 = a22 = 1.0; a21 = 0;
  REAL _a11, _a21, _a22;
  //REAL _lambda1, _lambda2, _v1[2], _v2[2];

  // For visualisation
  void *vw_win;
  arraypool *vw_imgs;

  // Construction / Destruction
  Triangulation();
  ~Triangulation();

  void initialize();
  void clean();

  void reset_options();

  int  check_mesh(int topo_only, int check_level);
  void quality_statistics();
  void memory_statistics();
  void mesh_statistics();

  // Input / Output (io.cpp)
  int  parse_commands(int argc, char* argv[]);
  int  read_nodes();  
  int  read_edge();
  int  read_ele();
  int  read_region();
  int  read_weights();
  int  read_metric();
  int  read_sol();
  int  read_grd();
  int  read_area();
  int  read_poly();
  int  read_inria_mesh();
  int  read_poly_mesh();
  int  read_point_array();
  int  read_mesh();
  int  remove_exteriors();
  void save_nodes(int Steiner_only);
  void save_weights(int Steiner_only);
  void save_metric(int Steiner_only);
  void save_poly(int Steiner_only);
  void save_triangulation();
  void save_edges();
  void save_smesh();
  void save_inria_mesh();
  void save_to_ucd(int meshidx, int save_val);
  void save_voronoi(int ucd = 0);
  void save_neighbors();

  // Metrics (metric.cpp)  
  REAL get_innerproduct(Vertex *v1, Vertex *v2);
  REAL get_distance(Vertex *v1, Vertex *v2);
  REAL get_angle(Vertex* v0, Vertex* v1, Vertex* v2); // angle at v0 in [0,pi].
  REAL get_cosangle(Vertex* v0, Vertex* v1, Vertex* v2);
  REAL get_tri_area(Vertex* pa, Vertex* pb, Vertex* pc);
  int  get_tri_normal(Vertex* pa, Vertex* pb, Vertex* pc, REAL normal[3]);
  REAL get_dihedral(Vertex* pa, Vertex* pb, Vertex* pc, Vertex* pd);
  REAL get_min_cosangle(Vertex* v0, Vertex* v1, Vertex* v2);

  // Flips (flips.cpp)
  int  flip13(Vertex *pt, TriEdge *tt);
  int  flip31(TriEdge *tt, Vertex **ppt);
  int  flip22(TriEdge *tt);
  int  flip24(Vertex *pt, TriEdge *tt);
  int  flip42(TriEdge *tt, Vertex **ppt);
  int  flip_check(TriEdge *te);
  int  flip(TriEdge tt[4], Vertex **ppt, int& fflag, arraypool* fqueue);
  int  first_triangle(Vertex *pa, Vertex *pb, Vertex *pc);

  // (Weighted) Delaunay triangulation (delaunay.cpp)
  int  locate_point(Vertex *pt, TriEdge &E, int encflag);
  bool regular_test(Vertex* pa, Vertex* pb, Vertex* pc, Vertex* pd);
  int  lawson_flip(Vertex *pt, int hullflag, arraypool *fqueue);
  int  sort_vertices(Vertex* vrtarray, int, Vertex**& permutarray);
  int  first_tri(Vertex **ptlist, int ptnum);
  int  incremental_delaunay();

  // Constrained Delaunay triangulation (constrained.cpp)
  bool is_fixed_vertex(Vertex *pt);
  int  get_edge(Vertex *e1, Vertex *e2, TriEdge& E);
  int  remove_segment(Triang* seg);
  int  insert_segment(TriEdge E, int stag, REAL val, Triang** pseg);
  int  find_direction(TriEdge& E, Vertex *pt);
  int  detect_intersection(Vertex *e1, Vertex *e2, TriEdge &S);
  int  recover_edge(Vertex *e1, Vertex *e2, TriEdge& E, arraypool*);
  int  insert_segment_intersect(Vertex *e1, Vertex *e2, int stag, REAL val);
  int  recover_segments();
  int  set_ridgevertices();
  int  set_subdomains(); // and mark exterior triangles
  int  reconstruct_mesh(int);

  // Power Voronoi diagram (voronoi.cpp)
  int  search_point(Vertex *pt, TriEdge &E, int encflag);
  int  get_boundary_cut_dualedge(Vertex& In_pt, Vertex& Out_pt, TriEdge& S);
  int  get_hulltri_orthocenter(Triang* hulltri);
  int  get_tri_orthocenter(Triang* tri);
  int  get_powercell(Vertex *mesh_vertex, Vertex** pptlist, int* ptnum);
  int  get_mass_center(Vertex* ptlist, int ptnum, REAL mc[2]);

  // Delaunay refinement (adding Steiner points) (refine.cpp)
  bool is_acute_vertex(TriEdge& S);
  void mark_acute_vertices();
  int  check_segment(Vertex *pt, TriEdge &S, arraypool *encsegs);
  int  split_enc_segment(TriEdge &S, Vertex *encpt, bool mtrflag, arraypool*, arraypool*, arraypool*);
  int  repair_encsegments(arraypool *encsegs, arraypool *enctris);
  void enq_triangle(Triang* tri, bool mtrflag, arraypool* enctris);
  void enq_vertex_star(Vertex *pt, arraypool* enctris);
  int  check_triangle(Triang *tri, arraypool* enctris);
  int  get_triangle(Vertex *e1, Vertex *e2, Vertex *e3, TriEdge& E);
  int  get_tri_circumcenter(Triang *tri, REAL ccent[3]);
  int  repair_triangles(arraypool *enctris);
  int  delaunay_refinement();

  // Experiments.
  REAL get_element_gradient(Triang* tri); // get fx, fy, saved in cct[0], [1]
  REAL get_Dirichlet_energy(); // of the whole triangulation
  REAL dump_Dirichlet_energy(); // for visualisation _energy.inp
  void dump_edge_energy_jump(); // for visualisation _jump.inp

  // Mesh adaptation (adapt.cpp)
  int  locate_hull_edge(Vertex *v, TriEdge &E); // for vertex interpolation.
  int  set_vertex_metric(Vertex *v, TriEdge &E, int &iloc); // interpolate vertex values.
  int  set_vertex_metrics();
  REAL get_vertex_min_edge_length(Vertex *pt); 
  int  remove_point(Vertex *pt, arraypool* fqueue);
  int  coarsen_mesh(); // Mesh coarsening
  int  get_powercell_mass_centers(Vertex *massptlist);
  int  get_laplacian_center(Vertex *mesh_vertex, Vertex *movept);
  int  get_laplacian_centers(Vertex *massptlist);
  REAL get_distmesh_target_length();
  int  get_distmesh_point(Vertex *mesh_vertex, Vertex *movept);
  int  get_distmesh_points(Vertex *moveptlist);
  int  smooth_vertices(); // Mesh smoothing
  int  mesh_adapt();

  // Debug functions
  Vertex* db_vrt(int v);
  TriEdge db_seg(int v1, int v2);
  TriEdge db_tri(int v1, int v2, int v3);
  void dump_vertex_star(int v);

  // Calculate linear transformation weights (in io.cpp).
  void linear_transform(int Steiner_only, double Cx, double Cy, double Wc);

  // flip recover (weighted) Delaunay (adapt.cpp).
  int  check_delaunay(); // debug
  void dump_missing_pt_triangles(Vertex *pt, arraypool *tris); // debug
  void save_uninserted_points(arraypool*); // output only
  int  insert_missing_weighted_point_old(Vertex *pt);
  bool insert_missing_weighted_point(Vertex *pt);
  void flip_recover_delaunay();
};

} // namespace detri2

#endif  // #ifndef __DETRI2_H__
