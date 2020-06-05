#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

//==============================================================================
#ifndef TRILIBRARY

//==============================================================================

int merge_two_triangulations(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage: merge inputfile1.ele inputfile2.ele\n");
    return 0;
  }

  int myargc = 2;
  char *myargv[2];
  myargv[0] = argv[0];
  myargv[1] = argv[1];

  Triangulation *Tr1 = new Triangulation();
  Tr1->parse_commands(myargc, myargv);
  if (!Tr1->read_mesh()) {
    printf("Failed to read input file %s.\n", argv[1]);
    delete Tr1;
    return 0;
  }
  
  myargv[1] = argv[2];
  Triangulation *Tr2 = new Triangulation();
  Tr2->parse_commands(myargc, myargv);
  if (!Tr2->read_mesh()) {
    printf("Failed to read input file %s.\n", argv[1]);
    delete Tr1;
    delete Tr2;
    return 0;
  }

  // Calculate the new bbox.
  REAL xmin = Tr1->io_xmin < Tr2->io_xmin ? Tr1->io_xmin : Tr2->io_xmin;
  REAL ymin = Tr1->io_ymin < Tr2->io_ymin ? Tr1->io_ymin : Tr2->io_ymin;
  //REAL zmin = Tr1->io_zmin < Tr2->io_zmin ? Tr1->io_zmin : Tr2->io_zmin;
  REAL xmax = Tr1->io_xmax > Tr2->io_xmax ? Tr1->io_xmax : Tr2->io_xmax;
  REAL ymax = Tr1->io_ymax > Tr2->io_ymax ? Tr1->io_ymax : Tr2->io_ymax;
  //REAL zmax = Tr1->io_zmax > Tr2->io_zmax ? Tr1->io_zmax : Tr2->io_zmax;
  REAL dx = xmax - xmin;
  REAL dy = ymax - ymin;
  //REAL dz = zmax - zmin;
  REAL dd = sqrt(dx*dx + dy*dy) * 0.2;

  REAL bbox[8][3];
  bbox[0][0] = xmin - dd; bbox[0][1] = ymin - dd; bbox[0][2] = -dd;
  bbox[1][0] = xmax + dd; bbox[1][1] = ymin - dd; bbox[1][2] = -dd;
  bbox[2][0] = xmax + dd; bbox[2][1] = ymax + dd; bbox[2][2] = -dd;
  bbox[3][0] = xmin - dd; bbox[3][1] = ymax + dd; bbox[3][2] = -dd;
  bbox[4][0] = xmin - dd; bbox[4][1] = ymin - dd; bbox[4][2] =  dd;
  bbox[5][0] = xmax + dd; bbox[5][1] = ymin - dd; bbox[5][2] =  dd;
  bbox[6][0] = xmax + dd; bbox[6][1] = ymax + dd; bbox[6][2] =  dd;
  bbox[7][0] = xmin - dd; bbox[7][1] = ymax + dd; bbox[7][2] =  dd;

  printf("Merge two meshes to file merge.smesh\n");

  FILE *fout = fopen("merge.smesh", "w");

  int nv = Tr1->ct_in_vrts + Tr2->ct_in_vrts + 8; // 8 corners

  fprintf(fout, "%d 3 0 0\n", nv);

  int idx = 1;
  for (int i = 0; i < Tr1->ct_in_vrts; i++) {
    Vertex *v = &(Tr1->in_vrts[i]);
    fprintf(fout, "%d %g %g %g\n", idx, v->crd[0], v->crd[1], 0.0);
    v->idx = idx;  
    idx++;
  }

  for (int i = 0; i < Tr2->ct_in_vrts; i++) {
    Vertex *v = &(Tr2->in_vrts[i]);
    fprintf(fout, "%d %g %g %g\n", idx, v->crd[0], v->crd[1], 0.0);
    v->idx = idx;  
    idx++;
  }

  for (int i = 0; i < 8; i++) {
    fprintf(fout, "%d %g %g %g\n", idx, bbox[i][0], bbox[i][1], bbox[i][2]);
    idx++;
  }

  fprintf(fout, "%d 1\n", Tr1->ct_in_tris + Tr2->ct_in_tris);

  for (int i = 0; i < Tr1->tr_tris->objects; i++) {
    Triang *t = (Triang *) Tr1->tr_tris->get(i);
    fprintf(fout, "3 %d %d %d  1\n", t->vrt[0]->idx, t->vrt[1]->idx, t->vrt[2]->idx); 
  }

  for (int i = 0; i < Tr2->tr_tris->objects; i++) {
    Triang *t = (Triang *) Tr2->tr_tris->get(i);
    fprintf(fout, "3 %d %d %d  2\n", t->vrt[0]->idx, t->vrt[1]->idx, t->vrt[2]->idx); 
  }

  fprintf(fout, "0\n");

  fclose(fout);

  delete Tr1;
  delete Tr2;
  return 1;
}

//==============================================================================
#define LINEBUF 1024

char* findnextnumber(char *string); // in io.cpp

int save_inp_to_m(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage: save_inp_to_m inputfile.inp scale_z\n");
    return 0;
  }

  // Read the file once, get the nnode, nface.
  char buf[LINEBUF]; //, new_buf[LINEBUF];
  char *pstr;
  //double x, y, z;
  int i;

  char filename1[LINEBUF];
  strcpy(filename1, argv[1]);

  FILE *infile = NULL;
  infile = fopen(filename1, "r");
  if (infile == NULL) {
    printf("Usage: save_inp_to_m inputfile.inp scale_z\n");
    return 0;
  }
  printf("Reading %s\n", filename1);

  // Get the number of nodes, faces.
  int nnode = 0, nface = 0, nflag;
  fgets(buf, LINEBUF, infile);
  pstr = buf;
  nnode = atoi(pstr);
  pstr = findnextnumber(pstr);
  nface = atoi(pstr);
  pstr = findnextnumber(pstr);
  nflag = atoi(pstr);

  if ((nnode == 0) || (nface == 0) || (nflag != 1)) {
    printf("!! Wrong nnode (%d) or nface (%d) or nflag (%d)\n", nnode, nface, nflag);
    fclose(infile);
    return 0;
  }

  double *node_x = new double[nnode];
  double *node_y = new double[nnode];
  double *node_z = new double[nnode];
  int *face_i = new int[nface];
  int *face_j = new int[nface];
  int *face_k = new int[nface];

  for (i = 0; i < nnode; i++) {
    fgets(buf, LINEBUF, infile);
    pstr = buf; // idx
    pstr = findnextnumber(pstr);
    node_x[i] = atof(pstr);
    pstr = findnextnumber(pstr);
    node_y[i] = atof(pstr);
  }

  // 1 1 tri 214 193 183
  for (i = 0; i < nface; i++) {
    fgets(buf, LINEBUF, infile);
    pstr = buf; // idx
    pstr = findnextnumber(pstr); // 1
    pstr = findnextnumber(pstr);
    face_i[i] = atof(pstr);
    pstr = findnextnumber(pstr);
    face_j[i] = atof(pstr);
    pstr = findnextnumber(pstr);
    face_k[i] = atof(pstr);
  }
  
  // Skip gthe following two lines
  fgets(buf, LINEBUF, infile);
  fgets(buf, LINEBUF, infile);
  //1 1
  //unknown, adim

  for (i = 0; i < nnode; i++) {
    fgets(buf, LINEBUF, infile);
    pstr = buf; // idx
    pstr = findnextnumber(pstr);
    node_z[i] = atof(pstr);
  }

  fclose(infile);

  double scale_z = atof(argv[2]);
  printf("scale_z = %g\n", scale_z);

  char filename2[LINEBUF];
  sprintf(filename2, "%s.m", filename1);
  FILE *outfile = NULL;
  outfile = fopen(filename2, "w");
  printf("Writing %d vertices, %d triangles to %s\n", nnode, nface, filename2);

  for (i = 0; i < nnode; i++) {
    fprintf(outfile, "Vertex %d %g %g %g\n", i+1, node_x[i], node_y[i],
            node_z[i] * scale_z);
  }
  for (i = 0; i < nface; i++) {
    fprintf(outfile, "Face %d %d %d %d\n", i+1, face_i[i], face_j[i], face_k[i]);
  }

  delete [] node_x;
  delete [] node_y;
  delete [] node_z;
  delete [] face_i;
  delete [] face_j;
  delete [] face_k;

  fclose(outfile);
  return 1;
}

//==============================================================================

int save_inp_to_smesh(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: save_inp_to_smesh inputfile.inp\n");
    return 0;
  }

  // Read the file once, get the nnode, nface.
  char buf[LINEBUF]; //, new_buf[LINEBUF];
  char *pstr;
  //double x, y, z;
  int i;

  char filename1[LINEBUF];
  strcpy(filename1, argv[1]);

  FILE *infile = NULL;
  infile = fopen(filename1, "r");
  if (infile == NULL) {
    printf("Usage: save_inp_to_smesh inputfile.inp\n");
    return 0;
  }
  printf("Reading %s\n", filename1);

  // Get the number of nodes, faces.
  int nnode = 0, nface = 0, nflag;
  fgets(buf, LINEBUF, infile);
  pstr = buf;
  nnode = atoi(pstr);
  pstr = findnextnumber(pstr);
  nface = atoi(pstr);
  pstr = findnextnumber(pstr);
  nflag = atoi(pstr);

  if ((nnode == 0) || (nface == 0) || (nflag != 1)) {
    printf("!! Wrong nnode (%d) or nface (%d) or nflag (%d)\n", nnode, nface, nflag);
    fclose(infile);
    return 0;
  }

  double *node_x = new double[nnode];
  double *node_y = new double[nnode];
  double *node_z = new double[nnode];
  int *face_i = new int[nface];
  int *face_j = new int[nface];
  int *face_k = new int[nface];

  for (i = 0; i < nnode; i++) {
    fgets(buf, LINEBUF, infile);
    pstr = buf; // idx
    pstr = findnextnumber(pstr);
    node_x[i] = atof(pstr);
    pstr = findnextnumber(pstr);
    node_y[i] = atof(pstr);
    pstr = findnextnumber(pstr);
    node_z[i] = atof(pstr);
  }

  // 1 1 tri 214 193 183
  for (i = 0; i < nface; i++) {
    fgets(buf, LINEBUF, infile);
    pstr = buf; // idx
    pstr = findnextnumber(pstr); // 1
    pstr = findnextnumber(pstr);
    face_i[i] = atof(pstr);
    pstr = findnextnumber(pstr);
    face_j[i] = atof(pstr);
    pstr = findnextnumber(pstr);
    face_k[i] = atof(pstr);
  }

  /*
  // Skip gthe following two lines
  fgets(buf, LINEBUF, infile);
  fgets(buf, LINEBUF, infile);
  //1 1
  //unknown, adim

  for (i = 0; i < nnode; i++) {
    fgets(buf, LINEBUF, infile);
    pstr = buf; // idx
    pstr = findnextnumber(pstr);
    node_z[i] = atof(pstr);
  }
  */

  fclose(infile);

  char filename2[LINEBUF];
  sprintf(filename2, "%s.smesh", filename1);
  FILE *outfile = NULL;
  outfile = fopen(filename2, "w");
  printf("Writing %d vertices, %d triangles to %s\n", nnode, nface, filename2);

  fprintf(outfile, "%d 3 0 0\n", nnode);

  for (i = 0; i < nnode; i++) {
    fprintf(outfile, "%d %g %g %g\n", i+1, node_x[i], node_y[i], node_z[i]);
  }

  fprintf(outfile, "%d 0\n", nface);

  for (i = 0; i < nface; i++) {
    fprintf(outfile, "3 %d %d %d\n", face_i[i], face_j[i], face_k[i]);
  }

  fprintf(outfile, "0\n");

  delete [] node_x;
  delete [] node_y;
  delete [] node_z;
  delete [] face_i;
  delete [] face_j;
  delete [] face_k;

  fclose(outfile);
  return 1;
}

//==============================================================================
int get_xy_uv_mesh(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: get_xyz_uv dump_bamg.uv.m\n");
    return 0;
  }

  class Vertex {
   public:
    int idx;
    double x, y, z;
    int farther;
    double r, g, b;
    double u, v;

    void init() {
      idx = 0;
      x = y = z = 0.0;
      farther = 0;
      r = g = b = 0.0;
      u = v = 0.0;
    }
  };

  class Face {
   public:
    int idx;
    int v1, v2, v3;
    int farther;
  };

  // Read the file once, get the nnode, nface.
  char buf[LINEBUF]; //, new_buf[LINEBUF];
  char *pstr;
  double x, y, z, u, v;
  int idx;

  int nnode, nface;
  int ncount, fcount;

  nnode = 0; // 3422;
  nface = 0; // 6674;
  ncount = fcount = 0; // Count the number vertices, faces.
  

  char filename1[LINEBUF];
  //strcpy(filename1, "dump_huang3.uv.m");
  //strcpy(filename1, "dump_bamg.uv.m");
  strcpy(filename1, argv[1]);

  FILE *infile = NULL;
  infile = fopen(filename1, "r");
  if (infile == NULL) {
    printf("Usage: get_xyz_uv dump_bamg.uv.m\n");
    return 0;
  }
  printf("Reading %s\n", filename1);
  while (fgets(buf, LINEBUF, infile) != NULL) {
    //sprintf(new_buf, "%3d %s", count, buf);
    pstr = buf;
    if (strstr(pstr, "Vertex")) {
      pstr = findnextnumber(pstr);
      idx = atoi(pstr);
      if (idx > nnode) nnode = idx;
      ncount++;
    } else if (strstr(pstr, "Face")) {
      //fputs(buf, outfile);
      pstr = findnextnumber(pstr);
      idx = atoi(pstr);
      if (idx > nface) nface = idx;
      fcount++;
    } else if (strstr(pstr, "Edge")) {
      //fputs(buf, outfile);
    }
  }
  printf("Number of nodes = %d, ncount=%d\n", nnode, ncount);
  printf("Number of faces = %d, fcount=%d\n", nface, fcount);
  fclose(infile);

  if (nnode != ncount) {
    printf("!! Warning:  wrong number of vertices.\n");
  }
  if (nface != fcount) {
    printf("!! Warning:  wrong number of faces. use fcount=%d\n", fcount);
    nface = fcount;
  }

  //=========================================
  
  infile = fopen(filename1, "r");
  printf("Reading %s\n", filename1);
  
  FILE* xynode = fopen("output_xy.node", "w");
  FILE* xyele  = fopen("output_xy.ele", "w");
  printf("Writing %s\n", "output_xy.node, output_xy.ele");

  FILE *uvnode = fopen("output_uv.node", "w");
  FILE *uvele  = fopen("output_uv.ele", "w");
  printf("Writing %s\n", "output_uv.node, output_uv.ele");

  FILE* xyznode = fopen("output_xyz.node", "w");
  FILE* xyzsmesh = fopen("output_xyz.smesh", "w");
  printf("Writing %s\n", "output_xyz.node, output_xyz.smesh");

  fprintf(xynode, "%d 2 0\n", nnode);
  fprintf(xyele,  "%d 3 0\n", nface);

  fprintf(uvnode, "%d 2 0\n", nnode);
  fprintf(uvele,  "%d 3 0\n", nface);

  fprintf(xyznode,  "%d 3 0\n", nnode);
  fprintf(xyzsmesh, "0 3 0\n");
  fprintf(xyzsmesh, "%d 0\n", nface);
  
  //int count = 1;
  while (fgets(buf, LINEBUF, infile) != NULL) {
    //sprintf(new_buf, "%3d %s", count, buf);
    pstr = buf;
    if (strstr(pstr, "Vertex")) {
      pstr = findnextnumber(pstr);
      idx = atoi(pstr);
      pstr = findnextnumber(pstr);
      x = atof(pstr);
      pstr = findnextnumber(pstr);
      y = atof(pstr);
      pstr = findnextnumber(pstr);
      z = atof(pstr);
      pstr = findnextnumber(pstr);
      u = atof(pstr);
      pstr = findnextnumber(pstr);
      v = atof(pstr);
      //sprintf(new_buf, "Vertex %d %g %g 0 {uv=(%g %g)}\n", idx, x, y, u, v);
      //fputs(new_buf, outfile);
      //sprintf(new_buf, "%d %g %g 0\n", idx, u, v);
      //fputs(new_buf, stdout);
      fprintf(xynode, "%d %g %g\n", idx, x, y);
      fprintf(uvnode, "%d %g %g\n", idx, u, v);
      fprintf(xyznode, "%d %g %g %g\n", idx, x, y, z);
    } else if (strstr(pstr, "Face")) {
      //fputs(buf, outfile);
      pstr = findnextnumber(pstr);
      idx = atoi(pstr);
      pstr = findnextnumber(pstr);
      int e1 = atoi(pstr);
      pstr = findnextnumber(pstr);
      int e2 = atoi(pstr);
      pstr = findnextnumber(pstr);
      int e3 = atoi(pstr);
      //sprintf(new_buf, "3 %d %d %d\n", e1, e2, e3);
      //fputs(new_buf, stdout);
      fprintf(xyele, "%d %d %d %d\n", idx, e1, e2, e3);
      fprintf(uvele, "%d %d %d %d\n", idx, e1, e2, e3);
      fprintf(xyzsmesh, "3 %d %d %d\n", e1, e2, e3);
    } else if (strstr(pstr, "Edge")) {
      //fputs(buf, outfile);
    }
  }

  fclose(infile);
  fclose(xynode);
  fclose(xyele);
  fclose(uvnode);
  fclose(uvele);
  fclose(xyznode);
  fclose(xyzsmesh);

  return 1;
}

//==============================================================================
// dot() returns the dot product: v1 dot v2.
double dot(double* v1, double* v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// cross() computes the cross product: n = v1 cross v2.
void cross(double* v1, double* v2, double* n)
{
  n[0] =   v1[1] * v2[2] - v2[1] * v1[2];
  n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
  n[2] =   v1[0] * v2[1] - v2[0] * v1[1];
}

double triarea3d(double* pa, double* pb, double* pc)
{
  double A[4][4];

  // Compute the coefficient matrix A (3x3).
  A[0][0] = pb[0] - pa[0];
  A[0][1] = pb[1] - pa[1];
  A[0][2] = pb[2] - pa[2]; // vector V1 (pa->pb)
  A[1][0] = pc[0] - pa[0];
  A[1][1] = pc[1] - pa[1];
  A[1][2] = pc[2] - pa[2]; // vector V2 (pa->pc)

  cross(A[0], A[1], A[2]); // vector V3 (V1 X V2)

  return 0.5 * sqrt(dot(A[2], A[2])); // The area of [a,b,c].
}

// Read two triangulation xy.node, xy.ele and uv.node, uv.ele
// Calculate the ratio of two triangles: Lambda = Area_uv / Area_xy
// Output a uv.inp mesh file
//
// The command line:
//   detri2 xy.ele -m uv.ele
//   detri2 xyz.smesh -m uv.ele scale
// where:
//   xyz.smesh is the source mesh (of a curved surface)
//   uv.ele is the mapped mesh on parameter domain (planar)
//   scale is a factor to scale the density function
//
// The mesh density on uv.ele is calulcated by the triangle area ratio
//   between triangles in uv and xzy,  (Lambda = Area_uv / Area_xyz)
//   The mesh density on each triangle in uv is:  1 / Lambda.
//   it is saved in the output file:  uv.area

int generate_density_from_area_ratio(int argc, char* argv[])
{
  if (argc < 4) {
    //printf("Usage: detri2 xy.ele uv.ele\n");
    printf("Usage: detri2 xyz.smesh uv.ele scale [max_min_ratio]\n");
    return 0;
  }

  // Read options.
  int myargc = 2;
  char *myargv[2];
  myargv[0] = argv[0];
  myargv[1] = argv[1];

  Triangulation *Tr_xy = new Triangulation();
  if (!Tr_xy->parse_commands(myargc, myargv)) {
    // No input or wrong parameters.
    //printf("Usage: detri2 xy.ele uv.ele\n");
    printf("Usage: detri2 xyz.smesh uv.ele\n");
    delete Tr_xy;
    return 0;
  }

  myargv[1] = argv[2];
  Triangulation *Tr_uv = new Triangulation();
  if (!Tr_uv->parse_commands(myargc, myargv)) {
    // No input or wrong parameters.
    //printf("Usage: detri2 xy.ele uv.ele\n");
    printf("Usage: detri2 xyz.smesh uv.ele\n");
    delete Tr_uv;
    return 0;
  }

  Tr_xy->read_mesh(); // Read xy.node, xy.smesh // xy.ele
  Tr_uv->read_mesh(); // Read uv.node, uv.ele
 
  // Calculate the ratio.
  int ntri = Tr_xy->tr_tris->objects;
  assert(Tr_uv->tr_tris->objects == ntri);
  double *Area_xy = new double[ntri];
  double *Area_uv = new double[ntri];
  double *ratio_uv_over_xy = new double[ntri];
  int i;

  for (i = 0; i < ntri; i++) {
    Triang *t = (Triang *) Tr_xy->tr_tris->get(i);
    //Area_xy[i] = Tr_xy->get_tri_area(t->vrt[0], t->vrt[1], t->vrt[2]);
    Area_xy[i] = triarea3d(t->vrt[0]->crd, t->vrt[1]->crd, t->vrt[2]->crd);
  }
  for (i = 0; i < ntri; i++) {
    Triang *t = (Triang *) Tr_uv->tr_tris->get(i);
    Area_uv[i] = Tr_xy->get_tri_area(t->vrt[0], t->vrt[1], t->vrt[2]);
  }
  double max_ratio, min_ratio;
  max_ratio = 0.0;
  min_ratio = 1.e30;
  for (i = 0; i < ntri; i++) {
    ratio_uv_over_xy[i] = Area_uv[i] / Area_xy[i];
    max_ratio = ratio_uv_over_xy[i] > max_ratio ? ratio_uv_over_xy[i] : max_ratio;
    min_ratio = ratio_uv_over_xy[i] < min_ratio ? ratio_uv_over_xy[i] : min_ratio;
  }

  printf("max ratio = %g\n", max_ratio);
  printf("min ratio = %g\n", min_ratio);

  double max_min_ratio = 1.0;
  if (argc == 5) {
    max_min_ratio = atof(argv[4]);
  }
  printf("max_min ratio = %g\n", max_min_ratio);
  // [Comment: 2018-08-30]
  //   Here we let the minimum area density be fixed, and limit the maximum
  //     area denisty to be  minimum_area_density * max_min_ratio/
  //     min_area_density = 1 / max_ratio
  //     max_area_denisty = 1 / min_ratio
  //     Hence we should limit min_ratio below.
  // Here we adjust the ratio (for bamg example)
  min_ratio = max_ratio / 50.0;
  //if (max_min_ratio > 1.0) {
  //  min_ratio = max_ratio / max_min_ratio;
  //}

  FILE *fout = fopen("area_xy_uv.txt", "w");
  for (i = 0; i < ntri; i++) {
    Triang *t = (Triang *) Tr_xy->tr_tris->get(i);
    fprintf(fout, "%d %d %d %d #%g %g #%g\n", i+1, t->vrt[0]->idx, t->vrt[1]->idx,
            t->vrt[2]->idx, Area_xy[i], Area_uv[i], ratio_uv_over_xy[i]);
  }
  fclose(fout);

  double scale = atof(argv[3]);
  printf("scale = %g\n", scale);

  double hmax, hmin;
  hmax = 0.0;
  hmin = 1.e30;

  // Save mesh to a .inp file.
  FILE *outfile = fopen("bgmesh.inp", "w");
  printf("Wrinting density to file bgmesh.inp for Paraview.\n");

  //int ntri = (int) tr_tris->objects - ct_hullsize;
  printf("Writing %d triangles to file bgmesh.inp.\n", ntri);
  int nv = Tr_uv->ct_in_vrts;

  fprintf(outfile, "%d %d 0 1 0\n", nv, ntri);

  int idx=1; // UCD index starts from 1.
  for (i = 0; i < Tr_uv->ct_in_vrts; i++) {
    //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
    fprintf(outfile, "%d %g %g 0\n", idx, Tr_uv->in_vrts[i].crd[0], Tr_uv->in_vrts[i].crd[1]);
    Tr_uv->in_vrts[i].idx = idx;
    idx++;
  }

  // UCD assumes vertex index starts from 1.
  int shift = (Tr_uv->io_firstindex == 1 ? 0 : 1);
  idx = 1;
  for (i = 0; i < Tr_uv->tr_tris->objects; i++) {
    Triang* tri = (Triang *) Tr_uv->tr_tris->get(i);
    //if (!tri->is_deleted()) {
      // ignore a hull triangle.
      //if (!tri->is_hulltri()) {
        fprintf(outfile, "%d %d tri %d %d %d\n", idx, tri->tag,
                tri->vrt[0]->idx + shift,
                tri->vrt[1]->idx + shift,
                tri->vrt[2]->idx + shift);
        tri->idx = idx;
        idx++;
      //}
    //}
  }

  // Output metric on nodes.
  fprintf(outfile, "1 1\n");
  fprintf(outfile, "Density (per triangle), adim\n");

  //int shift = (io_firstindex == 1 ? 0 : 1);
  TriEdge E;
  idx = 1;
  for (i = 0; i < Tr_uv->tr_tris->objects; i++) {
    E.tri = (Triang *) Tr_uv->tr_tris->get(i);
    //if (!E.tri->is_deleted()) {
      // ignore a hull triangle.
      //if (!E.tri->is_hulltri()) {
        double ratio = ratio_uv_over_xy[i];
        if (ratio < min_ratio) ratio = min_ratio;
        double dd = 1.0 / ratio;
        fprintf(outfile, "%d %g\n", idx, dd * scale);
        idx++;
      //}
    //}
  }

  fclose(outfile);

  // output_uv.area
  outfile = fopen("output_uv.area", "w");
  printf("Writing %d areas to file output_uv.area.\n", ntri);

  fprintf(outfile, "%d\n", ntri);

  idx = 1;
  for (i = 0; i < ntri; i++) {
    //double dd = 1.0 / ratio_uv_over_xy[i];
    double ratio = ratio_uv_over_xy[i];
    if (ratio < min_ratio) ratio = min_ratio;
    double dd = 1.0 / ratio;
    dd *= scale;
    hmax = hmax < dd ? dd : hmax;
    hmin = hmin > dd ? dd : hmin;
    fprintf(outfile, "%d %g\n", idx, dd);
    idx++;
  }

  fclose(outfile);

  printf("UV xmin = %g, xmax = %g,  width=%g\n",
         Tr_uv->io_xmin, Tr_uv->io_xmax, Tr_uv->io_xmax - Tr_uv->io_xmin);
  printf("UV ymin = %g, ymax = %g, height=%g\n",
         Tr_uv->io_ymin, Tr_uv->io_ymax, Tr_uv->io_ymax - Tr_uv->io_ymin);
  printf("hmin = %g, hmax = %g\n", hmin, hmax);

  delete [] Area_xy;
  delete [] Area_uv;
  delete [] ratio_uv_over_xy;

  delete Tr_xy;
  delete Tr_uv;
  return 1;
}

//==============================================================================
// The command line:
//   detri2 xyz.smesh uv.ele uv_adapt.ele
// where:
//   xyz.smesh    is the source mesh (of a curved surface)
//   uv.ele       is the mapped mesh on parameter domain (planar)
//   uv_adapt.ele is the adapted mesh to be pulled back to xyz.smesh
// Output:
//   uv_pullback.ele

int pull_back(int argc, char* argv[])
{
  if (argc < 4) {
    printf("Usage: detri2 xyz.smesh uv.ele uv_adapt.ele\n");
    return 0;
  }

  // Read options.
  int myargc = 2;
  char *myargv[2];
  myargv[0] = argv[0];

  myargv[1] = argv[1];
  Triangulation *Tr_xyz = new Triangulation();
  Tr_xyz->parse_commands(myargc, myargv);

  myargv[1] = argv[2];
  Triangulation *Tr_uv = new Triangulation();
  Tr_uv->parse_commands(myargc, myargv);

  myargv[1] = argv[3];
  Triangulation *Tr_uv_adapt = new Triangulation();
  Tr_uv_adapt->parse_commands(myargc, myargv);

  Tr_xyz->read_mesh(); // Read xy.node, xy.smesh // xy.ele
  Tr_uv->read_mesh(); // Read uv.node, uv.ele
  Tr_uv_adapt->read_mesh();

  Tr_xyz->reconstruct_mesh(0); // check_delaunay=0, no flips
  Tr_uv->reconstruct_mesh(0);
  Tr_uv_adapt->reconstruct_mesh(0);

  // Index all triangles.
  int i;
  for (i = 0; i < Tr_xyz->tr_tris->used_items; i++) {
    Triang *tri = (Triang *) Tr_xyz->tr_tris->get(i);
    tri->idx = i;
  }
  for (i = 0; i < Tr_uv->tr_tris->used_items; i++) {
    Triang *tri = (Triang *) Tr_uv->tr_tris->get(i);
    tri->idx = i;
  }
  for (i = 0; i < Tr_uv_adapt->tr_tris->used_items; i++) {
    Triang *tri = (Triang *) Tr_uv_adapt->tr_tris->get(i);
    tri->idx = i;
  }

  // Locate each point in Tr_uv_adapt in Tr_uv.
  Vertex *newptlist = new Vertex[Tr_uv_adapt->ct_in_vrts];
  
  TriEdge E;
  int loc;
  for (i = 0; i < Tr_uv_adapt->ct_in_vrts; i++) {
    Vertex *pt = &(Tr_uv_adapt->in_vrts[i]);
    E.tri = Tr_uv->tr_recnttri; // start from the latest searching
    printf("  Locate point %d: %g,%g\n", pt->idx, pt->crd[0], pt->crd[1]);
    if ((E.tri != NULL) && (!E.tri->is_hulltri())) {
      printf("  from triangle [%d,%d,%d]\n", E.org()->idx, E.dest()->idx, E.apex()->idx);
    }
    //loc = Tr_uv->locate_point(pt, E, 1, 0); // rndflag=1
    loc = Tr_uv->locate_point(pt, E, 0); // rndflag=1
    if (loc == LOC_ON_EDGE) {
      if (E.tri->is_hulltri()) {
        E = E.esym();
      }
    } else if (loc == LOC_ON_VERT) {
      if (E.tri->is_hulltri()) {
        do { // Search a non-hull tri.
          E = E.eprev_esym(); // ccw rotate
        } while (E.tri->is_hulltri());
      }
    }
    if (E.tri->is_hulltri()) {
      // search the point globally.
      printf("  !! Brute-force searching\n");
      for (int j = 0; j < Tr_uv->tr_tris->used_items; j++) {
        Triang *t = (Triang *) Tr_uv->tr_tris->get(j);
        if (!t->is_hulltri()) {
          Vertex *pa = t->vrt[0];
          Vertex *pb = t->vrt[1];
          Vertex *pc = t->vrt[2];
          double A  = Tr_uv->get_tri_area(pa, pb, pc);
          double Wa = Tr_uv->get_tri_area(pt, pb, pc) / A;
          double Wb = Tr_uv->get_tri_area(pa, pt, pc) / A;
          double Wc = Tr_uv->get_tri_area(pa, pb, pt) / A;
          if (fabs((Wa + Wb + Wc) - 1.0) < 1.e-3) {
            E.tri = t;
            break;
          }
        }
      }
    }
    if (!E.tri->is_hulltri()) {
      Vertex *pa = E.tri->vrt[0]; // E.org();
      Vertex *pb = E.tri->vrt[1]; // E.dest();
      Vertex *pc = E.tri->vrt[2]; // E.apex();
      printf("  UV  tri %d: [%d,%d,%d]\n", E.tri->idx, pa->idx, pb->idx, pc->idx);
      double A  = Tr_uv->get_tri_area(pa, pb, pc);
      double Wa = Tr_uv->get_tri_area(pt, pb, pc) / A;
      double Wb = Tr_uv->get_tri_area(pa, pt, pc) / A;
      double Wc = Tr_uv->get_tri_area(pa, pb, pt) / A;
      // Calculate the new position.
      Triang *tri = (Triang *) Tr_xyz->tr_tris->get(E.tri->idx);
      pa = tri->vrt[0]; // E.org();
      pb = tri->vrt[1]; // E.dest();
      pc = tri->vrt[2]; // E.apex();
      printf("  XYZ tri %d: [%d,%d,%d]\n", tri->idx, pa->idx, pb->idx, pc->idx);
      printf("  weights %g, %g, %g\n", Wa, Wb, Wc);
      Vertex *newpt = &(newptlist[i]);
      newpt->init();
      newpt->crd[0] = Wa * pa->crd[0] + Wb * pb->crd[0] + Wc * pc->crd[0];
      newpt->crd[1] = Wa * pa->crd[1] + Wb * pb->crd[1] + Wc * pc->crd[1];
      newpt->crd[2] = Wa * pa->crd[2] + Wb * pb->crd[2] + Wc * pc->crd[2];
      printf("  newpt %g, %g, %g\n", newpt->crd[0], newpt->crd[1], newpt->crd[2]);
      Tr_uv->tr_recnttri = E.tri;
    } else {
      assert(0); // Something is wrong.
    }
  }

  FILE *outfile = fopen("pullback_xyz.node", "w");
  printf("Output file pullback_xyz.node\n");
  fprintf(outfile, "%d 3 0\n", Tr_uv_adapt->ct_in_vrts);
  for (i = 0; i < Tr_uv_adapt->ct_in_vrts; i++) {
    Vertex *newpt = &(newptlist[i]);
    fprintf(outfile, "%d %g %g %g\n", i+1, newpt->crd[0], newpt->crd[1], newpt->crd[2]);
  }
  fclose(outfile);

  outfile = fopen("pullback_xyz.smesh", "w");
  printf("Output file pullback_xyz.smesh\n");
  fprintf(outfile, "0 3 0\n");
  fprintf(outfile, "%d 0\n", Tr_uv_adapt->ct_in_tris);
  for (i = 0; i < Tr_uv_adapt->ct_in_tris; i++) {
    Triang *t = (Triang *) Tr_uv_adapt->tr_tris->get(i);
    fprintf(outfile, "3  %d %d %d\n", t->vrt[0]->idx, t->vrt[1]->idx, t->vrt[2]->idx);
  }
  fprintf(outfile, "0\n");
  fclose(outfile);

  outfile = fopen("pullback_proj.node", "w");
  printf("Output file pullback_proj.node\n");
  fprintf(outfile, "%d 2 0\n", Tr_uv_adapt->ct_in_vrts);
  for (i = 0; i < Tr_uv_adapt->ct_in_vrts; i++) {
    Vertex *newpt = &(newptlist[i]);
    fprintf(outfile, "%d %g %g\n", i+1, newpt->crd[0], newpt->crd[1]);
  }
  fclose(outfile);
  outfile = fopen("pullback_proj.ele", "w");
  printf("Output file pullback_proj.ele\n");
  fprintf(outfile, "%d 3 0\n", Tr_uv_adapt->ct_in_tris);
  for (i = 0; i < Tr_uv_adapt->ct_in_tris; i++) {
    Triang *t = (Triang *) Tr_uv_adapt->tr_tris->get(i);
    fprintf(outfile, "%d  %d %d %d\n", i+1, t->vrt[0]->idx, t->vrt[1]->idx, t->vrt[2]->idx);
  }
  fclose(outfile);

  delete [] newptlist;
  delete Tr_xyz;
  delete Tr_uv;
  delete Tr_uv_adapt;
  return 1;
}


/*
//==============================================================================
int main(int argc, char* argv[])
{
  //save_inp_to_m(argc, argv);

  //get_xy_uv_mesh(argc, argv);
  generate_density_from_area_ratio(argc, argv);
  //generate_mesh_adapt(argc, argv);
  //pull_back(argc, argv);

  //generate_mesh_adapt(argc, argv);

  return 1;
}
*/

#endif // #ifndef TRILIBRARY

//==============================================================================
int generate_mesh(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: detri2 [-options] filename[.node, .ele]\n");
    return 0;
  }

  Triangulation *Tr = new Triangulation();

  // Read options.
  if (!Tr->parse_commands(argc, argv)) {
    // No input or wrong parameters.
    printf("Usage: detri2 [-options] filename[.node, .poly, .ele, .edge]\n");
    delete Tr;
    return 0;
  }

  // Read inputs.
  if (!Tr->read_mesh()) {
    printf("Failed to read input from file %s[.poly, .node, .ele, .edge]\n",
           Tr->io_infilename);
    delete Tr;
    return 0;
  }

  // Generate (constrained) (weighted) Delaunay triangulation.
  if (Tr->tr_tris == NULL) {
    if (Tr->incremental_delaunay()) {
      if (Tr->tr_segs != NULL) {
        Tr->recover_segments();
        Tr->set_subdomains();
      }
    } else {
      printf("Failed to create Delaunay (regular) triangulation.\n");
      delete Tr;
      return 0;
    }
  } else {
    Tr->reconstruct_mesh(1);
  }

  // Mesh refinement and adaptation.
  //if (Tr->tr_segs != NULL) {
  if (Tr->op_quality || (Tr->op_metric > 0)) {
    if (Tr->io_omtfilename[0] != '\0') {
      // A background mesh is supplied.
      Tr->OMT_domain = new Triangulation();
      int myargc = 2;
      char *myargv[2];
      myargv[0] = argv[0];
      myargv[1] = Tr->io_omtfilename;
      Tr->OMT_domain->parse_commands(myargc, myargv);
      Tr->OMT_domain->read_mesh();
      Tr->OMT_domain->reconstruct_mesh(0);
      Tr->op_metric = METRIC_Euclidean;
    } else {
      assert(Tr->OMT_domain == NULL);
    }
  
    if (Tr->op_metric) {
      Tr->set_vertex_metrics();
      Tr->coarsen_mesh();
    }
  
    Tr->delaunay_refinement();
  }

  // Mesh export (to files).
  if (Tr->tr_tris != NULL) {
    if (Tr->ct_exteriors > 0) { 
      Tr->remove_exteriors();
    }
    Tr->save_triangulation();
    if (Tr->io_outedges) {
      Tr->save_edges();
    }
  }

  Tr->mesh_statistics();

  delete Tr;
  return 1;
}

//==============================================================================
int adapt_mesh(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: detri2_adapt_mesh [-options] filename[.ele, .mesh]\n");
    return 0;
  }

  Triangulation *Tr = new Triangulation();

  // Read options.
  if (!Tr->parse_commands(argc, argv)) {
    // No input or wrong parameters.
    printf("Usage: detri2 [-options] filename[.node, .poly, .ele, .edge]\n");
    delete Tr;
    return 0;
  }

  // Read inputs.
  if (!Tr->read_mesh()) {
    printf("Failed to read input from file %s[.poly, .node, .ele, .edge]\n",
           Tr->io_infilename);
    delete Tr;
    return 0;
  }

  // Generate (constrained) (weighted) Delaunay triangulation.
  if (Tr->tr_tris == NULL) {
    printf("No input mesh.\n");
    delete Tr;
    return 0;
  }
  
  if (!Tr->io_with_metric) {
    printf("No metric.\n");
    delete Tr;
    return 0;
  }

  Tr->reconstruct_mesh(0);

  Tr->mesh_adapt();

  //Tr->mesh_statistics();

  // Save mesh for solver.
  Tr->save_inria_mesh(); 
  
  delete Tr;
  return 1;
}

//==============================================================================

int interpolate_solutions(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage: detri2_interpolate_solutions src.mesh dest.mesh\n");
    return 0;
  }

  printf("\n\nInterpolate solutions...\n");

  char fname1[256], fname2[256];
  strcpy(fname1, argv[1]);
  strcpy(fname2, argv[2]);

  char *pstr = strstr(fname1, ".mesh");
  if (pstr != NULL) {
    *pstr = '\0';
  }
  pstr = strstr(fname2, ".mesh");
  if (pstr != NULL) {
    *pstr = '\0';
  }

  Triangulation *srcTr = new Triangulation();
  strcpy(srcTr->io_infilename, fname1);
  sprintf(srcTr->io_outfilename, "%s_debug", fname1);

  srcTr->io_inria_mesh = 1;
  if (!srcTr->read_inria_mesh()) {
    printf("Failed to read source mesh %s.\n", argv[1]);
    delete srcTr;
    return 0;
  }

  Triangulation *dstTr = new Triangulation();
  strcpy(dstTr->io_infilename, fname2);
  sprintf(dstTr->io_outfilename, "%s_debug", fname2);

  dstTr->io_inria_mesh = 1;
  if (!dstTr->read_inria_mesh()) {
    printf("Failed to read dest mesh %s.\n", argv[2]);
    delete srcTr;
    delete dstTr;
    return 0;
  }

  //srcTr->op_convex = 1; // make it convex. This needs to be re-implemented later.
  //srcTr->reconstruct_mesh(0);
  // Make a convex hull of the vertices of the source mesh.
  if (srcTr->tr_tris != NULL) {
    delete srcTr->tr_tris;
    srcTr->tr_tris = NULL;
  }
  if (srcTr->tr_segs != NULL) {
    delete srcTr->tr_segs;
    srcTr->tr_segs = NULL;
  }
  srcTr->incremental_delaunay();
  // Save for debug
  srcTr->save_triangulation();

  // Read solutions on the source mesh.  
  FILE *fuh0 = fopen("../files/uh0.txt", "r");
  if (!fuh0) {
    printf("Failed to read soltion file uh0.txt");
    delete srcTr;
    delete dstTr;
    return 0;
  }
  double *uh0 = new double[srcTr->ct_in_vrts];
  for (int i = 0; i < srcTr->ct_in_vrts; i++) {
    fscanf(fuh0, "%lf\n", &(uh0[i]));
  }
  fclose(fuh0);

  FILE *fuh1 = fopen("../files/uh1.txt", "r");
  if (!fuh1) {
    printf("Failed to read soltion file uh1.txt");
    delete [] uh0;
    delete srcTr;
    delete dstTr;
    return 0;
  }
  double *uh1 = new double[srcTr->ct_in_vrts];
  for (int i = 0; i < srcTr->ct_in_vrts; i++) {
    fscanf(fuh1, "%lf\n", &(uh1[i]));
  }
  fclose(fuh1);  

  // Save files for debug
  for (int i = 0; i < srcTr->ct_in_vrts; i++) {
    srcTr->in_vrts[i].val = uh0[i];
  }
  srcTr->save_to_ucd(0, 1); // mshidx=0
  
  for (int i = 0; i < srcTr->ct_in_vrts; i++) {
    srcTr->in_vrts[i].val = uh1[i];
  }
  srcTr->save_to_ucd(1, 1); // mshidx=1

  // Interpolation
  dstTr->reconstruct_mesh(0);

  // The interpolated solution.
  double *iuh0 = new double[dstTr->ct_in_vrts];
  double *iuh1 = new double[dstTr->ct_in_vrts];

  // Interpolate solutions for dest mesh.
  // Locate points from dstTr in srcTr, then get its value.
  TriEdge E, N;
  int loc;
  for (int i = 0; i < dstTr->tr_tris->used_items; i++) {
    Triang *tri = (Triang *) dstTr->tr_tris->get(i);
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    E.tri = NULL;
    for (int j = 0; j < 3; j++) {
      if (!tri->vrt[j]->is_infected()) {
        int idx = tri->vrt[j]->idx - 1;
        loc = srcTr->locate_point(tri->vrt[j], E, false);
        if (loc == LOC_IN_OUTSIDE) {
          //assert(0); // Caused by rounding error, slightly outside.
          assert(E.tri->is_hulltri());
          // E should be an hull edge.
          assert(E.apex() == srcTr->tr_infvrt);
          // loc = srcTr->locate_hull_edge(tri->vrt[j], E);
          //E = E.esym();
          loc = LOC_ON_EDGE; // interpolate on this hull edge.
        } else {
          if (loc == LOC_IN_TRI) {
            // Approximating the location of the point w.r.t. this triangle.
            //
            //                    * v3
            //                  / | \
            //                 /  |  \
            //                /   |   \
            //               /a3  | a2 \
            //              /    *pt    \
            //             /    /    \   \
            //            /       a1      \
            //        v1 *-----------------* v2
            //                 -- E -->
            //
            Vertex *pt = tri->vrt[j];
            Vertex *v1 = E.org();
            Vertex *v2 = E.dest();
            Vertex *v3 = E.apex();
            double a1 = srcTr->get_tri_area(pt, v1, v2);
            double a2 = srcTr->get_tri_area(pt, v2, v3);
            double a3 = srcTr->get_tri_area(pt, v3, v1);
            double triarea = a1 + a2 + a3;
            if (triarea > 0.) {
              // Round the areas
              if (fabs(a1 / triarea - 1.0) < 1e-5) a1 = 0.0;
              if (fabs(a2 / triarea - 1.0) < 1e-5) a2 = 0.0;
              if (fabs(a3 / triarea - 1.0) < 1e-5) a3 = 0.0;
              if (a1 == 0) {
                if (a2 == 0) {
                  if (a3 > 0) {
                    // on v2.
                    E = E.enext(); // E.enextself();
                    loc = LOC_ON_VERT;
                  } else {
                    assert(0); // not possible
                  }
                } else if (a3 == 0) {
                  // on v1.
                  loc = LOC_ON_VERT;
                } else {
                  // on edge (v1, v2)
                  loc = LOC_ON_EDGE;
                }
              } else if (a2 == 0) {
                if (a3 == 0) {
                  // on v3
                  E = E.eprev();
                  loc = LOC_ON_VERT;
                } else {
                  // on edge (v2, v3).
                  E = E.enext();
                  loc = LOC_ON_EDGE;
                }
              } else if (a3 == 0) {
                // on edge (v3, v1).
                E = E.eprev();
                loc = LOC_ON_EDGE;
              }
            } else {
              // A degenerated triangle. Choose the longest edge
              a1 = srcTr->get_distance(v1, v2);
              a2 = srcTr->get_distance(v2, v3);
              a3 = srcTr->get_distance(v3, v1);
              if (a1 > a2) {
                if (a1 > a3) {
                  // Choose edge (v1, v2)
                  //loc = LOC_ON_EDGE;
                } else {
                  // Choose edge (v3, v1)
                  E = E.eprev();
                }
              } else {
                if (a2 > a3) {
                  // Choose edge (v2, v3)
                  E = E.enext();
                } else {
                  // Choose edge (v3, v1)
                  E = E.eprev();
                }
              }
              loc = LOC_ON_EDGE;
            }
          } // if (loc == LOC_IN_TRI)
        } // if (loc == LOC_IN_OUTSIDE)
        
        if (loc == LOC_ON_VERT) {
          // no need to interpolate.
          Vertex *v = E.org();
          iuh0[idx] = uh0[v->idx - 1];
          iuh1[idx] = uh1[v->idx - 1];
        } else if (loc == LOC_ON_EDGE) {
          // Linear interpolation on edge E.
          Vertex *v = tri->vrt[j];
          Vertex *pa = E.org();
          Vertex *pb = E.dest();
          double L  = srcTr->get_distance(pa, pb);
          double Wa = srcTr->get_distance( v, pb);
          double Wb = srcTr->get_distance(pa,  v);
          iuh0[idx] = (uh0[pa->idx - 1] * Wa + uh0[pb->idx - 1] * Wb) / L;
          iuh1[idx] = (uh1[pa->idx - 1] * Wa + uh1[pb->idx - 1] * Wb) / L;
        } else {
          if (E.tri->is_hulltri()) {
            E = E.esym();
          }
          Vertex *v = tri->vrt[j];
          Vertex *pa = E.org();
          Vertex *pb = E.dest();
          Vertex *pc = E.apex();
          double A  = srcTr->get_tri_area(pa, pb, pc);
          double Wa = srcTr->get_tri_area( v, pb, pc);
          double Wb = srcTr->get_tri_area(pa,  v, pc);
          double Wc = srcTr->get_tri_area(pa, pb,  v);
          //  v->val = (pa->val * Wa + pb->val * Wb + pc->val * Wc) / A;
          iuh0[idx] = (uh0[pa->idx - 1] * Wa + uh0[pb->idx - 1] * Wb + uh0[pc->idx - 1] * Wc) / A;
          iuh1[idx] = (uh1[pa->idx - 1] * Wa + uh1[pb->idx - 1] * Wb + uh1[pc->idx - 1] * Wc) / A;
        }
        tri->vrt[j]->set_infect();
      }
    }
  }

  // Save results for debugging.
  for (int i = 0; i < dstTr->ct_in_vrts; i++) {
    dstTr->in_vrts[i].val = iuh0[i];
  }
  dstTr->save_to_ucd(0, 1); // mshidx=0
  
  for (int i = 0; i < dstTr->ct_in_vrts; i++) {
    dstTr->in_vrts[i].val = iuh1[i];
  }
  dstTr->save_to_ucd(1, 1); // mshidx=1

  // write results.
  fuh0 = fopen("../files/uh0.1.txt", "w");
  for (int i = 0; i < dstTr->ct_in_vrts; i++) {
    fprintf(fuh0, "%.17g\n", iuh0[i]);
  }
  fclose(fuh0);

  fuh1 = fopen("../files/uh1.1.txt", "w");
  for (int i = 0; i < dstTr->ct_in_vrts; i++) {
    fprintf(fuh1, "%.17g\n", iuh1[i]);
  }
  fclose(fuh1);

  delete [] uh0;
  delete [] uh1;
  delete [] iuh0;
  delete [] iuh1;

  delete srcTr;
  delete dstTr;
  return 1;
}

//==============================================================================

int save_solution_to_paraview(int argc, char *argv[])
{
  if (argc < 3) {
    printf("Usage: detri2_save_solutions sol.mesh iter\n");
    return 0;
  }

  char fname[256];
  strcpy(fname, argv[1]);

  char *pstr = strstr(fname, ".mesh");
  if (pstr != NULL) {
    *pstr = '\0';
  }

  Triangulation *Tr = new Triangulation();
  strcpy(Tr->io_infilename, fname);
  sprintf(Tr->io_outfilename, "../output/sol");

  Tr->io_inria_mesh = 1;
  if (!Tr->read_inria_mesh()) {
    printf("Failed to read mesh %s.\n", argv[1]);
    delete Tr;
    return 0;
  }

  if (Tr->io_with_solution == 0) {
    printf("Failed to read solution with the mesh %s.\n", argv[1]);
    delete Tr;
    return 0;
  }

  int i;
  for (i = 0; i < Tr->ct_in_vrts; i++) {
    Tr->in_vrts[i].val = Tr->in_vrts[i].fval;
  }

  int iter = atoi(argv[2]);
  printf("Save solution %d to file\n", iter);

  Tr->save_to_ucd(iter, 1);

  delete Tr;
  return 1;
}

//==============================================================================
void detri2_draw_voronoi_mass_centers(Triangulation* Tr, Triangulation* OMT,
                                      bool show_powe_cell, int center_size,
                                      void* Painter, double Cx, double Cy, double Sxy)
{
    //QPainter *painter = (QPainter *) Painter;
    //qDebug()<<"detri2 draw voronoi";
    Tr->OMT_domain = OMT;

    //qDebug()<<"  calculating voronoi vertices";
    int i, idx;
    idx = Tr->io_firstindex;
    // Calculate circumcenters for hull triangles.
    for (i = 0; i < Tr->tr_tris->used_items; i++) {
      Triang* tri = (Triang *) Tr->tr_tris->get(i);
      // Ignore exterior triangles.
      if (tri->is_deleted() || tri->is_hulltri()) continue;
      Tr->get_tri_orthocenter(tri);
      tri->idx = idx; // hulltri is also indexed.
      idx++;
    }

    printf("cccccccccccc detri2_draw_voronoi_mass_centers()\n");

    // Calculate bisectors for hull triangles.
    for (i = 0; i < Tr->tr_tris->used_items; i++) {
      Triang* tri = (Triang *) Tr->tr_tris->get(i);
      // Ignore exterior triangles.
      if (tri->is_deleted()) continue;
      if (tri->is_hulltri()) { // A hull triangle.
        Tr->get_hulltri_orthocenter(tri);
      }
    }

    printf("aaaaaaaaaaa detri2_draw_voronoi_mass_centers()\n");

    // Draw power cells and mass centers.
    for (int i = 0; i < Tr->ct_in_vrts; i++) {
      Vertex *mesh_vertex = &(Tr->in_vrts[i]);
      if (mesh_vertex->typ == UNUSEDVERTEX) continue;
      //printf("  idx = %d\n", i);
      Vertex *ptlist = NULL;
      int ptnum = 0;
      if (Tr->get_powercell(mesh_vertex, &ptlist, &ptnum)) {
        if (show_powe_cell) {
          // Draw the ptnum points and line segments.
          for (int i = 0; i < ptnum; i++) {
            Vertex *v1 = &(ptlist[i]);
            Vertex *v2 = &(ptlist[(i+1)%ptnum]);
            int x1 = (int) ( Sxy * v1->crd[0] + Cx);
            int y1 = (int) (-Sxy * v1->crd[1] + Cy);
            int x2 = (int) ( Sxy * v2->crd[0] + Cx);
            int y2 = (int) (-Sxy * v2->crd[1] + Cy);
            //painter->drawLine(QPoint(x1, y1), QPoint(x2,y2));
          }
        }

        // Show mass center
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

        int px = (int) ( Sxy * mcx + Cx);
        int py = (int) (-Sxy * mcy + Cy);
        //painter->drawEllipse(QPoint(px,py), center_size, center_size);

        delete [] ptlist;
      } // if (Tr->get_powercell
    }

    printf("bbbbbbbbbbbbb detri2_draw_voronoi_mass_centers()\n");

    if (Tr->tr_steiners == NULL) {
      return;
    }

    for (int i = 0; i < Tr->tr_steiners->used_items; i++) {
      Vertex *mesh_vertex = (Vertex *) Tr->tr_steiners->get(i);
      if (mesh_vertex->is_deleted()) continue;
      //printf("  idx = %d\n", i);
      Vertex *ptlist = NULL;
      int ptnum = 0;

      printf("%d ddddddddd detri2_draw_voronoi_mass_centers()\n", mesh_vertex->idx);

      if (Tr->get_powercell(mesh_vertex, &ptlist, &ptnum)) {
        if (show_powe_cell) {
          // Draw the ptnum points and line segments.
          for (int i = 0; i < ptnum; i++) {
            Vertex *v1 = &(ptlist[i]);
            Vertex *v2 = &(ptlist[(i+1)%ptnum]);
            int x1 = (int) ( Sxy * v1->crd[0] + Cx);
            int y1 = (int) (-Sxy * v1->crd[1] + Cy);
            int x2 = (int) ( Sxy * v2->crd[0] + Cx);
            int y2 = (int) (-Sxy * v2->crd[1] + Cy);
            //painter->drawLine(QPoint(x1, y1), QPoint(x2,y2));
          }
        }

        // Show mass center
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

        int px = (int) ( Sxy * mcx + Cx);
        int py = (int) (-Sxy * mcy + Cy);
        //painter->drawEllipse(QPoint(px,py), center_size, center_size);

        delete [] ptlist;
      } // if (Tr->get_powercell
    }
}

int anisotropic_CVT(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: detri2 [-options] filename[.node, .ele]\n");
    return 0;
  }

  Triangulation *Tr = new Triangulation();

  // Read options.
  if (!Tr->parse_commands(argc, argv)) {
    // No input or wrong parameters.
    printf("Usage: detri2 [-options] filename[.node, .poly, .ele, .edge]\n");
    delete Tr;
    return 0;
  }

  // Read inputs.
  if (!Tr->read_mesh()) {
    printf("Failed to read input from file %s[.poly, .node, .ele, .edge]\n",
           Tr->io_infilename);
    delete Tr;
    return 0;
  }

  // Generate (constrained) (weighted) Delaunay triangulation.
  if (Tr->tr_tris == NULL) {
    if (Tr->incremental_delaunay()) {
      if (Tr->tr_segs != NULL) {
        Tr->recover_segments();
        Tr->set_subdomains();
      }
    } else {
      printf("Failed to create Delaunay (regular) triangulation.\n");
      delete Tr;
      return 0;
    }
  } else {
    Tr->reconstruct_mesh(1);
  }

  // Mesh refinement and adaptation.
  //if (Tr->tr_segs != NULL) {
  //if (Tr->op_quality || (Tr->op_metric > 0)) {
    if (Tr->io_omtfilename[0] != '\0') {
      // A background mesh is supplied.
      Tr->OMT_domain = new Triangulation();
      int myargc = 2;
      char *myargv[2];
      myargv[0] = argv[0];
      myargv[1] = Tr->io_omtfilename;
      Tr->OMT_domain->parse_commands(myargc, myargv);
      Tr->OMT_domain->read_mesh();
      Tr->OMT_domain->reconstruct_mesh(0);
      Tr->op_metric = METRIC_Euclidean;
    } else {
      assert(Tr->OMT_domain == NULL);
    }
  
    if (Tr->op_metric) {
      Tr->set_vertex_metrics();
      Tr->coarsen_mesh();
    }
  
    Tr->op_target_length = 0.3;
  
    Tr->delaunay_refinement();
    
    if (Tr->ct_exteriors > 0) { 
      Tr->remove_exteriors();
    }
    
    // Debug ouput
    Tr->save_triangulation();
    
    /*
    // Anisotropic
    Tr->_a11 = 0.333333;
    Tr->_a21 = -0.3849;
    Tr->_a22 = 0.777778;
    
    Tr->lawson_flip(NULL, 0, NULL);

    // Debug ouput
    Tr->save_triangulation();

    // Calculate the mass centers of the VD.
    detri2_draw_voronoi_mass_centers(Tr, Tr, true, 2, NULL, 0, 0, 0);
    */
    Tr->op_smooth_criterion = SMOOTH_DISTMESH;
    Tr->op_target_length = Tr->get_distmesh_target_length();
    Tr->op_smooth_deltat = 0.2;

    for (int it = 0; it < 5; it++) {
      Tr->smooth_vertices();
      Tr->save_triangulation();
    }
    
  //}

  // Mesh export (to files).
  if (Tr->tr_tris != NULL) {
    if (Tr->ct_exteriors > 0) { 
      Tr->remove_exteriors();
    }
    Tr->save_triangulation();
    if (Tr->io_outedges) {
      Tr->save_edges();
    }
  }

  Tr->mesh_statistics();

  delete Tr;
  return 1;
}

//==============================================================================

int main(int argc, char* argv[])
{
  //anisotropic_CVT(argc, argv);
  //generate_mesh(argc, argv);
  //save_inp_to_smesh(argc, argv);
  //merge_two_triangulations(argc, argv);
  
  int status = adapt_mesh(argc, argv);
  //int status = interpolate_solutions(argc, argv);
  //int status = save_solution_to_paraview(argc, argv);
  
  return status;
}
