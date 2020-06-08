#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "detri2.h"

using  namespace detri2;

int Triangulation::parse_commands(int argc, char* argv[])
{
  io_infilename[0] = '\0';
  io_outfilename[0] = '\0';

  char workstring[256];
  int j, k;

  for (int i = 1; i < argc; i++) {
    // Is this string a filename?
    if (argv[i][0] != '-') {
      strncpy(io_infilename, argv[i], 1024 - 1);
      io_infilename[1024 - 1] = '\0';
      continue;
    }
    // Parse the individual switch from the string.
    for (j = 1; argv[i][j] != '\0'; j++) {
      if (argv[i][j] == 'V') {
        op_db_verbose++;
      } else if (argv[i][j] == 'u') {
        op_dt_nearest = -1;
      } else if (argv[i][j] == 'G') {
        op_no_gabriel = 1;
      } else if (argv[i][j] == 'Y') {
        op_no_bisect = 1;
      } else if (argv[i][j] == 'F') {
        op_no_incremental_flip = 1;
      } else if (argv[i][j] == 'q') {
        op_quality = 1;
        if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
            (argv[i][j + 1] == '.')) {
          k = 0;
          while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                 (argv[i][j + 1] == '.') || (argv[i][j + 1] == 'e') ||
                 (argv[i][j + 1] == '-') || (argv[i][j + 1] == '+')) {
            j++;
            workstring[k] = argv[i][j];
            k++;
          }
          workstring[k] = '\0';
          op_minangle = atof(workstring);
        }
      } else if (argv[i][j] == 'L') {
        if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
            (argv[i][j + 1] == '.')) {
          k = 0;
          while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                 (argv[i][j + 1] == '.') || (argv[i][j + 1] == 'e') ||
                 (argv[i][j + 1] == '-') || (argv[i][j + 1] == '+')) {
            j++;
            workstring[k] = argv[i][j];
            k++;
          }
          workstring[k] = '\0';
          op_minlen = atof(workstring);
        }
      } else if (argv[i][j] == 'a') {
        if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
            (argv[i][j + 1] == '.')) {
          k = 0;
          while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                 (argv[i][j + 1] == '.') || (argv[i][j + 1] == 'e') ||
                 (argv[i][j + 1] == '-') || (argv[i][j + 1] == '+')) {
            j++;
            workstring[k] = argv[i][j];
            k++;
          }
          workstring[k] = '\0';
          op_maxarea = atof(workstring);
        }
      } else if (argv[i][j] == 'R') {
        //op_coarse_mesh = 1;
      } else if (argv[i][j] == 'm') {
        /* [2019-02-06] no use
        op_metric = METRIC_Euclidean;
        if (argc >= (i+2)) {
          if (argv[i+1][0] != '-') {
            // It is a filename following by -m
            strncpy(io_omtfilename, argv[i+1], 1024 - 1);
            io_omtfilename[1024 - 1] = '\0';
            i++; // Skip the next string.
            break; // j
          }
        }
        */
      } else if (argv[i][j] == 'S') { 
        // Sorting options.
        if (argv[i][j+1] == 'N') { // -SN
          so_nosort = 1; j++;
        } else if (argv[i][j+1] == 'R') { // -SR
          so_norandom = 1; j++;
        } else if (argv[i][j+1] == 'B') { // -SB
          so_nobrio = 1; j++;
        } else if (argv[i][j+1] == 'h') { // -Sh#,#
          // Parse options (if provided).
          j++;
          if ((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) {
            k = 0;
            while ((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) {
              j++; workstring[k] = argv[i][j]; k++;
            }
            workstring[k] = '\0';
            so_hilbert_order = atoi(workstring);
          }
          if ((argv[i][j + 1] == '/') || (argv[i][j + 1] == ',')) {
            j++;
            if ((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) {
              k = 0;
              while ((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) {
                j++; workstring[k] = argv[i][j]; k++;
              }
              workstring[k] = '\0';
              so_hilbert_limit = atoi(workstring);
            }
          }
        } else if (argv[i][j+1] == 'b') { // -Sb#,#
          // Parse options (if provided).
          j++;
          if ((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) {
            k = 0;
            while ((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) {
              j++; workstring[k] = argv[i][j]; k++;
            }
            workstring[k] = '\0';
            so_brio_threshold = atoi(workstring);
          }
          if ((argv[i][j + 1] == '/') || (argv[i][j + 1] == ',')) {
            j++;
            if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                (argv[i][j + 1] == '.')) {
              k = 0;
              while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                     (argv[i][j + 1] == '.') || (argv[i][j + 1] == 'e') ||
                     (argv[i][j + 1] == '-') || (argv[i][j + 1] == '+')) {
                j++; workstring[k] = argv[i][j]; k++;
              }
              workstring[k] = '\0';
              so_brio_ratio = atof(workstring);
            }
          }
        }
        // End of '-S'
      } else if (argv[i][j] == 'M') {
        if (argv[i][j+1] == 'L') { // -ML=#
          // Set target length (and tri area)
          j+=3; // skip 'ML='
          if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
              (argv[i][j + 1] == '.')) {
            k = 0;
            while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                   (argv[i][j + 1] == '.') || (argv[i][j + 1] == 'e') ||
                   (argv[i][j + 1] == '-') || (argv[i][j + 1] == '+')) {
              j++;
              workstring[k] = argv[i][j];
              k++;
            }
            workstring[k] = '\0';
            op_target_length = atof(workstring);
          }
        } else if ((argv[i][j+1] == 's') &&
                   (argv[i][j+2] == '1')) { // -Ms1=#
          // the s1 parameter for HDE
          j+=3; // skip 'Ms1='
          if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
              (argv[i][j + 1] == '.')) {
            k = 0;
            while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                   (argv[i][j + 1] == '.') || (argv[i][j + 1] == 'e') ||
                   (argv[i][j + 1] == '-') || (argv[i][j + 1] == '+')) {
              j++;
              workstring[k] = argv[i][j];
              k++;
            }
            workstring[k] = '\0';
            //op_hde_s1 = atof(workstring);
          }
        } else if ((argv[i][j+1] == 's') &&
                   (argv[i][j+2] == '2')) { // -Ms2=#
          // the s2 parameter for HDE
          j+=3; // skip 'Ms2='
          if (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
              (argv[i][j + 1] == '.')) {
            k = 0;
            while (((argv[i][j + 1] >= '0') && (argv[i][j + 1] <= '9')) ||
                   (argv[i][j + 1] == '.') || (argv[i][j + 1] == 'e') ||
                   (argv[i][j + 1] == '-') || (argv[i][j + 1] == '+')) {
              j++;
              workstring[k] = argv[i][j];
              k++;
            }
            workstring[k] = '\0';
            //op_hde_s2 = atof(workstring);
          }         
        } else if (argv[i][j+1] == 'f') { // -Mf=
          j+=3; // skip 'Mf='
          //op_test_fun = (int) (argv[i][j] - '0');
          j++;
        }
        // End of '-M'
      } else if (argv[i][j] == 'I') {
        // Input and output options
        if (argv[i][j+1] == 'N') { // -IN
          io_noindices = 1; j++;
        } else if (argv[i][j+1] == '0') { // -I0 (zero)
          io_firstindex = 0; j++; 
        } else if (argv[i][j+1] == '1') { // -I1 (one)
          io_firstindex = 1; j++; 
        } else if (argv[i][j+1] == 'v') { // -Iv
          io_out_voronoi = 1; j++;
        } else if (argv[i][j+1] == 'e') { // -Ie
          io_outedges = 1; j++;
        } else if (argv[i][j+1] == 'J') { // -IJ
          io_no_unused = 1; j++;
        } else if (argv[i][j+1] == 'd') { // -Id
          io_dump_to_ucd = 1; j++;
        } else if (argv[i][j+1] == 'l') { // -Il
          io_dump_lift_map = 1; j++;
        }
        // End of '-I'
      }
    } // for j
  } // for i

  if (io_infilename[0] == '\0') {
    // No input file name. Use a default output name.
    strcpy(io_outfilename, "output");
    return 0;
  }

  // Reconginze any file format.
  if (!strcmp(&io_infilename[strlen(io_infilename) - 5], ".node")) {
    io_infilename[strlen(io_infilename) - 5] = '\0';
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 4], ".ele")) {
    io_infilename[strlen(io_infilename) - 4] = '\0';
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 5], ".face")) {
    io_infilename[strlen(io_infilename) - 5] = '\0';
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 5], ".edge")) {
    io_infilename[strlen(io_infilename) - 5] = '\0';
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 5], ".poly")) {
    io_infilename[strlen(io_infilename) - 5] = '\0';
    io_poly = 1;
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 6], ".smesh")) {
    io_infilename[strlen(io_infilename) - 6] = '\0';
    io_poly = 1;
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 5], ".mesh")) {
    io_infilename[strlen(io_infilename) - 5] = '\0';
    io_inria_mesh = 1;
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 5], ".voro")) {
    io_infilename[strlen(io_infilename) - 5] = '\0';
    io_voronoi = 1;
  } else if (!strcmp(&io_infilename[strlen(io_infilename) - 4], ".txt")) {
    io_infilename[strlen(io_infilename) - 4] = '\0';
    io_point_array = 1;
  }

  int increment = 0;
  strcpy(workstring, io_infilename);
  j = 1;
  while (workstring[j] != '\0') {
    if ((workstring[j] == '.') && (workstring[j + 1] != '\0')) {
      increment = j + 1;
    }
    j++;
  }
  int meshnumber = 0;
  if (increment > 0) {
    j = increment;
    do {
      if ((workstring[j] >= '0') && (workstring[j] <= '9')) {
        meshnumber = meshnumber * 10 + (int) (workstring[j] - '0');
      } else {
        increment = 0;
      }
      j++;
    } while (workstring[j] != '\0');
  }
  if (increment == 0) {
    meshnumber = 0;
  } else {
    workstring[increment-1] = '\0';
  }
  sprintf(io_outfilename, "%s.%d", workstring, meshnumber + 1);

  return 1;
}

//==============================================================================

int Triangulation::read_point_array()
{
  FILE *infile;
  char line[1024]; //*pstr;
  //char delim[] = " ,\t";
  int pnum = 0, i;
  //int dim = 0; // dimension 2 or 3;
  // If dim == 3, the 3rd value is treated as height.

  char filename[256];
  strcpy(filename, io_infilename);
  strcat(filename, ".txt");
  infile = fopen(filename, "r");
  if (infile == NULL) {
    printf("Unable to open file %s.\n", filename);
    return 0;
  }

  arraypool *ptary = new arraypool(sizeof(double)*2, 10);
  while (fgets(line, 1024, infile)) {
    double *pt = (double *) ptary->alloc();
    sscanf(line, "%lf %lf", &(pt[0]), &(pt[1]));
  }

  pnum = ptary->objects;
  ct_in_vrts = pnum;
  in_vrts = new Vertex[pnum];

  io_firstindex = 1; // index is started from 1.

  printf("Reading %d points from file %s\n", pnum, filename);
  REAL x, y;

  for (i = 0; i < pnum; i++) {
    Vertex *vrt = &in_vrts[i];
    vrt->init();
    // Default vertex type is UNUSEDVERTEX (0)
    vrt->typ = UNUSEDVERTEX;
    ct_unused_vrts++;

    vrt->idx = i + (io_firstindex == 1 ? 1 : 0);

    double *pt = (double *) ptary->get(i);
    x = vrt->crd[0] = pt[0];
    y = vrt->crd[1] = pt[1];
    vrt->crd[2] = x*x + y*y; // height
    vrt->wei = 0.0;

    // Determine the smallest and largest x, and y coordinates.
    if (i == 0) {
      io_xmin = io_xmax = x;
      io_ymin = io_ymax = y;
    } else {
      io_xmin = (x < io_xmin) ? x : io_xmin;
      io_xmax = (x > io_xmax) ? x : io_xmax;
      io_ymin = (y < io_ymin) ? y : io_ymin;
      io_ymax = (y > io_ymax) ? y : io_ymax;
    }
  } // i

  double dx = io_xmax - io_xmin;
  double dy = io_ymax - io_ymin;
  io_diagonal2 = dx*dx + dy*dy;
  io_diagonal = sqrt(io_diagonal2);

  if (i < pnum) {
    printf("Missing %d points from file %s.\n", pnum - i, filename);
    fclose(infile);
    return 0;
  }

  delete ptary;

  fclose(infile);
  return 1;
}

//==============================================================================

int Triangulation::read_nodes()
{
  FILE *infile;
  char line[1024], *pstr;
  char delim[] = " ,\t";
  int pnum = 0, i;
  int dim = 0; // dimension 2 or 3;
  // If dim == 3, the 3rd value is treated as height.

  char filename[256];
  strcpy(filename, io_infilename);
  strcat(filename, ".node");
  infile = fopen(filename, "r");
  if (infile == NULL) {
    printf("Unable to open file %s.\n", filename);
    return 0;
  }

  fgets(line, 1024, infile); // The first line
  // Check if it is an output of rbox (qhull).
  if (strstr(line, "rbox") != NULL) {
    pstr = strtok(line, delim);
    dim = atoi(pstr);
    io_noindices = 1;
    fgets(line, 1024, infile);
    pstr = strtok(line, delim);
    pnum = atoi(pstr);
  } else {
    // Skip comments and empty lines.
    do {
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    } while (fgets(line, 1024, infile));
    /*
    while (fgets(line, 1024, infile)) {
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    */
    // Read the number of nodes.
    pnum = atoi(pstr);
    if (pnum <= 0) {
      printf("!! No points in file %s.\n", filename);
      fclose(infile);
      return 0;
    }
    pstr = strtok(NULL, delim);
    dim = atoi(pstr);
  }
  if ((dim != 2) && (dim != 3)) {
    printf("!! Wrong dimension (%d) (should be 2 or 3) in file %s.\n", dim, filename);
    fclose(infile);
    return 0;
  }

  ct_in_vrts = pnum;
  in_vrts = new Vertex[pnum];

  printf("Reading %d points from file %s\n", pnum, filename);
  REAL x, y;

  for (i = 0; i < pnum; i++) {
    while (fgets(line, 1024, infile)) {
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (feof(infile)) break;

    Vertex *vrt = &in_vrts[i];
    vrt->init();
    // Default vertex type is UNUSEDVERTEX (0)
    ct_unused_vrts++;
    if (!io_noindices) { // no -IN
      vrt->idx = atoi(pstr);
      if (i == 0) { // Set the first index (0 or 1).
        io_firstindex = vrt->idx;
      }
      pstr = strtok(NULL, delim); // Skip the index
    } else { // No index
      vrt->idx = i + (io_firstindex == 1 ? 1 : 0);
    }
    x = vrt->crd[0] = atof(pstr);
    pstr = strtok(NULL, delim);
    y = vrt->crd[1] = atof(pstr);
    if (dim == 3) {
      pstr = strtok(NULL, delim);
      vrt->crd[2] = atof(pstr); // height
      vrt->wei = x*x + y*y - vrt->crd[2];
      //vrt->wei = op_lambda1 * x*x + op_lambda2 * y*y - vrt->crd[2];
    } else {
      vrt->crd[2] = x*x + y*y; // height
      //vrt->crd[2] = op_lambda1 * x*x + op_lambda2 * y*y;
      vrt->wei = 0.0;
    }
    if (pstr != NULL) {
      vrt->tag = atoi(pstr);
    }
    // Determine the smallest and largest x, and y coordinates.
    if (i == 0) {
      io_xmin = io_xmax = x;
      io_ymin = io_ymax = y;
    } else {
      io_xmin = (x < io_xmin) ? x : io_xmin;
      io_xmax = (x > io_xmax) ? x : io_xmax;
      io_ymin = (y < io_ymin) ? y : io_ymin;
      io_ymax = (y > io_ymax) ? y : io_ymax;
    }
  } // i

  double dx = io_xmax - io_xmin;
  double dy = io_ymax - io_ymin;
  io_diagonal2 = dx*dx + dy*dy;
  io_diagonal = sqrt(io_diagonal2);

  if (i < pnum) {
    printf("Missing %d points from file %s.\n", pnum - i, filename);
    fclose(infile);
    return 0;
  }

  fclose(infile);

  // Initialise predicates.
  //void exactinit(int verbose, int noexact, int o3dfilter, int ispfilter,
  //               REAL maxx, REAL maxy, REAL maxz)
  exactinit(0, 0, 0, 0, io_xmax - io_xmin, io_ymax - io_ymax, 0.0);

  return 1;
}

//==============================================================================

int Triangulation::read_weights()
{
  char filename[256];
  FILE *infile = NULL;
  char line[1024], *pstr;
  char delim[] = " ,\t";
  int i;

  // Try to read a .weight file (if it exists).
  strcpy(filename, io_infilename);
  strcat(filename, ".weight");
  infile = fopen(filename, "r");
  if (infile != NULL) {
    // Read the number of segments.
    int wnum = 0;
    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      wnum = atoi(pstr);
    }
    if (wnum == ct_in_vrts) {
      // The number of points and weights must be equal.
      printf("Reading point weights from file %s\n", filename);
      for (i = 0; i < wnum; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        //seg->idx = atoi(pstr); // i
        pstr = strtok(NULL, delim);
        REAL w = atof(pstr);        
        in_vrts[i].wei = w;
        // Re-calculate the height of this vertex.
        REAL x = in_vrts[i].crd[0];
        REAL y = in_vrts[i].crd[1];
        in_vrts[i].crd[2] = x*x + y*y - w;
        //in_vrts[i].crd[2] = op_lambda1 * x*x + op_lambda2 * y*y - w;
      }

      if (i < wnum) {
        printf("Missing %d point weights from file %s.\n", wnum - i, filename);
      }
    } else {
      printf("Wrong number %d (should be %d) of point weights\n", wnum, ct_in_vrts);
    }
    fclose(infile);
  } // Read point weights

  return 1;
}

//==============================================================================
// Read a .mtr file.
// It should be called after read_nodes(), or read_mesh()

int Triangulation::read_metric()
{
  char filename[256];
  FILE *infile = NULL;
  char line[1024], *pstr;
  char delim[] = " ,\t";
  int i, mtrsize = 1; // default

  // Read point metrics (nodal mesh size)
  strcpy(filename, io_infilename);
  strcat(filename, ".mtr");
  infile = fopen(filename, "r");
  if (infile != NULL) {
    // Read the number of points.
    int mtrnum = 0; 
    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      mtrnum = atoi(pstr);
      // Read the number of metrics (1 or 3).
      pstr = strtok(NULL, delim);
      if (pstr != NULL) {
        mtrsize = atoi(pstr);
      }
    }
    if (mtrnum == ct_in_vrts) {
      // The number of points and weights must be equal.
      printf("Reading point metric from file %s\n", filename);
      for (i = 0; i < mtrnum; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        if (!io_noindices) { // no -IN
          //seg->idx = atoi(pstr);
          pstr = strtok(NULL, delim);
        }
        if (mtrsize == 1) {
          in_vrts[i].val = atof(pstr);
        } else if (mtrsize == 3) {
          // Read a metric tensor        
          // to do ...
        } else if (mtrsize == 4) {
          // (HDE metric)
          in_vrts[i].val = atof(pstr); 
          // to do ...
        }
      }
      if (i < mtrnum) {
        printf("Missing %d point value from file %s.\n", mtrnum - i, filename);
      }
      io_with_metric = 1;
    } else {
      printf("Wrong number %d (should be %d) of point values\n", mtrnum, ct_in_vrts);
    }
    fclose(infile);
  }  

  return 1;
}

//==============================================================================
// Read a .area file.
// It should be called after read_mesh()

int Triangulation::read_area()
{
  char filename[256];
  FILE *infile = NULL;
  char line[1024], *pstr;
  char delim[] = " ,\t";
  int i;

  if (tr_tris == NULL) {
    return 0;
  }
  if (ct_in_tris != tr_tris->objects) {
    return 0; // Should be the same.
  }

  // Read triangle area
  strcpy(filename, io_infilename);
  strcat(filename, ".area");
  infile = fopen(filename, "r");
  if (infile != NULL) {
    // Read the number of segments.
    int mtrnum = 0; 
    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      mtrnum = atoi(pstr);
    }
    if (mtrnum == ct_in_tris) {
      // The number of triangle and areas must be equal.
      printf("Reading point value from file %s\n", filename);
      for (i = 0; i < mtrnum; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        if (!io_noindices) { // no -IN
          //seg->idx = atoi(pstr);
          pstr = strtok(NULL, delim);
        }
        Triang *tri = (Triang *) tr_tris->get(i);
        tri->val = atof(pstr);
      }
      if (i < mtrnum) {
        printf("Missing %d area value from file %s.\n", mtrnum - i, filename);
      }
    } else {
      printf("Wrong number %d (should be %d) of area values\n", mtrnum, ct_in_tris);
    }
    fclose(infile);
  }

  return 1;
}

//==============================================================================

// It reads either a .poly or a .smesh file.
int Triangulation::read_poly()
{
  FILE *infile;
  char line[1024], *pstr;
  char delim[] = " ,\t";
  int smesh = 0;
  int i;

  char filename[256];
  strcpy(filename, io_infilename);
  strcat(filename, ".poly");
  infile = fopen(filename, "r");
  if (infile == NULL) {
    strcpy(filename, io_infilename);
    strcat(filename, ".smesh");
    infile = fopen(filename, "r");
    if (infile == NULL) {  
      // printf("Unable to open file %s.poly\n", infilename);
      return 0;
    } else {
      smesh = 1; // Read in an smesh (surface triangulation) file.
    }
  }

  // Read the number of nodes.
  while (fgets(line, 1024, infile)) {
    //printf("%s", line);
    pstr = strtok(line, delim);
    if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
        (pstr[0] != '#')) break;
  }
  if (feof(infile)) {
    printf("Failed to read the point list.\n");
    fclose(infile);
    return 0;
  }

  int pnum = atoi(pstr);
  if (pnum == 0) {
    if (!read_nodes()) {
      printf("Failed to read the point list.\n");
      fclose(infile);
      return 0;
    }
  } else {
    // get the dimension.
    pstr = strtok(NULL, delim);
    int dim = atoi(pstr);
    assert((dim == 2) || (dim == 3));
    
    ct_in_vrts = pnum;
    in_vrts = new Vertex[pnum];
  
    printf("Reading %d points from file %s\n", pnum, filename);
    REAL x, y, w;
  
    for (i = 0; i < pnum; i++) {
      while (fgets(line, 1024, infile)) {
        pstr = strtok(line, delim);
        if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
            (pstr[0] != '#')) break;
      }
      if (feof(infile)) break;
  
      Vertex *vrt = &in_vrts[i];
      vrt->init();
      // Default vertex type is UNUSEDVERTEX (0)
      ct_unused_vrts++;
      if (!io_noindices) { // no -IN
        vrt->idx = atoi(pstr);
        if (i == 0) { // Set the first index (0 or 1).
          io_firstindex = vrt->idx;
        }
        pstr = strtok(NULL, delim); // Skip the index
      } else { // No index
        vrt->idx = i + (io_firstindex == 1 ? 1 : 0);
      }
      x = vrt->crd[0] = atof(pstr);
      pstr = strtok(NULL, delim);
      y = vrt->crd[1] = atof(pstr);
      //pstr = strtok(NULL, delim);
      //z = vrt->crd[2] = atof(pstr);
      //vrt->crd[3] = x*x + y*y + z*z; // height
      if (dim == 3) {
        pstr = strtok(NULL, delim);
        vrt->crd[2] = atof(pstr); // height is the z-coordinate
        w = x*x + y*y - vrt->crd[2]; // calculate the weight
        //w = op_lambda1 * x*x + op_lambda2 * y*y - vrt->crd[2];
        vrt->wei = w;
      } else {
        vrt->crd[2] = x*x + y*y; // height
        //vrt->crd[2] = op_lambda1 * x*x + op_lambda2 * y*y;
        vrt->wei = 0.0;
      }
      if (pstr != NULL) {
        vrt->tag = atoi(pstr);
      }
      // Determine the smallest and largest x, and y coordinates.
      if (i == 0) {
        io_xmin = io_xmax = x;
        io_ymin = io_ymax = y;
      } else {
        io_xmin = (x < io_xmin) ? x : io_xmin;
        io_xmax = (x > io_xmax) ? x : io_xmax;
        io_ymin = (y < io_ymin) ? y : io_ymin;
        io_ymax = (y > io_ymax) ? y : io_ymax;
      }
    } // i

    double dx = io_xmax - io_xmin;
    double dy = io_ymax - io_ymin;
    io_diagonal2 = dx*dx + dy*dy;
    io_diagonal = sqrt(io_diagonal2);

    if (i < pnum) {
      printf("Missing %d points from file %s.\n", pnum - i, filename);
      fclose(infile);
      return 0;
    }

    exactinit(0, 0, 0, 0, io_xmax - io_xmin, io_ymax - io_ymax, 0.0);
  }

  if (smesh) {
    // Read the number of triangles.
    int trinum = 0, v1, v2, v3, tag;
    Vertex *p1, *p2, *p3;

    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      trinum = atoi(pstr);
    }  
    if (trinum <= 0) {
      printf("No triangles in file %s.\n", filename);
      fclose(infile);
      return 0;
    }

    if (trinum > 0) {
      printf("Reading %d triangle from file %s\n", trinum, filename);
      // remember the input number of triangles.
      ct_in_tris = trinum;
      int log2objperblk = 0;
      while (trinum >>= 1) log2objperblk++;
      tr_tris = new arraypool(sizeof(Triang), log2objperblk);
      for (i = 0; i < ct_in_tris; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        v1 = v2 = v3 = 0;
        //if (!io_noindices) { // no -IN
          //seg->idx = atoi(pstr);
          pstr = strtok(NULL, delim);
        //}
        v1 = atoi(pstr);
        pstr = strtok(NULL, delim);
        v2 = atoi(pstr);
        pstr = strtok(NULL, delim);
        v3 = atoi(pstr);
        pstr = strtok(NULL, delim);
        if (pstr != NULL) {
          tag = atoi(pstr);
        } else {
          tag = 0;
        }
        if ((v1 >= io_firstindex) && (v1 < ct_in_vrts + io_firstindex) &&
            (v2 >= io_firstindex) && (v2 < ct_in_vrts + io_firstindex) &&
            (v3 >= io_firstindex) && (v3 < ct_in_vrts + io_firstindex)) {
          p1 = &(in_vrts[v1 - io_firstindex]);
          p2 = &(in_vrts[v2 - io_firstindex]);
          p3 = &(in_vrts[v3 - io_firstindex]);
          // Make sure all tetrahedra are CCW oriented.
          REAL ori = Orient2d(p1, p2, p3);
          if (ori < 0) {
            // Swap the first two vertices.
            Vertex *swap = p1;
            p1 = p2;
            p2 = swap;
          }
          if (ori != 0) {
            Triang *tri = (Triang *) tr_tris->alloc();
            tri->init();
            tri->vrt[0] = p1;
            tri->vrt[1] = p2;
            tri->vrt[2] = p3;
            tri->tag = tag;
          } else {
            printf("!! Triangle #%d [%d,%d,%d] is degenerated.\n",
                   i + io_firstindex, v1, v2, v3);
          }
        } else {
          printf("!! Triangle #%d [%d,%d,%d] has invalid vertices.\n",
                 i + io_firstindex, v1, v2, v3);
        }
      }
    } // if (trinum > 0)
    fclose(infile);
  } else {
    // Read the number of segments.
    int snum = 0, e1, e2, tag;
  
    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      snum = atoi(pstr);
    }
    if (snum > 0) {
      printf("Reading %d segments from file %s\n", snum, filename);
      int est_size = snum;
      int log2objperblk = 0;
      while (est_size >>= 1) log2objperblk++;
      tr_segs = new arraypool(sizeof(Triang), log2objperblk);
      for (i = 0; i < snum; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        e1 = e2 = 0;
        //if (!io_noindices) { // no -IN
          //seg->idx = atoi(pstr);
          pstr = strtok(NULL, delim);
        //}
        e1 = atoi(pstr);
        pstr = strtok(NULL, delim);
        e2 = atoi(pstr);
        pstr = strtok(NULL, delim);
        if (pstr != NULL) {
          tag = atoi(pstr);
        } else {
          tag = -1; // default give it a boundary marker.
        }
        if (tag != 0) {
          // A segment must have non-zero tag.
          if ((e1 != e2) &&
              (e1 >= io_firstindex) && (e1 < ct_in_vrts + io_firstindex) &&
              (e2 >= io_firstindex) && (e2 < ct_in_vrts + io_firstindex)) {
            Triang *seg = (Triang *) tr_segs->alloc();
            seg->init();
            seg->vrt[0] = &(in_vrts[e1 - io_firstindex]);
            seg->vrt[1] = &(in_vrts[e2 - io_firstindex]);
            seg->tag = tag;
            //printf("  get a segment %d,%d, tag(%d), val(%g)\n",
            //       seg->vrt[0]->idx, seg->vrt[1]->idx, seg->tag, seg->val);
          } else {
            printf("Segment %d has invalid vertices.\n", i + io_firstindex);
          }
        }
      }
      //printf("Read %d segments from file %s\n", tr_segs->objects, filename);
    } // snum > 0
    //fclose(infile);
  } // if (!smesh)

  printf("debugging: io.cpp read_poly()  read holes\n");

  // Read the number of use-deined holes.
  int rnum = 0;
  while (fgets(line, 1024, infile)) {
    //printf("%s", line);
    pstr = strtok(line, delim);
    if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
        (pstr[0] != '#')) break;
  }
  if (pstr != NULL) {
    rnum = atoi(pstr);
  }
  if (rnum > 0) {
    printf("Reading %d holes from file %s\n", rnum, filename);
    ct_in_sdms = rnum;
    in_sdms = new Vertex[ct_in_sdms];
    for (i = 0; i < rnum; i++) {
      while (fgets(line, 1024, infile)) {
        //printf("%s", line);
        pstr = strtok(line, delim);
        if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
            (pstr[0] != '#')) break;
      }
      //if (feof(infile)) break;
      Vertex *vrt = &(in_sdms[i]);
      vrt->idx = atoi(pstr);
      pstr = strtok(NULL, delim);
      vrt->crd[0] = atof(pstr);
      pstr = strtok(NULL, delim);
      vrt->crd[1] = atof(pstr);
      pstr = strtok(NULL, delim);
      if (pstr != NULL) {
        vrt->tag = atoi(pstr);
        pstr = strtok(NULL, delim);
        if (pstr != NULL) {
          vrt->val = atof(pstr); // region maxarea.
        }
      } else {
        vrt->tag = 0; // Hole
      }
    } // i
  } // rnum > 0

  printf("debugging: io.cpp read_poly()  read regions\n");

  // Read the number of user-defined subdomains.
  rnum = 0;
  while (fgets(line, 1024, infile)) {
    //printf("%s", line);
    pstr = strtok(line, delim);
    if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
        (pstr[0] != '#')) break;
  }
  if (pstr != NULL) {
    rnum = atoi(pstr);
  }
  if (rnum > 0) {
    printf("Reading %d subdomains from file %s\n", rnum, filename);
    if (ct_in_sdms > 0) {
      Vertex *newsdm = new Vertex[ct_in_sdms + rnum];
      for (i = 0; i < ct_in_sdms; i++) {
        newsdm[i].init();
        newsdm[i].crd[0] = in_sdms[i].crd[0];
        newsdm[i].crd[1] = in_sdms[i].crd[1];
        newsdm[i].tag = in_sdms[i].tag;
      }
      delete [] in_sdms;
      in_sdms = newsdm;
    } else {
      in_sdms = new Vertex[rnum];
    }
    int idx = ct_in_sdms;
    ct_in_sdms += rnum;
    for (i = 0; i < rnum; i++) {
      while (fgets(line, 1024, infile)) {
        //printf("%s", line);
        pstr = strtok(line, delim);
        if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
            (pstr[0] != '#')) break;
      }
      //if (feof(infile)) break;
      Vertex *vrt = &(in_sdms[idx]);
      vrt->idx = atoi(pstr);
      pstr = strtok(NULL, delim);
      vrt->crd[0] = atof(pstr);
      pstr = strtok(NULL, delim);
      vrt->crd[1] = atof(pstr);
      pstr = strtok(NULL, delim);
      if (pstr != NULL) {
        vrt->tag = atoi(pstr);
        pstr = strtok(NULL, delim);
        if (pstr != NULL) {
          vrt->val = atof(pstr); // region maxarea.
        }
      } else {
        vrt->tag = 0; // Hole
      }
      idx++;
    } // i
  } // if (rnum > 0)

  fclose(infile);

  printf("debugging: io.cpp read_poly()  done!\n");

  // Read point weights.
  read_weights();
  // Read point metrics (.mtr)
  read_metric();
  // Read triangle areas (.area)
  read_area();
  
  read_sol();
  read_grd();

  return 1;
}

//==============================================================================

char* findnextnumber(char *string)
{
  char *result;

  result = string;
  // Skip the current field.  Stop upon reaching whitespace or a comma.
  while ((*result != '\0') && (*result != '#') && (*result != ' ') && 
         (*result != '\t') && (*result != ',') &&
         (*result != '(') && (*result != '{')) {
    result++;
  }
  // Now skip the whitespace and anything else that doesn't look like a
  //   number, a comment, or the end of a line. 
  while ((*result != '\0') && (*result != '#')
         && (*result != '.') && (*result != '+') && (*result != '-')
         && ((*result < '0') || (*result > '9'))) {
    result++;
  }
  // Check for a comment (prefixed with `#').
  if (*result == '#') {
    *result = '\0';
  }
  return result;
}

//==============================================================================

int Triangulation::read_inria_mesh()
{
  // Read mesh file(s)
  char filename[256];
  FILE *infile = NULL;
  char line[1024], *pstr;
  char delim[] = " ,\t";
  int i;

  // Try to read a .edge file (if it exists).
  strcpy(filename, io_infilename);
  strcat(filename, ".mesh");
  infile = fopen(filename, "r");
  if (infile == NULL) {
    return 0;
  }

  int pnum = 0, trinum = 0, snum = 0;

  while (fgets(line, 1024, infile)) {
    if (pnum == 0) {
      pstr = strstr(line, "Vertices");
      if (pstr) {
        // Read the number of vertices.
        pstr = findnextnumber(pstr); // Skip field "Vertices".
        if (*pstr == '\0') {
          // Read a non-empty line.
          pstr = fgets(line, 1024, infile);
        }
        pnum = atoi(pstr);
        if (pnum > 0) {
          ct_in_vrts = pnum;
          in_vrts = new Vertex[pnum];
        
          printf("Reading %d points from file %s\n", pnum, filename);
          REAL x, y;
        
          io_firstindex = 1; // .mesh use 1 as first index.
        
          for (i = 0; i < pnum; i++) {
            while (fgets(line, 1024, infile)) {
              pstr = strtok(line, delim);
              if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
                  (pstr[0] != '#')) break;
            }
            //if (feof(infile)) break;
            Vertex *vrt = &in_vrts[i];
            vrt->init();
            // Default vertex type is UNUSEDVERTEX (0)
            ct_unused_vrts++;
            if (0) { // .mesh has no index
              vrt->idx = atoi(pstr);
              if (i == 0) { // Set the first index (0 or 1).
                io_firstindex = vrt->idx;
              }
              pstr = strtok(NULL, delim); // Skip the index
            } else { // No index
              vrt->idx = i + (io_firstindex == 1 ? 1 : 0);
            }
            x = vrt->crd[0] = atof(pstr);
            pstr = strtok(NULL, delim);
            y = vrt->crd[1] = atof(pstr);
            //pstr = strtok(NULL, delim);
            //z = vrt->crd[2] = atof(pstr);
            //vrt->crd[3] = x*x + y*y + z*z; // height
            vrt->crd[2] = x*x + y*y; // height
            //vrt->crd[2] = op_lambda1 * x*x + op_lambda2 * y*y;
            vrt->wei = 0.0; // no weight
            //if (pstr != NULL) {
            //  vrt->tag = atoi(pstr);
            //}
            // Determine the smallest and largest x, and y coordinates.
            if (i == 0) {
              io_xmin = io_xmax = x;
              io_ymin = io_ymax = y;
            } else {
              io_xmin = (x < io_xmin) ? x : io_xmin;
              io_xmax = (x > io_xmax) ? x : io_xmax;
              io_ymin = (y < io_ymin) ? y : io_ymin;
              io_ymax = (y > io_ymax) ? y : io_ymax;
            }
          } // i

          double dx = io_xmax - io_xmin;
          double dy = io_ymax - io_ymin;
          io_diagonal2 = dx*dx + dy*dy;
          io_diagonal = sqrt(io_diagonal2);
        
          if (i < pnum) {
            printf("Missing %d points from file %s.\n", pnum - i, filename);
            fclose(infile);
            return 0;
          }

          exactinit(0, 0, 0, 0, io_xmax - io_xmin, io_ymax - io_ymax, 0.0);    
        } // pnum > 0
        continue;
      }
    }
    if (trinum == 0) {
      pstr = strstr(line, "Triangles");
      if (pstr) {
        // Read the number of vertices.
        pstr = findnextnumber(pstr); // Skip field "Vertices".
        if (*pstr == '\0') {
          // Read a non-empty line.
          pstr = fgets(line, 1024, infile);
        }
        trinum = atoi(pstr);
        if (trinum > 0) {
          printf("Reading %d triangle from file %s\n", trinum, filename);
          // remember the input number of triangles.
          int v1, v2, v3, tag;
          Vertex *p1, *p2, *p3;
          ct_in_tris = trinum;
          int log2objperblk = 0;
          while (trinum >>= 1) log2objperblk++;
          tr_tris = new arraypool(sizeof(Triang), log2objperblk);
          for (i = 0; i < ct_in_tris; i++) {
            while (fgets(line, 1024, infile)) {
              //printf("%s", line);
              pstr = strtok(line, delim);
              if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
                  (pstr[0] != '#')) break;
            }
            //if (feof(infile)) break;
            v1 = v2 = v3 = 0;
            if (0) { // no -IN
              //seg->idx = atoi(pstr);
              pstr = strtok(NULL, delim);
            }
            v1 = atoi(pstr);
            pstr = strtok(NULL, delim);
            v2 = atoi(pstr);
            pstr = strtok(NULL, delim);
            v3 = atoi(pstr);
            pstr = strtok(NULL, delim);
            if (pstr != NULL) {
              tag = atoi(pstr);
            } else {
              tag = 0;
            }
            if ((v1 >= io_firstindex) && (v1 < ct_in_vrts + io_firstindex) &&
                (v2 >= io_firstindex) && (v2 < ct_in_vrts + io_firstindex) &&
                (v3 >= io_firstindex) && (v3 < ct_in_vrts + io_firstindex)) {
              p1 = &(in_vrts[v1 - io_firstindex]);
              p2 = &(in_vrts[v2 - io_firstindex]);
              p3 = &(in_vrts[v3 - io_firstindex]);
              // Make sure all tetrahedra are CCW oriented.
              REAL ori = Orient2d(p1, p2, p3);
              if (ori < 0) {
                // Swap the first two vertices.
                Vertex *swap = p1;
                p1 = p2;
                p2 = swap;
              }
              if (ori != 0) {
                Triang *tri = (Triang *) tr_tris->alloc();
                tri->init();
                tri->vrt[0] = p1;
                tri->vrt[1] = p2;
                tri->vrt[2] = p3;
                tri->tag = tag;
              } else {
                printf("!! Triangle #%d [%d,%d,%d] is degenerated.\n",
                       i + io_firstindex, v1, v2, v3);
              }
            } else {
              printf("!! Triangle #%d [%d,%d,%d] has invalid vertices.\n",
                     i + io_firstindex, v1, v2, v3);
            }
          } 
        } // trinum > 0
        continue;
      }
    }
    if (snum == 0) {
      pstr = strstr(line, "Edges");
      if (pstr) {
        // Read the number of vertices.
        pstr = findnextnumber(pstr); // Skip field "Edges".
        if (*pstr == '\0') {
          // Read a non-empty line.
          pstr = fgets(line, 1024, infile);
        }
        snum = atoi(pstr);
        if (snum > 0) {
          printf("Reading %d edges from file %s\n", snum, filename);
          int e1, e2, tag;
          int est_size = snum;
          int log2objperblk = 0;
          while (est_size >>= 1) log2objperblk++;
          tr_segs = new arraypool(sizeof(Triang), log2objperblk);
          for (i = 0; i < snum; i++) {
            while (fgets(line, 1024, infile)) {
              //printf("%s", line);
              pstr = strtok(line, delim);
              if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
                  (pstr[0] != '#')) break;
            }
            //if (feof(infile)) break;
            e1 = e2 = 0;
            if (0) { // no -IN
              //seg->idx = atoi(pstr);
              pstr = strtok(NULL, delim);
            }
            e1 = atoi(pstr);
            pstr = strtok(NULL, delim);
            e2 = atoi(pstr);
            pstr = strtok(NULL, delim);
            if (pstr != NULL) {
              tag = atoi(pstr);
            } else {
              tag = 0;
            }
            if (tag != 0) {
              // A segment must have non-zero tag.
              if ((e1 != e2) &&
                  (e1 >= io_firstindex) && (e1 < ct_in_vrts + io_firstindex) &&
                  (e2 >= io_firstindex) && (e2 < ct_in_vrts + io_firstindex)) {
                Triang *seg = (Triang *) tr_segs->alloc();
                seg->init();
                seg->vrt[0] = &(in_vrts[e1 - io_firstindex]);
                seg->vrt[1] = &(in_vrts[e2 - io_firstindex]);
                seg->tag = tag;
              } else {
                printf("Segment %d has invalid vertices.\n", i + io_firstindex);
              }
            }
          }
          printf("Read %d segments from file %s\n", tr_segs->objects, filename);
        } // if (snum > 0)
        continue;
      }
    }
  } // while

  fclose(infile);

  if (pnum == 0) {
    // No point is found in this file.
    assert((trinum == 0) && (snum == 0)); // These numbers are read after pnum.
    return 0;
  }

  // Read point weights.
  read_weights();
  // Read point metrics (.mtr)
  read_metric();
  // Read triangle areas (.area)
  read_area();

  read_sol(); // .sol file
  read_grd(); // .grd file

  return 1;
}

//==============================================================================
// Read a .sol file (INRIA's .sol format).
// It should be called after read_nodes(), or read_mesh()

int Triangulation::read_sol()
{
  char filename[256];
  FILE *infile = NULL;
  char line[1024], *pstr;
  char delim[] = " ,\t";

  // Read point metrics (nodal mesh size)
  strcpy(filename, io_infilename);
  strcat(filename, ".sol");
  infile = fopen(filename, "r");
  if (infile == NULL) {
    return 0;
  }

  // Searching keyword ``SolAtVertices", and reading number of vertices.
  int pnum = 0;

  while (fgets(line, 1024, infile)) {
    if (pnum == 0) {
      pstr = strstr(line, "SolAtVertices");
      if (pstr) {
        // Read the number of vertices.
        pstr = findnextnumber(pstr); // Skip field "Vertices".
        if (*pstr == '\0') {
          // Read a non-empty line.
          pstr = fgets(line, 1024, infile);
        }
        pnum = atoi(pstr);
        if (pnum > 0) {
          break;
        }
      }
    } // if (pnum == 0)
  }

  if (pnum != ct_in_vrts) {
    printf("Wrong number %d (should be %d) of point values\n", pnum, ct_in_vrts);
    fclose(infile);
    return 0;
  }

  printf("Reading %d vertex solutions from file %s\n", pnum, filename);

  int i, mtrsize = 1; // default

  // Read number of solutions per vertex.
  while (fgets(line, 1024, infile)) { // Skip empty lines.
    //printf("%s", line);
    pstr = strtok(line, delim);
    if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
        (pstr[0] != '#')) break;
  }
  if (pstr != NULL) {
    //mtrnum = atoi(pstr); // skip 1
    // Read the number of metrics (1 or 3).
    pstr = strtok(NULL, delim);
    if (pstr != NULL) {
      mtrsize = atoi(pstr); // it is either 1 or 3.
    }
  }

  while (fgets(line, 1024, infile)) { // Skip empty lines.
    pstr = strtok(line, delim);
    if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
        (pstr[0] != '#')) break;
  }

  // Read the vertex soltions.
  for (i = 0; i < pnum; i++) {
    in_vrts[i].fval = atof(line);
    fgets(line, 1024, infile);
  }

  fclose(infile);

  io_with_sol = 1;

  return 1;
}

//==============================================================================
// Read a .grd file

int Triangulation::read_grd()
{
  char filename[256];
  FILE *infile = NULL;
  //char line[1024], *pstr;
  //char delim[] = " ,\t";

  // Read point metrics (nodal mesh size)
  strcpy(filename, io_infilename);
  strcat(filename, ".grd");
  infile = fopen(filename, "r");
  if (infile == NULL) {
    return 0;
  }

  int pnum = ct_in_vrts;
  
  printf("Reading %d vertex gradients from file %s\n", pnum, filename);

  for (int i = 0; i < pnum; i++) {
    Vertex *v = &(in_vrts[i]);
    fscanf(infile, "%lf %lf %lf\n", &(v->fval), &(v->grd[0]), &(v->grd[1]));
  }

  io_with_sol = 1;
  io_with_grd = 1;

  fclose(infile);
  return 1;
}

//==============================================================================

int Triangulation::read_poly_mesh()
{
  // Read mesh file(s)
  char filename[256];
  FILE *infile = NULL;
  char line[1024]; // *pstr;
  //char delim[] = " ,\t";
  int i;
  
  // First try to read a *.mesh file.
  strcpy(filename, io_infilename);
  strcat(filename, ".mesh");
  infile = fopen(filename, "r");
  if (infile == NULL) {
    // Try to read a *.voro file.
    strcpy(filename, io_infilename);
    strcat(filename, ".voro");
    infile = fopen(filename, "r");
    if (infile == NULL) {
      return 0;
    }
  }

  // Read the first line, get nv, ne, nc;
  int nv, ne, nc;
  fgets(line, 1024, infile);
  sscanf(line, "%d %d %d", &nv, &ne, &nc);

  if (nv == 0) {
    fclose(infile);
    return 0;
  }

  ct_in_vrts = nv;
  in_vrts = new Vertex[nv];
  io_firstindex = 1;

  printf("Reading %d points from file %s\n", nv, filename);
  float x, y;  
  int idx = io_firstindex;
  for (i = 0; i < nv; i++) {
    fgets(line, 1024, infile);
    Vertex *vrt = &in_vrts[i];
    vrt->init();
    // Default vertex type is UNUSEDVERTEX (0)
    ct_unused_vrts++;
    sscanf(line, "%f %f", &x, &y);
    vrt->crd[0] = (REAL) x;
    vrt->crd[1] = (REAL) y;
    vrt->crd[2] = (REAL) (x*x + y*y); // height
    //vrt->crd[2] = op_lambda1 * x*x + op_lambda2 * y*y;
    vrt->wei = 0.0;
    vrt->idx = idx;
    idx++;
    // Determine the smallest and largest x, and y coordinates.
    if (i == 0) {
      io_xmin = io_xmax = x;
      io_ymin = io_ymax = y;
    } else {
      io_xmin = (x < io_xmin) ? x : io_xmin;
      io_xmax = (x > io_xmax) ? x : io_xmax;
      io_ymin = (y < io_ymin) ? y : io_ymin;
      io_ymax = (y > io_ymax) ? y : io_ymax;
    }
  }

  double dx = io_xmax - io_xmin;
  double dy = io_ymax - io_ymin;
  io_diagonal2 = dx*dx + dy*dy;
  io_diagonal = sqrt(io_diagonal2);

  // Initialise predicates.
  //void exactinit(int verbose, int noexact, int o3dfilter, int ispfilter,
  //               REAL maxx, REAL maxy, REAL maxz)
  exactinit(0, 0, 0, 0, io_xmax - io_xmin, io_ymax - io_ymax, 0.0);

  printf("Reading %d edges from file %s\n", ne, filename);
  tr_segs = new arraypool(sizeof(Triang), 10);

  int e1, e2, tag;

  for (i = 0; i < ne; i++) {
    fgets(line, 1024, infile);
    sscanf(line, "%d %d %d", &e1, &e2, &tag);
    Triang *seg = (Triang *) tr_segs->alloc();
    seg->init();
    seg->vrt[0] = &(in_vrts[e1 - io_firstindex]);
    seg->vrt[1] = &(in_vrts[e2 - io_firstindex]);
    int mytag = tag; // tag should not be zero.
    if (mytag == 0) mytag = -1;
    seg->tag = mytag;
  }

  fclose(infile);

  return 1;
}

//==============================================================================
// Read the following files as long as they are provided:
//   .node, .edge, .ele, .region, .weight, ...

int Triangulation::read_mesh()
{
  if (io_poly) {
    if (!read_poly()) {
      printf("Fail to read %s.poly (or smesh) file.\n", io_infilename);
      return 0;
    }
    return 1;
  } else if (io_inria_mesh) {
    if (!read_inria_mesh()) {
      if (!read_poly_mesh()) {
        printf("Fail to read %s.mesh file.\n", io_infilename);
        return 0;
      }
    }
    return 1;
  } else if (io_voronoi) {
    if (!read_poly_mesh()) { // the same format as .mesh
      printf("Fail to read %s.voro file.\n", io_infilename);
      return 0;
    }
    return 1;
  } else if (io_point_array) {
    if (!read_point_array()) {
      printf("Fail to read %s.txt file.\n", io_infilename);
      return 0;
    }
    return 1;
  }

  // Read mesh file(s)
  char filename[256];
  FILE *infile = NULL;
  char line[1024], *pstr;
  char delim[] = " ,\t";
  int i;

  // Try to read .node file.
  if (!read_nodes()) {
    printf("Fail to read %s.node file.\n", io_infilename);
    return 0;
  }

  // Try to read a .edge file (if it exists).
  strcpy(filename, io_infilename);
  strcat(filename, ".edge");
  infile = fopen(filename, "r");
  if (infile != NULL) {
    // Read the number of segments.
    int snum = 0, e1, e2, tag;
    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      snum = atoi(pstr);
    }
    if (snum > 0) {
      printf("Reading %d edges from file %s\n", snum, filename);
      int est_size = snum;
      int log2objperblk = 0;
      while (est_size >>= 1) log2objperblk++;
      tr_segs = new arraypool(sizeof(Triang), log2objperblk);
      for (i = 0; i < snum; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        e1 = e2 = 0;
        if (!io_noindices) { // no -IN
          //seg->idx = atoi(pstr);
          pstr = strtok(NULL, delim);
        }
        e1 = atoi(pstr);
        pstr = strtok(NULL, delim);
        e2 = atoi(pstr);
        pstr = strtok(NULL, delim);
        if (pstr != NULL) {
          tag = atoi(pstr);
        } else {
          tag = 0;
        }
        if (tag != 0) {
          // A segment must have non-zero tag.
          if ((e1 != e2) &&
              (e1 >= io_firstindex) && (e1 < ct_in_vrts + io_firstindex) &&
              (e2 >= io_firstindex) && (e2 < ct_in_vrts + io_firstindex)) {
            Triang *seg = (Triang *) tr_segs->alloc();
            seg->init();
            seg->vrt[0] = &(in_vrts[e1 - io_firstindex]);
            seg->vrt[1] = &(in_vrts[e2 - io_firstindex]);
            seg->tag = tag;
          } else {
            printf("Segment %d has invalid vertices.\n", i + io_firstindex);
          }
        }
      }
      printf("Read %d segments from file %s\n", tr_segs->objects, filename);
    } // snum > 0
    fclose(infile);
  } // Read segments.

  // Try to read a .ele file (if it exists).
  strcpy(filename, io_infilename);
  strcat(filename, ".ele");
  infile = fopen(filename, "r");
  if (infile != NULL) {
    // Read the number of triangles.
    int trinum = 0, v1, v2, v3, tag;
    Vertex *p1, *p2, *p3;
    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      trinum = atoi(pstr);
    }
    if (trinum > 0) {
      printf("Reading %d triangle from file %s\n", trinum, filename);
      // remember the input number of triangles.
      assert(tr_tris == NULL);
      ct_in_tris = trinum;
      int log2objperblk = 0;
      while (trinum >>= 1) log2objperblk++;
      tr_tris = new arraypool(sizeof(Triang), log2objperblk);
      for (i = 0; i < ct_in_tris; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        v1 = v2 = v3 = 0;
        if (!io_noindices) { // no -IN
          //seg->idx = atoi(pstr);
          pstr = strtok(NULL, delim);
        }
        v1 = atoi(pstr);
        pstr = strtok(NULL, delim);
        v2 = atoi(pstr);
        pstr = strtok(NULL, delim);
        v3 = atoi(pstr);
        pstr = strtok(NULL, delim);
        if (pstr != NULL) {
          tag = atoi(pstr);
        } else {
          tag = 0;
        }
        if ((v1 >= io_firstindex) && (v1 < ct_in_vrts + io_firstindex) &&
            (v2 >= io_firstindex) && (v2 < ct_in_vrts + io_firstindex) &&
            (v3 >= io_firstindex) && (v3 < ct_in_vrts + io_firstindex)) {
          p1 = &(in_vrts[v1 - io_firstindex]);
          p2 = &(in_vrts[v2 - io_firstindex]);
          p3 = &(in_vrts[v3 - io_firstindex]);
          // Make sure all tetrahedra are CCW oriented.
          REAL ori = Orient2d(p1, p2, p3);
          if (ori < 0) {
            // Swap the first two vertices.
            Vertex *swap = p1;
            p1 = p2;
            p2 = swap;
          }
          if (ori != 0) {
            Triang *tri = (Triang *) tr_tris->alloc();
            tri->init();
            tri->vrt[0] = p1;
            tri->vrt[1] = p2;
            tri->vrt[2] = p3;
            tri->tag = tag;
          } else {
            printf("!! Triangle #%d [%d,%d,%d] is degenerated.\n",
                   i + io_firstindex, v1, v2, v3);
          }
        } else {
          printf("!! Triangle #%d [%d,%d,%d] has invalid vertices.\n",
                 i + io_firstindex, v1, v2, v3);
        }
      }
    } // if (trinum > 0)
    fclose(infile);
  } // Read triangles.

  // Try to read a .region file (if it exists).
  strcpy(filename, io_infilename);
  strcat(filename, ".region");
  infile = fopen(filename, "r");
  if (infile != NULL) {
    // Read the number of regions (holes).
    int rnum = 0;
    while (fgets(line, 1024, infile)) {
      //printf("%s", line);
      pstr = strtok(line, delim);
      if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
          (pstr[0] != '#')) break;
    }
    if (pstr != NULL) {
      rnum = atoi(pstr);
    }
    if (rnum > 0) {
      printf("Reading %d regions (and holes) from file %s\n", rnum, filename);
      ct_in_sdms = rnum;
      in_sdms = new Vertex[ct_in_sdms];
      for (i = 0; i < rnum; i++) {
        while (fgets(line, 1024, infile)) {
          //printf("%s", line);
          pstr = strtok(line, delim);
          if ((pstr != NULL) && (pstr[0] != '\r') && (pstr[0] != '\n') &&
              (pstr[0] != '#')) break;
        }
        //if (feof(infile)) break;
        Vertex *vrt = &(in_sdms[i]);
        vrt->idx = atoi(pstr);
        pstr = strtok(NULL, delim);
        vrt->crd[0] = atof(pstr);
        pstr = strtok(NULL, delim);
        vrt->crd[1] = atof(pstr);
        pstr = strtok(NULL, delim);
        if (pstr != NULL) {
          vrt->tag = atoi(pstr);
          pstr = strtok(NULL, delim);
          if (pstr != NULL) {
            vrt->val = atof(pstr); // region maxarea.
          }
        } else {
          vrt->tag = 0; // Hole
        }
      } // i
    } // rnum > 0
    fclose(infile);
  } // Read .region file.

  // Read point weights.
  read_weights();
  // Read point metrics (.mtr)
  read_metric();
  // Read triangle areas (.area)
  read_area();

  return 1;
}

//==============================================================================

int Triangulation::remove_exteriors()
{
  int i;

  // Loop through all triangles, deleted exterior and hull triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang *tri = (Triang *) tr_tris->get(i);
    if (tri->is_deleted()) continue;
    if (tri->is_hulltri() || tri->is_exterior()) {
      tri->set_deleted();
      tr_tris->dealloc(tri);
    }
  }
  ct_exteriors = 0;

  // Recreate the hull triangles.
  int log2objperblk = 1;
  while (ct_hullsize >>= 1) log2objperblk++;
  arraypool *hulltris = new arraypool(sizeof(TriEdge), log2objperblk);
  TriEdge E, N;

  for (i = 0; i < tr_tris->used_items; i++) {
    E.tri = (Triang *) tr_tris->get(i);
    if (E.tri->is_deleted()) continue;
    for (E.ver = 0; E.ver < 3; E.ver++) {
      if (E.esym().tri->is_deleted()) {
        * (TriEdge *) hulltris->alloc() = E;
      }
    }
  }

  ct_hullsize = hulltris->objects; // The new hullsize.

  // Create the hull triangles.
  for (i = 0; i < hulltris->objects; i++) {
    TriEdge *parytri = (TriEdge *) hulltris->get(i);
    E = *parytri;
    N.tri = (Triang *) tr_tris->alloc();
    N.tri->init();
    N.set_vertices(E.dest(), E.org(), tr_infvrt);
    N.tri->set_hullflag();
    N.connect(E);
    if (E.is_segment()) N.set_segment();
    // Update the vertex-to-tri map.
    E.org()->adj  = E;
    E.dest()->adj = N;
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

  // Exterior vertices are removed as well.
  for (i = 0; i < ct_in_vrts; i++) {
    if (in_vrts[i].typ != UNUSEDVERTEX) {
      if (in_vrts[i].adj.tri->is_deleted() ||
          (in_vrts[i].adj.org() != &(in_vrts[i]))) {
        in_vrts[i].typ = UNUSEDVERTEX;
        ct_unused_vrts++;
      }
    }
  }
  if (tr_steiners) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *v = (Vertex *) tr_steiners->get(i);
      if (v->is_deleted()) continue;
      if ((v->adj.tri->is_deleted()) || (v->adj.org() != v)) {
        v->set_deleted();
        tr_steiners->dealloc(v);
      }
    }
  }

  delete hulltris;

  // The mesh is non-convex (for point location).
  tr_nonconvex = true;

  return 1;
}

//==============================================================================
// if Steiner_only > 0, only save Steiner points (used by Detri2Gui)
// [Comment: 2018-06-28] Do not save weights or heights.
//   - Use save_weight() to save weights into a .weight file.
//   - Use save_smesh() to save heights (z-coordinate) into a .smesh file.

void Triangulation::save_nodes(int Steiner_only)
{
  int i, idx;
  char filename[256];
  strcpy(filename, io_outfilename);
  strcat(filename, ".node");
  FILE *outfile = fopen(filename, "w");

  int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
  if (Steiner_only > 0) {
    nv -= ct_in_vrts;
  } else {
    nv -= ct_unused_vrts;
  }
  printf("Writing %d nodes to file %s.\n", nv, filename);
  fprintf(outfile, "%d 2 0 0\n", nv);
  
  idx = io_firstindex;
  if (!(Steiner_only > 0)) {
    for (i = 0; i < ct_in_vrts; i++) {
      //if (io_no_unused) { // -IJ
        if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      //}
      double x = in_vrts[i].crd[0];
      double y = in_vrts[i].crd[1];
      fprintf(outfile, "%d %g %g\n", idx, x, y);
      in_vrts[i].idx = idx;
      idx++;
    }
  }
  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      double x = vrt->crd[0];
      double y = vrt->crd[1];
      //double h = vrt->crd[2];
      fprintf(outfile, "%d %g %g\n", idx, x, y);
      vrt->idx = idx;
      idx++;
    }
  }
  fclose(outfile);
}

//==============================================================================

void Triangulation::save_weights(int Steiner_only)
{
  int i; //, idx;
  char wfilename[256];
  strcpy(wfilename, io_outfilename);
  strcat(wfilename, ".weight");
  FILE *woutfile = fopen(wfilename, "w");

  int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
  if (Steiner_only > 0) {
    nv -= ct_in_vrts;
  } else {
    nv -= ct_unused_vrts;
  }
  printf("Writing %d weights to file %s.\n", nv, wfilename);
  fprintf(woutfile, "%d\n", nv);
  
  int idx = io_firstindex;
  if (!(Steiner_only > 0)) {
    for (i = 0; i < ct_in_vrts; i++) {
      //if (io_no_unused) { // -IJ
        if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      //}
      fprintf(woutfile, "%d %g\n", idx, in_vrts[i].wei);
      idx++;
    }
  }
  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      fprintf(woutfile, "%d %g\n", idx, vrt->wei);
      idx++;
    }
  }
  fclose(woutfile);
}

//==============================================================================

void Triangulation::save_metric(int Steiner_only)
{
  int i; //, idx;
  char wfilename[256];
  strcpy(wfilename, io_outfilename);
  strcat(wfilename, ".mtr");
  FILE *woutfile = fopen(wfilename, "w");

  int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
  if (Steiner_only > 0) {
    nv -= ct_in_vrts;
  } else {
    nv -= ct_unused_vrts;
  }
  int mtrsize = 1;
  if (op_metric == METRIC_Riemannian) {
    mtrsize = 3;
  } else if (op_metric == METRIC_HDE) {
    mtrsize = 4;  
  }
  printf("Writing %d weights to file %s.\n", nv, wfilename);
  fprintf(woutfile, "%d  %d\n", nv, mtrsize);

  int idx = io_firstindex;
  if (!(Steiner_only > 0)) {
    for (i = 0; i < ct_in_vrts; i++) {
      //if (io_no_unused) { // -IJ
        if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      //}
      if (mtrsize == 1) {
        fprintf(woutfile, "%d %g\n", idx, in_vrts[i].val);
      } else if (mtrsize == 3) {
        fprintf(woutfile, "%d %g %g %g\n", idx, 
                in_vrts[i].mtr[0], in_vrts[i].mtr[1], in_vrts[i].mtr[2]);    
      } else if (mtrsize == 4) {
        fprintf(woutfile, "%d %g %g %g %g\n", idx, in_vrts[i].val,
                in_vrts[i].mtr[0], in_vrts[i].mtr[1], in_vrts[i].mtr[2]); 
      }
      idx++;
    }
  }
  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      if (mtrsize == 1) {
        fprintf(woutfile, "%d %g\n", idx, vrt->val);
      } else if (mtrsize == 3) {
        fprintf(woutfile, "%d %g %g %g\n", idx, 
                vrt->mtr[0], vrt->mtr[1], vrt->mtr[2]);    
      } else if (mtrsize == 4) {
        fprintf(woutfile, "%d %g %g %g %g\n", idx, vrt->val,
                vrt->mtr[0], vrt->mtr[1], vrt->mtr[2]); 
      }
      idx++;
    }
  }
  fclose(woutfile);
}

//==============================================================================
// This function also output exterior triangles.
// If you don't want them, call remove_exterior() first.

void Triangulation::save_triangulation()
{
  int i, idx;
  char filename[256];

  save_nodes(0);

  strcpy(filename, io_outfilename);
  strcat(filename, ".ele");
  FILE *outfile = fopen(filename, "w");
  int nt = tr_tris->objects - ct_hullsize; // - ct_exteriors;
  printf("Writing %d triangles to file %s.\n", nt, filename);
  fprintf(outfile, "%d 3 1\n", nt);
  idx = io_firstindex;
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    //if (tri->is_deleted() || tri->is_hulltri() || tri->is_exterior()) continue;
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    fprintf(outfile, "%d  %d %d %d  %d\n", idx, tri->vrt[0]->idx,
            tri->vrt[1]->idx, tri->vrt[2]->idx, tri->tag);
    tri->idx = idx;
    idx++;
  }
  fclose(outfile);
}

//==============================================================================

void Triangulation::save_edges()
{
  int i, idx;
  char filename[256];
  FILE *outfile = NULL;
  strcpy(filename, io_outfilename);
  strcat(filename, ".edge");
  outfile = fopen(filename, "w");

  int ne = 0;
  if (io_outedges) { // -Ie
    ne = (3 * (tr_tris->objects - ct_hullsize) + ct_hullsize) / 2;
    printf("Writing %d edges to file %s.\n", ne, filename);
  } else {
    ne = ct_segments;
    printf("Writing %d segments to file %s.\n", ne, filename);
  }

  fprintf(outfile, "%d 1\n", ne);

  TriEdge E, S;

  idx = io_firstindex;
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    if (tri->is_deleted()) continue;
    E.tri = tri;
    if (!E.tri->is_hulltri()) {
      for (E.ver = 0; E.ver < 3; E.ver++) {
        if (E.esym().tri->is_hulltri() || !E.esym().tri->is_infected()) {
          // A segment will always be output.
          if (E.is_segment()) {
            S.tri = E.get_segment();
            if (S.tri->tag == 0) S.tri->tag = -1;
            fprintf(outfile, "%d  %d %d  %d\n", idx, E.org()->idx,
                    E.dest()->idx, S.tri->tag);
            idx++;
          } else {
            if (io_outedges) { // -Ie
              //if (!E.tri->is_hulltri()) {
                // Output an interior edge.
                fprintf(outfile, "%d  %d %d  0\n", idx, E.org()->idx,
                        E.dest()->idx);
                idx++;
              //}
            }
          }
        }
      }
      E.tri->set_infect();
    }
  }
  
  // Uninfect all triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
	Triang* tri = (Triang *) tr_tris->get(i);
	if (tri->is_deleted()) continue;
	if (!tri->is_hulltri()) {
      tri->clear_infect();
	} // if (!tri->is_hulltri()) {
  }

  fclose(outfile);
}

//==============================================================================

void Triangulation::save_poly(int Steiner_only)
{
  char filename[256];
  strcpy(filename, io_outfilename);
  strcat(filename, ".poly");
  FILE *outfile = fopen(filename, "w");

  int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
  if (Steiner_only > 0) {
    nv -= ct_in_vrts;
  } else {
    nv -= ct_unused_vrts;
  }
  int nseg = (tr_segs != NULL ? tr_segs->objects : 0);
  printf("Writing %d vertices, %d segments, %d subdomains to file %s.\n",
         nv, nseg, ct_in_sdms, filename);

  fprintf(outfile, "%d 2 0 0\n", nv);
  int i, idx=io_firstindex;
  if (!(Steiner_only > 0)) {
    for (i = 0; i < ct_in_vrts; i++) {
      if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      fprintf(outfile, "%d %g %g\n", idx, in_vrts[i].crd[0], in_vrts[i].crd[1]);
      in_vrts[i].idx = idx;
      idx++;
    }
  }
  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      fprintf(outfile, "%d %g %g\n", idx, vrt->crd[0], vrt->crd[1]);
      vrt->idx = idx;
      idx++;
    }
  }
  
  fprintf(outfile, "%d 1\n", nseg);
  
  if (tr_segs != 0) {
    idx = io_firstindex;
    for (i = 0; i < tr_segs->used_items; i++) {
      Triang *seg = (Triang *) tr_segs->get(i);
      if (seg->is_deleted()) continue;
      fprintf(outfile, "%d  %d %d %d\n", idx, seg->vrt[0]->idx, seg->vrt[1]->idx, seg->tag);
      idx++;
    }
  }

  fprintf(outfile, "%d\n", ct_in_sdms);

  idx = io_firstindex;
  for (i = 0; i < ct_in_sdms; i++) {
    fprintf(outfile, "%d  %g %g %d %g\n", idx, in_sdms[i].crd[0], in_sdms[i].crd[1],
            in_sdms[i].tag, in_sdms[i].val);
    idx++;
  }

  fclose(outfile);
}

//==============================================================================
// This function also output exterior triangles.
// If you don't want them, call remove_exterior() first.
// [Comment: 2018-06-28] This function save 3d coordinates (with heights as z-coordinates).

void Triangulation::save_smesh()
{
  char filename[256];
  strcpy(filename, io_outfilename);
  strcat(filename, ".smesh");
  FILE *outfile = fopen(filename, "w");

  int ntri = (int) tr_tris->objects - ct_hullsize; // - ct_exteriors;
  printf("Writing %d triangles to file %s.\n", ntri, filename);
  int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
  //nv -= ct_unused_vrts;

  fprintf(outfile, "%d 3 0 0\n", nv);
  int i, idx=io_firstindex;
  for (i = 0; i < ct_in_vrts; i++) {
    //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
    REAL val = in_vrts[i].crd[2]; // default
    if (op_metric == METRIC_Euclidean) {
      val = in_vrts[i].val;
    } else if (op_metric == METRIC_HDE) {
      val = in_vrts[i].mtr[0];
    }
    fprintf(outfile, "%d %g %g %g\n", idx, in_vrts[i].crd[0], in_vrts[i].crd[1], val);
    in_vrts[i].idx = idx;
    idx++;
  }
  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      REAL val = vrt->crd[2]; // default
      if (op_metric == METRIC_Euclidean) {
        val = vrt->val;
      } else if (op_metric == METRIC_HDE) {
        val = vrt->mtr[0];
      }
      fprintf(outfile, "%d %g %g %g\n", idx, vrt->crd[0], vrt->crd[1], val);
      vrt->idx = idx;
      idx++;
    }
  }

  fprintf(outfile, "%d 0\n", ntri);
  idx = io_firstindex;
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    //if (tri->is_deleted() || tri->is_hulltri() || tri->is_exterior()) continue;
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    fprintf(outfile, "3 %d %d %d  %d\n",
            tri->vrt[0]->idx, tri->vrt[1]->idx, tri->vrt[2]->idx, tri->tag);
    tri->idx = idx;
    idx++;
  }

  fprintf(outfile, "0\n");
  fclose(outfile);
}

//==============================================================================
// This function also output exterior triangles.
// If you don't want them, call remove_exterior() first.

void Triangulation::save_inria_mesh()
{
  int i, idx;
  char filename[256];
  strcpy(filename, io_outfilename);
  strcat(filename, ".mesh");
  FILE *outfile = fopen(filename, "w");

  int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
  //if (io_no_unused)
  nv -= ct_unused_vrts;
  printf("Writing %d nodes to file %s.\n", nv, filename);

  fprintf(outfile, "MeshVersionFormatted 1\n");
  fprintf(outfile, "\n");
  fprintf(outfile, "Dimension\n");
  fprintf(outfile, "2\n");
  fprintf(outfile, "\n");

  fprintf(outfile, "Vertices\n");
  fprintf(outfile, "%d\n", nv);

  idx = io_firstindex;
  for (i = 0; i < ct_in_vrts; i++) {
    //if (io_no_unused) { // -IJ
      if (in_vrts[i].typ == UNUSEDVERTEX) continue;
    //}
    int tag = 0;
    //if ((in_vrts[i].typ == RIDGEVERTEX) || (in_vrts[i].typ == SEGMENTVERTEX)) {
    if (in_vrts[i].typ == SEGMENTVERTEX) {
      TriEdge seg = in_vrts[i].on_bd;
      if (seg.tri != NULL) tag = seg.tri->tag;
    }
    fprintf(outfile, "%.17g  %.17g  %d\n", in_vrts[i].crd[0], in_vrts[i].crd[1], tag);
    in_vrts[i].idx = idx;
    idx++;
  }
  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      int tag = 0;
      if (vrt->on_bd.tri != NULL) {
        tag = vrt->on_bd.tri->tag;
      }
      fprintf(outfile, "%.17g  %.17g  %d\n", vrt->crd[0], vrt->crd[1], tag);
      vrt->idx = idx;
      idx++;
    }
  }

  int nt = tr_tris->objects - ct_hullsize;//  - ct_exteriors;
  printf("Writing %d triangles to file %s.\n", nt, filename);

  fprintf(outfile, "Triangles\n");
  fprintf(outfile, "%d\n", nt);

  idx = io_firstindex;
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    //if (tri->is_deleted() || tri->is_hulltri() || tri->is_exterior()) continue;
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    fprintf(outfile, "%d %d %d  %d\n", tri->vrt[0]->idx,
            tri->vrt[1]->idx, tri->vrt[2]->idx, tri->tag);
    tri->idx = idx;
    idx++;
  }

  int ne = 0;
  if (io_outedges) { // -Ie
    ne = (3 * (tr_tris->objects - ct_hullsize) + ct_hullsize) / 2;
    printf("Writing %d edges to file %s.\n", ne, filename);
  } else {
    ne = ct_segments;
    printf("Writing %d segments to file %s.\n", ne, filename);
  }

  fprintf(outfile, "Edges\n");
  fprintf(outfile, "%d\n", ne);

  TriEdge E, S;

  idx = io_firstindex;
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    if (tri->is_deleted() || tri->is_hulltri()) continue;
    E.tri = tri;
    for (E.ver = 0; E.ver < 3; E.ver++) {
      if (!E.esym().tri->is_infected()) {
        // A segment will always be output.
        if (E.is_segment()) {
          S.tri = E.get_segment();
          if (S.tri->tag == 0) S.tri->tag = -1;
          fprintf(outfile, "%d %d  %d\n", E.org()->idx, E.dest()->idx, S.tri->tag);
          idx++;
        } else {
          if (io_outedges) { // -Ie
            // Output an interior edge.
            fprintf(outfile, "%d %d  0\n", E.org()->idx, E.dest()->idx);
            idx++;
          }
        }
      }
    }
    E.tri->set_infect();
  }
  assert((idx - io_firstindex) == ne);

  // Uninfect all (non-hull) triangles.
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    if (tri->is_deleted() || E.tri->is_hulltri()) continue;
    E.tri->clear_infect();
  }

  fprintf(outfile, "End\n");

  fclose(outfile);
}

//==============================================================================

void Triangulation::save_to_ucd(int meshidx, int save_val)
{
  char filename[256];
  sprintf(filename, "%s_%d.inp", io_outfilename, meshidx);
  FILE *outfile = fopen(filename, "w");

  int ntri = (int) tr_tris->objects - ct_hullsize - ct_exteriors;
  printf("Writing %d triangles to file %s.\n", ntri, filename);
  int nv = ct_in_vrts + (tr_steiners != NULL ? tr_steiners->objects : 0);
  //nv -= ct_unused_vrts;

  fprintf(outfile, "%d %d %d 0 0\n", nv, ntri, save_val);

  int i, idx=1; // UCD index starts from 1.
  for (i = 0; i < ct_in_vrts; i++) {
    //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
    fprintf(outfile, "%d %g %g 0\n", idx, in_vrts[i].crd[0], in_vrts[i].crd[1]);
    in_vrts[i].idx = idx;
    idx++;
  }
  if (tr_steiners != NULL) {
    for (i = 0; i < tr_steiners->used_items; i++) {
      Vertex *vrt = (Vertex *) tr_steiners->get(i);
      if (vrt->is_deleted()) continue;
      fprintf(outfile, "%d %g %g 0\n", idx, vrt->crd[0], vrt->crd[1]);
      vrt->idx = idx;
      idx++;
    }
  }

  // UCD assumes vertex index starts from 1.
  //int shift = (io_firstindex == 1 ? 0 : 1);
  idx = 1;
  for (i = 0; i < tr_tris->used_items; i++) {
    Triang* tri = (Triang *) tr_tris->get(i);
    if (!tri->is_deleted()) {
      // ignore a hull triangle.
      if (!tri->is_hulltri() && !tri->is_exterior()) {
        fprintf(outfile, "%d %d tri %d %d %d\n", idx, tri->tag,
                tri->vrt[0]->idx, // + shift,
                tri->vrt[1]->idx, // + shift,
                tri->vrt[2]->idx); // + shift);
        tri->idx = idx;
        idx++;
      }
    }
  }

  if (save_val > 0) {
    // Output metric on nodes.
    fprintf(outfile, "1 1\n");
    fprintf(outfile, "unknown, adim\n");

    idx=1; 
    for (i = 0; i < ct_in_vrts; i++) {
      //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      REAL val = in_vrts[i].val;
      if (fabs(val) < 1.e-15) val = 0.0;
      fprintf(outfile, "%d %g\n", idx, val);
      idx++;
    }
    if (tr_steiners != NULL) {
      for (i = 0; i < tr_steiners->used_items; i++) {
        Vertex *vrt = (Vertex *) tr_steiners->get(i);
        if (vrt->is_deleted()) continue;
        REAL val = vrt->val;
        if (fabs(val) < 1.e-15) val = 0.0;
        fprintf(outfile, "%d %g\n", idx, val);
        idx++;
      }
    }
  }

  fclose(outfile);

  if (io_dump_lift_map) { // -Il
    sprintf(filename, "%s_lift_%d.inp", io_outfilename, meshidx);
    outfile = fopen(filename, "w");

    fprintf(outfile, "%d %d %d 0 0\n", nv, ntri, 0);

    idx=1; // UCD index starts from 1.
    for (i = 0; i < ct_in_vrts; i++) {
      //if (in_vrts[i].typ == UNUSEDVERTEX) continue;
      fprintf(outfile, "%d %g %g %g\n", idx, in_vrts[i].crd[0], in_vrts[i].crd[1], in_vrts[i].crd[2]);
      in_vrts[i].idx = idx;
      idx++;
    }
    if (tr_steiners != NULL) {
      for (i = 0; i < tr_steiners->used_items; i++) {
        Vertex *vrt = (Vertex *) tr_steiners->get(i);
        if (vrt->is_deleted()) continue;
        fprintf(outfile, "%d %g %g %g\n", idx, vrt->crd[0], vrt->crd[1], vrt->crd[2]);
        vrt->idx = idx;
        idx++;
      }
    }

    idx = 1;
    for (i = 0; i < tr_tris->used_items; i++) {
      Triang* tri = (Triang *) tr_tris->get(i);
      if (!tri->is_deleted()) {
        // ignore a hull triangle.
        if (!tri->is_hulltri() && !tri->is_exterior()) {
          fprintf(outfile, "%d %d tri %d %d %d\n", idx, tri->tag,
                  tri->vrt[0]->idx, // + shift,
                  tri->vrt[1]->idx, // + shift,
                  tri->vrt[2]->idx); // + shift);
          tri->idx = idx;
          idx++;
        }
      }
    }

    fclose(outfile);
  }
}
