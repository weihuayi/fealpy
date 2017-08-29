

#include <p4est.h>
#include <p4est_vtk.h>
#include <p4est_mesh.h>

#include <cmath>
#include <iostream>

using namespace std;

template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}


static int
uniform_refine_fn(p4est_t * p4est, p4est_topidx_t which_tree,
           p4est_quadrant_t * quadrant)

{
    return 1;
}


static int
refine_fn (p4est_t * p4est, p4est_topidx_t which_tree,
           p4est_quadrant_t * quadrant)
{
    double xmin = -1.0;
    double ymin = -1.0;
    double xmax = 1.0;
    double ymax = 1.0;

    double x = 0;
    double y = 0;

    int sum  = 0;
    double phi = 0.0;

    int8_t l = quadrant->level;

    if( l > 8)
        return 0;

    p4est_qcoord_t qx = quadrant->x;
    p4est_qcoord_t qy = quadrant->y;
    double vxy[3] = {0.0, 0.0, 0.0};
    p4est_qcoord_to_vertex(p4est->connectivity, which_tree, qx, qy, vxy);

    double h = 1.0/pow(2, l);
    x = xmin + vxy[0]*(xmax - xmin);
    y = ymin + vxy[1]*(ymax - ymin);
    phi = sqrt(x*x + y*y) - 0.8;
    sum += sign(phi);

    x = xmin + (vxy[0]+h)*(xmax - xmin);
    y = ymin + vxy[1]*(ymax - ymin);
    phi = sqrt(x*x + y*y) - 0.8;
    sum += sign(phi);

    x = xmin + vxy[0]*(xmax - xmin);
    y = ymin + (vxy[1]+h)*(ymax - ymin);
    phi = sqrt(x*x + y*y) - 0.8;
    sum += sign(phi);

    x = xmin + (vxy[0]+h)*(xmax - xmin);
    y = ymin + (vxy[1]+h)*(ymax - ymin);
    phi = sqrt(x*x + y*y) - 0.8;
    sum += sign(phi);

    if(abs(sum) < 4)
        return 1;
    else 
        return 0;
}

int main(int argc, char **argv)
{

    int                 mpiret;
    int                 recursive, partforcoarsen, balance;
    sc_MPI_Comm         mpicomm;
    p4est_t            *p4est;
    p4est_connectivity_t *conn;

    /* Initialize MPI; see sc_mpi.h.
     * If configure --enable-mpi is given these are true MPI calls.
     * Else these are dummy functions that simulate a single-processor run. */
    mpiret = sc_MPI_Init (&argc, &argv);
    SC_CHECK_MPI (mpiret);
    mpicomm = sc_MPI_COMM_WORLD;
    conn = p4est_connectivity_new_unitsquare( );

    /* Create a forest that is not refined; it consists of the root octant. */
    p4est = p4est_new (mpicomm, conn, 0, NULL, NULL);

    for(int i = 0; i < 1; i++)
        p4est_refine(p4est, 0, uniform_refine_fn, NULL);

    for(int i = 0; i < 2; i++)
        p4est_refine(p4est, 0, refine_fn, NULL);



    /* Partition: The quadrants are redistributed for equal element count.  The
     * partition can optionally be modified such that a family of octants, which
     * are possibly ready for coarsening, are never split between processors. */
    partforcoarsen = 0;
    p4est_partition(p4est, partforcoarsen, NULL);

    /* Write the forest to disk for visualization, one file per processor. */
    p4est_vtk_write_file(p4est, NULL, "test");

    /* Destroy the p4est and the connectivity structure. */
    p4est_destroy(p4est);
    p4est_connectivity_destroy(conn);

    /* Verify that allocations internal to p4est and sc do not leak memory.
     * This should be called if sc_init () has been called earlier. */
    sc_finalize();

    /* This is standard MPI programs.  Without --enable-mpi, this is a dummy. */
    mpiret = sc_MPI_Finalize();
    return 0;
}
