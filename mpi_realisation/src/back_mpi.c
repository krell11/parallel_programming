#define _POSIX_C_SOURCE 200809L
#include "fem.h"
#ifdef FEM_WITH_MPI
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void make_counts_displs(size_t n, int size, int **counts_out, int **displs_out){
    int *counts = (int*)malloc(sizeof(int)*size);
    int *displs = (int*)malloc(sizeof(int)*size);
    size_t base = n / (size_t)size;
    size_t rem  = n % (size_t)size;
    size_t off = 0;
    for(int r=0;r<size;++r){
        size_t c = base + (r < (int)rem ? 1 : 0);
        counts[r] = (int)c;
        displs[r] = (int)off;
        off += c;
    }
    *counts_out = counts;
    *displs_out = displs;
}

int fem_solve_mpi_jacobi(const fem_problem_t* p,
                         const double* b_global,
                         double* x_global,
                         int max_iter, double tol,
                         MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const size_t n = (p->n_nodes >= 2) ? (p->n_nodes - 2) : 0;
    if (n == 0) return 0;

    int *counts = NULL, *displs = NULL;
    make_counts_displs(n, size, &counts, &displs);
    const int local_n = counts[rank];

    double *b_local = (double*)malloc(sizeof(double)* (size_t)local_n);
    double *x_old   = (double*)calloc((size_t)local_n, sizeof(double));
    double *x_new   = (double*)calloc((size_t)local_n, sizeof(double));
    if (!b_local || !x_old || !x_new){
        free(b_local); free(x_old); free(x_new);
        free(counts); free(displs);
        return -2;
    }

    MPI_Scatterv(b_global, counts, displs, MPI_DOUBLE,
                 b_local, local_n, MPI_DOUBLE,
                 0, comm);

    const double h    = p->h;
    const double diag = 2.0 / h;
    const double off  = -1.0 / h;

    const int left  = (rank > 0)       ? rank-1 : MPI_PROC_NULL;
    const int right = (rank+1 < size)  ? rank+1 : MPI_PROC_NULL;

    int stop = 0;
    for (int it = 0; it < max_iter; ++it) {
        if (stop) break;

        double halo_left_old = 0.0, halo_right_old = 0.0;

        if (local_n > 0) {
            MPI_Sendrecv(x_old, 1, MPI_DOUBLE, left,  100,
                         &halo_right_old, 1, MPI_DOUBLE, right, 100,
                         comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(x_old + (local_n-1), 1, MPI_DOUBLE, right, 200,
                         &halo_left_old, 1, MPI_DOUBLE, left,  200,
                         comm, MPI_STATUS_IGNORE);
        } else {
        }

        for (int i = 0; i < local_n; ++i) {
            double s = b_local[i];
            double xl = (i > 0) ? x_old[i-1] : halo_left_old;
            double xr = (i+1 < local_n) ? x_old[i+1] : halo_right_old;
            s -= off * xl;
            s -= off * xr;
            x_new[i] = s / diag;
        }

        double halo_left_new = 0.0, halo_right_new = 0.0;
        if (local_n > 0) {
            MPI_Sendrecv(x_new, 1, MPI_DOUBLE, left,  300,
                         &halo_right_new, 1, MPI_DOUBLE, right, 300,
                         comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(x_new + (local_n-1), 1, MPI_DOUBLE, right, 400,
                         &halo_left_new, 1, MPI_DOUBLE, left,  400,
                         comm, MPI_STATUS_IGNORE);
        }

        double r2_local = 0.0;
        for (int i = 0; i < local_n; ++i) {
            double xl = (i > 0) ? x_new[i-1] : halo_left_new;
            double xr = (i+1 < local_n) ? x_new[i+1] : halo_right_new;
            double Ki = diag * x_new[i] + off * xl + off * xr;
            double ri = Ki - b_local[i];
            r2_local += ri * ri;
        }

        double r2_total = 0.0;
        MPI_Allreduce(&r2_local, &r2_total, 1, MPI_DOUBLE, MPI_SUM, comm);
        stop = (sqrt(r2_total) < tol);

        if (!stop && local_n > 0) {
            memcpy(x_old, x_new, sizeof(double)*(size_t)local_n);
        }
    }

    if (x_global) {
        MPI_Gatherv(x_new, local_n, MPI_DOUBLE,
                    x_global, counts, displs, MPI_DOUBLE,
                    0, comm);
    } else {
        MPI_Gatherv(x_new, local_n, MPI_DOUBLE,
                    NULL, counts, displs, MPI_DOUBLE,
                    0, comm);
    }

    free(b_local);
    free(x_old);
    free(x_new);
    free(counts);
    free(displs);
    return 0;
}
#endif // FEM_WITH_MPI
