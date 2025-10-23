#include "fem.h"
#ifdef FEM_WITH_MPI
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t Ne = (argc>1) ? strtoull(argv[1], NULL, 10) : 200;  // число элементов
    int max_iter = (argc>2) ? atoi(argv[2]) : 20000;
    double tol   = (argc>3) ? atof(argv[3]) : 1e-8;

    fem_problem_t P;
    if (rank==0) {
        fem_init_poisson_1d(&P, Ne);
    }
    // рассылаем параметры задачи простым способом
    MPI_Bcast(&P, sizeof(P), MPI_BYTE, 0, MPI_COMM_WORLD);

    const size_t n = (P.n_nodes>=2)?(P.n_nodes-2):0;

    double* b = NULL;
    double* x = NULL;
    if (rank==0) {
        b = (double*)malloc(sizeof(double)*n);
        x = (double*)malloc(sizeof(double)*n);
        fem_build_rhs(&P, b);
    }

    fem_solve_mpi_jacobi(&P, b, x, max_iter, tol, MPI_COMM_WORLD);

    if (rank==0) {
        printf("n=%zu, h=%.6g, x[0]=%.6f, x[mid]=%.6f, x[last]=%.6f\n",
               n, P.h, x[0], x[n/2], x[n-1]);
        free(x);
        free(b);
    }

    MPI_Finalize();
    return 0;
}
#else
int main(){ return 0; }
#endif
