#include "fem.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>   
#include <omp.h>    

int main(int argc, char** argv) {

    size_t Ne    = (argc > 1) ? strtoull(argv[1], NULL, 10) : 200;
    int max_iter = (argc > 2) ? atoi(argv[2]) : 20000;
    double tol   = (argc > 3) ? atof(argv[3]) : 1e-8;

    fem_problem_t P;
    fem_init_poisson_1d(&P, Ne);

    size_t n = (P.n_nodes >= 2) ? (P.n_nodes - 2) : 0;

    printf("Starting OpenMP FEM 1D Poisson solver...\n");
    printf("Nodes: %zu, Max Iter: %d, Tol: %g\n", n, max_iter, tol);

    double* b = (double*)malloc(sizeof(double) * n);
    double* x = (double*)malloc(sizeof(double) * n);
    if (!b || !x) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    fem_build_rhs(&P, b);

    double start_time = omp_get_wtime();
    
    int err = fem_solve_openmp_jacobi(&P, b, x, max_iter, tol);
    
    double end_time = omp_get_wtime();

    if (err != 0) {
        fprintf(stderr, "Solver failed with error code %d\n", err);
    } else {
        printf("Solver finished in %.4f seconds.\n", end_time - start_time);
        
        printf("n=%zu, h=%.6g, x[0]=%.6f, x[mid]=%.6f, x[last]=%.6f\n",
               n, P.h, x[0], x[n/2], x[n-1]);
    }

    free(x);
    free(b);
    fem_free(&P);

    return 0;
}