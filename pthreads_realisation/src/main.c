#define _POSIX_C_SOURCE 199309L // Для clock_gettime
#include "fem.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

int main(int argc, char** argv){

    size_t Ne    = (argc > 1) ? strtoull(argv[1], NULL, 10) : 200;
    int max_iter = (argc > 2) ? atoi(argv[2]) : 20000;
    double tol   = (argc > 3) ? atof(argv[3]) : 1e-8;
    int nthreads = (argc > 4) ? atoi(argv[4]) : 4; 

    fem_problem_t P; 
    fem_init_poisson_1d(&P, Ne);
    
    size_t n = P.n_nodes - 2;
    double* b = (double*)malloc(sizeof(double)*n);
    double* x = (double*)malloc(sizeof(double)*n);
    
    if (!b || !x) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    fem_build_rhs(&P, b);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    fem_solve_pthreads_jacobi(&P, b, x, max_iter, tol, nthreads);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Solver finished in %.4f seconds\n", time_spent);

    printf("n=%zu, h=%.3g, x[0]=%.6f, x[mid]=%.6f, x[last]=%.6f\n",
           n, P.h, x[0], x[n/2], x[n-1]);

    free(x); 
    free(b); 
    fem_free(&P);
    return 0; 
}