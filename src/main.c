#include "fem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv){
    size_t Ne = (argc>1) ? strtoull(argv[1], NULL, 10) : 200; // элементов
    int nthreads = (argc>2) ? atoi(argv[2]) : 4;

    fem_problem_t P; fem_init_poisson_1d(&P, Ne);
    size_t n = P.n_nodes - 2;
    double* b = (double*)malloc(sizeof(double)*n);
    double* x = (double*)malloc(sizeof(double)*n);
    fem_build_rhs(&P, b);

    // выбери один:
    // fem_solve_serial_jacobi(&P, b, x, 20000, 1e-8);
    fem_solve_pthreads_jacobi(&P, b, x, 20000, 1e-8, nthreads);

    printf("n=%zu, h=%.3g, x[0]=%.6f, x[mid]=%.6f, x[last]=%.6f\n",
           n, P.h, x[0], x[n/2], x[n-1]);

    free(x); free(b); fem_free(&P);
    return 0; // t
}