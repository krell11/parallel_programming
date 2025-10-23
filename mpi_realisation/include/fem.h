#ifndef FEM_H
#define FEM_H
#include <stddef.h>

typedef struct {
    size_t n_nodes; // число узлов (включая граничные)
    double h;       // шаг сетки
} fem_problem_t;

void fem_init_poisson_1d(fem_problem_t* p, size_t n_elements);
void fem_free(fem_problem_t* p);

void fem_build_rhs(const fem_problem_t* p, double* b);

int fem_solve_serial_jacobi(const fem_problem_t* p, const double* b, double* x,
                            int max_iter, double tol);

int fem_solve_pthreads_jacobi(const fem_problem_t* p, const double* b, double* x,
                              int max_iter, double tol, int nthreads);


#ifdef FEM_WITH_MPI
#include <mpi.h>

int fem_solve_mpi_jacobi(const fem_problem_t* p,
                         const double* b_global,  
                         double* x_global,        
                         int max_iter, double tol,
                         MPI_Comm comm);
#endif

#endif 
