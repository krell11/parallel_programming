#ifndef FEM_H
#define FEM_H

#include <stddef.h>

typedef struct {
    size_t n_nodes; // число узлов (включая граничные)
    double h;       // шаг сетки
} fem_problem_t;

// инициализация простой 1D задачи Пуассона на [0,1] с u(0)=u(1)=0, f=1
void fem_init_poisson_1d(fem_problem_t* p, size_t n_elements);
void fem_free(fem_problem_t* p);

// формирование правой части для внутренних узлов
void fem_build_rhs(const fem_problem_t* p, double* b);

// последовательный Якоби (для сравнения/тестов)
int fem_solve_serial_jacobi(const fem_problem_t* p, const double* b,
                            double* x, int max_iter, double tol);

// pthreads-версия Якоби
int fem_solve_pthreads_jacobi(const fem_problem_t* p, const double* b,
                              double* x, int max_iter, double tol, int nthreads);

#endif // FEM_H
