#include "fem.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void fem_init_poisson_1d(fem_problem_t* p, size_t n_elements) {
    if (n_elements < 2) n_elements = 2;
    p->n_nodes = n_elements + 1; 
    p->h = 1.0 / (double)n_elements;
}
void fem_free(fem_problem_t* p) { (void)p; }

void fem_build_rhs(const fem_problem_t* p, double* b) {
    // неизвестных n = n_nodes-2 (внутренние)
    size_t n = (p->n_nodes >= 2) ? (p->n_nodes - 2) : 0;
    for (size_t i = 0; i < n; ++i) b[i] = p->h; // f - 1, вклад h для каждого внутреннего узла
}

int fem_solve_serial_jacobi(const fem_problem_t* p, const double* b,
                            double* x, int max_iter, double tol) {
    const size_t n = (p->n_nodes >= 2) ? (p->n_nodes - 2) : 0; // число неизвестных
    if (n == 0) return 0;
    const double h = p->h;
    const double diag = 2.0 / h, off = -1.0 / h;

    double* x_new = (double*)calloc(n, sizeof(double));
    if (!x_new) return -2;
    memset(x, 0, sizeof(double) * n);

    for (int it = 0; it < max_iter; ++it) {
        // x_new[i] = (b[i] - off*x[i-1] - off*x[i+1]) / diag
        for (size_t i = 0; i < n; ++i) {
            double s = b[i];
            if (i > 0)     s -= off * x[i - 1];
            if (i + 1 < n) s -= off * x[i + 1];
            x_new[i] = s / diag;
        }
        // невязка r = Kx - b, считаем ||r||_2
        double r2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double Ki = diag * x_new[i]
                      + (i > 0     ? off * x_new[i - 1] : 0.0)
                      + (i + 1 < n ? off * x_new[i + 1] : 0.0);
            double ri = Ki - b[i];
            r2 += ri * ri;
        }
        memcpy(x, x_new, sizeof(double) * n);
        if (sqrt(r2) < tol) break;
    }
    free(x_new);
    return 0;
}
