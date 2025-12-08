
#define _XOPEN_SOURCE 700
#include "fem.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const fem_problem_t* p; 
    const double* b;        
    double* x;              
    double* x_new;          

    double* r2_accum;

    size_t n;               
    int max_iter;           
    double tol;             
    int nthreads;           

    pthread_barrier_t barrier; 
    volatile int stop;         
} ctx_t;

typedef struct {
    ctx_t* C;
    int tid;           
    size_t i0, i1;     
} task_t;


static void* worker(void* arg) {
    task_t* T = (task_t*)arg;
    ctx_t* C = T->C;

    const double h = C->p->h;
    const double diag = 2.0 / h;
    const double off  = -1.0 / h;

    for (int it = 0; it < C->max_iter; ++it) {
        if (C->stop) break;

        for (size_t i = T->i0; i < T->i1; ++i) {
            double s = C->b[i];
            if (i > 0)         s -= off * C->x[i - 1];
            if (i + 1 < C->n)  s -= off * C->x[i + 1];
            C->x_new[i] = s / diag;
        }

        pthread_barrier_wait(&C->barrier);

        double r2_local = 0.0;
        for (size_t i = T->i0; i < T->i1; ++i) {
            double Ki =
                diag * C->x_new[i]
              + (i > 0        ? off * C->x_new[i - 1] : 0.0)
              + (i + 1 < C->n ? off * C->x_new[i + 1] : 0.0);
            double ri = Ki - C->b[i];
            r2_local += ri * ri;
        }
        C->r2_accum[T->tid] = r2_local;

        pthread_barrier_wait(&C->barrier);

        if (T->tid == 0) {
            double r2_total = 0.0;
            for (int t = 0; t < C->nthreads; ++t) r2_total += C->r2_accum[t];
            C->stop = (sqrt(r2_total) < C->tol);
        }

        pthread_barrier_wait(&C->barrier);
        if (C->stop) break;

        for (size_t i = T->i0; i < T->i1; ++i) C->x[i] = C->x_new[i];

        pthread_barrier_wait(&C->barrier);
    }

    return NULL;
}


int fem_solve_pthreads_jacobi(const fem_problem_t* p, const double* b,
                              double* x, int max_iter, double tol, int nthreads)
{
    const size_t n = (p->n_nodes >= 2) ? (p->n_nodes - 2) : 0;
    if (n == 0) return 0;
    if (nthreads < 1) nthreads = 1;

    double* x_new   = (double*)calloc(n, sizeof(double));
    double* r2buf   = (double*)calloc((size_t)nthreads, sizeof(double));
    pthread_t* th   = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)nthreads);
    task_t* tasks   = (task_t*)malloc(sizeof(task_t) * (size_t)nthreads);
    if (!x_new || !r2buf || !th || !tasks) {
        free(x_new); free(r2buf); free(th); free(tasks);
        return -2;
    }

    memset(x, 0, sizeof(double) * n);

    ctx_t C;
    C.p = p; C.b = b; C.x = x; C.x_new = x_new; C.r2_accum = r2buf;
    C.n = n; C.max_iter = max_iter; C.tol = tol; C.nthreads = nthreads; C.stop = 0;

    pthread_barrier_init(&C.barrier, NULL, (unsigned)nthreads);

    const size_t chunk = (n + (size_t)nthreads - 1) / (size_t)nthreads;

    for (int t = 0; t < nthreads; ++t) {
        size_t i0 = (size_t)t * chunk;
        size_t i1 = i0 + chunk; if (i1 > n) i1 = n;
        tasks[t].C = &C;
        tasks[t].tid = t;
        tasks[t].i0 = i0;
        tasks[t].i1 = i1;
        pthread_create(&th[t], NULL, worker, &tasks[t]);
    }

    for (int t = 0; t < nthreads; ++t) pthread_join(th[t], NULL);

    pthread_barrier_destroy(&C.barrier);
    free(tasks);
    free(th);
    free(r2buf);
    free(x_new);
    return 0;
}
