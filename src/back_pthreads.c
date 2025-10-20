// src/back_pthreads.c
// Реализация итераций Якоби для 1D Пуассона с распараллеливанием на pthreads.
// Матрица K трёхдиагональная: diag=2/h, off=-1/h (дискретизация -u'' = 1)
// Неизвестные — только внутренние узлы (n = n_nodes - 2).
#define _XOPEN_SOURCE 700
#include "fem.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// ------------------------- служебные структуры ------------------------------

typedef struct {
    const fem_problem_t* p; // параметры задачи (n_nodes, h)
    const double* b;        // правая часть размером n
    double* x;              // текущее приближение (размер n)
    double* x_new;          // следующее приближение (размер n)

    // аккумулируем локальные суммы ||r||^2 от каждого потока
    // (здесь можно было бы сделать редукцию, но проще — общий буфер)
    double* r2_accum;

    size_t n;               // число неизвестных (n_nodes - 2)
    int max_iter;           // максимум итераций
    double tol;             // порог по ||r||_2
    int nthreads;           // число потоков

    pthread_barrier_t barrier; // барьер на nthreads участников
    volatile int stop;         // флаг «хватит итераций»
} ctx_t;

typedef struct {
    ctx_t* C;
    int tid;           // id потока [0..nthreads-1]
    size_t i0, i1;     // участок индексов [i0, i1) этого потока
} task_t;

// ---------------------------- рабочая функция -------------------------------

static void* worker(void* arg) {
    task_t* T = (task_t*)arg;
    ctx_t* C = T->C;

    const double h = C->p->h;
    const double diag = 2.0 / h;
    const double off  = -1.0 / h;

    for (int it = 0; it < C->max_iter; ++it) {
        if (C->stop) break;

        // 1) локально считаем x_new на своём отрезке из "старого" x
        for (size_t i = T->i0; i < T->i1; ++i) {
            double s = C->b[i];
            if (i > 0)         s -= off * C->x[i - 1];
            if (i + 1 < C->n)  s -= off * C->x[i + 1];
            C->x_new[i] = s / diag;
        }

        // 2) ждём, пока все потоки посчитают x_new
        pthread_barrier_wait(&C->barrier);

        // 3) считаем локальную невязку r = K*x_new - b на своём отрезке
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

        // 4) барьер — гарантируем, что все записали свои r2
        pthread_barrier_wait(&C->barrier);

        // 5) поток 0 суммирует r2 и решает, останавливаемся ли
        if (T->tid == 0) {
            double r2_total = 0.0;
            for (int t = 0; t < C->nthreads; ++t) r2_total += C->r2_accum[t];
            C->stop = (sqrt(r2_total) < C->tol);
        }

        // 6) барьер — чтобы все увидели обновлённый C->stop
        pthread_barrier_wait(&C->barrier);
        if (C->stop) break;

        // 7) обмен x <- x_new на своём отрезке
        for (size_t i = T->i0; i < T->i1; ++i) C->x[i] = C->x_new[i];

        // 8) барьер — перед следующей итерацией все должны завершить копирование
        pthread_barrier_wait(&C->barrier);
    }

    return NULL;
}

// --------------------------- публичная функция ------------------------------

int fem_solve_pthreads_jacobi(const fem_problem_t* p, const double* b,
                              double* x, int max_iter, double tol, int nthreads)
{
    // число внутренних узлов
    const size_t n = (p->n_nodes >= 2) ? (p->n_nodes - 2) : 0;
    if (n == 0) return 0;
    if (nthreads < 1) nthreads = 1;

    // рабочие буферы
    double* x_new   = (double*)calloc(n, sizeof(double));
    double* r2buf   = (double*)calloc((size_t)nthreads, sizeof(double));
    pthread_t* th   = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)nthreads);
    task_t* tasks   = (task_t*)malloc(sizeof(task_t) * (size_t)nthreads);
    if (!x_new || !r2buf || !th || !tasks) {
        free(x_new); free(r2buf); free(th); free(tasks);
        return -2;
    }

    // начальное приближение: нули
    memset(x, 0, sizeof(double) * n);

    // контекст
    ctx_t C;
    C.p = p; C.b = b; C.x = x; C.x_new = x_new; C.r2_accum = r2buf;
    C.n = n; C.max_iter = max_iter; C.tol = tol; C.nthreads = nthreads; C.stop = 0;

    // инициализируем барьер на nthreads участников
    // (важно собирать с -pthread и определить _XOPEN_SOURCE>=600 в флагах компиляции)
    pthread_barrier_init(&C.barrier, NULL, (unsigned)nthreads);

    // равномерно делим диапазон [0, n) по потокам
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
