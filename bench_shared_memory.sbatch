#!/bin/bash
#SBATCH --job-name=bench_shmem
#SBATCH --partition=tornado
#SBATCH --output=bench_shmem_%j.out
#SBATCH --error=bench_shmem_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

module purge
module load compiler/gcc/11

CSV="results_shmem.csv"
echo "Type,Cores,Time" > $CSV

NE=100000
ITER=10000
TOL=1e-8

echo "=== КОМПИЛЯЦИЯ ==="

gcc openmp_realisation/src/main.c openmp_realisation/src/back_openmp.c -o fem_omp -I openmp_realisation/include -fopenmp -lm -O3

gcc pthreads_realisation/src/main.c pthreads_realisation/src/back_pthreads.c pthreads_realisation/src/core.c -o fem_pth -I pthreads_realisation/include -pthread -lm -O3

THREADS="1 2 4 8 16 24 32 40 48"

for T in $THREADS; do
    echo "Running OpenMP with $T threads..."
    export OMP_NUM_THREADS=$T
    # Запускаем и вырезаем время (4-е слово в строке "Solver finished in X s")
    VAL=$(./fem_omp $NE $ITER $TOL | grep "finished in" | awk '{print $4}')
    
    if [[ ! -z "$VAL" ]]; then
        echo "OpenMP,$T,$VAL" >> $CSV
    else
        echo "Error parsing time for OpenMP $T"
    fi
done

for T in $THREADS; do
    echo "Running Pthreads with $T threads..."
    VAL=$(./fem_pth $NE $ITER $TOL $T | grep "finished in" | awk '{print $4}')
    
    if [[ ! -z "$VAL" ]]; then
        echo "Pthreads,$T,$VAL" >> $CSV
    else
        echo "Error parsing time for Pthreads $T"
    fi
done

echo "Shared Memory Benchmark Done."
