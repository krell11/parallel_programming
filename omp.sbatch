#!/bin/bash
#SBATCH --job-name=check_omp
#SBATCH --partition=tornado
#SBATCH --output=out_omp.txt
#SBATCH --error=err_omp.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

module purge
module load compiler/gcc/11

echo "Компиляция OpenMP..."
gcc openmp_realisation/src/main.c openmp_realisation/src/back_openmp.c -o fem_omp -I openmp_realisation/include -fopenmp -lm -O3

echo "Запуск OpenMP..."
export OMP_NUM_THREADS=48
./fem_omp 100000 20000 1e-8