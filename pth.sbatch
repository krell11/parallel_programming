#!/bin/bash
#SBATCH --job-name=check_pth
#SBATCH --partition=tornado
#SBATCH --output=out_pth.txt
#SBATCH --error=err_pth.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

module purge
module load compiler/gcc/11

echo "Компиляция Pthreads..."
gcc pthreads_realisation/src/main.c pthreads_realisation/src/back_pthreads.c pthreads_realisation/src/core.c -o fem_pth -I pthreads_realisation/include -pthread -lm -O3

echo "Запуск Pthreads..."
./fem_pth 100000 20000 1e-8 48