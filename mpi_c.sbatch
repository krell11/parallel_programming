#!/bin/bash
#SBATCH --job-name=check_mpi_c
#SBATCH --partition=tornado
#SBATCH --output=out_mpi_c.txt
#SBATCH --error=err_mpi_c.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1

set -x

module purge
module load mpi/openmpi/4.1.6/gcc/11

echo "Компиляция MPI C..."
mpicc -DFEM_WITH_MPI mpi_realisation/src/main.c mpi_realisation/src/back_mpi.c pthreads_realisation/src/core.c -o fem_mpi -I mpi_realisation/include -lm -O3

echo "Запуск MPI C..."
mpiexec ./fem_mpi 100000 20000 1e-8
