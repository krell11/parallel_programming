#!/bin/bash
#SBATCH --job-name=check_mpi_py
#SBATCH --partition=tornado
#SBATCH --output=out_mpi_py.txt
#SBATCH --error=err_mpi_py.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1

module purge
module load python/3.13
module load mpi/openmpi/4.1.6/gcc/11

source .env/bin/activate

echo "Запуск MPI Python..."
mpiexec python3 python_realisation/fem_mpi.py 100000 20000 1e-8