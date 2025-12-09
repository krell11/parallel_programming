#!/bin/bash
#SBATCH --job-name=bench_mpi
#SBATCH --partition=tornado
#SBATCH --output=bench_mpi_%j.out
#SBATCH --error=bench_mpi_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1

module purge
module load compiler/gcc/11
module load mpi/openmpi/4.1.6/gcc/11
module load python/3.11

source .env/bin/activate

CSV="results_mpi.csv"
echo "Type,Cores,Time" > $CSV

NE=100000
ITER=10000
TOL=1e-8

rm -f fem_mpi

echo "=== КОМПИЛЯЦИЯ MPI C ==="
mpicc -DFEM_WITH_MPI mpi_realisation/src/main.c mpi_realisation/src/back_mpi.c pthreads_realisation/src/core.c -o fem_mpi -I mpi_realisation/include -lm -O3

PROCS="1 2 4 8 16 32 48 64 80 96 112"

for P in $PROCS; do
    echo "Testing MPI C with $P processes..."
    TIME=$(mpiexec -n $P ./fem_mpi $NE $ITER $TOL | grep "finished in" | awk '{print $4}')
    if [[ ! -z "$TIME" ]]; then
        echo "MPI_C,$P,$TIME" >> $CSV
    fi

    echo "Testing MPI Python with $P processes..."
    TIME_PY=$(mpiexec -n $P python3 python_realisation/fem_mpi.py $NE $ITER $TOL | grep "finished in" | awk '{print $4}')
    if [[ ! -z "$TIME_PY" ]]; then
        echo "MPI_Py,$P,$TIME_PY" >> $CSV
    fi
done

echo "MPI Benchmark Done."
