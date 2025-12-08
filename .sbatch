#!/bin/bash
#SBATCH --job-name=fem_mpi_bench
#SBATCH --partition=tornado        
#SBATCH --output=mpi_out_%j.txt
#SBATCH --error=mpi_err_%j.txt
#SBATCH --time=01:00:00

#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1

module purge
module load python/3.13
module load mpi/openmpi/4.1.6/gcc/11
source .env/bin/activate

CSV_FILE="results_mpi.csv"
echo "Type,Cores,Time" > $CSV_FILE

NE=100000
ITER=20000
TOL=1e-8

PROCS_LIST="1 2 4 8 16 32 48 64 96 112"

echo "=== STARTING MPI BENCHMARK ==="

for P in $PROCS_LIST; do
    echo "Running MPI C with $P processes..."
    T_C=$(mpiexec -n $P ./mpi_realisation/fem_mpi $NE $ITER $TOL | grep "finished in" | awk '{print $4}')
    
    if [ ! -z "$T_C" ]; then
        echo "MPI_C,$P,$T_C" >> $CSV_FILE
    fi

    echo "Running MPI Python with $P processes..."

    T_PY=$(mpiexec -n $P python3 python_realisation/fem_mpi.py $NE $ITER $TOL | grep "finished in" | awk '{print $4}') # Убедитесь, что скрипт печатает время!

    if [ ! -z "$T_PY" ]; then
        echo "MPI_Py,$P,$T_PY" >> $CSV_FILE
    else

        echo "Warning: Python parsing failed for $P"
    fi
done

echo "MPI Done."