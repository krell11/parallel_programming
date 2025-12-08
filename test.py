import subprocess
import re
import csv
import os

NE = 100000           
MAX_ITER = 20000      
TOL = 1e-8            
OUTPUT_FILE = "benchmark_results.csv"

THREADS_LIST = [1, 2, 4, 8, 12, 16, 24, 32, 48]
THREADS_LIST = [1, 2, 4]
MPI_PROCS_LIST = [1, 2, 4, 8, 16, 32, 48, 64, 96, 112]
MPI_PROCS_LIST = [1, 2 ,4]


EXEC_OMP = "./openmp_realisation/fem_omp"
EXEC_PTH = "./pthreads_realisation/fem_pth"
EXEC_MPI_C = "./mpi_realisation/fem_mpi"
EXEC_MPI_PY = "./python_realisation/fem_mpi.py" 

results = []

def parse_time(output_str):
    match = re.search(r"finished in ([\d\.]+) seconds", output_str)
    if match:
        return float(match.group(1))
    return None

def run_command(cmd, env=None):
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
            
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            env=full_env
        )
        if result.returncode != 0:
            print(f"Error running {' '.join(cmd)}: {result.stderr}")
            return None
        return parse_time(result.stdout)
    except Exception as e:
        print(f"Exception: {e}")
        return None

print(f"Starting Benchmarks. Ne={NE}, Iter={MAX_ITER}")


print("\n--- Running OpenMP ---")
for t in THREADS_LIST:
    env = {"OMP_NUM_THREADS": str(t)}
    time = run_command([EXEC_OMP, str(NE), str(MAX_ITER), str(TOL)], env=env)
    if time:
        print(f"OpenMP Threads {t}: {time:.4f} s")
        results.append(["OpenMP", t, time])

print("\n--- Running Pthreads ---")
for t in THREADS_LIST:
    time = run_command([EXEC_PTH, str(NE), str(MAX_ITER), str(TOL), str(t)])
    if time:
        print(f"Pthreads Threads {t}: {time:.4f} s")
        results.append(["Pthreads", t, time])

print("\n--- Running MPI C ---")
for p in MPI_PROCS_LIST:
    cmd = ["mpiexec", "--oversubscribe", "-n", str(p), EXEC_MPI_C, str(NE), str(MAX_ITER), str(TOL)]
    
    time = run_command(cmd)
    if time:
        print(f"MPI C Procs {p}: {time:.4f} s")
        results.append(["MPI_C", p, time])
    else:
        print(f"MPI C Procs {p}: Failed (check compilation or mpiexec)")

print("\n--- Running MPI Python ---")
for p in MPI_PROCS_LIST:
    cmd = ["mpiexec", "--oversubscribe", "-n", str(p), "python3", EXEC_MPI_PY, str(NE), str(MAX_ITER), str(TOL)]
    
    time = run_command(cmd)
    if time:
        print(f"MPI Py Procs {p}: {time:.4f} s")
        results.append(["MPI_Py", p, time])
    else:
         print(f"MPI Py Procs {p}: Failed")

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Type", "Count", "Time"])
    writer.writerows(results)

print(f"\nDone! Results saved to {OUTPUT_FILE}")