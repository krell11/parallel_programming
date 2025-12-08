import numpy as np
from mpi4py import MPI
import sys

def fem_solve_mpi_jacobi(n_global, h, b_local, x_local, max_iter, tol, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_n = len(b_local)
    
    diag = 2.0 / h
    off  = -1.0 / h
    
    left_neighbor  = rank - 1 if rank > 0 else MPI.PROC_NULL
    right_neighbor = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    
    x_new = np.zeros_like(x_local)
    
    halo_left_val  = np.array([0.0], dtype=np.float64)
    halo_right_val = np.array([0.0], dtype=np.float64)
    
    for it in range(max_iter):

        send_buf_first = x_local[0:1] if local_n > 0 else np.array([0.0])
        comm.Sendrecv(sendbuf=send_buf_first, dest=left_neighbor,  sendtag=10,
                      recvbuf=halo_right_val, source=right_neighbor, recvtag=10)
        
        send_buf_last = x_local[-1:] if local_n > 0 else np.array([0.0])
        comm.Sendrecv(sendbuf=send_buf_last, dest=right_neighbor,  sendtag=20,
                      recvbuf=halo_left_val,  source=left_neighbor,  recvtag=20)
        
        halo_left = halo_left_val[0]
        halo_right = halo_right_val[0]


        if local_n > 0:
            x_left_shifted  = np.empty(local_n)
            x_right_shifted = np.empty(local_n)
            
            x_left_shifted[1:] = x_local[:-1]
            x_left_shifted[0]  = halo_left
            
            x_right_shifted[:-1] = x_local[1:]
            x_right_shifted[-1]  = halo_right
            
            s = b_local - off * x_left_shifted - off * x_right_shifted
            x_new[:] = s / diag

        send_buf_first = x_new[0:1] if local_n > 0 else np.array([0.0])
        comm.Sendrecv(sendbuf=send_buf_first, dest=left_neighbor,  sendtag=30,
                      recvbuf=halo_right_val, source=right_neighbor, recvtag=30)
        
        send_buf_last = x_new[-1:] if local_n > 0 else np.array([0.0])
        comm.Sendrecv(sendbuf=send_buf_last, dest=right_neighbor,  sendtag=40,
                      recvbuf=halo_left_val,  source=left_neighbor,  recvtag=40)
                      
        halo_left_new = halo_left_val[0]
        halo_right_new = halo_right_val[0]
        
        r2_local = 0.0
        if local_n > 0:
            x_left_shifted[1:] = x_new[:-1]
            x_left_shifted[0]  = halo_left_new
            
            x_right_shifted[:-1] = x_new[1:]
            x_right_shifted[-1]  = halo_right_new
            
            Ki = diag * x_new + off * x_left_shifted + off * x_right_shifted
            ri = Ki - b_local
            r2_local = np.sum(ri**2)
            
        r2_total = comm.allreduce(r2_local, op=MPI.SUM)
        
        if np.sqrt(r2_total) < tol:
            x_local[:] = x_new
            break
            
        x_local[:] = x_new

    return x_local

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    Ne = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    
    h = 1.0 / Ne
    n_global = Ne - 1 
    
    if n_global <= 0:
        if rank == 0: print("Grid too small")
        MPI.Finalize()
        return

    base = n_global // size
    rem = n_global % size
    
    counts = []
    displs = []
    offset = 0
    for r in range(size):
        c = base + (1 if r < rem else 0)
        counts.append(c)
        displs.append(offset)
        offset += c
        
    local_n = counts[rank]
    
    b_local = np.full(local_n, h, dtype=np.float64)
    x_local = np.zeros(local_n, dtype=np.float64)
    
    comm.Barrier() 
    t_start = MPI.Wtime()
    
    fem_solve_mpi_jacobi(n_global, h, b_local, x_local, max_iter, tol, comm)

    comm.Barrier() 
    t_end = MPI.Wtime()

    if rank == 0:
        print(f"Solver finished in {t_end - t_start:.4f} seconds")

    if rank == 0:
        x_global = np.zeros(n_global, dtype=np.float64)
    else:
        x_global = None
        
    recvbuf = [x_global, counts, displs, MPI.DOUBLE] if rank == 0 else None
    
    comm.Gatherv(x_local, recvbuf, root=0)
    
    if rank == 0:
        print(f"n={n_global}, h={h:.6g}, x[0]={x_global[0]:.6f}, "
              f"x[mid]={x_global[n_global//2]:.6f}, x[last]={x_global[-1]:.6f}")

if __name__ == "__main__":
    main()