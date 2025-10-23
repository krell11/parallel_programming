# ---- MPI BUILD ----
MPICC   = mpicc
CFLAGS  = -O2 -std=c11 -Wall -Wextra -Iinclude -I../pthreads_realisation/include
LDFLAGS = -lm

# общий core.c берём из pthreads-проекта
SRC_CORE = ../pthreads_realisation/src/core.c

SRC = $(SRC_CORE) src/back_mpi.c src/main.c
BIN = fem_mpi

all: $(BIN)

$(BIN): $(SRC)
	$(MPICC) $(CFLAGS) -DFEM_WITH_MPI $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(BIN) *.o

.PHONY: all clean
