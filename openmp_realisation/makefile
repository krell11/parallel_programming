CC = gcc
CFLAGS = -O3 -Wall -fopenmp -Iinclude
LDFLAGS = -lm -fopenmp

SRC = src/main.c src/back_openmp.c

TARGET = fem_omp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)