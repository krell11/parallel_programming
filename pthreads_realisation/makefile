CC=gcc
CFLAGS=-O2 -std=c11 -Wall -Wextra -Wpedantic -Iinclude -D_XOPEN_SOURCE=700
LDFLAGS=-pthread -lm

SRC=src/core.c src/back_pthreads.c src/main.c
BIN=fem_pth

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

run: all
	./$(BIN) 200 4

clean:
	rm -f $(BIN)
