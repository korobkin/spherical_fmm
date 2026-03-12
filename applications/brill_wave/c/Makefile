CC ?= g++
CFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
LDFLAGS ?=
LDLIBS ?= -lm -lstdc++

BIN := conformal_demo
OBJ := src/main.o src/conformal_solver.o

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)

src/%.o: src/%.c include/conformal_solver.hpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(BIN) $(OBJ)
