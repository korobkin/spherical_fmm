CXX ?= g++
# NOTE: do not use -Ofast/-ffast-math here. It lets the compiler assume no NaNs
# and reorder the reductions, which changes the results relative to the Python
# reference.
#
# Add -ffp-contract=off to make the grid coordinates bit-identical to the
# notebook's: by default GCC contracts `-L + dx*i` into an FMA, which differs
# from numpy by 1 ulp. It does not improve agreement in psi (that floor is set
# by summation order: numpy sums the O(N^2) kernel through BLAS, this code
# through a sequential loop), and FMA is both faster and slightly more
# accurate, so it is left enabled.
CXXFLAGS ?= -O3 -march=native -std=c++20 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
LDFLAGS ?=
LDLIBS ?= -lm

BIN := conformal_demo
SRC := src/main.cpp src/conformal_solver.cpp
OBJ := $(SRC:.cpp=.o)
HDR := include/conformal_solver.hpp

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)

src/%.o: src/%.cpp $(HDR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(BIN) $(OBJ)
