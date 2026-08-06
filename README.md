
# spherical_fmm

A high-performance Fast Multipole Method (FMM) library supporting SIMD and CUDA code generation.

## Cloning

This repository uses Git submodules. Clone it with:

```bash
git clone --recurse-submodules git@github.com:dmarce1/spherical_fmm.git
cd spherical_fmm
````

If you already cloned the repository without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

## Building

Create an out-of-source build directory:

```bash
mkdir release
cd release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

or equivalently:

```bash
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release
cmake --build release -j
```

## Build Requirements

* CMake ≥ 3.12
* GCC (or another C++ compiler with C++20 support)
* NVIDIA CUDA Toolkit (if CUDA support is enabled)
* GMP
* MPFR

## Executables

After building, the following executables are produced:

* `test` — General library test program.
* `brille` — Brill wave example and validation test.

They can be run with:

```bash
./release/test
./release/brille
```

## Updating Submodules

To pull the latest changes from all submodules:

```bash
git submodule update --remote --merge
```

After updating a submodule, commit the updated submodule pointer in the parent repository.

