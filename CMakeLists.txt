
cmake_minimum_required(VERSION 3.12.4)

project(
    sfmm
    LANGUAGES
        C
        CXX
        CUDA
)

# Additional project-specific CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# CUDA is already enabled as a project language above.
# CUDAToolkit provides imported targets and toolkit metadata.
find_package(CUDAToolkit REQUIRED)


# -----------------------------------------------------------------------------
# SFMM configuration
# -----------------------------------------------------------------------------

# These are strings rather than BOOL options because their numeric values are
# passed directly to the code generator as preprocessor definitions.
set(SFMM_USE_DOUBLE "1" CACHE STRING "Enable double-precision generation")
set(SFMM_USE_FLOAT  "1" CACHE STRING "Enable single-precision generation")
set(SFMM_USE_SIMD   "1" CACHE STRING "Enable SIMD generation")
set(SFMM_USE_CUDA   "1" CACHE STRING "Enable CUDA generation")

set(SFMM_PMIN "3" CACHE STRING "Minimum expansion order")
set(SFMM_PMAX "3" CACHE STRING "Maximum expansion order")

set(
    SFMM_DOUBLE_SIMD_WIDTH
    "4"
    CACHE STRING
    "Width of double-precision SIMD vectors, except for M2M"
)

set(
    SFMM_FLOAT_SIMD_WIDTH
    "8"
    CACHE STRING
    "Width of single-precision SIMD vectors, except for M2M"
)

message(STATUS "SFMM minimum expansion order: ${SFMM_PMIN}")
message(STATUS "SFMM maximum expansion order: ${SFMM_PMAX}")
message(STATUS "SFMM double SIMD width:      ${SFMM_DOUBLE_SIMD_WIDTH}")
message(STATUS "SFMM float SIMD width:       ${SFMM_FLOAT_SIMD_WIDTH}")


# -----------------------------------------------------------------------------
# SIMD dependency
# -----------------------------------------------------------------------------

add_subdirectory(external/simd)

# Using the namespaced target causes configuration to fail immediately if the
# SIMD dependency is not set up correctly, rather than producing a later
# linker error involving an accidental '-lsimd'.
if(NOT TARGET simd::simd)
    message(
        FATAL_ERROR
        "The SIMD submodule did not define the target simd::simd"
    )
endif()


# -----------------------------------------------------------------------------
# SFMM code generation
# -----------------------------------------------------------------------------

set(SFMM_CODEGEN_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/sfmm_codegen")
set(SFMM_GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_code")
set(SFMM_GENERATED_INCLUDE_DIRECTORY
    "${SFMM_GENERATED_DIRECTORY}/include"
)

message(STATUS "Compiling SFMM code generator")

execute_process(
    COMMAND
        "${CMAKE_CXX_COMPILER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp"
        -march=native
        "-DUSE_CUDA_FLAG=${SFMM_USE_CUDA}"
        "-DUSE_FLOAT_FLAG=${SFMM_USE_FLOAT}"
        "-DUSE_SIMD_FLAG=${SFMM_USE_SIMD}"
        "-DUSE_DOUBLE_FLAG=${SFMM_USE_DOUBLE}"
        "-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
        "-DPMIN=${SFMM_PMIN}"
        "-DPMAX=${SFMM_PMAX}"
        "-DSIMD_FLOAT_WIDTH=${SFMM_FLOAT_SIMD_WIDTH}"
        "-DSIMD_DOUBLE_WIDTH=${SFMM_DOUBLE_SIMD_WIDTH}"
        -o
        "${SFMM_CODEGEN_EXECUTABLE}"
    WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE
        SFMM_CODEGEN_COMPILE_RESULT
    OUTPUT_VARIABLE
        SFMM_CODEGEN_COMPILE_OUTPUT
    ERROR_VARIABLE
        SFMM_CODEGEN_COMPILE_ERROR
)

if(NOT SFMM_CODEGEN_COMPILE_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "Failed to compile the SFMM code generator.\n"
        "Exit result: ${SFMM_CODEGEN_COMPILE_RESULT}\n"
        "stdout:\n${SFMM_CODEGEN_COMPILE_OUTPUT}\n"
        "stderr:\n${SFMM_CODEGEN_COMPILE_ERROR}"
    )
endif()

message(STATUS "Removing previously generated SFMM files")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E remove_directory
        "${SFMM_GENERATED_DIRECTORY}"
    RESULT_VARIABLE
        SFMM_CODEGEN_CLEAN_RESULT
)

if(NOT SFMM_CODEGEN_CLEAN_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "Failed to remove the old SFMM generated-code directory"
    )
endif()

message(STATUS "Running SFMM code generator")

execute_process(
    COMMAND
        "${SFMM_CODEGEN_EXECUTABLE}"
    WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE
        SFMM_CODEGEN_RUN_RESULT
    OUTPUT_VARIABLE
        SFMM_CODEGEN_RUN_OUTPUT
    ERROR_VARIABLE
        SFMM_CODEGEN_RUN_ERROR
)

if(NOT SFMM_CODEGEN_RUN_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "SFMM code generation failed.\n"
        "Exit result: ${SFMM_CODEGEN_RUN_RESULT}\n"
        "stdout:\n${SFMM_CODEGEN_RUN_OUTPUT}\n"
        "stderr:\n${SFMM_CODEGEN_RUN_ERROR}"
    )
endif()

message(STATUS "SFMM code generation complete")


# -----------------------------------------------------------------------------
# Common target settings
# -----------------------------------------------------------------------------

# Apply the generated include directory, expansion-order definitions, SIMD
# dependency, and architecture options consistently to each SFMM executable.
function(sfmm_configure_target target_name)
    target_include_directories(
        "${target_name}"
        PRIVATE
            "${SFMM_GENERATED_INCLUDE_DIRECTORY}"
    )

    target_compile_definitions(
        "${target_name}"
        PRIVATE
            PMIN=${SFMM_PMIN}
            PMAX=${SFMM_PMAX}
    )

    target_compile_options(
        "${target_name}"
        PRIVATE
            -march=native
    )

    target_link_libraries(
        "${target_name}"
        PRIVATE
            simd::simd
    )
endfunction()


# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

add_executable(
    test
    src/tree_code.cpp
)

sfmm_configure_target(test)

add_executable(
    brille
    src/brille.cpp
)

sfmm_configure_target(brille)


# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

install(
    FILES
        "${SFMM_GENERATED_INCLUDE_DIRECTORY}/sfmm.hpp"
    DESTINATION
        include
)

install(
    FILES
        "${SFMM_GENERATED_INCLUDE_DIRECTORY}/detail/sfmm_detail.hpp"
    DESTINATION
        include/detail
)
