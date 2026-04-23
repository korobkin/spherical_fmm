cmake_minimum_required(VERSION 3.18)

project(spherical_fmm LANGUAGES C CXX Fortran)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(FMM_WITH_CUDA "Enable CUDA" ON)

set(FMM_PMIN "3" CACHE STRING "pmin")
set(FMM_PMAX "10" CACHE STRING "pmax")

message(STATUS "pmin = ${FMM_PMIN}")
message(STATUS "pmax = ${FMM_PMAX}")

if(FMM_WITH_CUDA)
	enable_language(CUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS FALSE)

	execute_process(
		COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		RESULT_VARIABLE CUDA_DETECT_BUILD_RESULT
	)

	if(NOT CUDA_DETECT_BUILD_RESULT EQUAL 0)
		message(FATAL_ERROR "Failed to build cuda_detect")
	endif()

	execute_process(
		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/cuda_detect
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		OUTPUT_VARIABLE CUDA_ARCH
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE CUDA_DETECT_RUN_RESULT
	)

	if(NOT CUDA_DETECT_RUN_RESULT EQUAL 0)
		message(FATAL_ERROR "Failed to run cuda_detect")
	endif()

	message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated_code)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
	${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
	COPYONLY
)

execute_process(
	COMMAND ${CMAKE_CXX_COMPILER}
		-O3
		-DFMM_PMIN=${FMM_PMIN}
		-DFMM_PMAX=${FMM_PMAX}
		-o ${CMAKE_CURRENT_BINARY_DIR}/codegen
		${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	RESULT_VARIABLE CODEGEN_BUILD_RESULT
)

if(NOT CODEGEN_BUILD_RESULT EQUAL 0)
	message(FATAL_ERROR "Failed to build code generator")
endif()

execute_process(
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/codegen
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	RESULT_VARIABLE CODEGEN_RUN_RESULT
)

if(NOT CODEGEN_RUN_RESULT EQUAL 0)
	message(FATAL_ERROR "Failed to run code generator")
endif()

file(GLOB GENERATED_SRC_FILES1 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*.cpp)
file(GLOB GENERATED_SRC_FILES2 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*/*.cpp)
file(GLOB GENERATED_CUDA_SRC_FILES1 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*.cu)
file(GLOB GENERATED_CUDA_SRC_FILES2 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*/*.cu)
file(GLOB GENERATED_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)
file(GLOB GENERATED_DETAIL_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/detail/*)

add_library(spherical_fmm
	${GENERATED_SRC_FILES1}
	${GENERATED_SRC_FILES2}
	${GENERATED_HEADER_FILES}
	${GENERATED_DETAIL_HEADER_FILES}
)

target_include_directories(spherical_fmm
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
)

target_compile_options(spherical_fmm
	PUBLIC
		-march=native
)

target_compile_definitions(spherical_fmm
	PUBLIC
		FMM_PMIN=${FMM_PMIN}
		FMM_PMAX=${FMM_PMAX}
)

if(FMM_WITH_CUDA)
	add_library(cuda_spherical_fmm
		${GENERATED_CUDA_SRC_FILES1}
		${GENERATED_CUDA_SRC_FILES2}
		${GENERATED_HEADER_FILES}
		${GENERATED_DETAIL_HEADER_FILES}
	)

	target_include_directories(cuda_spherical_fmm
		PUBLIC
			${CMAKE_CURRENT_SOURCE_DIR}/include
			${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
	)

	target_compile_definitions(cuda_spherical_fmm
		PRIVATE
			FMM_PMIN=${FMM_PMIN}
			FMM_PMAX=${FMM_PMAX}
	)

	target_compile_options(cuda_spherical_fmm
		PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
	)

	set_property(TARGET cuda_spherical_fmm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(test_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_include_directories(test_float
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
)
target_link_libraries(test_float PUBLIC spherical_fmm)
target_compile_options(test_float PUBLIC -march=native)
target_compile_definitions(test_float PUBLIC
	FMM_PMIN=${FMM_PMIN}
	FMM_PMAX=${FMM_PMAX}
	TEST_TYPE_FLOAT
)

add_executable(test_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_include_directories(test_double
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
)
target_link_libraries(test_double PUBLIC spherical_fmm)
target_compile_options(test_double PUBLIC -march=native)
target_compile_definitions(test_double PUBLIC
	FMM_PMIN=${FMM_PMIN}
	FMM_PMAX=${FMM_PMAX}
	TEST_TYPE_DOUBLE
)

add_executable(test_vec_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_include_directories(test_vec_float
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
)
target_link_libraries(test_vec_float PUBLIC spherical_fmm)
target_compile_options(test_vec_float PUBLIC -march=native)
target_compile_definitions(test_vec_float PUBLIC
	FMM_PMIN=${FMM_PMIN}
	FMM_PMAX=${FMM_PMAX}
	TEST_TYPE_VEC_FLOAT
)

add_executable(test_vec_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_include_directories(test_vec_double
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
)
target_link_libraries(test_vec_double PUBLIC spherical_fmm)
target_compile_options(test_vec_double PUBLIC -march=native)
target_compile_definitions(test_vec_double PUBLIC
	FMM_PMIN=${FMM_PMIN}
	FMM_PMAX=${FMM_PMAX}
	TEST_TYPE_VEC_DOUBLE
)

if(FMM_WITH_CUDA)
	add_executable(test_cuda_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cu)
	target_include_directories(test_cuda_float
		PUBLIC
			${CMAKE_CURRENT_SOURCE_DIR}/include
			${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
	)
	target_link_libraries(test_cuda_float PUBLIC cuda_spherical_fmm)
	target_compile_definitions(test_cuda_float PUBLIC
		FMM_PMIN=${FMM_PMIN}
		FMM_PMAX=${FMM_PMAX}
		TEST_TYPE_FLOAT
	)
endif()