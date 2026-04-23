cmake_minimum_required(VERSION 3.18)

project(spherical_fmm LANGUAGES C CXX Fortran)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(FMM_WITH_CUDA "Enable CUDA" ON)

set(FMM_PMIN "3" CACHE STRING "pmin")
set(FMM_PMAX "6" CACHE STRING "pmax")

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

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated_code)

add_executable(codegen src/codegen.cpp)
target_compile_options(codegen PRIVATE -O3)
target_compile_definitions(codegen PRIVATE
	FMM_PMIN=${FMM_PMIN}
	FMM_PMAX=${FMM_PMAX}
)

set(GENERATED_STAMP ${CMAKE_CURRENT_BINARY_DIR}/generated_code/codegen.stamp)

add_custom_command(
	OUTPUT ${GENERATED_STAMP}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated_code
	COMMAND $<TARGET_FILE:codegen>
	COMMAND ${CMAKE_COMMAND} -E touch ${GENERATED_STAMP}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	DEPENDS
		codegen
		${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
	COMMENT "Generating spherical_fmm sources"
	VERBATIM
)

add_custom_target(generate_spherical_fmm_code DEPENDS ${GENERATED_STAMP})

file(GLOB GENERATED_SRC_FILES1 CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*.cpp)
file(GLOB GENERATED_SRC_FILES2 CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*/*.cpp)
file(GLOB GENERATED_CUDA_SRC_FILES1 CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*.cu)
file(GLOB GENERATED_CUDA_SRC_FILES2 CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*/*.cu)
file(GLOB GENERATED_HEADER_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)
file(GLOB GENERATED_DETAIL_HEADER_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/detail/*)

set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
file(WRITE ${DUMMY_CPP} "void spherical_fmm_dummy() {}\n")

if(FMM_WITH_CUDA)
	set(DUMMY_CU ${CMAKE_CURRENT_BINARY_DIR}/dummy.cu)
	file(WRITE ${DUMMY_CU} "__global__ void spherical_fmm_dummy_kernel() {}\n")
endif()

add_library(spherical_fmm
	${DUMMY_CPP}
	${GENERATED_SRC_FILES1}
	${GENERATED_SRC_FILES2}
	${GENERATED_HEADER_FILES}
	${GENERATED_DETAIL_HEADER_FILES}
)

add_dependencies(spherical_fmm generate_spherical_fmm_code)

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
		${DUMMY_CU}
		${GENERATED_CUDA_SRC_FILES1}
		${GENERATED_CUDA_SRC_FILES2}
		${GENERATED_HEADER_FILES}
		${GENERATED_DETAIL_HEADER_FILES}
	)

	add_dependencies(cuda_spherical_fmm generate_spherical_fmm_code)

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


add_executable(brill_test ${CMAKE_CURRENT_SOURCE_DIR}/src/brill_wave_test.cpp)

add_dependencies(brill_test generate_spherical_fmm_code)

target_include_directories(brill_test
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/generated_code/include
)

target_link_libraries(brill_test PUBLIC spherical_fmm)

target_compile_options(brill_test PUBLIC -march=native)

target_compile_definitions(brill_test PUBLIC
	FMM_PMIN=${FMM_PMIN}
	FMM_PMAX=${FMM_PMAX}
)
