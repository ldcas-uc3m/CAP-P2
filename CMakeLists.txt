cmake_minimum_required(VERSION 3.1)

project(
  contrast
  VERSION 1.0
  LANGUAGES CXX)


find_package(MPI)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(contrast src/contrast-enhancement.cpp src/histogram-equalization.cpp src/contrast.cpp)
add_executable(contrast-omp src/contrast-enhancement-omp.cpp src/histogram-equalization.cpp src/contrast.cpp)
add_executable(contrast-mpi src/contrast-enhancement-mpi.cpp src/histogram-equalization.cpp src/contrast.cpp)
add_executable(contrast-mpi-omp src/contrast-enhancement-mpi-omp.cpp src/histogram-equalization.cpp src/contrast.cpp)

target_link_libraries(contrast ${MPI_C_LIBRARIES})
