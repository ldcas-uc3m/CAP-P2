cmake_minimum_required(VERSION 3.1)

project(
  contrast
  VERSION 1.0
  LANGUAGES CXX
)


find_package(MPI)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# TODO: move each one to its own folder... while still leaving the executables in the same directory...

add_executable(
  contrast
  src/serial/contrast-enhancement.cpp
  src/serial/histogram-equalization.cpp
  src/serial/contrast.cpp
)
add_executable(
  contrast-mpi-omp
  src/mpi-omp/contrast-enhancement.cpp
  src/mpi-omp/histogram-equalization.cpp
  src/mpi-omp/contrast.cpp
)
add_executable(
  contrast-mpi
  src/mpi/contrast-enhancement.cpp
  src/mpi/histogram-equalization.cpp
  src/mpi/contrast.cpp
)
add_executable(
  contrast-omp
  src/omp/contrast-enhancement.cpp
  src/omp/histogram-equalization.cpp
  src/omp/contrast.cpp
)

target_link_libraries(contrast-mpi MPI::MPI_CXX)
target_link_libraries(contrast-mpi-omp MPI::MPI_CXX)
