cmake_minimum_required(VERSION 3.16)
project(CahnHilliard2D CXX)

# Point CMake to your deal.II installation path
find_package(deal.II 9.7.1 REQUIRED
  HINTS /home/grathod/Downloads/dealii/dealii/installation_files
)

# Initialize cached variables required by deal.II
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Define your executable and source file
add_executable(main main.cpp)

# Link deal.II and all dependencies (MPI, PETSc, Trilinos, p4est)
DEAL_II_SETUP_TARGET(main)
