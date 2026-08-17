#include <deal.II/grid/tria.h> //triangulation
#include <deal.II/dofs/dof_handler.h> //enumeration of degrees of freedom
#include <deal.II/grid/grid_generator.h> //grid generation

#include <deal.II/fe/fe_q.h> //Lagrange finite elements
#include <deal.II/dofs/dof_tools.h> //DoFHandler tools

#include <deal.II/fe/fe_values.h> //used to assemble matrix using quadrature on each cell
#include <deal.II/base/quadrature_lib.h> //quadrature rules

//need thiese three for treatment of boundary values
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

//linear algebra
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

//output
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

//include files we have created
#include <Poisson.hpp>

int main(){
    Poisson poisson_problem;
    poisson_problem.run();

    return 0;
}