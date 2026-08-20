//Implicit Cahn-Hilliard - DealII

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
// #include <deal.II/lac/solver_cg.h>
// #include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/precondition.h>

//output
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <filesystem>

//random number generator
#include <random>

using namespace dealii;

#include <CahnHilliard.hpp>
#include <OutputWriter.hpp>

int main(){
    constexpr unsigned int Nsd = 2; //2 for 2D problem, 3 for 3D problem
    constexpr unsigned int BfOrder = 1; //1 for linear elements, 2 for quadratic

    unsigned int NT = 100; //number of timesteps to solve for

    //Conjugate Solver parameters
    unsigned int NCG = 1000;
    double epsilonCG = 1e-6;

    unsigned int quadOrder = 2; //gaussian quadrature 

    //Problem parameters
    double Mobility = 1.0;

    //assumptions
    double epsilon = 0.1;
    double dt = 1e-7;

    //bulk free energy function
    auto fFunc = [](double c){ return 0.25*pow((c*c - 1),2); };
    auto fFuncDerivative = [](double c){ return c*(c*c - 1); };

    //domain and mesh parameters
    double x_ll = 0.0, x_ul = 1.0; //square domain for 2D, cube domain for 3D

    OutputWriter<Nsd,BfOrder> output_writer("output");
    CahnHilliard<Nsd,BfOrder> problem(x_ll, x_ul, quadOrder, fFunc, fFuncDerivative, NT, Mobility, epsilon, dt, NCG, epsilonCG, output_writer);
    
    problem.run();
}
