#pragma once

class Poisson{
    public:
        Poisson(); // class constructor
        void run();
    
    private:
        void make_grid();
        void setup_system();
        void assemble_system();
        void solve();
        void output_results() const;

        Triangulation<2> triangulation;
        const FE_Q<2> fe;
        DoFHandler<2> dof_handler;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
};

#include "Poisson.tpp" //class constructor definition, and public functions definition lies here

#include "MakeGrid.tpp" //generate the triangulation and number each vertex with the degree of freedom
#include "SetupSystem.tpp" //enumerate all the dofs and setup matrix and vector objects to hold the system data
#include "AssembleSystem.tpp" //compute the entries of system_matrix and the system_rhs vector from which we compute the solution
#include "Solver.tpp" //solve the discretised equation system_matrix * solution = system_rhs
#include "OutputResults.tpp"