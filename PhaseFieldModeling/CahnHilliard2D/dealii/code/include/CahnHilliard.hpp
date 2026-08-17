#pragma once

template<unsigned int Nsd, unsigned int BfOrder>
class CahnHilliard{
    public:
        CahnHilliard(const double x_ll, const double x_ul, const double quadOrder, const std::function<double(double)> fFuncDerivative); //constructor
        void run(); //execute the problem
    
    private:
        const double x_ll_, x_ul_, quadOrder_;
        const std::function<double(double)> fFuncDerivative_;

        void make_grid();
        void setup_system();
        void compute_element(FEValues<Nsd>& fe_values, FullMatrix<double>& Mlocal, FullMatrix<double>& Klocal, Vector<double>& Bglobal, std::vector<types::global_dof_index>& local_dof_indices);
        void assemble_system();
        // void solve();
        // void output_writer();

        Triangulation<Nsd> triangulation;
        const FE_Q<Nsd> fe;
        DoFHandler<Nsd> dof_handler;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> Mglobal;
        SparseMatrix<double> Kglobal;
        Vector<double> Bglobal;
        Vector<double> c;
        Vector<double> mu;
};

#include "CahnHilliard.tpp"

#include "MakeGrid.tpp"
#include "SetupSystem.tpp"
#include "AssembleSystem.tpp"