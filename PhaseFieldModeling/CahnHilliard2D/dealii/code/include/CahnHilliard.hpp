#pragma once

#include "OutputWriter.hpp"

template<unsigned int Nsd, unsigned int BfOrder>
class CahnHilliard{
    public:
        CahnHilliard(const double x_ll, const double x_ul, const double quadOrder, const std::function<double(double)> fFuncDerivative, 
                    const unsigned int NT, const double Mobility, const double epsilon, const double dt,
                    const unsigned int NCG, const double epsilonCG,
                    OutputWriter<Nsd,BfOrder>& output_writer
                ); //constructor
        void run(); //execute the problem
    
    private:
        const double x_ll_, x_ul_, quadOrder_;
        const std::function<double(double)> fFuncDerivative_;
        const unsigned int NT_;
        const double Mobility_, epsilon_, dt_;
        const unsigned int NCG_;
        const double epsilonCG_;
        OutputWriter<Nsd,BfOrder>& output_writer_;

        void make_grid();
        void setup_system();
        void compute_element(const typename DoFHandler<Nsd>::active_cell_iterator& elem, FEValues<Nsd>& fe_values, FullMatrix<double>& Mlocal, FullMatrix<double>& Klocal, Vector<double>& Blocal, std::vector<types::global_dof_index>& local_dof_indices);
        void assemble_system();
        void solve();

        Triangulation<Nsd> triangulation;
        const FE_Q<Nsd> fe;
        DoFHandler<Nsd> dof_handler;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> Mglobal;
        SparseMatrix<double> Kglobal;
        Vector<double> Bglobal;
        Vector<double> c, c_np1;
        Vector<double> mu, mu_np1;
        Vector<double> RHS1;
        Vector<double> RHS2, RHS2_;
};

#include "CahnHilliard.tpp"

#include "MakeGrid.tpp"
#include "SetupSystem.tpp"
#include "AssembleSystem.tpp"
#include "Solver.tpp"