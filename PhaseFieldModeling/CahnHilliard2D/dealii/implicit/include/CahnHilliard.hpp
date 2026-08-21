#pragma once

#include "OutputWriter.hpp"

template<unsigned int Nsd, unsigned int BfOrder>
class CahnHilliard{
    public:
        CahnHilliard(const double x_ll, const double x_ul, const double quadOrder, const std::function<double(double)> fFunc, const std::function<double(double)> fFuncDerivative, const std::function<double(double)> fFuncDoubleDerivative,
                    const unsigned int NT, const double Mobility, const double epsilon, const double dt,
                    const unsigned int N_NR, const double epsilon_NR, const double theta,
                    OutputWriter<Nsd,BfOrder>& output_writer
                ); //constructor
        void run(); //execute the problem
    
    private:
        const double x_ll_, x_ul_, quadOrder_;
        const std::function<double(double)> fFunc_;
        const std::function<double(double)> fFuncDerivative_;
        const std::function<double(double)> fFuncDoubleDerivative_;
        const unsigned int NT_;
        const double Mobility_, epsilon_, dt_;
        const unsigned int N_NR_;
        const double epsilon_NR_;
        const double theta_;
        OutputWriter<Nsd,BfOrder>& output_writer_;

        void make_grid();
        void setup_system();
        void compute_element(const typename DoFHandler<Nsd>::active_cell_iterator& elem, FEValues<Nsd>& fe_values, FullMatrix<double>& Mlocal, FullMatrix<double>& Klocal, std::vector<types::global_dof_index>& local_dof_indices);
        void compute_element_B(const typename DoFHandler<Nsd>::active_cell_iterator& elem, FEValues<Nsd>& fe_values, Vector<double>& Blocal, FullMatrix<double>& dBlocal_dc, std::vector<types::global_dof_index>& local_dof_indices);
        void assemble_system();
        void assemble_system_B();
        void solve();
        void debug_system();

        Triangulation<Nsd> triangulation;
        const FE_Q<Nsd> fe;
        DoFHandler<Nsd> dof_handler;
        unsigned int N; //total number of degrees of freedom

        SparsityPattern sparsity_pattern, jacobian_sparsity_pattern;
        SparseMatrix<double> Mglobal;
        SparseMatrix<double> Kglobal;
        Vector<double> Bglobal;
        SparseMatrix<double> dBglobal_dc;
        Vector<double> c, c_k, c_np1, delta_c;
        Vector<double> mu, mu_k, mu_np1, delta_mu;
        Vector<double> RHS1;
        Vector<double> RHS2, RHS2_;
        SparseMatrix<double> NR_jacobian, J_cc, J_cmu, J_muc, J_muc_term1, J_mumu;
        Vector<double> NR_update, NR_residual, G_c, G_mu;
};

#include "CahnHilliard.tpp"

#include "MakeGrid.tpp"
#include "SetupSystem.tpp"
#include "AssembleSystem.tpp"
#include "Solver.tpp"