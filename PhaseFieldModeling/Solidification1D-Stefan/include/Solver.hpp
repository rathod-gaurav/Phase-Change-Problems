#pragma once

#include <Eigen/Dense>
#include "Assembler.hpp"
#include "BoundaryConditions.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class CoupledPhaseFieldSolver{
    public:
        CoupledPhaseFieldSolver(
            const double tau,
            const double epsilon,
            const double dt,
            const unsigned int NT,
            const unsigned int maxIncr //in case incremental loading is required for dirischlet BCs
        );

        void solve(
            Eigen::VectorXd& phi,
            Eigen::VectorXd& T,
            const Assembler<Nsd,Nne,BfOrder>& assembler,
            const BoundaryConditions<Nsd,Nne>& bcs_phi,
            const BoundaryConditions<Nsd,Nne>& bcs_T,
            std::function<void(unsigned int, double, const Eigen::VectorXd&, const Eigen::VectorXd&)> iterCallback = nullptr //optional callback function for monitoring
        );
    
    private:
        const double tau_;
        const double epsilon_;
        const double dt_;
        const unsigned int NT_;
        const unsigned int maxIncr_;
};

#include "Solver.tpp"