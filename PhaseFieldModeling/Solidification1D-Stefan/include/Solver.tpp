#pragma once

#include "Solver.h"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
PhaseFieldSolver<Nsd,Nne,BfOrder>::PhaseFieldSolver(
    const double tau,
    const double epsilon,
    const double dt,
    const unsigned int NT,
    const unsigned int maxIncr
) :
    tau_(tau),
    epsilon_(epsilon),
    dt_(dt),
    NT_(NT)
    maxIncr_(maxIncr)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void PhaseFieldSolver<Nsd,Nne,BfOrder>::solve(
    Eigen::VectorXd& phi,
    Eigen::VectorXd& T,
    const Assembler<Nsd,Nne,BfOrder>& assembler,
    const BoundaryConditions<Nsd,Nne>& bcs_phi,
    const BoundaryConditions<Nsd,Nne>& bcs_T,
    std::function<void(unsigned int, unsigned int, double)> iterCallback = nullptr; //optional callback function for monitoring
){
    Eigen::MatrixXd Mphi = Eigen::MatrixXd::Zero(Nt,Nt);
    Eigen::MatrixXd Kphi = Eigen::MatrixXd::Zero(Nt,Nt);
    Eigen::VectorXd Rphi = Eigen::VectorXd::Zero(Nt);
    Eigen::MatrixXd MT = Eigen::MatrixXd::Zero(Nt,Nt);
    Eigen::MatrixXd KT = Eigen::MatrixXd::Zero(Nt,Nt);
    Eigen::VectorXd RT = Eigen::VectorXd::Zero(Nt);

    double t = dt_;
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){
        double incrFraction = 1.0;
        bcs_phi.applyDirischletToSolution(phi,incrFraction);

        assembler.assembleSystem(
            phi,
            T,
            Mphi,
            Kphi,
            Rphi,
            MT,
            KT,
            RT
        );

        assembler.partition(Mphi, Kphi, Rphi, bcs_phi, MphiUU, MphiUD, KphiUU, KphiUD, RphiI);
        
        Eigen::VectorXd phi_np1 = LHS.PartialPivLu().solve(RHS);

    }
}