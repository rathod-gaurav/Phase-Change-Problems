#pragma once

#include "Solver.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
CoupledPhaseFieldSolver<Nsd,Nne,BfOrder>::CoupledPhaseFieldSolver(
    const double tau,
    const double epsilon,
    const double dt,
    const unsigned int NT,
    const unsigned int maxIncr
) :
    tau_(tau),
    epsilon_(epsilon),
    dt_(dt),
    NT_(NT),
    maxIncr_(maxIncr)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void CoupledPhaseFieldSolver<Nsd,Nne,BfOrder>::solve(
    Eigen::VectorXd& phi,
    Eigen::VectorXd& T,
    const Assembler<Nsd,Nne,BfOrder>& assembler,
    const BoundaryConditions<Nsd,Nne>& bcs_phi,
    const BoundaryConditions<Nsd,Nne>& bcs_T,
    std::function<void(unsigned int, unsigned int, double)> iterCallback //optional callback function for monitoring
){
    Eigen::MatrixXd Mphi, MphiUU, MphiUD;
    Eigen::MatrixXd Kphi, KphiUU, KphiUD;
    Eigen::VectorXd Rphi, RphiU;
    Eigen::MatrixXd MT, MTUU, MTUD;
    Eigen::MatrixXd KT, KTUU, KTUD;
    Eigen::VectorXd RT, RTU;

    Eigen::VectorXd& phiU, phi_np1U, phi_np1; 
    Eigen::VectorXd& TU, T_np1U, T_np1;

    double t = dt_;
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){
        double incrFraction = 1.0;
        //Phase field equation
        bcs_phi.applyDirischletToSolution(phi,incrFraction);

        assembler.assembleSystem_phi(
            Mphi,
            Kphi,
            Rphi
        );

        assembler.partition(Mphi, Kphi, Rphi, phi, bcs_phi, MphiUU, MphiUD, KphiUU, KphiUD, RphiU, phiU);
        
        Eigen::MatrixXd LHS_phi = tau_*epsilon_*epsilon_*MphiUU + dt_*epsilon_*epsilon_*KphiUU;
        Eigen::VectorXd RHS_phi = tau_*epsilon_*epsilon_*MphiUU*phiU + dt_*RphiU;

        phi_np1U = LHS_phi.partialPivLu().solve(RHS_phi);
        phi_np1.resize(phi.size());
        const auto& unknownIndexes = bcs_phi.getUnknownIndexes();
        for(unsigned int i = 0 ; i < unknownIndexes.size() ; i++){
            phi_np1(unknownIndexes[i]) = phi_np1U[i];
        }
        bcs_phi.applyDirischletToSolution(phi_np1,incrFraction);

        //Temperature equation
        bcs_T.applyDirischletToSolution(T,incrFraction);

        assembler.assembleSystem_T(
            MT,
            KT,
            RT,
            phi_np1,
            dt_
        );

        assembler.partition(MT, KT, RT, T, bcs_T, MTUU, MTUD, KTUU, KTUD, RTU, TU);

        Eigen::MatrixXd LHS_T = MTUU + dt_*KTUU;
        Eigen::VectorXd RHS_T = MTUU*TU + dt_*RTU;
        T_np1U = LHS_T.partialPivLu().solve(RHS_T);
        T_np1.resize(T.size());
        for(unsigned int i = 0 ; i < unknownIndexes.size() ; i++){
            T_np1(unknownIndexes[i]) = T_np1U[i];
        }
        bcs_T.applyDirischletToSolution(T_np1, incrFraction);

        phi = phi_np1;
        T = T_np1;

    }
}