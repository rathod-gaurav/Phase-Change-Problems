#pragma once

#include "Solver.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
CoupledPhaseFieldSolver<Nsd,Nne,BfOrder>::CoupledPhaseFieldSolver(
    const double tau,
    const double epsilon,
    const double dt,
    const unsigned int NT,
    const unsigned int incrSteps
) :
    tau_(tau),
    epsilon_(epsilon),
    dt_(dt),
    NT_(NT),
    incrSteps_(incrSteps)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void CoupledPhaseFieldSolver<Nsd,Nne,BfOrder>::solve(
    Eigen::VectorXd& phi,
    Eigen::VectorXd& T,
    const Assembler<Nsd,Nne,BfOrder>& assembler,
    const BoundaryConditions<Nsd,Nne>& bcs_phi,
    const BoundaryConditions<Nsd,Nne>& bcs_T,
    std::function<void(unsigned int, double, const Eigen::VectorXd&, const Eigen::VectorXd&)> iterCallback //optional callback function for monitoring
){
    Eigen::MatrixXd Mphi;
    Eigen::MatrixXd Kphi;
    Eigen::VectorXd Rphi;
    Eigen::MatrixXd MT;
    Eigen::MatrixXd KT;
    Eigen::VectorXd RT;

    Eigen::VectorXd phi_np1U, phi_np1; 
    Eigen::VectorXd T_np1U, T_np1;

    Eigen::MatrixXd LHS_phiUU, LHS_phiUD, LHS_TUU, LHS_TUD;
    Eigen::VectorXd RHS_phiU, RHS_TU;

    double incrFraction = 1.0;
    bcs_phi.applyDirischletToSolution(phi,incrFraction);
    bcs_T.applyDirischletToSolution(T,incrFraction);    

    double t = dt_;
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){        
        //Phase field equation

        assembler.assembleSystem_phi(
            Mphi,
            Kphi,
            Rphi
        );

        Eigen::MatrixXd LHS_phi = tau_*epsilon_*epsilon_*Mphi;
        Eigen::VectorXd RHS_phi = (tau_*epsilon_*epsilon_*Mphi - dt_*epsilon_*epsilon_*Kphi)*phi + dt_*Rphi;

        assembler.partition(LHS_phi, RHS_phi, bcs_phi, LHS_phiUU, LHS_phiUD, RHS_phiU);
        assembler.make_lumped(LHS_phi);

        phi_np1U = LHS_phiUU.fullPivLu().solve(RHS_phiU);
        phi_np1.resize(phi.size());
        const auto& unknownIndexes_phi = bcs_phi.getUnknownIndexes();
        for(unsigned int i = 0 ; i < unknownIndexes_phi.size() ; i++){
            phi_np1(unknownIndexes_phi[i]) = phi_np1U[i];
        }
        bcs_phi.applyDirischletToSolution(phi_np1,incrFraction);
        // phi_np1 = phi;

        //Temperature equation

        // assembler.assembleSystem_T(
        //     MT,
        //     KT,
        //     RT,
        //     phi_np1,
        //     dt_
        // );

        // Eigen::MatrixXd LHS_T = MT;
        // Eigen::VectorXd RHS_T = (MT - dt_*KT)*T + dt_*RT;

        // assembler.partition(LHS_T, RHS_T, bcs_T, LHS_TUU, LHS_TUD, RHS_TU);

        // T_np1U = LHS_TUU.fullPivLu().solve(RHS_TU);
        // T_np1.resize(T.size());
        // const auto& unknownIndexes_T = bcs_T.getUnknownIndexes();
        // for(unsigned int i = 0 ; i < unknownIndexes_T.size() ; i++){
        //     T_np1(unknownIndexes_T[i]) = T_np1U[i];
        // }
        // bcs_T.applyDirischletToSolution(T_np1, incrFraction);

        phi = phi_np1;
        // T = T_np1;

        if (iterCallback) {
            iterCallback(timestep, t, phi, T);
        }

        bool is_phi_bounded = (phi.array() >= 0.0).all() && (phi.array() <= 1.0).all();
        if (!is_phi_bounded) {
            std::cout << "Out of bounds phi detected at timestep: " << timestep << " time: " << t << std::endl;
            // break;
        }

        t+=dt_;
    }
}