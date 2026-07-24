#pragma once

#include "Solver.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
CoupledPhaseFieldSolver<Nsd,Nne,BfOrder>::CoupledPhaseFieldSolver(
    const double tau,
    const double epsilon,
    const double dt,
    const unsigned int NT,
    const unsigned int incrSteps,
    const Mesh<Nsd,Nne>& mesh //debug : remove mesh
) :
    tau_(tau),
    epsilon_(epsilon),
    dt_(dt),
    NT_(NT),
    incrSteps_(incrSteps),
    mesh_(mesh)
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

        //debug
        // 1. Interface width (phi=0.88 to phi=0.12 distance)
        double x_88 = -1, x_12 = -1;
        double h = mesh_.nodes[1].x1 - mesh_.nodes[0].x1;
        for(int i = 0; i < mesh_.Nnodes()-1; i++){
            if(phi[i] >= 0.88 && phi[i+1] < 0.88)
                x_88 = mesh_.nodes[i].x1 + (0.88-phi[i])/(phi[i+1]-phi[i])*h;
            if(phi[i] >= 0.12 && phi[i+1] < 0.12)
                x_12 = mesh_.nodes[i].x1 + (0.12-phi[i])/(phi[i+1]-phi[i])*h;
        }
        double width = (x_88>0 && x_12>0) ? x_12-x_88 : -1;

        // 2. Temperature at interface centre
        int iface = 0;
        double min_diff = 1.0;
        for(int i=0; i<mesh_.Nnodes(); i++)
            if(fabs(phi[i]-0.5) < min_diff){ min_diff=fabs(phi[i]-0.5); iface=i; }
        double T_iface = T[iface];

        // 3. Undercooling at interface
        double undercooling = 273.15 - T_iface;

        std::cout << "t=" << t
                << "  width=" << width
                << "  T_iface=" << T_iface
                << "  undercooling=" << undercooling
                << std::endl;

        if (iterCallback) {
            iterCallback(timestep, t, phi, T);
        }

        bool is_phi_bounded = (phi.array() >= 0.0).all() && (phi.array() <= 1.0).all();
        if (!is_phi_bounded) {
            // std::cout << "Out of bounds phi detected at timestep: " << timestep << " time: " << t << std::endl;
            // break;
        }

        t+=dt_;
    }
}