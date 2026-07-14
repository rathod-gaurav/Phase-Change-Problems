#pragma once

#include "Solver.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
CoupledPhaseFieldSolver<Nsd,Nne,BfOrder>::CoupledPhaseFieldSolver(
    const double tau,
    const double epsilon,
    const double dt,
    const unsigned int NT,
    const unsigned int incrSteps,
    const unsigned int maxIter,
    const double epsilon_NR
) :
    tau_(tau),
    epsilon_(epsilon),
    dt_(dt),
    NT_(NT),
    incrSteps_(incrSteps),
    maxIter_(maxIter),
    epsilon_NR_(epsilon_NR)
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

    Eigen::VectorXd phi_k, T_k;
    Eigen::VectorXd G, G_phi, G_T;

    Eigen::MatrixXd Jacobian, J_phiphi, J_phiT, J_Tphi, J_TT;
    Eigen::MatrixXd Dphiphi, DphiT, DTphi;   

    Eigen::VectorXd delta_phi, delta_T;

    double incrFraction = 1.0;
    bcs_phi.applyDirischletToSolution(phi,incrFraction);
    bcs_T.applyDirischletToSolution(T,incrFraction);
    phi_k = phi;
    T_k = T;
    double t = dt_;
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){        
        
        //start newton raphson iterations
        for(unsigned int k = 1 ; k < maxIter_ ; k++){

            assembler.assembleSystem_phi(
                Mphi,
                Kphi,
                Rphi,
                phi_k,
                T_k,
                Dphiphi,
                DphiT
            );

            G_phi = ((tau_*epsilon_*epsilon_)/dt_)*Mphi*(phi_k - phi) + epsilon_*epsilon_*Kphi*phi_k - Rphi;

            assembler.assembleSystem_T(
                MT,
                KT,
                RT,
                phi_k,
                T_k,
                dt_,
                DTphi
            );

            G_T = (1/dt_)*MT*(T_k - T) + KT*T_k - RT;
            // std::cout << "G(100): " << G_T(100) << std::endl;
            G_T(100) = 0.0;

            double error = G_phi.norm() + G_T.norm();
            std::cout << "timestep: " << timestep << " | iteration: " << k << " | error: " << error << std::endl;

            if(error < epsilon_NR_){
                phi = phi_k;
                T = T_k;
                std::cout << "convergence achieved for timestep " << timestep << " in " << k << " iterations" << std::endl;
                break;
            }
            else{
                J_phiphi = ((tau_*epsilon_*epsilon_)/dt_)*Mphi + epsilon_*epsilon_*Kphi - Dphiphi;
                J_phiT = -DphiT;
                J_Tphi = DTphi;
                J_TT = (1/dt_)*MT + KT;

                //apply dirischlet boundary conditions
                const auto& dirischletIndexes = bcs_T.getDirischletIndexes();
                for(unsigned int i = 0 ; i < dirischletIndexes.size() ; i++){
                    J_phiT.col(dirischletIndexes[i]).setZero();
                    J_Tphi.row(dirischletIndexes[i]).setZero();
                    J_TT.col(dirischletIndexes[i]).setZero();
                    J_TT.row(dirischletIndexes[i]).setZero();
                    J_TT(dirischletIndexes[i],dirischletIndexes[i]) = 1.0;
                    // G_T(dirischletIndexes[i]) = 0.0;
                }

                unsigned int Nt = phi_k.size();
                Jacobian.resize(2*Nt,2*Nt);
                Jacobian << J_phiphi , J_phiT,
                            J_Tphi, J_TT;

                G.resize(2*Nt);
                G << G_phi , G_T;

                Eigen::VectorXd delta_soln(2*Nt);
                delta_soln = Jacobian.fullPivLu().solve(-G);

                delta_phi.resize(Nt);
                delta_T.resize(Nt);
                delta_phi = delta_soln.segment(0,Nt);
                delta_T = delta_soln.segment(Nt,Nt);
                
                phi_k += delta_phi;
                T_k += delta_T;

                bcs_phi.applyDirischletToSolution(phi_k,incrFraction);
                bcs_T.applyDirischletToSolution(T_k,incrFraction);
            }
        }

        phi = phi_k;
        T = T_k;
        
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