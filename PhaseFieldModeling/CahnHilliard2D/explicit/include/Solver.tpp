#pragma once

#include <Solver.hpp>

template<unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
Solver<Nsd,Nne,BfOrder>::Solver(
    const double Mobility,
    const double epsilon,
    const double dt,
    const unsigned int NT,
    const Mesh<Nsd,Nne>& mesh
) : 
    Mobility_(Mobility), 
    epsilon_(epsilon), 
    dt_(dt), 
    NT_(NT), 
    mesh_(mesh) 
{}

template<unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void Solver<Nsd,Nne,BfOrder>::solve(
    Eigen::VectorXd& c,
    Eigen::VectorXd& mu,
    const Assembler<Nsd,Nne,BfOrder>& assembler,
    const std::function<double(double)>& fFunc,
    const std::function<double(double)>& fFuncDerivative,
    std::function<void(double)> iterCallback
){
    Eigen::MatrixXd Mglobal, Kglobal;
    Eigen::VectorXd Bglobal;

    Eigen::VectorXd c_np1, mu_np1;

    double t = dt_;
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){
        assembler.assembleSystem(Mglobal, Kglobal, Bglobal);
        
        Eigen::VectorXd RHS1 = epsilon_*epsilon_*Kglobal*c + Bglobal;
        mu = Mglobal.fullPivLu().solve(RHS1);

        Eigen::VectorXd RHS2 = Mglobal*c - dt_*Mobility_*Kglobal*mu;
        c = Mglobal.fullPivLu().solve(RHS2);

        if (iterCallback) {
            iterCallback(timestep);
        }

        t+= dt_;
    }
}