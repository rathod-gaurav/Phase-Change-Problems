#pragma once

#include "AssembleSystem.tpp"

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::solve(){    

    std::cout << "Initiating solver..." << std::endl;

    double t = dt_;
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){
        assemble_system();
        
        // RHS1 = Kglobal*(epsilon_*epsilon_)*c + Bglobal;
        Kglobal.vmult(RHS1,c);
        RHS1 *= (epsilon_*epsilon_);
        RHS1 += Bglobal;
        SolverControl solver1_control(NCG_, epsilonCG_*RHS1.l2_norm());
        SolverCG<Vector<double>> solver1(solver1_control);
        solver1.solve(Mglobal, mu_np1, RHS1, PreconditionIdentity());

        // RHS2 = Mglobal*c - Kglobal*dt_*Mobility_*mu_np1;
        Kglobal.vmult(RHS2,mu_np1);
        RHS2 *= (-1*dt_*Mobility_);
        Mglobal.vmult(RHS2_,c);
        // RHS2 = RHS2_ - RHS2;
        RHS2.add(1,RHS2_);
        SolverControl solver2_control(NCG_, epsilonCG_*RHS2.l2_norm());
        SolverCG<Vector<double>> solver2(solver2_control);
        solver2.solve(Mglobal, c_np1, RHS2, PreconditionIdentity()); 

        mu = mu_np1;
        c = c_np1;        

        t+= dt_;
    }

    std::cout << "Solve completed." << std::endl;

}