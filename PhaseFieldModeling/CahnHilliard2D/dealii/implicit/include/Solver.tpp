#pragma once

#include "AssembleSystem.tpp"
#include "DebugSystem.tpp"

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::solve(){    

    std::cout << "Initiating solver..." << std::endl;

    std::ofstream pvd_output("final_solution.pvd");
    output_writer_.initiate_pvd(pvd_output); //initiated pvd file with headers
    output_writer_.write_vtu_and_pvd(dof_handler, c, mu, 0, pvd_output);

    double t = dt_;
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){
        Mglobal = 0.0;
        Kglobal = 0.0;
        Bglobal = 0.0;
        assemble_system();
        
        // SparseILU<double> preconditioner;
        // preconditioner.initialize(Mglobal);

        // // RHS1 = Kglobal*(epsilon_*epsilon_)*c + Bglobal;
        Kglobal.vmult(RHS1,c);
        RHS1 *= (epsilon_*epsilon_);
        RHS1 += Bglobal;

        SparseDirectUMFPACK directsolver1;
        directsolver1.initialize(Mglobal);
        directsolver1.vmult(mu_np1, RHS1);


        // SolverControl solver1_control(NCG_, epsilonCG_*RHS1.l2_norm());
        // SolverGMRES<Vector<double>> solver1(solver1_control);
        // // std::cout << "||A||_F=" << Mglobal.frobenius_norm()
        // //   << "  ||b||="  << RHS1.l2_norm()
        // //   << "  ||x0||=" << c.l2_norm() << std::endl;
        // solver1.solve(Mglobal, mu_np1, RHS1, PreconditionIdentity());

        // // RHS2 = Mglobal*c - Kglobal*dt_*Mobility_*mu_np1;
        Kglobal.vmult(RHS2,mu_np1);
        RHS2 *= (-1*dt_*Mobility_);
        Mglobal.vmult(RHS2_,c);
        RHS2 += RHS2_;

        // SparseDirectUMFPACK directsolver2;
        // directsolver2.initialize(Mglobal);
        directsolver1.vmult(c_np1, RHS2);

        // RHS2.add(1,RHS2_);
        // SolverControl solver2_control(NCG_, epsilonCG_*RHS2.l2_norm());
        // SolverGMRES<Vector<double>> solver2(solver2_control);
        // solver2.solve(Mglobal, c_np1, RHS2, PreconditionIdentity()); 

        mu = mu_np1;
        c = c_np1;  

        // debug_system();
        
        output_writer_.write_vtu_and_pvd(dof_handler, c, mu, timestep, pvd_output);
        std::cout << "Solve completed for timestep " << timestep << std::endl;
        t+= dt_;
    }

    output_writer_.finish_pvd(pvd_output); //finishes pvd file with footer

    std::cout << "Solve completed." << std::endl;

}
