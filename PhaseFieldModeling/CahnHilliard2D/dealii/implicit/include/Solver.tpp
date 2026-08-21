#pragma once

#include "AssembleSystem.tpp"
#include "AssembleSystemB.tpp"
#include "DebugSystem.tpp"

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::solve(){    

    std::cout << "Initiating solver..." << std::endl;

    std::ofstream pvd_output("final_solution.pvd");
    output_writer_.initiate_pvd(pvd_output); //initiated pvd file with headers
    output_writer_.write_vtu_and_pvd(dof_handler, c, mu, 0, pvd_output);

    double t = dt_;
    Mglobal = 0.0;
    Kglobal = 0.0;
    assemble_system(); //assemble Mglobal and Kglobal only once - as they are constant matrices
    
    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){
        Bglobal = 0.0;

        assemble_system_B();

        // // RHS1 = Kglobal*(epsilon_*epsilon_)*c + Bglobal;
        Kglobal.vmult(RHS1,c);
        RHS1 *= (epsilon_*epsilon_);
        RHS1 += Bglobal;

        SparseDirectUMFPACK directsolver1;
        directsolver1.initialize(Mglobal);
        directsolver1.vmult(mu_np1, RHS1);

        // // RHS2 = Mglobal*c - Kglobal*dt_*Mobility_*mu_np1;
        Kglobal.vmult(RHS2,mu_np1);
        RHS2 *= (-1*dt_*Mobility_);
        Mglobal.vmult(RHS2_,c);
        RHS2 += RHS2_;

        directsolver1.vmult(c_np1, RHS2);

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
