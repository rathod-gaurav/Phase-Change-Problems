#pragma once

#include "AssembleSystem.tpp"
#include "AssembleSystemB.tpp"
#include "DebugSystem.tpp"

template <unsigned int Nsd, unsigned int BfOrder>
void AllenCahn<Nsd,BfOrder>::solve(){    

    std::cout << "Initiating solver..." << std::endl;

    double t = dt_;
    Mglobal = 0.0;
    Kglobal = 0.0;
    assemble_system(); //assemble Mglobal and Kglobal only once - as they are constant matrices

    std::ofstream pvd_output("final_solution.pvd");
    output_writer_.initiate_pvd(pvd_output); //initiated pvd file with headers
    output_writer_.write_vtu_and_pvd(dof_handler, phi, 0, pvd_output);

    //NR jacobian
    jacobian_term1 = 0.0; NR_jacobian = 0.0;
    jacobian_term1.copy_from(Mglobal);
    jacobian_term1.add(dt_*Mobility_*epsilon_*epsilon_, Kglobal);

    // std::cout << "Mglobal frobenius norm : " << Mglobal.frobenius_norm() << std::endl;

    // debug_system();

    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){        

        //NR loops
        for(unsigned int k = 0 ; k < N_NR_ ; k++){
            Bglobal = 0.0;
            dBglobal_dphi = 0.0;

            NR_jacobian = 0.0;
            NR_jacobian.copy_from(jacobian_term1);

            assemble_system_B(); //it uses phi_k //calculates Bglobal and dBglobal_dphi

            //residual
            // // G_phi = (Mglobal + dt*mobility*epsilon*epsilon*Kglobal)phi_k - Mglobal*phi + Mobility*dt*Bglobal
            jacobian_term1.vmult(G_phi, phi_k);

            Vector<double> tmp;
            tmp.reinit(phi);
            tmp.equ(-1, phi);
            Mglobal.vmult_add(G_phi, tmp);
            tmp.equ(Mobility_*dt_, Bglobal);
            G_phi += tmp;

            double error_phi = G_phi.l2_norm();

            std::cout << "timestep: " << timestep << " | iteration: " << k << " | phi_error: " << error_phi << std::endl;
            if(error_phi < epsilon_NR_){
                phi = phi_k;
                std::cout << "convergence achieved for timestep " << timestep << " in " << k << " iterations" << std::endl;
                break;
            }
            else{
                NR_update = 0.0;
                NR_residual = 0.0;
                //Residual
                std::copy(G_phi.begin(), G_phi.end(), NR_residual.begin());
                NR_residual *= -1;

                //Jacobian
                NR_jacobian.add(Mobility_*dt_, dBglobal_dphi);                

                // std::cout << "NR jacobian Frobenius norm : " << NR_jacobian.frobenius_norm() << std::endl;

                //Solve
                SparseDirectUMFPACK directsolver;
                directsolver.initialize(NR_jacobian);
                directsolver.vmult(NR_update, NR_residual);

                // std::cout << "NR update L2 norm : " << NR_update.l2_norm() << std::endl;

                for(unsigned int i = 0 ; i < N ; i++){
                    phi_k(i) += NR_update(i);
                }

                // std::cout << "phi_k l2 norm : " << phi_k.l2_norm() << std::endl;
            }
        }
        
        output_writer_.write_vtu_and_pvd(dof_handler, phi, timestep, pvd_output);
        t+= dt_;
    }

    output_writer_.finish_pvd(pvd_output); //finishes pvd file with footer

    std::cout << "Solve completed." << std::endl;

}
