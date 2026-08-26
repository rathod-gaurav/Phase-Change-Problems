#pragma once

#include "AssembleSystem.tpp"
#include "AssembleSystemB.tpp"
#include "DebugSystem.tpp"

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::solve(){    

    std::cout << "Initiating solver..." << std::endl;

    double t = dt_;
    Mglobal = 0.0;
    Kglobal = 0.0;
    assemble_system(); //assemble Mglobal and Kglobal only once - as they are constant matrices

    // //initialize mu using initialized c
    c_k = c; //just because compute_element_B is configured to use c_k to compute Bglobal
    Bglobal = 0.0;
    dBglobal_dc = 0.0;
    assemble_system_B();
    
    // // Mglobal*mu = epsilon_*epsilon_*Kglobal*c  Bglobal;
    RHS1 = 0.0;
    Kglobal.vmult(RHS1, c);
    RHS1 *= epsilon_*epsilon_;
    RHS1 += Bglobal;
    SparseDirectUMFPACK directsolver0;
    directsolver0.initialize(Mglobal);
    directsolver0.vmult(mu,RHS1);

    std::ofstream pvd_output("final_solution.pvd");
    output_writer_.initiate_pvd(pvd_output); //initiated pvd file with headers
    output_writer_.write_vtu_and_pvd(dof_handler, c, mu, 0, pvd_output);

    //NR monolithic jacobian block matrices
    J_cc = 0.0; J_cmu = 0.0; J_muc = 0.0; J_muc_term1 = 0.0; J_mumu = 0.0;
    J_cc.copy_from(Mglobal);
    J_cmu.add(dt_*Mobility_*theta_, Kglobal);
    J_muc_term1.add(-1*epsilon_*epsilon_, Kglobal);
    J_mumu.copy_from(Mglobal);

    // std::cout << "Mglobal frobenius norm : " << Mglobal.frobenius_norm() << std::endl;

    //lambda function to copy these matrices into the system monolothic jacobian
    auto copy_block = [](const SparseMatrix<double>& child_matrix, const unsigned int row_offset, const unsigned int col_offset, SparseMatrix<double>& result_matrix){
        for(unsigned int i = 0; i < child_matrix.m(); i++){
            for(auto it = child_matrix.begin(i) ; it != child_matrix.end(i) ; it++){
                result_matrix.add(i + row_offset, it->column() + col_offset, it->value());
            }
        }
    };

    //zero a particular block from system monolithic jacobian
    auto zero_block = [](const SparseMatrix<double>& child_matrix, const unsigned int row_offset, const unsigned int col_offset, SparseMatrix<double>& result_matrix){
        for(unsigned int i = 0; i < child_matrix.m(); i++){
            for(auto it = child_matrix.begin(i) ; it != child_matrix.end(i) ; it++){
                result_matrix(i + row_offset, it->column() + col_offset) = 0.0;
            }
        }
    };

    NR_jacobian = 0.0;
    copy_block(J_cc, 0, 0, NR_jacobian);
    copy_block(J_cmu, 0, N, NR_jacobian);
    copy_block(J_mumu, N, N, NR_jacobian);

    // debug_system();

    for(unsigned int timestep = 1 ; timestep < NT_ ; timestep++){
        c_k = c;
        mu_k = mu;

        //NR loops
        for(unsigned int k = 0 ; k < N_NR_ ; k++){

            Bglobal = 0.0;
            dBglobal_dc = 0.0;
            J_muc.copy_from(J_muc_term1);

            assemble_system_B(); //it uses c_k //calculates Bglobal and dBglobal_dc

            //residual
            // // G_c = Mglobal*(c_k - c) + dt_*Mobility_*Kglobal*((1-theta_)*mu + theta_*mu_k);
            Vector<double> tmp;
            tmp.reinit(c);

            tmp = c_k;
            tmp -= c;
            Mglobal.vmult(G_c, tmp);
            tmp.equ(dt_*Mobility_*(1-theta_), mu);
            tmp.add(dt_*Mobility_*theta_, mu_k);
            Kglobal.vmult_add(G_c, tmp);

            // // G_mu = Mglobal*mu_k - epsilon_*epsilon_*Kglobal*c_k - Bglobal
            Kglobal.vmult(G_mu, c_k);
            G_mu *= -epsilon_ * epsilon_;
            Mglobal.vmult_add(G_mu, mu_k);
            G_mu -= Bglobal;

            double error_c = G_c.l2_norm();
            double error_mu = G_mu.l2_norm();

            std::cout << "timestep: " << timestep << " | iteration: " << k << " | c_error: " << error_c << " | mu_error: " << error_mu << std::endl;
            if(error_c < epsilon_NR_ && error_mu < epsilon_NR_){
                c = c_k;
                mu = mu_k;
                std::cout << "convergence achieved for timestep " << timestep << " in " << k << " iterations" << std::endl;
                break;
            }
            else{
                NR_update = 0.0;
                NR_residual = 0.0;
                //Residual
                for(unsigned int i = 0 ; i < N ; i++){
                    NR_residual(i) = G_c(i);
                    NR_residual(i + N) = G_mu(i);
                }
                // std::copy(G_c.begin(), G_c.end(), NR_residual.begin());
                // std::copy(G_mu.begin(), G_mu.end(), NR_residual.begin() + N);
                NR_residual *= -1;

                //Jacobian
                J_muc.add(-1.0, dBglobal_dc);

                // std::cout << "J_cc jacobian Frobenius norm : " << J_cc.frobenius_norm() << std::endl;
                // std::cout << "J_cmu jacobian Frobenius norm : " << J_cmu.frobenius_norm() << std::endl;
                // std::cout << "J_muc jacobian Frobenius norm : " << J_muc.frobenius_norm() << std::endl;
                // std::cout << "J_mumu jacobian Frobenius norm : " << J_mumu.frobenius_norm() << std::endl;

                zero_block(J_muc, N, 0, NR_jacobian);
                copy_block(J_muc, N, 0, NR_jacobian);

                // std::cout << "NR jacobian Frobenius norm : " << NR_jacobian.frobenius_norm() << std::endl;

                //Solve
                SparseDirectUMFPACK directsolver;
                directsolver.initialize(NR_jacobian);
                directsolver.vmult(NR_update, NR_residual);

                // std::cout << "NR update L2 norm : " << NR_update.l2_norm() << std::endl;

                for(unsigned int i = 0 ; i < N ; i++){
                    c_k(i) += NR_update(i);
                    mu_k(i) += NR_update(i + N);
                }

                // std::cout << "c_k l2 norm : " << c_k.l2_norm() << std::endl;
                // std::cout << "mu_k l2 norm : " << mu_k.l2_norm() << std::endl;
            }
        }
        
        output_writer_.write_vtu_and_pvd(dof_handler, c, mu, timestep, pvd_output);
        t+= dt_;
    }

    output_writer_.finish_pvd(pvd_output); //finishes pvd file with footer

    std::cout << "Solve completed." << std::endl;

}
