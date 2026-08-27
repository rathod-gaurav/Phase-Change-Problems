#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
void AllenCahn<Nsd,BfOrder>::setup_system(){
    dof_handler.distribute_dofs(fe);
    N = dof_handler.n_dofs();

    std::cout << "Number of degrees of freedom: " << N << std::endl;

    DynamicSparsityPattern dsp(N);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    Mglobal.reinit(sparsity_pattern);
    Kglobal.reinit(sparsity_pattern);

    Bglobal.reinit(N);
    dBglobal_dphi.reinit(sparsity_pattern);

    phi.reinit(N);

    phi_k.reinit(N);

    NR_update.reinit(N);
    NR_residual.reinit(N);
    
    NR_jacobian.reinit(sparsity_pattern);
    jacobian_term1.reinit(sparsity_pattern);

    G_phi.reinit(N);

    //Initial conditions
    phi = 0.0;
    //random noise initial condition //results stored in output 2
    // // std::random_device rd;
    std::default_random_engine gen(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for(unsigned int i = 0 ; i < N; i++){
        phi(i) = 0.63 + 0.02*(0.5 - dist(gen));
    }

    
    const std::map<types::global_dof_index, Point<2>> dof_locations_map = DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler);
    
    //equilibrium profile tanh initial condition
    // const double x0 = 0.50;
    // for(const auto& [dof_index, point] : dof_locations_map){
    //     c(dof_index) = std::tanh((point[0] - x0)/sqrt(2*epsilon_*epsilon_));
    // }

    //circular droplet initial condition //results stored in output
    // const double x1c = 0.5*x_ul_;
    // const double x2c = 0.5*x_ul_;
    // const double R = 0.25*x_ul_;

    // for(const auto& [dof_index, point] : dof_locations_map){
    //     double ri = sqrt(pow((point[0] - x1c),2) + pow((point[1] - x2c),2));
    //     c(dof_index) = std::tanh((R - ri)/sqrt(2*epsilon_*epsilon_));
    // }

    //two circular droplets of different size ////results stored in output1
    // const double x1c1 = 0.70*x_ul_;
    // const double x2c1 = 0.30*x_ul_;
    // const double R1 = 0.25*x_ul_;

    // const double x1c2 = 0.25*x_ul_;
    // const double x2c2 = 0.75*x_ul_;
    // const double R2 = 0.10*x_ul_;

    // for(const auto& [dof_index, point] : dof_locations_map){
    //     double ri1 = sqrt(pow((point[0] - x1c1),2) + pow((point[1] - x2c1),2));
    //     double ri2 = sqrt(pow((point[0] - x1c2),2) + pow((point[1] - x2c2),2));
    //     double di1 = 0.5*(1 + std::tanh((R1 - ri1)/sqrt(2*epsilon_*epsilon_)));
    //     double di2 = 0.5*(1 + std::tanh((R2 - ri2)/sqrt(2*epsilon_*epsilon_)));
    //     c(dof_index) = std::max(di1,di2);
    // }

    std::cout << "All global system matrices and vectors initialized" << std::endl;
}
