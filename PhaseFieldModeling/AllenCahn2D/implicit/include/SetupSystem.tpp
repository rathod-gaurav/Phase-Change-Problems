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
    
    const std::map<types::global_dof_index, Point<Nsd>> dof_locations_map = DoFTools::map_dofs_to_support_points(MappingQ1<Nsd>(), dof_handler);

    //Initial conditions
    phi = 0.0;
    //vertical separation initial condition
    // for(const auto& [dof_index,point] : dof_locations_map){
    //     if(point[0] <= 0.5){ phi(dof_index) = 1.0; }
    //     else{ phi(dof_index) = -1.0; }
    // }
    

    //circular droplet initial condition
    const double x1c = 0.5*x_ul_;
    const double x2c = 0.5*x_ul_;
    const double R = 0.45*x_ul_;

    for(const auto& [dof_index, point] : dof_locations_map){
        double ri = sqrt(pow((point[0] - x1c),2) + pow((point[1] - x2c),2));
        if(ri <= R){
            phi(dof_index) = 1.0;
        }
        else{
            phi(dof_index) = -1.0;
        }
    }

    std::cout << "All global system matrices and vectors initialized" << std::endl;
}
