#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::setup_system(){
    dof_handler.distribute_dofs(fe);
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    Mglobal.reinit(sparsity_pattern);
    Kglobal.reinit(sparsity_pattern);

    Bglobal.reinit(dof_handler.n_dofs());
    c.reinit(dof_handler.n_dofs());
    mu.reinit(dof_handler.n_dofs());

    c_np1.reinit(dof_handler.n_dofs());
    mu_np1.reinit(dof_handler.n_dofs());

    RHS1.reinit(dof_handler.n_dofs());
    RHS2.reinit(dof_handler.n_dofs());
    RHS2_.reinit(dof_handler.n_dofs());

    //Initial conditions
    c = 0.0;
    //random noise initial condition
    // // std::random_device rd;
    // std::default_random_engine gen(123);
    // std::uniform_real_distribution<double> dist(-1.0, std::nextafter(1, std::numeric_limits<double>::max()));
    // for(unsigned int i = 0 ; i < dof_handler.n_dofs(); i++){
    //     c(i) = dist(gen);
    // }

    
    const std::map<types::global_dof_index, Point<2>> dof_locations_map = DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler);
    
    //equilibrium profile tanh initial condition
    // const double x0 = 0.50;
    // for(const auto& [dof_index, point] : dof_locations_map){
    //     c(dof_index) = std::tanh((point[0] - x0)/sqrt(2*epsilon_*epsilon_));
    // }

    //circular droplet initial condition
    const double x1c = 0.5*x_ul_;
    const double x2c = 0.5*x_ul_;
    const double R = 0.25*x_ul_;

    for(const auto& [dof_index, point] : dof_locations_map){
        double ri = sqrt(pow((point[0] - x1c),2) + pow((point[1] - x2c),2));
        c(dof_index) = std::tanh((R - ri)/sqrt(2*epsilon_*epsilon_));
    }


    mu = 0.0;

    std::cout << "All global system matrices and vectors initialized" << std::endl;
}