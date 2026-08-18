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

    std::cout << "All global system matrices and vectors initialized" << std::endl;
}