#pragma once

void Poisson::setup_system(){
    dof_handler.distribute_dofs(fe); //enumerate all the dofs

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    //we setup the sparsity pattern in the following fashion 
    //first, we create a temporary structure, and tag those entries that might be nonzero
    //we then copy the data over to sparsitypattern object that can then be used by the system matrix
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    //initialize the system_matrix matrix object with the created sparsity pattern
    system_matrix.reinit(sparsity_pattern);

    //set the sizes of RHS vector and the solution vector
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}