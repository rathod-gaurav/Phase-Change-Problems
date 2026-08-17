#pragma once

void Poisson::assemble_system(){

    const QGauss<2> quadrature_formula(fe.degree + 1); //gaussian quadrature points
    
    // the following class handles three things at once - finite element(basis functions), quadrature, and mapping from parent to real domains
    FEValues<2> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values); //the list of what kind of information we need on each cell is given as a collection of flags as the third argument to the constructor of FEValues class

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    //we first compute the elements of each element in a small matrix, and then transfer them to the global matrix
    FullMatrix<double> elem_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> elem_rhs(dofs_per_cell);

    //we now loop over all elements
    for(const auto &elem : dof_handler.active_cell_iterators()){
        // we use fe_values object recompute the required information on each cell
        fe_values.reinit(elem);
        
        //reset the element matrix and rhs vector to zero
        elem_matrix = 0;
        elem_rhs = 0;

        //start gaussian integration over the cell - by loping over the quadrature points
        for(const unsigned int q_index : fe_values.quadrature_point_indices()){
            for(const unsigned int i : fe_values.dof_indices()){
                for(const unsigned int j : fe_values.dof_indices()){
                    elem_matrix(i,j) += (fe_values.shape_grad(i,q_index)*fe_values.shape_grad(j,q_index)*fe_values.JxW(q_index)); 
                }
                elem_rhs(i) += (fe_values.shape_value(i,q_index)*1.0*fe_values.JxW(q_index)); //here, we are using a constant f = 1.0 (RHS of Poisson's equation)
            }
        }

        //before we transfer the element matrix and rhs to the global variables, we first have to find which global numbers does the degrees of freedom on this cell have
        elem->get_dof_indices(local_dof_indices);
        //now again loop over all shape functions and transfer the element information to the global matrices
        for(const unsigned int i : fe_values.dof_indices()){
            for(const unsigned int j : fe_values.dof_indices()){
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], elem_matrix(i,j));
            }
            system_rhs(local_dof_indices[i]) += elem_rhs(i);
        }
    }

    //Boundary conditions
    std::map<types::global_dof_index, double> boundary_values; //list of pairs of global dof numbers and their boundary values
    VectorTools::interpolate_boundary_values(
        dof_handler, //to get the global numbers of dofs on the boundary
        types::boundary_id(0), //the component of boundary where the boundary values shall be interpolated
        Functions::ZeroFunction<2>(), //the boundary value function //Here, it is a function which is zero everywhere
        boundary_values //the output object
    );

    // we now use the boundary Dofs and their respective boundary values to modify the system of equations accordingly
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

}