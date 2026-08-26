#pragma once

#include "ComputeElementB.tpp"

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::assemble_system_B(){
    const QGauss<Nsd> quadrature_formula(quadOrder_);

    // the following class handles three things at once - finite element(basis functions), quadrature, and mapping from parent to real domains
    FEValues<Nsd> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values); //the list of what kind of information we need on each cell is given as a collection of flags as the third argument to the constructor of FEValues class

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    Vector<double> Blocal(dofs_per_cell);
    FullMatrix<double> dBlocal_dc(dofs_per_cell, dofs_per_cell);

    for(const typename DoFHandler<Nsd>::active_cell_iterator &elem : dof_handler.active_cell_iterators()){
        Blocal = 0.0;
        dBlocal_dc = 0.0;

        elem->get_dof_indices(local_dof_indices);
        compute_element_B(elem, fe_values, Blocal, dBlocal_dc, local_dof_indices);

        for(const unsigned int i : fe_values.dof_indices()){
            Bglobal(local_dof_indices[i]) += Blocal(i);
            for(const unsigned int j : fe_values.dof_indices()){
                dBglobal_dc.add(local_dof_indices[i], local_dof_indices[j], dBlocal_dc(i,j));
            }
        }
    }

    // std::cout << "System assembly completed" << std::endl;
}