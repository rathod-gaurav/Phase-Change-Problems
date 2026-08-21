#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::compute_element_B(const typename DoFHandler<Nsd>::active_cell_iterator& elem, FEValues<Nsd>& fe_values, Vector<double>& Blocal, std::vector<types::global_dof_index>& local_dof_indices){
    fe_values.reinit(elem);

    for(const unsigned int q_index : fe_values.quadrature_point_indices()){
        for(const unsigned int i : fe_values.dof_indices()){
            double c_e = c(local_dof_indices[i]);
            Blocal(i) += fFuncDerivative_(c_e)*fe_values.shape_value(i,q_index)*fe_values.JxW(q_index);
        }
    }
}