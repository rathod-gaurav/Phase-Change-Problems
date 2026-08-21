#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::compute_element_B(const typename DoFHandler<Nsd>::active_cell_iterator& elem, FEValues<Nsd>& fe_values, Vector<double>& Blocal, FullMatrix<double>& dBlocal_dc, std::vector<types::global_dof_index>& local_dof_indices){
    fe_values.reinit(elem);

    for(const unsigned int q_index : fe_values.quadrature_point_indices()){
        double c_h = 0.0;
        for(const unsigned int i : fe_values.dof_indices()){
            c_h += fe_values.shape_value(i,q_index)*c_k(local_dof_indices[i]); //note : here we use c_k instead of c
        }

        for(const unsigned int i : fe_values.dof_indices()){
            Blocal(i) += fFuncDerivative_(c_h)*fe_values.shape_value(i,q_index)*fe_values.JxW(q_index);
            for(const unsigned int j : fe_values.dof_indices()){
                dBlocal_dc(i,j) += fFuncDoubleDerivative_(c_h)*fe_values.shape_value(i,q_index)*fe_values.shape_value(j,q_index)*fe_values.JxW(q_index);
            }
        }
    }
}