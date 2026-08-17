#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::compute_element(FEValues<Nsd>& fe_values, SparseMatrix<double>& Mglobal, SparseMatrix<double>& Kglobal, Vector<double>& Bglobal, std::vector<types::global_dof_index>& local_dof_indices){
    for(const unsigned int q_index : fe_values.quadrature_point_indices()){
        for(const unsigned int i : fe_values.dof_indices()){
            for(const unsigned int j : fe_values.dof_indices()){
                Kglobal(i,j) += (fe_values.shape_grad(i,q_index)*fe_values.shape_grad(j,q_index)*fe_values.JxW(q_index));
                Mglobal(i,j) +=  (fe_values.shape_value(i,q_index)*fe_values.shape_value(i,q_index)*fe_values.JxW(q_index))
            }
            double c_e = c(local_dof_indices[i]);
            Blocal(A) += fFuncDerivative_(c_e)*fe_values.shape_value(i,q_index)*fe_values.JxW(q_index);
        }
    }
}