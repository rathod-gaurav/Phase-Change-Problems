#pragma once

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
ElementEvaluator<Nsd,Nne,BfOrder>::ElementEvaluator(
    const Mesh<Nsd,Nne>& mesh,
    const QuadratureRule<Nsd,Nne>& quadRule
) :
    mesh_(mesh),
    quadRule_(quadRule)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
typename ElementEvaluator<Nsd,Nne,BfOrder>::MatrixNsd
ElementEvaluator<Nsd,Nne,BfOrder>::computeJacobian(unsigned int e, const VectorNsd& xi_vec) const{
    MatrixNsd J = MatrixNsd::Zero();

    if constexpr (Nsd == 1){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);

            unsigned int Aglobal = mesh_.elements[e].node[A];
            J(0,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x1;
        }
    }
    
    return J;
}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void ElementEvaluator<Nsd,Nne,BfOrder>::computeElement(
    unsigned int e,
    const Eigen::VectorXd& phi_e,
    const Eigen::VectorXd& T_e,
    Eigen::MatrixXd& MPhi_e,
    Eigen::MatrixXd& KPhi_e,
    Eigen::MatrixXd& RPhi_e,
    Eigen::MatrixXd& MT_e,
    Eigen::MatrixXd& KT_e,
    Eigen::MatrixXd& RT_e
) const{
      MPhi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      KPhi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      RPhi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      MT_e = Eigen::MatrixXd::Zero(Nne,Nne);
      KT_e = Eigen::MatrixXd::Zero(Nne,Nne);
      RT_e = Eigen::MatrixXd::Zero(Nne,Nne);

      if constexpr (Nsd == 1){
        if constexpr (Nne == 2){
            const auto& quad_points = quadRule_.points;
            const auto& quad_weights = quadRule.weights;
            unsigned int quadOrder = quad_points.size();

            for(unsigned int I = 0 ; I < quadOrder ; I++){
                double xi1 = quad_points[I];
                double weight = quad_weights[I];

                VectorNsd xi_vec(xi1);

                MatrixNsd Jac = computeJacobian(e, xi_vec);
                double JacDet = Jac.determinant();
                MatrixNsd JacInv = Jac.inverse();

                for(unsigned int A = 0 ; A < Nne ; A++){
                    for(unsigned int B = 0 ; B < Nne ; B++){
                        double N_A = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec);
                        double N_B = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(B, xi_vec);
                        VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                        VectorNsd basis_gradient_vecB = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(B, xi_vec);

                        double Mphi_AB = N_A * N_B * JacDet * weight;
                        double Kphi_AB = basis_gradient_vecA*JacInv*basis_gradient_vecB*JacInv*JacDet*weight;
                    }
                }
            }
        }
        else{
            throw std::invalid_argument("yet to implement element computation for given Nne");
        }
      }
};