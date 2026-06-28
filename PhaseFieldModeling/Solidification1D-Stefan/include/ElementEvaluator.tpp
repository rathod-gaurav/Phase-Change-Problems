#pragma once

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
ElementEvaluator<Nsd,Nne,BfOrder>::ElementEvaluator(
    const Mesh<Nsd,Nne>& mesh,
    const QuadratureRule<Nsd,Nne>& quadRule,
    const double& rho,
    const double& W,
    const double& lambda,
    const double& LatentHeat,
    const double& Tm,
    const Eigen::VectorXd& phi,
    const Eigen::VectorXd& T,
    const std::function<double(double)> gFunc,
    const std::function<double(double)> pFunc,
    const std::function<double(double)> gFuncDerivative,
    const std::function<double(double)> pFuncDerivative,
    const std::function<double(double)> Cphi,
    const std::function<double(double)> Kphi
) :
    mesh_(mesh),
    quadRule_(quadRule),
    rho_(rho),
    W_(W),
    lambda_(lambda),
    LatentHeat_(LatentHeat),
    Tm_(Tm),
    phi_(phi),
    T_(T),
    gFunc_(gFunc),
    pFunc_(pFunc),
    gFuncDerivative_(gFuncDerivative),
    pFuncDerivative_(pFuncDerivative),
    Cphi_(Cphi),
    Kphi_(Kphi)
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
    Eigen::MatrixXd& Mphi_e,
    Eigen::MatrixXd& Kphi_e,
    Eigen::MatrixXd& Rphi_e,
    Eigen::MatrixXd& MT_e,
    Eigen::MatrixXd& KT_e,
    Eigen::MatrixXd& RT_e
) const{
      Mphi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      Kphi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      Rphi_e = Eigen::VectorXd::Zero(Nne,Nne);
      MT_e = Eigen::MatrixXd::Zero(Nne,Nne);
      KT_e = Eigen::MatrixXd::Zero(Nne,Nne);
      RT_e = Eigen::VectorXd::Zero(Nne,Nne);

      if constexpr (Nsd == 1){
        if constexpr (Nne == 2){
            const auto& quad_points = quadRule_.points;
            const auto& quad_weights = quadRule_.weights;
            unsigned int quadOrder = quad_points.size();

            for(unsigned int I = 0 ; I < quadOrder ; I++){
                double xi1 = quad_points[I];
                double weight = quad_weights[I];

                VectorNsd xi_vec(xi1);

                double phi_h = 0.0;
                double T_h = 0.0;

                for(unsigned int A = 0 ; A < Nne ; A++){
                    unsigned int globalNodeIndex = mesh_.elements[e].node[A];
                    phi_h += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*phi_[globalNodeIndex];
                    T_h += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*T_[globalNodeIndex];
                }

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
                        double Kphi_AB = JacInv*basis_gradient_vecA*JacInv*basis_gradient_vecB*JacDet*weight;
                        Mphi_e(A,B) = Mphi_AB;
                        Kphi_e(A,B) = Kphi_AB;

                        double MT_AB = rho_*Cphi_(phi_h)*N_A*N_B*JacDet*weight;
                        double KT_AB = Kphi_(phi_h)*JacInv*basis_gradient_vecA*JacInv*basis_gradient_vecB*JacDet*weight;
                        MT_e(A,B) = MT_AB;
                        KT_e(A,B) = KT_AB;
                    }
                    double Rphi_multiplier = -1*W_*gFuncDerivative_(phi_h) + lambda_*pFuncDerivative_(phi_h)*LatentHeat_*((T_h - Tm_)/Tm_);
                    double Rphi_A = Rphi_multiplier*ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A,xi_vec)*JacDet*weight;
                    Rphi_e(A) = Rphi_A;
                }
            }
        }
        else{
            throw std::invalid_argument("yet to implement element computation for given Nne");
        }
      }
};