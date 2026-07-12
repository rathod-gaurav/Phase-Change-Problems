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
    const std::function<double(double)> gFuncDoubleDerivative,
    const std::function<double(double)> pFuncDerivative,
    const std::function<double(double)> pFuncDoubleDerivative,
    const std::function<double(double)> Cphi,
    const std::function<double(double)> CphiDerivative,
    const std::function<double(double)> Kphi,
    const std::function<double(double)> KphiDerivative
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
    gFuncDoubleDerivative_(gFuncDoubleDerivative),
    pFuncDerivative_(pFuncDerivative),
    pFuncDoubleDerivative_(pFuncDoubleDerivative),
    Cphi_(Cphi),
    CphiDerivative_(CphiDerivative),
    Kphi_(Kphi),
    KphiDerivative_(KphiDerivative)
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
void ElementEvaluator<Nsd,Nne,BfOrder>::computeElement_phi(
    unsigned int e,
    Eigen::MatrixXd& Mphi_e,
    Eigen::MatrixXd& Kphi_e,
    Eigen::VectorXd& Rphi_e,
    Eigen::VectorXd& phi_k,
    Eigen::VectorXd& T_k,
    Eigen::MatrixXd& Dphiphi_e,
    Eigen::MatrixXd& DphiT_e
) const{
      Mphi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      Kphi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      Rphi_e = Eigen::VectorXd::Zero(Nne);
      Dphiphi_e = Eigen::MatrixXd::Zero(Nne,Nne);
      DphiT_e = Eigen::MatrixXd::Zero(Nne,Nne);

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
                    phi_h += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*phi_k[globalNodeIndex];
                    T_h += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*T_k[globalNodeIndex];
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
                        double Kphi_AB = (JacInv*basis_gradient_vecA*JacInv*basis_gradient_vecB*JacDet*weight).value();
                        Mphi_e(A,B) += Mphi_AB;
                        Kphi_e(A,B) += Kphi_AB;

                        double Dphiphi_multiplier = -1*W_*gFuncDoubleDerivative_(phi_h) + lambda_*pFuncDoubleDerivative_(phi_h)*LatentHeat_*((T_h - Tm_)/Tm_);
                        double Dphiphi_AB = Dphiphi_multiplier*N_A*N_B*JacDet*weight;
                        Dphiphi_e(A,B) += Dphiphi_AB;

                        double DphiT_AB = lambda_*pFuncDerivative_(phi_h)*(LatentHeat_/Tm_)*N_A*N_B*JacDet*weight;
                        DphiT_e(A,B) += DphiT_AB;
                    }
                    double Rphi_multiplier = -1*W_*gFuncDerivative_(phi_h) + lambda_*pFuncDerivative_(phi_h)*LatentHeat_*((T_h - Tm_)/Tm_);
                    double Rphi_A = Rphi_multiplier*ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A,xi_vec)*JacDet*weight;
                    Rphi_e(A) += Rphi_A;                    
                }
            }
        }
        else{
            throw std::invalid_argument("yet to implement element computation for given Nne");
        }
      }
};


template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void ElementEvaluator<Nsd,Nne,BfOrder>::computeElement_T(
    unsigned int e,
    Eigen::MatrixXd& MT_e,
    Eigen::MatrixXd& KT_e,
    Eigen::VectorXd& RT_e,
    Eigen::VectorXd& phi_k,
    Eigen::VectorXd& T_k,
    const double& dt,
    Eigen::MatrixXd& DTphi_e
) const{
      MT_e = Eigen::MatrixXd::Zero(Nne,Nne);
      KT_e = Eigen::MatrixXd::Zero(Nne,Nne);
      RT_e = Eigen::VectorXd::Zero(Nne);
      DTphi_e = Eigen::MatrixXd::Zero(Nne,Nne);

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
                double phi_h_t = 0.0;
                double T_h = 0.0;
                double T_n_h = 0.0;
                double dT_h = 0.0;

                for(unsigned int A = 0 ; A < Nne ; A++){
                    unsigned int globalNodeIndex = mesh_.elements[e].node[A];
                    phi_h += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*phi_k[globalNodeIndex];

                    double phi_t = (phi_k[globalNodeIndex] - phi_[globalNodeIndex])/dt;
                    phi_h_t += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*phi_t;

                    T_h += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*T_k[globalNodeIndex];
                    T_n_h += ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*T_[globalNodeIndex];

                    VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                    dT_h += (basis_gradient_vecA*T_k[globalNodeIndex]).value();
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

                        double MT_AB = rho_*Cphi_(phi_h)*N_A*N_B*JacDet*weight;
                        double KT_AB = (Kphi_(phi_h)*JacInv*basis_gradient_vecA*JacInv*basis_gradient_vecB*JacDet*weight).value();
                        MT_e(A,B) += MT_AB;
                        KT_e(A,B) += KT_AB;

                        double DTphi_term1 = (1/dt)*rho_*CphiDerivative_(phi_h)*N_A*N_B*(T_h - T_n_h)*JacDet*weight;
                        double DTphi_term2 = (KphiDerivative_(phi_h)*N_B*basis_gradient_vecA*JacInv*dT_h*JacInv*JacDet*weight).value();
                        double DTphi_term3 = rho_*LatentHeat_*(pFuncDoubleDerivative_(phi_h)*phi_h_t + (pFuncDerivative_(phi_h)/dt))*N_A*N_B*JacDet*weight;
                        DTphi_e(A,B) += DTphi_term1 + DTphi_term2 - DTphi_term3;
                    }
                    double RT_multiplier = rho_*LatentHeat_*pFuncDerivative_(phi_h)*phi_h_t;
                    double RT_A = RT_multiplier*ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec)*JacDet*weight;
                    RT_e(A) += RT_A;
                }
            }
        }
        else{
            throw std::invalid_argument("yet to implement element computation for given Nne");
        }
      }
};