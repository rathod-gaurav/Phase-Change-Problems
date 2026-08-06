#pragma once
#include <ElementEvaluator.hpp>

template<unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
ElementEvaluator<Nsd,Nne,BfOrder>::ElementEvaluator(
    const Mesh<Nsd,Nne>& mesh,
    const QuadratureRule<Nsd,Nne> quadRule,
    const Eigen::VectorXd& c, //solution vector for concentration field
    const Eigen::VectorXd& mu, //solution vector for chemical potential field
    const std::function<double(double)> fFunc,
    const std::function<double(double)> fFuncDerivative
) : 
    mesh_(mesh), 
    quadRule_(quadRule),
    c_(c),
    mu_(mu),
    fFunc_(fFunc),
    fFuncDerivative_(fFuncDerivative) 
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
typename ElementEvaluator<Nsd,Nne,BfOrder>::MatrixNsd
ElementEvaluator<Nsd,Nne,BfOrder>::computeJacobian(unsigned int e, const VectorNsd& xi_vec) const{
    MatrixNsd J = MatrixNsd::Zero();

    if constexpr (Nsd == 2){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
            
            unsigned int Aglobal = mesh_.elements[e].node[A];
            J(0,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x1; //dx1/dxi2
            J(1,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x2; //dx2/dxi2
        }
    }
    else if constexpr (Nsd == 3){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
            
            unsigned int Aglobal = mesh_.elements[e].node[A];
            J(0,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x1; //dx1/dxi2
            J(0,2) += basis_gradient_vec(2)*mesh_.nodes[Aglobal].x1; //dx1/dxi3
            J(1,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x2; //dx2/dxi2
            J(1,2) += basis_gradient_vec(2)*mesh_.nodes[Aglobal].x2; //dx2/dxi3
            J(2,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x3; //dx3/dxi1
            J(2,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x3; //dx3/dxi2
            J(2,2) += basis_gradient_vec(2)*mesh_.nodes[Aglobal].x3; //dx3/dxi3
        }
    }
    
    return J;
}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void ElementEvaluator<Nsd,Nne,BfOrder>::computeElement(
    unsigned int e,
    Eigen::MatrixXd& Mlocal, //local mass matrix
    Eigen::MatrixXd& Klocal, //local stiffness matrix
    Eigen::VectorXd& Blocal //local nonlinear integral vector
) const{
    Mlocal = Eigen::MatrixXd::Zero(Nne, Nne);
    Klocal = Eigen::MatrixXd::Zero(Nne, Nne);
    Blocal = Eigen::VectorXd::Zero(Nne);

    if constexpr(Nsd == 2){
        if constexpr(Nne == 3){
            const auto& quad_points_x1 = quadRule_.points_x1;
            const auto& quad_points_x2 = quadRule_.points_x2;
            const auto& quad_weights = quadRule_.weights;
            unsigned int quadOrder = quad_points_x1.size(); //number of quadrature points in each direction

            for(unsigned int I = 0 ; I < quadOrder ; I++){
                double xi1 = quad_points_x1[I]; 
                double xi2 = quad_points_x2[I];
                
                double weight = quad_weights[I];

                VectorNsd xi_vec(xi1,xi2);

                MatrixNsd Jac = computeJacobian(e, xi_vec); //compute the Jacobian matrix at the quadrature point
                double JacDet = Jac.determinant(); //compute the determinant of the Jacobian
                MatrixNsd JacInv = Jac.inverse(); //compute the inverse of the Jacobian

                for(unsigned int A = 0 ; A < Nne ; A++){
                    double NA = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec);

                    VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                    VectorNsd dNA_dx = JacInv * basis_gradient_vecA; //gradient of basis function A with respect to x1 and x2

                    for(unsigned int B = 0 ; B < Nne ; B++){
                        double NB = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(B, xi_vec);
                        VectorNsd basis_gradient_vecB = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(B, xi_vec);
                        VectorNsd dNB_dx = JacInv * basis_gradient_vecB; //gradient of basis function B with respect to x1 and x2

                        Mlocal(A,B) += NA * NB * weight * JacDet;
                        Klocal(A,B) += (dNA_dx.transpose() * dNB_dx) * weight * JacDet;
                    }

                    double c_e = c_(mesh_.elements[e].node[A]);
                    Blocal(A) += fFuncDerivative_(c_e) * NA * weight * JacDet;
                }
            }
        }
        else if constexpr(Nne == 4 || Nne == 9){
            const auto& quad_points = quadRule_.points;
            const auto& quad_weights = quadRule_.weights;
            unsigned int quadOrder = quad_points.size(); //number of quadrature points in each direction

            for(unsigned int I = 0 ; I < quadOrder ; I++){
                for(unsigned int J = 0 ; J < quadOrder ; J++){
                    double xi1 = quad_points[I]; 
                    double xi2 = quad_points[J];

                    double weight = quad_weights[I] * quad_weights[J];

                    VectorNsd xi_vec(xi1,xi2);

                    MatrixNsd Jac = computeJacobian(e, xi_vec);
                    double JacDet = Jac.determinant();
                    MatrixNsd JacInv = Jac.inverse();

                    for(unsigned int A = 0 ; A < Nne ; A++){
                        double NA = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec);

                        VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                        VectorNsd dNA_dx = JacInv * basis_gradient_vecA; //gradient of basis function A with respect to x1 and x2

                        for(unsigned int B = 0 ; B < Nne ; B++){
                            double NB = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(B, xi_vec);
                            VectorNsd basis_gradient_vecB = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(B, xi_vec);
                            VectorNsd dNB_dx = JacInv * basis_gradient_vecB; //gradient of basis function B with respect to x1 and x2

                            Mlocal(A,B) += NA * NB * weight * JacDet;
                            Klocal(A,B) += (dNA_dx.transpose() * dNB_dx) * weight * JacDet;
                        }

                        double c_e = c_(mesh_.elements[e].node[A]);
                        Blocal(A) += fFuncDerivative_(c_e) * NA * weight * JacDet;
                    }

                }
            }
        }
    }
    else if constexpr(Nsd == 3){
        const auto& quad_points = quadRule_.points;
        const auto& quad_weights = quadRule_.weights;
        unsigned int quadOrder = quad_points.size(); //number of quadrature points in each direction

        for(unsigned int I = 0 ; I < quadOrder ; I++){
            for(unsigned int J = 0 ; J < quadOrder ; J++){
                for(unsigned int K = 0 ; K < quadOrder ; K++){
                    double xi1 = quad_points[I]; 
                    double xi2 = quad_points[J];
                    double xi3 = quad_points[K];

                    double weight = quad_weights[I] * quad_weights[J] * quad_weights[K];

                    VectorNsd xi_vec(xi1,xi2,xi3);

                    MatrixNsd Jac = computeJacobian(e, xi_vec);
                    double JacDet = Jac.determinant();
                    MatrixNsd JacInv = Jac.inverse();

                    for(unsigned int A = 0 ; A < Nne ; A++){
                        double NA = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec);

                        VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                        VectorNsd dNA_dx = JacInv * basis_gradient_vecA; //gradient of basis function A with respect to x1 and x2

                        for(unsigned int B = 0 ; B < Nne ; B++){
                            double NB = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(B, xi_vec);
                            VectorNsd basis_gradient_vecB = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(B, xi_vec);
                            VectorNsd dNB_dx = JacInv * basis_gradient_vecB; //gradient of basis function B with respect to x1 and x2

                            Mlocal(A,B) += NA * NB * weight * JacDet;
                            Klocal(A,B) += (dNA_dx.transpose() * dNB_dx) * weight * JacDet;
                        }

                        double c_e = c_(mesh_.elements[e].node[A]);
                        Blocal(A) += fFuncDerivative_(c_e) * NA * weight * JacDet;
                    }
                }
            }
        }
    }

}

