#pragma once

#include <Eigen/Dense>
#include <Mesh.hpp>
#include <Quadrature.hpp>
#include <ShapeFunction.hpp>


template<unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class ElementEvaluator {
    public:
        ElementEvaluator( //default constructor
            const Mesh<Nsd,Nne>& mesh,
            const QuadratureRule<Nsd,Nne> quadRule,
            const Eigen::VectorXd& c, //solution vector for concentration field
            const Eigen::VectorXd& mu, //solution vector for chemical potential field
            const std::function<double(double)> fFunc,
            const std::function<double(double)> fFuncDerivative
        );

        void computeElement(
            unsigned int e,
            Eigen::MatrixXd& Mlocal, //local mass matrix
            Eigen::MatrixXd& Klocal, //local stiffness matrix
            Eigen::VectorXd& Blocal //local nonlinear integral vector
        ) const;

    private:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        using VectorNsd = Eigen::Vector<double, Nsd>;

        MatrixNsd computeJacobian(unsigned int e, const VectorNsd& xi_vec) const; //function to compute the Jacobian matrix for the element at given quadrature point (xi1, xi2, xi3)

        const Mesh<Nsd,Nne>& mesh_; //reference to the mesh object
        const QuadratureRule<Nsd,Nne>& quadRule_; //reference to the quadrature rule object
        const Eigen::VectorXd& c_; //reference to the solution vector for concentration field
        const Eigen::VectorXd& mu_; //reference to the solution vector for chemical potential field
        const std::function<double(double)> fFunc_;
        const std::function<double(double)> fFuncDerivative_;
};

#include <ElementEvaluator.tpp>