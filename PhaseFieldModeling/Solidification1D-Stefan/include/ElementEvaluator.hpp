#pragma once

#include <Eigen/Dense>
#include "Mesh.hpp"
#include "ShapeFunction.hpp"
#include "Quadrature.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class ElementEvaluator{
    public:
        ElementEvaluator(//default constructor
            const Mesh<Nsd,Nne>& mesh,
            const QuadratureRule<Nsd,Nne>& quadRule
        );

        void computeElement(
            unsigned int e,
            const Eigen::VectorXd& phi_e,
            const Eigen::VectorXd& T_e,
            Eigen::MatrixXd& MPhi_e,
            Eigen::MatrixXd& KPhi_e,
            Eigen::MatrixXd& RPhi_e,
            Eigen::MatrixXd& MT_e,
            Eigen::MatrixXd& KT_e,
            Eigen::MatrixXd& RT_e
        ) const;
    
    private:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        using VectorNsd = Eigen::Vector<double, Nsd>;

        MatrixNsd computeJacobian(unsigned int e, const VectorNsd& xi_vec) const;

        const Mesh<Nsd,Nne>& mesh_;
        const QuadratureRule<Nsd,Nne>& quadRule_;
};

#include "ElementEvaluator.tpp"