#pragma once
using namespace std;
#include <Eigen/Dense>
#include "Mesh.hpp"
#include "ShapeFunction.hpp"
#include "Quadrature.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class ElementEvaluator{
    public:
        ElementEvaluator(//default constructor
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
        );

        void computeElement_phi(
            unsigned int e,
            Eigen::MatrixXd& Mphi_e,
            Eigen::MatrixXd& Kphi_e,
            Eigen::VectorXd& Rphi_e,
            Eigen::VectorXd& phi_k,
            Eigen::VectorXd& T_k,
            Eigen::MatrixXd& Dphiphi_e,
            Eigen::MatrixXd& DphiT_e
        ) const;

        void computeElement_T(
            unsigned int e,
            Eigen::MatrixXd& MT_e,
            Eigen::MatrixXd& KT_e,
            Eigen::VectorXd& RT_e,
            Eigen::VectorXd& phi_k,
            Eigen::VectorXd& T_k,
            const double& dt,
            Eigen::MatrixXd& DTphi_e
        ) const;
    
    private:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        using VectorNsd = Eigen::Vector<double, Nsd>;

        MatrixNsd computeJacobian(unsigned int e, const VectorNsd& xi_vec) const;

        const Mesh<Nsd,Nne>& mesh_;
        const QuadratureRule<Nsd,Nne>& quadRule_;
        const double& rho_;
        const double& W_;
        const double& lambda_;
        const double& LatentHeat_;
        const double& Tm_;
        const Eigen::VectorXd& phi_;
        const Eigen::VectorXd& T_;
        const std::function<double(double)> gFunc_;
        const std::function<double(double)> pFunc_;
        const std::function<double(double)> gFuncDerivative_;
        const std::function<double(double)> gFuncDoubleDerivative_;
        const std::function<double(double)> pFuncDerivative_;
        const std::function<double(double)> pFuncDoubleDerivative_;
        const std::function<double(double)> Cphi_;
        const std::function<double(double)> CphiDerivative_;
        const std::function<double(double)> Kphi_;
        const std::function<double(double)> KphiDerivative_;
};

#include "ElementEvaluator.tpp"