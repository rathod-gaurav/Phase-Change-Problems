#pragma once

#include <Eigen/Dense>
#include "Mesh.hpp"
#include "ElementEvaluator.hpp"
#include "BoundaryConditions.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class Assembler{
    public:
        Assembler(
            const Mesh<Nsd,Nne>& mesh,
            const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator
        );

        void assembleSystem_phi(
            Eigen::MatrixXd& Mphi,
            Eigen::MatrixXd& Kphi,
            Eigen::VectorXd& Rphi
        ) const;

        void assembleSystem_T(
            Eigen::MatrixXd& MT,
            Eigen::MatrixXd& KT,
            Eigen::VectorXd& RT,
            Eigen::VectorXd& phi_np1,
            const double& dt
        ) const;

        void partition(
            Eigen::MatrixXd& LHS,
            Eigen::VectorXd& RHS,
            const BoundaryConditions<Nsd,Nne>& bcs,
            Eigen::MatrixXd& LHSUU,
            Eigen::MatrixXd& LHSUD,
            Eigen::VectorXd& RHSU
        ) const;
    
    private:
        Eigen::MatrixXd extractSubmatrix(
            const Eigen::MatrixXd& K,
            const std::vector<unsigned int>& rows,
            const std::vector<unsigned int>& cols
        ) const ;

        const Mesh<Nsd,Nne>& mesh_;
        const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator_;

};

#include "Assembler.tpp"