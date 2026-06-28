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

        void assembleSystem(
            const Eigen::VectorXd& phi,
            const Eigen::VectorXd& T,
            Eigen::MatrixXd& Mphi,
            Eigen::MatrixXd& Kphi,
            Eigen::VectorXd& Rphi,
            Eigen::MatrixXd& MT,
            Eigen::MatrixXd& KT,
            Eigen::VectorXd& RT
        );

        // void partition(

        // )
    
    private:
        const Mesh<Nsd,Nne>& mesh_;
        const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator_;

};

#include "Assembler.tpp"