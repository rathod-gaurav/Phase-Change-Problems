#pragma once //include this only once during compilation

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Mesh.hpp"
#include <ElementEvaluator.hpp>
#include <BoundaryConditions.hpp>
#include <unordered_set>

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class Assembler{
    public: 
        Assembler(
            const Mesh<Nsd,Nne>& mesh, 
            const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator
        ); //constructor to initialize the assembler with the mesh, element evaluator, and diffusion evaluator

        void assembleSystem(
            Eigen::MatrixXd<double>& Mglobal, //global mass matrix (Nnodes x Nnodes matrix)
            Eigen::MatrixXd<double>& Kglobal, //global stiffness  matrix (Nnodes x Nnodes matrix)
            Eigen::VectorXd& Bglobal //global nonlinear bulk free energy vector (Nnodes x 1 vector)
        ) const;
    
    private:    
        const Mesh<Nsd,Nne>& mesh_; //reference to the mesh object
        const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator_; //reference to the element evaluator object

};

#include "Assembler.tpp" //include the implementation of the Assembler class