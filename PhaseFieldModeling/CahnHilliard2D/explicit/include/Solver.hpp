#pragma once

#include <Eigen/Dense>
#include <Assembler.hpp>
#include <Mesh.hpp>

template<unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class Solver {
    public:
        Solver(    
            const double Mobility,
            const double epsilon,
            const double dt,
            const unsigned int NT,
            const Mesh<Nsd,Nne>& mesh
        );

        void solve(
            Eigen::VectorXd& c,
            Eigen::VectorXd& mu,
            const Assembler<Nsd,Nne,BfOrder>& assembler,
            const std::function<double(double)>& fFunc,
            const std::function<double(double)>& fFuncDerivative,
            std::function<void(double)> iterCallback = nullptr
        );
    
    private:
        const double Mobility_;
        const double epsilon_;
        const double dt_;
        const unsigned int NT_;
        const Mesh<Nsd,Nne>& mesh_;
        
};

#include <Solver.tpp>