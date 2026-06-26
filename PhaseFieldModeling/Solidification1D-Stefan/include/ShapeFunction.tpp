#pragma once

#include <stdexcept>
#include <ShapeFunction.hpp>

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
typename ShapeFunction<Nsd,Nne,BfOrder>::VectorNsd
ShapeFunction<Nsd,Nne,BfOrder>::xi_at_node(unsigned int node){
    if constexpr (BfOrder == 1){
        if constexpr (Nsd == 1){
            double xi1;
            if constexpr (Nne == 2){
                switch (node){
                    case 0:
                        xi1 = -1.0;
                        break;
                    case 1:
                        xi1 = 1.0;
                        break;
                    default:
                        throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
                }
                return VectorNsd(xi1);
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }
    else{
        throw std::invalid_argument("yet to implement higher order basis functions");
    }
};

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
double ShapeFunction<Nsd,Nne,BfOrder>::basis_function(unsigned int node, const VectorNsd& xi){
    if constexpr (BfOrder == 1){
        if constexpr (Nsd == 1){
            if constexpr (Nne == 2){
                double value = 0.0;
                switch (node){
                    case 0:
                        return 0.5*(1.0 - xi(0));
                    case 1:
                        return 0.5*(1.0 + xi(0));
                    default:
                        throw std::invalid_argument("cannot evaluate basis function value for out of bound local node number");
                }
                return value;
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }
    else{
        throw std::invalid_argument("yet to implement higher order basis functions");
    }           
};

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
typename ShapeFunction<Nsd,Nne,BfOrder>::VectorNsd
ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(unsigned int node, const VectorNsd& xi_vec){
    if constexpr (BfOrder == 1){
        if constexpr (Nsd == 1){
            if constexpr (Nne == 2){
                double basis_gradient_xi1;
                switch (node){
                    case 0:
                        basis_gradient_xi1 = -0.5;
                        break;
                    case 1:
                        basis_gradient_xi1 = 0.5;
                        break;
                    default:
                        throw std::invalid_argument("cannot evaluate basis function gradient for out of bound local node number");
                }
                VectorNsd basis_gradient_vec(basis_gradient_xi1);
                return basis_gradient_vec;
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }
    else{
        throw std::invalid_argument("yet to implement higher order basis functions");
    }
};