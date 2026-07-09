#pragma once

#include <tuple>

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class ShapeFunction{
    public:
        using VectorNsd = Eigen::Vector<double, Nsd>;

        static VectorNsd xi_at_node(unsigned int node);

        static double basis_function(unsigned int node, const VectorNsd& xi);

        static VectorNsd basis_gradient(unsigned int node, const VectorNsd& xi_vec);
};

#include "ShapeFunction.tpp"