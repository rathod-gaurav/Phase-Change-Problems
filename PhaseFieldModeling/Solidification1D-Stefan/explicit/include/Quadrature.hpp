#pragma once

#include <vector>

template <unsigned int Nsd, unsigned int Nne>
struct QuadratureRule{
    std::vector<double> points;
    std::vector<double> weights;
};

template <unsigned int Nsd, unsigned int Nne>
class Quadrature{
    public:
        static QuadratureRule<Nsd,Nne> gauss_legendre(unsigned int n);
};

#include "Quadrature.tpp"