#pragma once
#include "Mesh.hpp"

template<unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class MeshGenerator;

//MeshGenerator for 1D
template<unsigned int Nne, unsigned int BfOrder>
class MeshGenerator<1,Nne,BfOrder>{
    public:
        static constexpr unsigned int Nsd = 1; //1D problem

        MeshGenerator(
            double x1_ll, double x1_ul, //domain parameters
            unsigned int Nel_x1 //mesh parameters
        );

        Mesh<Nsd,Nne> buildMesh() const;
    
    private:
        double x1_ll_, x1_ul_; //domain parameters
        unsigned int Nel_x1_; //mesh parameters
};

#include "MeshGenerator.tpp"