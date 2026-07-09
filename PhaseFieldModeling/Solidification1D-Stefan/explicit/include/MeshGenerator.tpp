#pragma once

#include "MeshGenerator.hpp"

//1D//
template<unsigned int Nne, unsigned int BfOrder>
MeshGenerator<1,Nne,BfOrder>::MeshGenerator(
    double x1_ll, double x1_ul,
    unsigned int Nel_x1
):
    x1_ll_(x1_ll), x1_ul_(x1_ul),
    Nel_x1_(Nel_x1)
{}

template<unsigned int Nne, unsigned int BfOrder>
Mesh<1,Nne> MeshGenerator<1,Nne,BfOrder>::buildMesh() const{
    Mesh<Nsd,Nne> mesh;

    if constexpr (BfOrder == 1){
        unsigned int Nnodes_x1 = Nel_x1_ + 1;

        unsigned int Nt = Nnodes_x1; //total number of nodes
        
        double dx1 = (x1_ul_ - x1_ll_)/(Nel_x1_);

        //build the nodes list of the mesh
        mesh.nodes.reserve(Nt);

        
        for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
            Node<Nsd> n;
            n.x1 = x1_ll_ + i*dx1;
            mesh.nodes.push_back(n);
        }

        //variables required for element connectivity
        unsigned int Nel_t = Nel_x1_;
        mesh.elements.reserve(Nel_t);
        if constexpr (Nne == 2){
            for(unsigned int i = 0 ; i < Nel_t ; i++){
                Element<Nne> e;
                e.node[0] = i;
                e.node[1] = i + 1;
                mesh.elements.push_back(e);
            }
        }
        else{
            throw std::runtime_error("Only 2 node elements are supported in 1D.");
        }
    }
    else{
        throw std::runtime_error("Only linear basis functions are supported in 1D.");
    }

    return mesh;
}