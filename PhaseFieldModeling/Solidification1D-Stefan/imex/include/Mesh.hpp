#pragma once
#include <vector>
#include <string>

template <unsigned int Nsd>
struct Node{
    static_assert(Nsd == 1, "Only 1D nodes are supported.");
};

template<>
struct Node<1>{
    double x1;
};

template <unsigned int Nne>
struct Element{
    unsigned int node[Nne];
};

template <unsigned int Nsd, unsigned int Nne>
class Mesh{
    public:
        //mesh class holds the nodes and elements lists
        std::vector<Node<Nsd>> nodes;
        std::vector<Element<Nne>> elements;

        //functions to return number of nodes and elements in the mesh
        unsigned int Nnodes() const { return nodes.size(); } //const here means this functon does not modify the state of the object. It is a read only function
        unsigned int Nelements() const { return elements.size(); }

        void writeToFiles(const std::string& dir) const; //function to write mesh into files

};

#include "Mesh.tpp"
