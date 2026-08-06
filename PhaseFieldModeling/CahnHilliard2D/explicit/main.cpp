#include <iostream>
#include <stdexcept>
#include <math.h>
#include <cmath>
#include <MeshGenerator.hpp>

int main() {
    constexpr unsigned int Nsd = 2;
    constexpr unsigned int BfOrder = 1;
    constexpr unsigned int Nne = 4;

    //number of timesteps to solve for
    unsigned int NT = 1000;

    //quadrature order
    unsigned int quadOrder = 2;

    std::cout << "Solving " << Nsd << "D problem with " << Nne << " node elements and " << BfOrder << " order basis functions." << std::endl;

    //Problem parameters
    double Mobility = 1.0;

    //assumptions
    double epsilon = 1e-2;
    double dt = 1e-5;

    //domain and mesh parameters
    double x1_ll = 0.0, x1_ul = 1.0;
    double x2_ll = 0.0, x2_ul = 1.0;
    double Nel_x1 = 100, Nel_x2 = 100;
    MeshGenerator<Nsd,Nne,BfOrder> meshGen(x1_ll, x1_ul, x2_ll, x2_ul, Nel_x1, Nel_x2);
    Mesh<Nsd,Nne> mesh = meshGen.buildMesh();
    mesh.writeToFiles("mesh");

    std::cout << "Mesh built: " << mesh.Nnodes() << " nodes, " << mesh.Nelements() << " elements" << std::endl;
    std::cout << "--------------------" << std::endl;
    
}
