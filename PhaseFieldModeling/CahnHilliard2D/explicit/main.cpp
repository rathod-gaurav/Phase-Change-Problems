#include <iostream>
#include <stdexcept>
#include <math.h>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <MeshGenerator.hpp>
#include <BoundaryConditions.hpp>
#include <Quadrature.hpp>
#include <OutputWriter.hpp>
#include <ElementEvaluator.hpp>

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

    //bulk free energy function
    auto fFunc = [](double c){ return 0.25*pow((c*c - 1),2); };
    auto fFuncDerivative = [](double c){ return c*(c*c - 1); };

    //domain and mesh parameters
    double x1_ll = 0.0, x1_ul = 1.0;
    double x2_ll = 0.0, x2_ul = 1.0;
    // double x3_ll = 0.0, x3_ul = 1.0;
    double Nel_x1 = 10, Nel_x2 = 10;
    MeshGenerator<Nsd,Nne,BfOrder> meshGen(x1_ll, x1_ul, x2_ll, x2_ul, Nel_x1, Nel_x2);
    Mesh<Nsd,Nne> mesh = meshGen.buildMesh();
    mesh.writeToFiles("mesh");

    std::cout << "Mesh built: " << mesh.Nnodes() << " nodes, " << mesh.Nelements() << " elements" << std::endl;
    std::cout << "--------------------" << std::endl;
    
    //initialize the solution vector
    Eigen::VectorXd c = Eigen::VectorXd::Zero(mesh.Nnodes());
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(mesh.Nnodes());
    
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, std::nextafter(1, std::numeric_limits<double>::max()));
    for(unsigned int i = 0 ; i < mesh.Nnodes(); i++){
        c(i) = dist(gen);
    }

    OutputWriter<Nsd,Nne> writer("output");
    writer.writeVTU(mesh, c, 0);
    QuadratureRule<Nsd,Nne>             quadRule = Quadrature<Nsd,Nne>::gauss_legendre(quadOrder); //get the quadrature points and weights for the specified quadrature order
    ElementEvaluator<Nsd,Nne,BfOrder>   elemEval(mesh, quadRule, c, mu, fFunc, fFuncDerivative);
    
}
