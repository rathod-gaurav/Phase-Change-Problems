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
#include <Assembler.hpp>
#include <Solver.hpp>

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
    double epsilon = sqrt(2.0);
    double dt = 1e-7;

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

    double h = std::min((x1_ul - x1_ll)/Nel_x1, (x2_ul - x2_ll)/Nel_x2);
    double CFL = pow(h,4)/(Mobility*pow(epsilon,2));
    std::cout << "CFL condition: dt <= " << CFL << std::endl;
    std::cout << "Using dt = " << dt << std::endl;
    std::cout << "Mesh spacing h = " << h << std::endl;
    
    //initialize the solution vectors
    Eigen::VectorXd c = Eigen::VectorXd::Zero(mesh.Nnodes());
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(mesh.Nnodes());
    
    // std::random_device rd;
    std::default_random_engine gen(123);
    std::uniform_real_distribution<double> dist(-1.0, std::nextafter(1, std::numeric_limits<double>::max()));
    for(unsigned int i = 0 ; i < mesh.Nnodes(); i++){
        c(i) = dist(gen);
    }

    QuadratureRule<Nsd,Nne>             quadRule = Quadrature<Nsd,Nne>::gauss_legendre(quadOrder); //get the quadrature points and weights for the specified quadrature order
    ElementEvaluator<Nsd,Nne,BfOrder>   elemEval(mesh, quadRule, c, mu, fFunc, fFuncDerivative);
    Assembler<Nsd,Nne,BfOrder>          assembler(mesh, elemEval);
    Solver<Nsd,Nne,BfOrder>             solver(Mobility, epsilon, dt, NT, mesh);
    OutputWriter<Nsd,Nne>               writer("output");

    writer.writeVTU(mesh, c, 0);

    std::cout << "Starting the solver..." << std::endl;

    solver.solve(c, mu, assembler, fFunc, fFuncDerivative,
                [&](double timestep){
                    writer.writeVTU(mesh, c, timestep);
                }    
        );
    
    writer.writePVD("final_solution.pvd");

    std::cout << "--------------------" << std::endl;
    std::cout << "Solve completed." << std::endl;
    
}
