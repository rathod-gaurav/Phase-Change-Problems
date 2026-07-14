#include <iostream>
#include <stdexcept>
#include <math.h>
#include <cmath>
#include <MeshGenerator.hpp>
#include <Quadrature.hpp>
#include <BoundaryConditions.hpp>
#include <ElementEvaluator.hpp>
#include <Assembler.hpp>
#include <Solver.hpp>
#include <OutputWriter.hpp>

int main(){

    constexpr unsigned int Nsd = 1;
    constexpr unsigned int BfOrder = 1;
    constexpr unsigned int Nne = 2;

    //Number of timesteps to solve for
    unsigned int NT = 200;
    unsigned int incrSteps = 1;
    unsigned int maxIter = 20; //maximum number of allowed iterations for Newton-Raphson
    double epsilon_NR = 1e-3; //convergence criteria for Newton-Raphson 

    //Quadrature order
    unsigned int quadOrder = 2;

    std::cout << "Solving " << Nsd << "D problem with " << Nne << " node elements and " << BfOrder << " order basis functions." << std::endl;

    //Problem parameters
    double rho = 917.0; //Kg/m3
    double Cs = 2090.0; //J/Kg-K
    double Cl = 4186.0; //J/Kg-K 
    double Ks = 2.22; //W/m-K
    double Kl = 0.556; //W/m-K
    double LatentHeat = 334000.0; //J/Kg
    double Tm = 273.15; //K
    double Tcold = 263.15; //K

    double sigma = 0.033; //J/m2
    double mu = 1e-8; //m/s-K

    //Assumptions
    double epsilon = 5*1e-4;
    //Derived quantities
    double W = 1.0;
    double delta = epsilon*sqrt(2.0/W);
    double lambda = (5.0/8.0)*(epsilon*sqrt(2*W)*rho*((Cs+Cl)/2)*Tm)/LatentHeat;
    double tau = (15*rho*((Cs+Cl)/2)*Tm)/(4*mu*LatentHeat);
    double dt = 0.1;

    std::cout << "----------------------" << std::endl;
    std::cout << "Problem parameters:" << std::endl;
    std::cout << "W: " << W << std::endl;
    std::cout << "Delta: " << delta << std::endl;
    std::cout << "Lambda: " << lambda << std::endl;
    std::cout << "Tau: " << tau << std::endl;
    std::cout << "Timestep size dt: " << dt << std::endl;
    std::cout << "----------------------" << std::endl;

    //PhaseField Model functions
    auto gFunc = [](double phi){ return phi*phi*(1 - phi)*(1 - phi); };
    auto pFunc = [](double phi){ return pow(phi,3)*(10 - 15*phi + 6*pow(phi,2)); };
    auto gFuncDerivative = [](double phi){ return 2*phi*(1 - phi)*(1 - 2*phi); };
    auto gFuncDoubleDerivative = [](double phi){ return 2*(1 - 6*phi + 6*pow(phi,2)); };
    auto pFuncDerivative = [gFunc](double phi){ return 30*gFunc(phi); };
    auto pFuncDoubleDerivative = [gFuncDerivative](double phi){ return 30*gFuncDerivative(phi); };
    auto Cphi = [Cs, Cl, pFunc](double phi){ return Cs + (Cl - Cs)*pFunc(phi); };
    auto CphiDerivative = [Cs, Cl, pFuncDerivative](double phi){ return (Cl - Cs)*pFuncDerivative(phi); };
    auto Kphi = [Ks, Kl, pFunc](double phi){ return Ks + (Kl - Ks)*pFunc(phi); };
    auto KphiDerivative = [Ks, Kl, pFuncDerivative](double phi){ return (Kl - Ks)*pFuncDerivative(phi); };

    //Mesh
    double x1_ll = 0.0, x1_ul = 0.01;
    double Nel_x1 = 100;
    double h = (x1_ul - x1_ll)/Nel_x1;
    std::cout << "Mesh size h: " << h << std::endl;
    std::cout << "Delta: " << delta << std::endl;
    std::cout << "Timestep size dt: " << dt << std::endl;
    if(delta < 5*h){
        throw std::runtime_error("Delta condition not satisfied. Mesh is too coarse for the given epsilon. Please refine the mesh.");
    }

    //Mesh generation
    MeshGenerator<Nsd, Nne, BfOrder> meshGen(x1_ll, x1_ul, Nel_x1);
    Mesh<Nsd,Nne> mesh = meshGen.buildMesh();
    mesh.writeToFiles("mesh");

    std::cout << "Mesh built: " << mesh.Nnodes() << " nodes, " << mesh.Nelements() << " elements" << std::endl;
    std::cout << "--------------------" << std::endl;

    //Boundary Conditions on T
    BoundaryConditions<Nsd,Nne> bcs_T(mesh);
    for(unsigned int i = 0 ; i < mesh.Nnodes() ; i++){
        if(mesh.nodes[i].x1 == x1_ul){
            bcs_T.addDirischlet(i , 0 , Tcold);
        }
        if(mesh.nodes[i].x1 == x1_ll){
            bcs_T.addNeumann(i , 0 , 0.0);
        }
    }
    bcs_T.buildBCs();
    std::cout << "Boundary conditions on T:" << std::endl;
    bcs_T.printSummary(); //print a summary of the boundary conditions
    std::cout << "--------------------" << std::endl;

    //Boundary conditions on phi
    BoundaryConditions<Nsd,Nne> bcs_phi(mesh);
    for(unsigned int i = 0 ; i < mesh.Nnodes() ; i++){
        if(mesh.nodes[i].x1 ==  x1_ll || mesh.nodes[i].x1 == x1_ul){
            bcs_phi.addNeumann(i , 0 , 0.0);
        }
    }
    bcs_phi.buildBCs();
    std::cout << "Boundary conditions on phi:" << std::endl;
    bcs_phi.printSummary(); //print a summary of the boundary conditions
    std::cout << "--------------------" << std::endl;

    //Initialize the phi and T global vectors
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(mesh.Nnodes());
    Eigen::VectorXd T = Eigen::VectorXd::Zero(mesh.Nnodes());
    double X0 = 0.007;
    for(unsigned int i = 0 ; i < mesh.Nnodes() ; i++){
        phi(i) = 0.5*(1 - std::tanh((mesh.nodes[i].x1 - X0)/(delta)));
        T(i) = Tm;
    }

    //Problem physics stack
    QuadratureRule<Nsd,Nne>                     quadRule = Quadrature<Nsd,Nne>::gauss_legendre(quadOrder);
    ElementEvaluator<Nsd,Nne,BfOrder>           elemEval(mesh, quadRule, rho, W, lambda, LatentHeat, Tm, phi, T, gFunc, pFunc, gFuncDerivative, gFuncDoubleDerivative, pFuncDerivative, pFuncDoubleDerivative, Cphi, CphiDerivative, Kphi, KphiDerivative);
    Assembler<Nsd,Nne,BfOrder>                  assembler(mesh,elemEval);
    CoupledPhaseFieldSolver<Nsd,Nne,BfOrder>    solver(tau, epsilon, dt, NT, incrSteps, maxIter, epsilon_NR);
    OutputWriter<Nsd,Nne>                       writer(mesh, "output_data", "localhost", 8000);

    writer.writeAndSend(0,0,phi,T);

    std::cout << "Starting the solver..." << std::endl;
    solver.solve(phi, T, assembler, bcs_phi, bcs_T,
                [&](unsigned int timestep, double time, const Eigen::VectorXd& phi, const Eigen::VectorXd& T){
                    writer.writeAndSend(timestep, time, phi, T);
                }
    );
    std::cout << "Solve completed." << std::endl;

}