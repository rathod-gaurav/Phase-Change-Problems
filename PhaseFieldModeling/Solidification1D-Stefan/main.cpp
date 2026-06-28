#include <iostream>
#include <stdexcept>
#include <math.h>
#include <MeshGenerator.hpp>
#include <Quadrature.hpp>
#include <BoundaryConditions.hpp>
#include <ElementEvaluator.hpp>
#include <Assembler.hpp>

int main(){

    constexpr unsigned int Nsd = 1;
    constexpr unsigned int BfOrder = 1;
    constexpr unsigned int Nne = 2;

    //Quadrature order
    unsigned int quadOrder = 2;

    std::cout << "Solving " << Nsd << "D problem with " << Nne << " node elements and " << BfOrder << " order basis functions." << std::endl;

    //Problem parameters
    double rho = 1000.0; //Kg/m3
    double Cs = 2000.0; //J/Kg-K
    double Cl = 4000.0; //J/Kg-K 
    double Ks = 2.0; //W/m-K
    double Kl = 0.6; //W/m-K
    double LatentHeat = 1e5; //J/Kg
    double Tm = 0.0; //K
    double Tcold = -10.0; //K

    double sigma = 0.01; //J/m2
    double mu = 1e-2; //m/s-K

    //Assumptions
    double epsilon = 4*1e-4;
    //Derived quantities
    double W = 2.0;
    double delta = epsilon*sqrt(2/W);
    double lambda = (5/8)*(epsilon*sqrt(2*W)*rho*((Cs+Cl)/2)*Tm)/LatentHeat;
    double tau = (15*rho*((Cs+Cl)/2)*Tm)/(4*mu*LatentHeat);

    //PhaseField Model functions
    auto gFunc = [](double phi){ return phi*phi*(1 - phi)*(1 - phi); };
    auto pFunc = [](double phi){ return pow(phi,3)*(10 - 15*phi + 6*pow(phi,2)); };
    auto gFuncDerivative = [](double phi){ return 2*phi*(1 - phi)*(1 - 2*phi); };
    auto pFuncDerivative = [gFunc](double phi){ return 30*gFunc(phi); };
    auto Cphi = [Cs, Cl, pFunc](double phi){ return Cs + (Cl - Cs)*pFunc(phi); };
    auto Kphi = [Ks, Kl, pFunc](double phi){ return Ks + (Kl - Ks)*pFunc(phi); };

    //Mesh
    double x1_ll = 0.0, x1_ul = 0.01;
    double Nel_x1 = 100;
    double h = (x1_ul - x1_ll)/Nel_x1;
    std::cout << "Mesh size h: " << h << std::endl;
    std::cout << "Delta: " << delta << std::endl;
    if(delta < 4*h){
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
        if(mesh.nodes[i].x1 == x1_ll){
            bcs_T.addDirischlet(i , 0 , Tcold);
        }
        if(mesh.nodes[i].x1 == x1_ul){
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

    //Problem physics stack
    QuadratureRule<Nsd,Nne>                 quadRule = Quadrature<Nsd,Nne>::gauss_legendre(quadOrder);
    ElementEvaluator<Nsd,Nne,BfOrder>       elemEval(mesh, quadRule, rho, W, lambda, LatentHeat, Tm, phi, T, gFunc, pFunc, gFuncDerivative, pFuncDerivative, Cphi, Kphi);
    Assembler<Nsd,Nne,BfOrder>              assembler(mesh,elemEval);
    
}