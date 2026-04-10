// #stefan problem 1D - 1 phase
#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <fstream>
#include <string>

struct Node{
    double x1; //location of the node in the x1 direction
};

template<unsigned int Nne>
struct Element{
    unsigned int node[Nne]; //global node numbers of the element
};

struct ProblemData {
    double Kl, rho, Cl, alpha, LatentHeat;
    double T0, T1, Tm, h, fint;
    double L, x1_ll, x1_ul;
    unsigned int Nt, Nel_t, quadRule;
    double dx1, he, Jac, Jac_inv;
    std::vector<Node> nodes;
    std::vector<Element<2>> elements;
    std::vector<double> qpoints, qweights;
    double alpha_t, dt;
};

ProblemData pd;

double xi_at_node(unsigned int node){ //function to return xi1 and xi2 for given node A
        double xi;
        switch(node){
            case 0:
                xi = -1.0;
                break;
            case 1:
                xi = 1.0;
                break;
            default:
                throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
        }
        return xi;
};

double basis_function(unsigned int node, double xi){
        double xi_node = xi_at_node(node);
        double value = 0.50*(1 + xi*xi_node);
        return value;
};

double basis_gradient(unsigned int node, double xi){
    double xi_node = xi_at_node(node);
    double basis_gradient_xi = 0.50*xi_node;
    return basis_gradient_xi;
}

struct QuadratureRule {
    std::vector<double> points;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(unsigned int n) {
    QuadratureRule rule;

    switch(n) {
        case 1:
            rule.points  = { 0.0 };
            rule.weights = { 2.0 };
            break;

        case 2:
            rule.points  = { -0.5773502691896257,  0.5773502691896257 };
            rule.weights = {  1.0,                 1.0 };
            break;

        case 3:
            rule.points  = { -0.7745966692414834, 0.0, 0.7745966692414834 };
            rule.weights = {  0.5555555555555556, 0.8888888888888888, 0.5555555555555556 };
            break;

        case 4:
            rule.points  = { -0.8611363115940526, -0.3399810435848563,
                              0.3399810435848563,  0.8611363115940526 };
            rule.weights = {  0.3478548451374539,  0.6521451548625461,
                              0.6521451548625461,  0.3478548451374539 };
            break;

        case 5:
            rule.points  = { -0.9061798459386640, -0.5384693101056831,
                              0.0,
                              0.5384693101056831,  0.9061798459386640 };
            rule.weights = {  0.2369268850561891,  0.4786286704993665,
                              0.5688888888888889,  0.4786286704993665,
                              0.2369268850561891 };
            break;

        case 6:
            rule.points  = { -0.9324695142031521, -0.6612093864662645,
                             -0.2386191860831969,  0.2386191860831969,
                              0.6612093864662645,  0.9324695142031521 };
            rule.weights = {  0.1713244923791704,  0.3607615730481386,
                              0.4679139345726910,  0.4679139345726910,
                              0.3607615730481386,  0.1713244923791704 };
            break;

        default:
            throw std::invalid_argument("Gauss-Legendre quadrature not implemented for this n");
    }

    return rule;
}


Eigen::MatrixXd extractSubmatrix(const Eigen::MatrixXd& OriginalMatrix , const std::vector<unsigned int> rows , const std::vector<unsigned int> cols){
    Eigen::MatrixXd subMatrix(rows.size(), cols.size());

    for(unsigned int i = 0 ; i < rows.size() ; i++){
        for(unsigned int j = 0 ; j < cols.size() ; j++){
            subMatrix(i,j) = OriginalMatrix(rows[i],cols[j]);
        }
    }
    return subMatrix;
}

void writeArrayToOutFile(const std::string& filename, const Eigen::VectorXd& vec, int size) {
    // Open the file
    std::ofstream outFile(filename);

    // In C++, checking the stream object itself (or !outFile.fail()) 
    // is the standard way to verify it's ready for writing.
    if (outFile) {
        for (int i = 0; i < size; ++i) {
            // Using vec(i) for Eigen vector access
            outFile << vec(i) << "\n";
        }
        outFile.close();
        // std::cout << "Data written to " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}

Eigen::VectorXd SolveHeatEqn(unsigned int Nt_active){
    unsigned int Nt = Nt_active;
    unsigned int Nel_t = Nt_active - 1;
    //Heat equation solving starts
    Eigen::MatrixXd Kglobal = Eigen::MatrixXd::Zero(Nt, Nt); //global stiffness matrix
    Eigen::MatrixXd Mglobal = Eigen::MatrixXd::Zero(Nt, Nt); //global mass matrix
    Eigen::VectorXd Fglobal = Eigen::VectorXd::Zero(Nt); //global force vector

    //Global node locations having Neumann Boundary conditions specified on them
    std::vector<unsigned int> nodeLocationsN;
    for(int i = 0; i < Nt; i++){
        if(nodes[i].x1 == x1_ul){
            nodeLocationsN.push_back(i);
        }
    }
    std::vector<bool> isNeumann(Nt,false);
    for(unsigned int& nodeLocation : nodeLocationsN){
        isNeumann[nodeLocation] = true;
    }

    std::vector<unsigned int> nodeLocationsD; //knowns (dirichlet node locations)
    nodeLocationsD.push_back(0); //dirischlet boundary at x = 0
    nodeLocationsD.push_back(Nt_active - 1); //dirischlet boundary at x = s (interface position)

    std::vector<bool> isDirischlet(Nt,false);
    for(unsigned int& nodeLocation : nodeLocationsD){
        isDirischlet[nodeLocation] = true;
    }

    Eigen::VectorXd dirischletVal(nodeLocationsD.size());
    dirischletVal[0] = T1; //high temperature BC at the left boundary
    dirischletVal[1] = Tm; //melt temperature BC at the interface

    Eigen::VectorXd dirischletValDot = Eigen::VectorXd::Zero(nodeLocationsD.size()); //rate of change of temperature on the dirischlet boundary

    //assembly
    for(unsigned int e = 0 ; e < Nel_t ; e++){
        Eigen::MatrixXd Klocal = Eigen::MatrixXd::Zero(Nne, Nne); //local stiffness matrix
        Eigen::MatrixXd Mlocal = Eigen::MatrixXd::Zero(Nne, Nne); //local mass matrix
        Eigen::VectorXd Flocal_int = Eigen::VectorXd::Zero(Nne); //local force vector
        Eigen::VectorXd Flocal_h = Eigen::VectorXd::Zero(Nne); //local force vector for neumann boundary condition

        for(unsigned int A = 0 ; A < Nne ; A++){
            for(unsigned int B = 0 ; B < Nne ; B++){
                for(unsigned int I = 0 ; I < quadRule ; I++){
                    double xi = points[I];
                    double w = weights[I];

                    double bfgradientA_xi = basis_gradient(A, xi);
                    double bfgradientB_xi = basis_gradient(B, xi);

                    double Kvalue = bfgradientA_xi * Kl * bfgradientB_xi * Jac_inv * w;
                    Klocal(A,B) += Kvalue;

                    double Mvalue = basis_function(A, xi) * rho * Cl * basis_function(B, xi) * Jac * w;
                    Mlocal(A,B) += Mvalue;
                }
            }
            for(unsigned int I = 0 ; I < quadRule ; I++){
                double xi = points[I];
                double w = weights[I];

                double bfA = basis_function(A, xi);

                Flocal_int(A) += bfA * fint * Jac * w; //contribution to the local force vector from the initial condition
            }
            if(isNeumann[elements[e].node[A]]){
                Flocal_h(A) += basis_function(A, 1.0) * h * he/2.0; //contribution to the local force vector from the neumann boundary condition
            }
        }

        //assemble local contributions into global matrices and vector
        for(int A = 0 ; A < Nne ; A++){
            int Aglobal = elements[e].node[A];
            for(int B = 0 ; B < Nne ; B++){
                int Bglobal = elements[e].node[B];
                Kglobal(Aglobal,Bglobal) += Klocal(A,B);
                Mglobal(Aglobal,Bglobal) += Mlocal(A,B);
            }
            Fglobal(Aglobal) += Flocal_int(A) - Flocal_h(A); //this now contains contribution from neumann boundary condition as well
        }
    }

    //Applying Dirischlet Boundary Conditions
    std::vector<unsigned int> nodeLocationsU; //unknown node locations - node locations where fleid value is unknown
    for(int i = 0 ; i < Nt ; i++){
        if(!isDirischlet[i]){
            nodeLocationsU.push_back(i);
        }
    }

    Eigen::MatrixXd KUU = extractSubmatrix(Kglobal, nodeLocationsU, nodeLocationsU); //extract from Kglobal - only rows and columns pertaining to unknown node locations
    Eigen::MatrixXd KUD = extractSubmatrix(Kglobal, nodeLocationsU, nodeLocationsD); //extract from Kglobal - only columns corresponding to Dirischlet node locations, for rows corresponding to unknown node locations

    Eigen::MatrixXd MUU = extractSubmatrix(Mglobal, nodeLocationsU, nodeLocationsU); //extract from Mglobal - only rows and columns pertaining to unknown node locations
    Eigen::MatrixXd MUD = extractSubmatrix(Mglobal, nodeLocationsU, nodeLocationsD); //extract from Mglobal - only columns corresponding to Dirischlet node locations, for rows corresponding to unknown node locations

    Eigen::VectorXd FU(nodeLocationsU.size()); //extract from Fglobal - only rows corresponding to unknown node locations
    for(int i = 0; i < nodeLocationsU.size(); i++){
        FU(i) = Fglobal(nodeLocationsU[i]);
    }
    
    Eigen::VectorXd F(FU.size()); //create final forcing function vector
    F = FU - KUD*dirischletVal - MUD*dirischletValDot;
    
    //initial condition
    Eigen::VectorXd D0 = Eigen::VectorXd::Zero(Nt);
    Eigen::VectorXd V0 = Eigen::VectorXd::Zero(Nt);
    for(int i = 0 ; i < Nt ; i++){
        D0(i) = T0; //initial temperature at all nodes is T0
    }

    Eigen::VectorXd Dn(nodeLocationsU.size());
    Eigen::VectorXd Vn(nodeLocationsU.size());
    for(int i = 0; i < nodeLocationsU.size() ; i++){
        Dn(i) = D0(nodeLocationsU[i]);
    }

    Eigen::LDLT<Eigen::MatrixXd> solver1(MUU);
    Vn = solver1.solve(F - KUU*Dn); //find V0 to initiate the time stepping process
    // cout << Vn << endl;
    Eigen::MatrixXd lhs = MUU + alpha_t*dt*KUU;
    Eigen::LDLT<Eigen::MatrixXd> solver(lhs);

    //Final solution stored in D
    Eigen::VectorXd D = Eigen::VectorXd::Zero(Nt);

    Eigen::VectorXd predictor = Dn + dt*(1 - alpha_t)*Vn;
    Eigen::VectorXd rhs = alpha_t*dt*F + MUU*predictor;

    Eigen::VectorXd Dnp1 = solver.solve(rhs);

    Dn = Dnp1;
    Vn = (Dnp1 - predictor)/(alpha_t*dt);

        // apply boundary conditions to obtain final solution
    for(int i = 0 ; i < nodeLocationsD.size() ; i++){
        int indexD = nodeLocationsD[i];
        D[indexD] = dirischletVal[i];
    }
    for(int i = 0 ; i < nodeLocationsU.size() ; i++){
        int indexD = nodeLocationsU[i];
        D[indexD] = Dn[i];
    }

    return D;
}

int main(){
    unsigned int Nd = 1; //1D problem
    constexpr int Nne = 2; //number of nodes per element
    unsigned int quadRule = 2; //number of quadrature points per element
    
    //problem variables
    double T0 = 0.0; //initial temperature
    double T1 = 25.0; //boundary temperature at x=0
    double h = 0.0; //neumann boundary
    double fint = 0.0; //internal forcing function

    double Kl = 0.564; //thermal conductivity of the liquid phase
    double rho = 1000.0; //density of the liquid phase
    double Cl = 4186.8; //specific heat capacity of the liquid phase
    double alpha = Kl/(rho*Cl); //thermal diffusivity of the liquid phase

    double LatentHeat = 333400.0; //latent heat of fusion for the phase change material

    //domain
    double L = 0.01; //length of the domain
    double x1_ll = 0.0; //left boundary of the domain
    double x1_ul = L; //right boundary of the domain

    //create mesh
    unsigned int Nt = 100; //total number of nodes in the mesh
    double dx1 = (x1_ul - x1_ll)/(Nt-1); //spacing between nodes
    unsigned int Nel_t = Nt - 1; //total number of elements in the mesh

    //global node locations
    std::vector<Node> nodes;
    nodes.reserve(Nt);
    for(unsigned int i=0; i<Nt; ++i){
        Node n;
        n.x1 = x1_ll + i*dx1;
        nodes.push_back(n);
    }

    using Element1D = Element<Nne>;
    std::vector<Element1D> elements;
    elements.reserve(Nel_t);
    for(unsigned int i=0; i<Nel_t; ++i){
        Element1D e;
        e.node[0] = i; //global node number of the first node of the element
        e.node[1] = i+1; //global node number of the second node of the element
        elements.push_back(e);
    }

    std::ofstream points_file("points.txt");
    for(auto& node : nodes){
        points_file << node.x1 << "\n";
    }

    std::ofstream linears_file("linears.txt");
    for(auto& elem : elements){
        linears_file << elem.node[0] << " " << elem.node[1] << "\n";
    }

    //Quadrature rule
    QuadratureRule q = gauss_legendre(quadRule);
    std::vector<double> points(quadRule), weights(quadRule);
    points = q.points;
    weights = q.weights;

    double he = nodes[1].x1 - nodes[0].x1;//node spacing in x1 direction : to be used for computation of neumann boundary condition term

    double Jac = he/2.0; //Jacobian of the transformation from reference element to physical element
    double Jac_inv = 1.0/Jac;

    //initialize the interface just after the leftmost node
    double s = he;

    //time integration using backward Euler method
    double alpha_t = 1.0; //weighting parameter for time integration scheme - 1.0 for backward Euler
    double dt = 1.0; //time step size
    unsigned int NT = 100; //number of time steps

    Eigen::VectorXd D_full = Eigen::VectorXd::Zero(Nt);

    for(unsigned int n = 0 ; n < NT ; n++){
        
        unsigned int InterfaceNode = static_cast<int>(s/he) + 1;
        unsigned int Nt_active = InterfaceNode + 1;//active number of nodes for solving heat equation

        Eigen::VectorXd D_active = SolveHeatEqn(Nt_active);

        D_full.head(Nt_active) = D_active;
        D_full.tail(Nt - Nt_active).setConstant(T0);

        std::string filename = "solutions/solution_t_" + std::to_string(n) + ".out";
        writeArrayToOutFile(filename, D_full, Nt);

        double dTdx = (D_full(InterfaceNode) - D_full(InterfaceNode - 1))/he;
        double InterfaceSpeed = -1*(Kl/(rho*LatentHeat))*dTdx;

        s += InterfaceSpeed*dt;
    }

}