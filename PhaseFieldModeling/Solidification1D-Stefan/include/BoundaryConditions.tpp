#pragma once
#include <BoundaryConditions.hpp>

template <unsigned int Nsd, unsigned int Nne>
BoundaryConditions<Nsd,Nne>::BoundaryConditions(
    const Mesh<Nsd,Nne>& mesh
):
    mesh_(mesh),
    totalDOFs_(mesh.Nnodes())
{}

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditions<Nsd,Nne>::addDirischlet(unsigned int node, int dof, double value){
    unsigned int globalNodeIndex = node + dof;
    dirischletVals_[globalNodeIndex] = value;
};

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditions<Nsd,Nne>::addNeumann(unsigned int node, int dof, double value){
    unsigned int globalNodeIndex = node + dof;
    if(dirischletVals_.find(globalNodeIndex) != dirischletVals_.end()){
        throw std::runtime_error("Cannot add Neumann boundary condition to a node that already has a Dirichlet boundary condition.");
    }
    // Store the Neumann boundary condition (implementation details depend on your specific needs)
    if(globalNodeIndex == 0){ //first node || only customised for 1D problem
        neumannVals_[globalNodeIndex] = solution_[globalNodeIndex + 1] - value*(mesh_.nodes[globalNodeIndex + 1].x1 - mesh_.nodes[globalNodeIndex].x1); //assuming uniform mesh and linear elements
    }
    if(globalNodeIndex == totalDOFs_ - 1){ //last node || only customised for 1D problem
        neumannVals_[globalNodeIndex] = solution_[globalNodeIndex - 1] + value*(mesh_.nodes[globalNodeIndex].x1 - mesh_.nodes[globalNodeIndex - 1].x1); //assuming uniform mesh and linear elements
    }
}

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditions<Nsd,Nne>::buildBCs(){
    isDirischlet_.assign(totalDOFs_, false); //resize the isDirichlet vector to the total number of degrees of freedom and initialize all values to false
    isNeumann_.assign(totalDOFs_, false); //resize the isNeumann vector to the total number of degrees of freedom and initialize all values to false
    dirischletIndexes_.clear(); //clear the dirischletIndexes vector to prepare for building the boundary conditions
    neumannIndexes_.clear(); //clear the neumannIndexes vector to prepare for building the boundary conditions
    unknownIndexes_.clear(); //clear the unknownIndexes vector to prepare for building the boundary conditions

    //mark dirischlet indexes
    for(const auto& [dof,val] : dirischletVals_){
        isDirischlet_[dof] = true; //mark the degrees of freedom that are subject to dirischlet boundary conditions as true in the isDirichlet vector
        dirischletIndexes_.push_back(dof); //add the degree of freedom to the dirischletIndexes vector
    }
    //mark neumann indexes
    for(const auto& [dof,val] : neumannVals_){
        isNeumann_[dof] = true; //mark the degrees of freedom that are subject to neumann boundary conditions as true in the isNeumann vector
        neumannIndexes_.push_back(dof); //add the degree of freedom to the neumannIndexes vector
    }
    //everything else is free
    for(unsigned int i = 0 ; i < totalDOFs_ ; i++){
        if(!isDirischlet_[i] && !isNeumann_[i]){ //check if the degree of freedom is not subject to dirischlet or neumann boundary conditions
            unknownIndexes_.push_back(i); //add the degree of freedom to the unknownIndexes vector if it is not subject to dirischlet boundary conditions
        }
    }
};

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditions<Nsd,Nne>::applyToSolution(Eigen::VectorXd& solution, double incrementFraction) const{
    for(const auto& [dof,val] : dirischletVals_){
        solution(dof) = val * incrementFraction; //apply the dirischlet boundary conditions to the solution vector based on the current increment fraction
    }
    for(const auto& [dof,val] : neumannVals_){
        solution(dof) = val; //apply the neumann boundary conditions to the solution vector based on the current increment fraction
    }
};

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditions<Nsd,Nne>::printSummary() const{
    std::cout << "Boundary Conditions Summary:" << std::endl;
    std::cout << "Total DOFs: " << totalDOFs_ << std::endl;
    std::cout << "Dirischlet DOFs: " << dirischletIndexes_.size() << std::endl;
    std::cout << "Neumann DOFs: " << neumannIndexes_.size() << std::endl;
    std::cout << "Unknown DOFs: " << unknownIndexes_.size() << std::endl;

    for(const auto& [dof,val] : neumannVals_){
        std::cout << dof << " : " << val << std::endl;
    }
};