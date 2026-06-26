#pragma once

#include <Mesh.hpp>
#include <map>
#include <vector>
#include <Eigen/Dense>

template <unsigned int Nsd, unsigned int Nne>
class BoundaryConditions{
    public:
        BoundaryConditions(const Mesh<Nsd,Nne>& mesh); //default constructor

        //function to store dirischlet boundary conditions map
        void addDirischlet(unsigned int node, int dof, double value);
        //function to store neumann boundary conditions map
        void addNeumann(unsigned int node, int dof, double value);

        void buildBCs(); //function to build the boundary conditions

        void applyToSolution(Eigen::VectorXd& solution, double incrementFraction) const; //function to apply the boundary conditions to the solution vector u based on the current increment fraction (used for incremental loading)

        //Query methods
        const std::vector<unsigned int>& getUnknownIndexes() const { return unknownIndexes_; } //function to return indexes of the unknown degrees of freedom
        const std::vector<unsigned int>& getDirischletIndexes() const { return dirischletIndexes_; } //function to return the indexes of the dirischlet degrees of freedom
        const std::vector<unsigned int>& getNeumannIndexes() const { return neumannIndexes_; } //function to return the indexes of the neumann degrees of freedom

        bool isDirischlet(unsigned int globalDOF) const {return isDirischlet_[globalDOF];} //function to check if a given global degree of freedom is subject to dirischlet boundary conditions
        bool isNeumann(unsigned int globalDOF) const {return isNeumann_[globalDOF];} //function to check if a given global degree of freedom is subject to neumann boundary conditions

        void printSummary() const; //function to print a summary of the boundary conditions
    
    private:
        const Mesh<Nsd,Nne>& mesh_; //reference to the mesh object
        unsigned int totalDOFs_;

        std::map<unsigned int, double> dirischletVals_;
        std::vector<bool> isDirischlet_;
        std::vector<unsigned int> dirischletIndexes_;
        std::vector<unsigned int> unknownIndexes_;
        std::map<unsigned int, double> neumannVals_;
        std::vector<bool> isNeumann_;
        std::vector<unsigned int> neumannIndexes_;
};

#include "BoundaryConditions.tpp"