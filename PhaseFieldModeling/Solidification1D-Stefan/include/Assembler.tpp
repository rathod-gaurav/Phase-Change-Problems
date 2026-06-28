#pragma once

#include "Assembler.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
Assembler<Nsd,Nne,BfOrder>::Assembler(
    const Mesh<Nsd,Nne>& mesh,
    const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator
) :
    mesh_(mesh),
    elem_evaluator_(elem_evaluator)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
Eigen::MatrixXd Assembler<Nsd,Nne,BfOrder>::extractSubmatrix(
    const Eigen::MatrixXd& K,
    const std::vector<unsigned int>& rows,
    const std::vector<unsigned int>& cols) const {
        
    // 1. Allocate the destination dense matrix with the target dimensions
    Eigen::MatrixXd sub(rows.size(), cols.size());

    // 2. Map the entries directly from K into the submatrix
    for (unsigned int i = 0; i < rows.size(); ++i) {
        for (unsigned int j = 0; j < cols.size(); ++j) {
            // rows[i] maps the local row 'i' to the global row in K
            // cols[j] maps the local col 'j' to the global col in K
            sub(i, j) = K(rows[i], cols[j]);
        }
    }

    return sub;
}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void Assembler<Nsd,Nne,BfOrder>::assembleSystem(
    const Eigen::VectorXd& phi,
    const Eigen::VectorXd& T,
    Eigen::MatrixXd& Mphi,
    Eigen::MatrixXd& Kphi,
    Eigen::VectorXd& Rphi,
    Eigen::MatrixXd& MT,
    Eigen::MatrixXd& KT,
    Eigen::VectorXd& RT
){
    unsigned int Nt = mesh_.Nnodes();
    unsigned int Nel_t = mesh_.Nelements();

    Eigen::MatrixXd Mphi_e = Eigen::MatrixXd::Zero(Nne,Nne);
    Eigen::MatrixXd Kphi_e = Eigen::MatrixXd::Zero(Nne,Nne);
    Eigen::VectorXd Rphi_e = Eigen::VectorXd::Zero(Nne);
    Eigen::MatrixXd MT_e = Eigen::MatrixXd::Zero(Nne,Nne);
    Eigen::MatrixXd KT_e = Eigen::MatrixXd::Zero(Nne,Nne);
    Eigen::VectorXd RT_e = Eigen::VectorXd::Zero(Nne);

    for(unsigned int e = 0 ; e < Nel_t ; e++){
        elem_evaluator_.compute_element(
            e,
            Mphi_e,
            Kphi_e,
            Rphi_e,
            MT_e,
            KT_e,
            RT_e
        );

        //Assemble
        unsigned int Aglobal_e = mesh_.elements[e].node[0];
        Mphi.block(Aglobal_e,Aglobal_e,Nne,Nne) += Mphi_e;
        Kphi.block(Aglobal_e,Aglobal_e,Nne,Nne) += Kphi_e;
        Rphi.segment(Aglobal_e,Nne) += Rphi_e;
    }
}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void Assembler<Nsd,Nne,BfOrder>::partition(
    Eigen::MatrixXd& M,
    Eigen::MatrixXd& K,
    Eigen::VectorXd& R,
    Eigen::VectorXd& solution,
    BoundaryConditions<Nsd,Nne>& bcs,
    Eigen::MatrixXd& MUU,
    Eigen::MatrixXd& MUD,
    Eigen::MatrixXd& KUU,
    Eigen::MatrixXd& KUD,
    Eigen::VectorXd& RU
){
    const auto& dirischletIndexes = bcs.getDirischletIndexes();
    const auto& unknownIndexes = bcs.getUnknownIndexes();

    MUU = extractSubmatrix(M, unknownIndexes, unknownIndexes);
    MUD = extractSubmatrix(M, unknownIndexes, dirischletIndexes);
    KUU = extractSubmatrix(K, unknownIndexes, unknownIndexes);
    KUD = extractSubmatrix(K, unknownIndexes, dirischletIndexes);

    RU.resize(unknownIndexes.size());
    for(unsigned int i = 0 ; i < unknownIndexes.size() ; i++){
        RU(i) = Rglobal(unknownIndexes[i]);
    }
}