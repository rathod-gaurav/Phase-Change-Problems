#pragma once

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
Assembler<Nsd,Nne,BfOrder>::Assembler(
    const Mesh<Nsd,Nne>& mesh, 
    const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator
) : 
    mesh_(mesh), 
    elem_evaluator_(elem_evaluator) 
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void Assembler<Nsd,Nne,BfOrder>::assembleSystem(
    Eigen::MatrixXd<double>& Mglobal, //global mass matrix (Nnodes x Nnodes matrix)
    Eigen::MatrixXd<double>& Kglobal, //global stiffness  matrix (Nnodes x Nnodes matrix)
    Eigen::VectorXd& Bglobal //global nonlinear bulk free energy vector (Nnodes x 1 vector)
) const {
    unsigned int Nt = mesh_.Nnodes();
    unsigned int Nel_t = mesh_.Nelements();
    Mglobal = Eigen::MatrixXd::Zero(Nt,Nt);
    Kglobal = Eigen::MatrixXd::Zero(Nt,Nt);
    Bglobal = Eigen::VectorXd::Zero(Nt);

    Eigen::MatrixXd M_local = Eigen::MatrixXd::Zero(Nne,Nne);
    Eigen::MatrixXd K_local = Eigen::MatrixXd::Zero(Nne,Nne);
    Eigen::VectorXd B_local = Eigen::VectorXd::Zero(Nne);

    for(unsigned int e = 0 ; e < Nel_t ; e++){
        elem_evaluator_.computeElement(
            e,
            M_local,
            K_local,
            B_local
        );

        //Assemble
        unsigned int Aglobal_e = mesh_.elements[e].node[0];
        Mglobal.block(Aglobal_e,Aglobal_e,Nne,Nne) += M_local;
        Kglobal.block(Aglobal_e,Aglobal_e,Nne,Nne) += K_local;
        Bglobal.segment(Aglobal_e,Nne) += B_local;
    }
}