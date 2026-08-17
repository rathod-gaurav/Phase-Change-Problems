#pragma once

void Poisson::solve(){
    SolverControl solver_control(1000, 1e-6*system_rhs.l2_norm()); //tells the ConjugateGradient algorithm when to stop
    SolverCG<Vector<double>> solver(solver_control);

    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

    std::cout << solver_control.last_step() << " CG iterations needed to achieve convergence" << std::endl;
}