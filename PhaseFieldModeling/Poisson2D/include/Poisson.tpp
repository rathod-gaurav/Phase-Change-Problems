#pragma once

Poisson::Poisson() : 
    fe(1), //bilinear polynomial lagrange basis functions
    dof_handler(triangulation) //associae the dof_handler variable to the triangulation we use
{}

void Poisson::run(){
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
}