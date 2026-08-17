#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
CahnHilliard<Nsd,BfOrder>::CahnHilliard():
    fe(BfOrder),
    dof_handler(triangulation)
{}

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::run(){
    std::cout << "I ran successfully" << std::endl;
    // make_grid();
    // setup_system();
    // assemble_system();
    // solve();
    // output_writer();
}
