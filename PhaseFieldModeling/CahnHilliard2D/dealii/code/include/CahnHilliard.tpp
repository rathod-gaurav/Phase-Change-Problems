#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
CahnHilliard<Nsd,BfOrder>::CahnHilliard(
    double x_ll, double x_ul
):
    x_ll_(x_ll),
    x_ul_(x_ul),
    fe(BfOrder),
    dof_handler(triangulation)
{}

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::run(){
    std::cout << "I ran successfully" << std::endl;
    make_grid();
    // setup_system();
    // assemble_system();
    // solve();
    // output_writer();
}
