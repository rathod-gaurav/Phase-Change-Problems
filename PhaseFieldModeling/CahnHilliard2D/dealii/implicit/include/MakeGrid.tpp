#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::make_grid(){
    GridGenerator::hyper_cube(triangulation, x_ll_, x_ul_); //domain is [x_ll,x_ul] in Nsd dimensions
    triangulation.refine_global(7); //refine the grid 7 times

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}