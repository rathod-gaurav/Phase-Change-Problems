#pragma once

void Poisson::make_grid(){
    GridGenerator::hyper_cube(triangulation, -1, 1); //domain is [-1,1]
    triangulation.refine_global(5); //refine the grid 5 times

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}