//grid
#include <deal.II/grid/tria.h> //for the triangulation class
#include <deal.II/grid/grid_generator.h> //generate a grid
#include <deal.II/grid/grid_out.h> //write the grid to a file

//dofs
#include <deal.II/dofs/dof_handler.h> //to associate dofs to vertices, lines, and cells
#include <deal.II/fe/fe_q.h> //this file has description of bilinear finite element
#include <deal.II/dofs/dof_tools.h> //several tools for manipulating dofs
#include <deal.II/fe/mapping_q1.h> //required to call one of the functions in dof_tools.h

#include <deal.II/lac/sparse_matrix.h> //sparse matrix class
#include <deal.II/lac/dynamic_sparsity_pattern.h> //dynamic sparsity pattern class
#include <deal.II/dofs/dof_renumbering.h> //algorithms for renumbering of dofs


#include <iostream>
#include <fstream>

using namespace dealii;

void generate_grid(){
    Triangulation<2> triangulation; //2D triangulation object
    GridGenerator::hyper_cube(triangulation); //fill the triangulation with a single cell for a square domain
    triangulation.refine_global(2); //refine the triangulation globally
    grid_out.write_vtu(triangulation, "grid.vtu"); //write the triangulation to a vtu file
    std::cout << "Grid written to grid.vtu" << std::endl;
}

void write_dof_locations(const DoFHandler<2> &dof_handler, const std::string &filename){ //function to write dof locations to a file : global node indes : x1 coordinate, x2 coordinate
    const std::map<types::global_dof_index, Point<2>> dof_locations_map = DofTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler);

    std::ofstream dof_locations_file(filename);
    DofTools::write_gnuplot_dof_support_point_info(dof_locations_file, dof_locations_map);
}

int main(){
    generate_grid(); 
    return 0;
}