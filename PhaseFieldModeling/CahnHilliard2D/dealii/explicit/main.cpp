//grid
#include <deal.II/grid/tria.h> //for the triangulation class
#include <deal.II/grid/grid_generator.h> //generate a grid
#include <deal.II/grid/grid_out.h> //write the grid to a file

//dofs
#include <deal.II/dofs/dof_handler.h> //to associate dofs to vertices, lines, and cells
#include <deal.II/fe/fe_q.h> //this file has description of bilinear finite element
#include <deal.II/dofs/dof_tools.h> //several tools for manipulating dofs
#include <deal.II/fe/mapping_q1.h> //required to call one of the functions in dof_tools.h
#include <deal.II/dofs/dof_renumbering.h> //algorithms for renumbering of dofs

//sparse matrices
#include <deal.II/lac/sparse_matrix.h> //sparse matrix class
#include <deal.II/lac/dynamic_sparsity_pattern.h> //dynamic sparsity pattern class

#include <iostream>
#include <fstream>

using namespace dealii;

//generate the mesh grid and output it into a gnuplot file
void generate_grid(Triangulation<2>& triangulation){
    GridGenerator::hyper_cube(triangulation); //fill the triangulation with a single cell for a square domain
    triangulation.refine_global(2); //refine the triangulation globally
    std::ofstream mesh_file("mesh.gnuplot");
    GridOut().write_gnuplot(triangulation, mesh_file); //write the triangulation to a gnuplot file
    std::cout << "Mesh written to mesh.gnuplot" << std::endl;
}

//function to write dof locations to a file : global node indes : x1 coordinate, x2 coordinate
void write_dof_locations(const DoFHandler<2> &dof_handler, const std::string &filename){ //DofHandler object knows where each Dof is located
    const std::map<types::global_dof_index, Point<2>> dof_locations_map = DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler); //map_dofs_to_support_points returns a list of locations in the formof a map

    std::ofstream dof_locations_file(filename);
    DoFTools::write_gnuplot_dof_support_point_info(dof_locations_file, dof_locations_map); //this function writes the dof location information to a file in a format that is understandable to the gnuplot program
}

//creation of a dof handler
//associate degrees of freedom to each vertex (or line, a cell in case we are using higher order elements) -- to later describe matrices and vectors which describe the finite element field on the triangulation
void distribute_dofs(DoFHandler<2> &dof_handler){
    const FE_Q<2> finite_elements(1);  //FE_Q is a derived class that describes Lagrange elements. Its constructor takes one argument that specifies the polynomial degree of element. Here, 1 indicates a bilinear element
    dof_handler.distribute_dofs(finite_elements); //we first create a object of FE_Q class and pass it on to the DofHandler object to allocate storage for the degrees of freedom
    write_dof_locations(dof_handler, "dof_locations.gnuplot"); //we have now associated a degree of freedom with a global node number to each vertex. this line of code outputs the information to a file using the function declared above
    
    //Sparsity pattern - we first need to create a structure which we use to store the places of nonzero elements 
    //this can later be used by one or more sparse matrix objects that store the values of the entries in the locations provided by this sparsity pattern
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(), dof_handler.n_dofs()); //this uses an internal data structure that we can later copy into the SparsityPattern object without much overhead. In order to initialize this data structue, we first need to give it the size of the matrix- which in this case will be square with as many rows and columns as the number of Dofs on the grid
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern); //this fills the dynamic_sparsity_pattern object with with the places where nonzero elements will be located
    //now we are ready to create the actual sparsity pattern that we could later use for our matrices
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dynamic_sparsity_pattern); //the sparsity_pattern contains the data already assembled in the DynamicSparsityPattern
    //write sparsity_pattern results to a file
    std::ofstream out("sparsity_pattern.svg");
    sparsity_pattern.print_svg(out);
}

int main(){
    Triangulation<2> triangulation; //2D triangulation object
    generate_grid(triangulation);
    
    DoFHandler<2> dof_handler(triangulation);
    distribute_dofs(dof_handler);
 
    return 0;
}
