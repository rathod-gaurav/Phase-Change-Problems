#pragma once

void Poisson::output_results() const {
    DataOut<2> data_out; //an object that knows about the output formats

    //we tell the data_out object of which DoFHandler object to yse, and which solution vector to use
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    data_out.build_patches();

    const std::string filename = "solution.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    std::cout << "Output written to " << filename << std::endl;
}