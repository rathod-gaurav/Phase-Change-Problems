#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
OutputWriter<Nsd,BfOrder>::OutputWriter(const std::string& output_dir)
: output_dir_(output_dir)
{
    std::filesystem::create_directories(output_dir_);
}

template <unsigned int Nsd, unsigned int BfOrder>
void OutputWriter<Nsd,BfOrder>::write_vtu_and_pvd(DoFHandler<Nsd>& dof_handler, Vector<double>& solution1, Vector<double>& solution2, const unsigned int timestep, const double dt, std::ofstream &pvd_output){
    DataOut<Nsd> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution1, "concentration");
    data_out.add_data_vector(solution2, "chemical potential");

    data_out.build_patches();

    std::string filename = output_dir_ + "/solution-" + dealii::Utilities::int_to_string(timestep, 4) + ".vtu"; //4 is for padding integer names in the file names
    std::ofstream vtu_output(filename);
    data_out.write_vtu(vtu_output);
    pvd_output << "    <DataSet timestep=\"" << timestep << "\" group=\"\" part=\"0\" file=\"" << filename << "\"/>\n"; //append to pvd file
}

template <unsigned int Nsd, unsigned int BfOrder>
void OutputWriter<Nsd,BfOrder>::initiate_pvd(std::ofstream &pvd_output){
    pvd_output << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"Collection\" version=\"0.1\" "
               << "byte_order=\"LittleEndian\" "
               << "compressor=\"vtkZLibDataCompressor\">\n"
               << "  <Collection>\n";
}

template <unsigned int Nsd, unsigned int BfOrder>
void OutputWriter<Nsd,BfOrder>::finish_pvd(std::ofstream &pvd_output){
    pvd_output << "  </Collection>\n" << "</VTKFile>\n";
    pvd_output.close();
}

