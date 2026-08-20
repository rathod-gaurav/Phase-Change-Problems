#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
class OutputWriter{
    public:
        explicit OutputWriter(const std::string& output_dir);

        void write_vtu_and_pvd(DoFHandler<Nsd>& dof_handler, Vector<double>& solution1, Vector<double>& solution2, const unsigned int timestep, std::ofstream &pvd_output);

        void initiate_pvd(std::ofstream &pvd_output);
        void finish_pvd(std::ofstream &pvd_output);

        const std::string& get_output_dir() const { return output_dir_; }//return output directory location

    private:
        std::string output_dir_;
};

#include "OutputWriter.tpp"