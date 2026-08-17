#pragma once

template<unsigned int Nsd, unsigned int BfOrder>
class CahnHilliard{
    public:
        CahnHilliard(); //constructor
        void run(); //execute the problem
    
    private:
        // void make_grid();
        // void setup_system();
        // void compute_element();
        // void assemble_system();
        // void solve();
        // void output_writer();

        Triangulation<Nsd> triangulation;
        const FE_Q<Nsd> fe;
        DoFHandler<Nsd> dof_handler;

};


#include "CahnHilliard.tpp"

