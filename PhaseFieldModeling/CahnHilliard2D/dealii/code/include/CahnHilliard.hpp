#pragma once

template<unsigned int Nsd, unsigned int BfOrder>
class CahnHilliard{
    public:
        CahnHilliard(double x_ll, double x_ul); //constructor
        void run(); //execute the problem
    
    private:
        double x_ll_, x_ul_;

        void make_grid();
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

#include "MakeGrid.tpp"

