#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
CahnHilliard<Nsd,BfOrder>::CahnHilliard(
    const double x_ll, const double x_ul, const double quadOrder, const std::function<double(double)> fFunc, const std::function<double(double)> fFuncDerivative, 
    const unsigned int NT, const double Mobility, const double epsilon, const double dt,
    const unsigned int NCG, const double epsilonCG,
    OutputWriter<Nsd,BfOrder>& output_writer
):
    x_ll_(x_ll),
    x_ul_(x_ul),
    quadOrder_(quadOrder),
    fFunc_(fFunc),
    fFuncDerivative_(fFuncDerivative),
    NT_(NT),
    Mobility_(Mobility),
    epsilon_(epsilon),
    dt_(dt),
    NCG_(NCG),
    epsilonCG_(epsilonCG),
    output_writer_(output_writer),
    fe(BfOrder),
    dof_handler(triangulation)
{}

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::run(){
    make_grid();
    setup_system();
    solve();
    // output_writer();
}
