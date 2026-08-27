#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
AllenCahn<Nsd,BfOrder>::AllenCahn(
    const double x_ll, const double x_ul, const double quadOrder, const std::function<double(double)> fFunc, const std::function<double(double)> fFuncDerivative, const std::function<double(double)> fFuncDoubleDerivative,
    const unsigned int NT, const double Mobility, const double epsilon, const double dt,
    const unsigned int N_NR, const double epsilon_NR,
    OutputWriter<Nsd,BfOrder>& output_writer
):
    x_ll_(x_ll),
    x_ul_(x_ul),
    quadOrder_(quadOrder),
    fFunc_(fFunc),
    fFuncDerivative_(fFuncDerivative),
    fFuncDoubleDerivative_(fFuncDoubleDerivative),
    NT_(NT),
    Mobility_(Mobility),
    epsilon_(epsilon),
    dt_(dt),
    N_NR_(N_NR),
    epsilon_NR_(epsilon_NR),
    output_writer_(output_writer),
    fe(BfOrder),
    dof_handler(triangulation)
{}

template <unsigned int Nsd, unsigned int BfOrder>
void AllenCahn<Nsd,BfOrder>::run(){
    make_grid();
    setup_system();
    solve();
    // output_writer();
}
