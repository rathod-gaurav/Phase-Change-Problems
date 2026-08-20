#pragma once

template <unsigned int Nsd, unsigned int BfOrder>
void CahnHilliard<Nsd,BfOrder>::debug_system(){
    std::cout << "------------------------" << std::endl;
    std::cout << "Debug info" << std::endl;

    Vector<double> test_vector(dof_handler.n_dofs());
    test_vector = 1.0;

    //check 1 to see if stiffness matrix kills constants : Kglobal*1 = 0
    Vector<double> test1(dof_handler.n_dofs());
    Kglobal.vmult(test1,test_vector);
    double tol = 1e-12;
    if(test1.linfty_norm() < tol){
        std::cout << "Kglobal test successful" << std::endl;
    }
    else{
        std::cout << "Kglobal test failed" << std::endl;
    }

    //check 2 to see if mass matrix Mglobal row sums give the domain area
    Vector<double> test2(dof_handler.n_dofs());
    Mglobal.vmult(test2,test_vector);
    double test2_val = test_vector*test2;
    // std::cout << "Test 2: 1T*M*1 = " << test2_val << std::endl;
    if(test2_val == 1){//1 is area of our 2D domain in this case
        std::cout << "Mglobal test successful" << std::endl;
    }
    else{
        std::cout << "Mglobal test failed" << std::endl;
    }

    //check 3 to see if matrices Mglobal and Kglobal are symmetric or not
    Vector<double> u(Mglobal.m()), v(Mglobal.m());
    for (unsigned int i = 0; i < Mglobal.m(); ++i) {
        u(i) = Utilities::generate_normal_random_number(0, 1);
        v(i) = Utilities::generate_normal_random_number(0, 1);
    }
    const double uAv = Mglobal.matrix_scalar_product(u, v);
    const double vAu = Mglobal.matrix_scalar_product(v, u);
    std::cout << "relative asymmetry in Mglobal: "
            << std::abs(uAv - vAu) / std::max(std::abs(uAv), 1e-30) << '\n';


    const double uAv_K = Kglobal.matrix_scalar_product(u, v);
    const double vAu_K = Kglobal.matrix_scalar_product(v, u);
    std::cout << "relative asymmetry in Kglobal: "
            << std::abs(uAv_K - vAu_K) / std::max(std::abs(uAv_K), 1e-30) << '\n';

    //check 4 to see if total mass of the system is conserver through the timesteps
    const double test4 = Mglobal.matrix_scalar_product(test_vector,c);
    std::cout << "total mass at current timestep : " << test4 << std::endl;

    //check 5 to see if energy decreases monotonically
    Vector<double> fFuncVec(Mglobal.m());
    for(unsigned int i = 0 ; i < Mglobal.m(); i++){
        fFuncVec(i) = fFunc_(c(i));
    }
    const double test5 = Mglobal.matrix_scalar_product(test_vector,fFuncVec) + (0.5*epsilon_*epsilon_)*Kglobal.matrix_scalar_product(c,c);
    std::cout << "total energy at current timestep : " << test5 << std::endl;

    std::cout << "------------------------" << std::endl;

}