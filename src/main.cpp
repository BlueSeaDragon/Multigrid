#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "dirac.h"
#include "Field.h"
#include "mesh.h"
#include "ConjugateGradient.h"
#include "MG.h"

#include <chrono>
#include <vector>




int main(int argc, char* argv[]) {

    // process args
    int mesh_size = 32;
    for(int i= 1; i < argc; ++i){
        std::string arg = std::string(argv[i]);
        std::string key = arg.substr(0, arg.find('='));
        std::string value = arg.substr(arg.find('=') + 1);
        if( key == "mesh_size")
            mesh_size = std::stoi(value); //set mesh size
    }


    std::cout << "Testing with mesh size of " << mesh_size << std::endl;
    Mesh mesh(mesh_size);
    /*
    FermionField phi(Eigen::VectorXcd::Random(mesh.get_size()*2));
    BosonField U(Eigen::VectorXcd::Random(mesh.get_size()*2));
     */

    // initialise fermion and boson field
    FermionField phi(mesh);
    BosonField U(mesh);
    phi.init_rand();
    U.init_uniform(1.); // expect Q=0, periodic boundary condition
    //U.printA();
    std::cout<< "Initial charge Q = " << U.Q() << std::endl;

    DiracOperator d(&mesh, 1.,&U);

    std::cout << "----------------------------------" << std::endl;
    std::cout << "--- Testing Direct Solver ---" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    DirectSolver solver(d);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Solver took " << duration << "ms to build " << std::endl;

    start = std::chrono::high_resolution_clock::now();

    auto solution = solver.solve(phi);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    //std::cout << "Solution: " << solution.get_data()<<std::endl;
    std::cout<< "Solution computed in " << duration  << "ms" << std::endl;

    FermionField::vector_type r = d.get_matrix() * solution.get_data() - phi.get_data();
    std::cout << "residual of norm " << r.norm() << std::endl;

    std::cout << "----------------------------------" << std::endl;
    std::cout << "--- Testing conjugate gradient ---" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    CGSolver cgSolver(d);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Solver built in " << duration << " ms" << std::endl;


    start = std::chrono::high_resolution_clock::now();

    auto solution2 = cgSolver.solve(phi);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    //std::cout << "Solution: " << solution.get_data()<<std::endl;
    std::cout<< "Solution computed in " << duration  << "ms" << std::endl;

    FermionField::vector_type r2 = d.get_matrix() * solution2.get_data() - phi.get_data();
    std::cout << "residual of norm " << r2.norm() << std::endl;

    FermionField::vector_type sol_diff = solution.get_data() - solution2.get_data();
    std::cout << "Difference between solutions: " << sol_diff.norm()<< std::endl;


    std::cout << "----------------------------------" << std::endl;
    std::cout << "--- Testing adaptive multigrid ---" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    MG mgSOLVER((d.get_matrix().transpose().conjugate()*d.get_matrix()), 1, 4);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Solver built in " << duration << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXcd sol3 = phi.get_data();
    sol3.setZero();
    sol3 = mgSOLVER.recursive_solve_vcycle(d.get_matrix().transpose().conjugate() * phi.get_data(), sol3,
                                               d.get_matrix().transpose().conjugate() * d.get_matrix(), 1);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<< "Solution computed in " << duration  << "ms" << std::endl;
    FermionField::vector_type r3 = d.get_matrix() * sol3 - phi.get_data();
    std::cout << "residual of norm " << r3.norm() << std::endl;
    FermionField::vector_type sol_diff2 = sol3 - solution.get_data();
    std::cout << "Difference between solutions: " << sol_diff2.norm()<< std::endl;
    std::cout << "Relative difference between solutions: " << sol_diff2.norm() / solution.get_data().norm() * 100 << "%" << std::endl;

    return 0;
}
