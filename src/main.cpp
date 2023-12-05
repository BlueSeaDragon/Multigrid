#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "dirac.h"
#include "Field.h"
#include "mesh.h"
#include <chrono>




int main() {
    Mesh mesh(8);
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

    DiracOperator d(&mesh,0.,&U);

    auto start = std::chrono::high_resolution_clock::now();
    DirectSolver solver(d);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    std::cout << "Solution: " << solver.solve(phi).get_data()<<std::endl;
    std::cout<< "computed in " << duration  << "ms" << std::endl;

    return 0;
}
