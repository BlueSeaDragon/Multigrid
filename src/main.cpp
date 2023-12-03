#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "dirac.h"
#include "Field.h"
#include "mesh.h"
#include "chrono"




int main() {
    Mesh mesh(128);
    FermionField phi(Eigen::VectorXcd::Random(mesh.get_size()*2));
    BosonField U(Eigen::VectorXcd::Random(mesh.get_size()*2));


    DiracOperator d(&mesh,0.,&U);

    auto start = std::chrono::high_resolution_clock::now();
    DirectSolver solver(d);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end-start);

    std::cout << "Solution: " << solver.solve(phi).get_data()<<std::endl;
    std::cout<< "computed in " << duration  << std::endl;

    return 0;
}
