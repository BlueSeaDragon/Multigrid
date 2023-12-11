//
// Created by jing2li on 10/12/23.
//

#ifndef MULTIGRID_GAUSSSEIDEL_H
#define MULTIGRID_GAUSSSEIDEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

class GaussSeidel {
public:
    GaussSeidel(){}
    Eigen::VectorXcd solve_forward(Eigen::SparseMatrix<std::complex<double>> M, Eigen::VectorXcd rhs, Eigen::VectorXcd x, double TOL, int max_iterations);
    Eigen::VectorXcd solve_backward(Eigen::SparseMatrix<std::complex<double>> M, Eigen::VectorXcd rhs, Eigen::VectorXcd x, double TOL, int max_iterations);

};


#endif //MULTIGRID_GAUSSSEIDEL_H
