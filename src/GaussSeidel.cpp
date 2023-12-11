//
// Created by jing2li on 10/12/23.
//

#include "GaussSeidel.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXcd GaussSeidel::solve_forward(Eigen::SparseMatrix<std::complex<double>> M, Eigen::VectorXcd rhs, Eigen::VectorXcd x, double TOL, int max_iterations) {
    double deltanorm;
    int count = 0;
    do {
        deltanorm = 0.;
        int const N = M.rows();
        for (int i=0; i< N; i++) {
            std::complex<double> corr = 0.;
            for (int j = 0; j < i; j++) {
                corr += M.coeff(i, j) * x(j);
            }
            corr = M.coeff(i, i) * (rhs(i) - corr);
            x(i) += corr;
            deltanorm += (conj(corr) * corr).real();
        }
        count ++;
    } while (deltanorm < TOL * x.norm() && count <max_iterations);
}

Eigen::VectorXcd GaussSeidel::solve_backward(Eigen::SparseMatrix<std::complex<double>> M, Eigen::VectorXcd rhs, Eigen::VectorXcd x, double TOL, int max_iterations) {
    double deltanorm;
    int count = 0;
    do {
        deltanorm = 0.;
        int const N = M.rows();
        for (int i=N-1; i>=0; i--) {
            std::complex<double> corr = 0.;
            for (int j = N-1; j > i; j--) {
                corr += M.coeff(i, j) * x(j);
            }
            corr = M.coeff(i, i) * (rhs(i) - corr);
            x(i) += corr;
            deltanorm += (conj(corr) * corr).real();
        }
        count ++;
    } while (deltanorm < TOL * x.norm() && count <max_iterations);
}