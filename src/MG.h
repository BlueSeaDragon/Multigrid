//
// Created by jing2li on 08/12/23.
//

#ifndef MULTIGRID_MG_H
#define MULTIGRID_MG_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "ConjugateGradient.h"

class MG {
public:
    // constructor: matrix, smoother, number of levels
    MG(Eigen::SparseMatrix<std::complex<double>> M, int levels, int coarsening);

    // solver for Mx = y: V-cycle
    Eigen::VectorXcd recursive_solve_vcycle(Eigen::VectorXcd y, Eigen::VectorXcd x, Eigen::SparseMatrix<std::complex<double>> M, int level);

    // compute prolongation operators: Mx = 0
    void compute_prolongation(Eigen::SparseMatrix<std::complex<double>> M, int coarsening);

private:
    int _level; // number of V-cycle levels
    ConjugateGradient<Eigen::SparseMatrix<std::complex<double>>, Eigen::VectorXcd> _K;  // smoother
    ConjugateGradient<Eigen::SparseMatrix<std::complex<double>>, Eigen::VectorXcd> _CG; // direct solver
    // int _p_dim = 10; // prolongation operator dim (at each level)
    std::vector<std::vector<Eigen::MatrixXcd>> _P;// array of prolongation operators _level * #sub-blocks * (size * #eigenvec)
    // dimension of the sub-block
    int _coarsening;
    // mapping from local to global
    std::vector<std::vector<std::vector<int>>> _map;
};


#endif //MULTIGRID_MG_H
