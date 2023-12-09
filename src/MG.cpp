//
// Created by jing2li on 08/12/23.
//

#include "MG.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>



template <typename SOLVER>
MG<SOLVER>::MG(Eigen::SparseMatrix<std::complex<double>> M, SOLVER K, int levels) {
    _K = K;
    _level = levels;

    // precompute prolongation operators
    compute_prolongation(M, 1e-5);
}

template <typename SOLVER>
void MG<SOLVER>::compute_prolongation(Eigen::SparseMatrix<std::complex<double>> M, double TOL) {
    Eigen::SparseMatrix<std::complex<double>> M_h = M;
    int size = M_h.cols();
    for (int i=0; i<_level; i++) {
        Eigen::VectorXcd x = Eigen::VectorXcd::Random(size); // random guess x
        Eigen::VectorXcd y = Eigen::VectorXcd::Zero(size); // rhs = 0
        Eigen::MatrixXcd error(size, 1); // store error directions
        int dir_count = 1;
        Eigen::VectorXcd corr_total;

        // store error directions (c.f. power iterations)
        do {
            error.resize(size, dir_count);
            Eigen::VectorXcd x_old = x;

            // 1. pre-smoothing: forward solver (space V_h, fine)
            Eigen::VectorXcd pre_corr_h = _K.solve(M_h, y - M_h * x);
            x += pre_corr_h;

            // 2. post-smoothing: backward solver (space V_h, fine)
            Eigen::VectorXcd post_corr_h = _K.solve(M_h.transpose(), y - M_h * x);
            x += post_corr_h;

            corr_total = x - x_old;
            error.col(dir_count) = corr_total;
            dir_count ++;
        } while(corr_total.norm() > TOL * x.norm());

        // QR decomposition to find orthogonal directions of error (prolongation operator)
        Eigen::HouseholderQR<Eigen::MatrixXcd> qr(error);
        _P.emplace_back(qr.householderQ());
        M_h = (_P[i].transpose().conjugate() * M_h * _P[i]).sparseView();
    }
}


template <typename SOLVER>
void MG<SOLVER>::recursive_solve_vcycle(Eigen::VectorXcd y, Eigen::VectorXcd x,
                                        Eigen::SparseMatrix<std::complex<double>> M, int level) {
    if (level==0) {
        // direct solve
        x = _CG.solve(y);
    }
    else {
        // restrict the matrix to the level
        Eigen::SparseMatrix<std::complex<double>> M_h = (_P[level].transpose().conjugate() * M * _P[level]).sparseView();

        // 1. pre-smoothing: forward solver (space V_h, fine)
        Eigen::VectorXcd pre_corr_h = _K.solve(M_h, y - M_h * x);
        x += pre_corr_h;

        // 2. corase grid correction (space V_H, coarse)
        // residual
        Eigen::VectorXcd r_h = y - M_h * x;
        // project residual onto coarse grid
        Eigen::VectorXcd r_H = _P[level-1].transpose().conjugate() * r_h;
        // initialise correction
        Eigen::VectorXcd coarse_corr_H(r_H.size());
        recursive_solve_vcycle(r_H, coarse_corr_H, M_h, level - 1);
        // update
        x += _P[level-1] * coarse_corr_H;

        // 3. post-smoothing: backward solver (space V_h, fine)
        Eigen::VectorXcd post_corr_h = _K.solve(M_h.transpose().conjugate(), y - M_h * x);
        x += post_corr_h;
    }
}
