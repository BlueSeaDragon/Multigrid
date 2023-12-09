//
// Created by jing2li on 08/12/23.
//

#ifndef MULTIGRID_MG_H
#define MULTIGRID_MG_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
// #include "conjugate-graident"

template <typename SOLVER>
class MG {
public:
    //constructor
    MG(Eigen::SparseMatrix<std::complex<double>> M, SOLVER K, int levels);

    // solver for Mx = y: V-cycle
    void recursive_solve_vcycle(Eigen::VectorXcd y, Eigen::VectorXcd x, Eigen::SparseMatrix<std::complex<double>> M, int level);

    // compute prolongation operators: Mx = 0
    void compute_prolongation(Eigen::SparseMatrix<std::complex<double>> M, double TOL);

private:
    int _level; // number of V-cycle levels
    SOLVER _K;  // smoother
    SOLVER _CG; // direct solver
    int _p_dim = 10; // prolongation operator dim (at each level)
    std::vector<Eigen::MatrixXd> _P;// array of prolongation operators _level * (? * dim)

};


#endif //MULTIGRID_MG_H
