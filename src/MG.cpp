//
// Created by jing2li on 08/12/23.
//

#include "MG.h"
#include "ConjugateGradient.h"
#include "mesh.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include "GaussSeidel.h"



MG::MG(Eigen::SparseMatrix<std::complex<double>> M, int levels, int coarsening) {
    // _K = K;
    _level = levels;
    _coarsening = coarsening;

    // precompute prolongation operators
    compute_prolongation(M, _coarsening);
}

void MG::compute_prolongation(Eigen::SparseMatrix<std::complex<double>> M, int coarsening) {

    //std::vector<Mesh> meshes;
    std::vector<std::vector<std::vector<int>>> blocks(_level);
    std::vector<int> dim_count(_level, 0);

    int dim = M.rows();
    for(int i = 0; i <_level; i++){
        // meshes.emplace_back(Mesh(dim));
        Mesh mesh_tmp(std::sqrt(dim/2.));
        auto block = mesh_tmp.block(coarsening);
        blocks[i] = block;
        dim /= coarsening;
        //initialize _P
        std::vector<Eigen::MatrixXcd> tmp (block.size());
        _P.emplace_back(tmp);
    }
    int size = M.cols();
    // std::vector<std::vector<Eigen::MatrixXcd>> error;
    do{
        Eigen::SparseMatrix<std::complex<double>> M_h = M;
        Eigen::VectorXcd x = Eigen::VectorXcd::Random(size); // random guess x
        Eigen::VectorXcd y = Eigen::VectorXcd::Zero(size); // rhs = 0

        for (int i=0; i<_level; i++) {
             // store error directions
            //Eigen::VectorXcd corr_total;
            double omega = 0.01;
            // store error directions (c.f. power iterations)
            Eigen::VectorXcd x_old;

            for (int j = 0; j < 1; j++) {
                x_old = x;
                // 1. pre-smoothing: forward solver (space Vcorr_total_h, fine)
                /* // the BiCGSTAB solver
                Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>>> biCG;
                biCG.compute(M_h);
                Eigen::VectorXcd pre_corr_h = biCG.solve(M_h * x);
                */
                //GaussSeidel GS;
                //Eigen::VectorXcd pre_corr_h = GS.solve_forward(M_h, -M_h * x, x, 1e-10, 10);
                //Eigen::VectorXcd pre_corr_h = M_h.triangularView<Eigen::Lower>().solve(M_h * x);
                // Eigen::VectorXcd pre_corr_h = _K.solve(M_h.triangularView<Eigen::Lower>(), M_h * x, x, 1e-10, 1);
                Eigen::VectorXcd pre_corr_h = omega * (M_h * x);
                x -= pre_corr_h;


                // 2. post-smoothing: backward solver (space V_h, fine)
                /* // BiCGSTAB
                Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>>> biCG;
                biCG.compute(M_h);
                Eigen::VectorXcd post_corr_h = biCG.solve(M_h * x);
                */
                //Eigen::VectorXcd post_corr_h = GS.solve_backward(M_h, M_h * x, x, 1e-10, 10);
                //Eigen::VectorXcd post_corr_h = M_h.triangularView<Eigen::Upper>().solve(M_h * x);
                //Eigen::VectorXcd post_corr_h = _K.solve(M_h.triangularView<Eigen::Upper>(), M_h * x, x, 1e-10, 1);
                Eigen::VectorXcd post_corr_h = omega * (M_h * x);

                x -= post_corr_h;
            }
                //corr_total = x - x_old;
            if (((x-x_old).dot(M_h*(x-x_old))).real() > 0.1 * (x_old.dot(M_h*x_old)).real()) {
                //_P[i].resize(size, (dir_count+1) * blocks[i].size());
                //Eigen::VectorXcd x_new(blocks[i].size(), dim_count[i]);
                ++dim_count[i];

                // loop over number of sub-blocks
                for(int k = 0; k < blocks[i].size(); ++k) {
                    // candidate error
                    Eigen::VectorXcd x_l(blocks[i][k].size() * 2);
                    for (int j = 0; j < blocks[i][k].size(); ++j) {
                        auto idx = blocks[i][k][j];
                        x_l(2*j) = x(2*idx);
                        x_l(2*j+1) = x(2*idx + 1);
                    }

                    /*
                    // subtract previous eigenvector directions
                    for (int l = 0; l < dim_count[i] -1; l++) {
                        Eigen::VectorXcd debug = _P[i][k].col(l) * (x_l.transpose().conjugate() * _P[i][k].col(l));
                        x_l -= _P[i][k].col(l) * (x_l.transpose().conjugate() * _P[i][k].col(l)) ;
                        //x -= x.dot(error.col(l)) * error.col(l);
                        //Eigen::VectorXcd error_comp = error[l][k].dot(x[]) * error[l][k];
                    }
                     */


                    // attach normalised new direction
                    _P[i][k].resize(x_l.size(), dim_count[i]);
                    //_P[i][k].col(dim_count[i] - 1) = (x_l / x_l.norm());
                    _P[i][k].col(dim_count[i] - 1) = (x_l);
                }
            }
            else {
                break;
            }

            //} while(corr_total.norm() > TOL * x.norm() && dir_count < );


            /*Eigen::MatrixXcd error_mat(size, error.size());
            for(int l = 0; l < error[i][k].size(); ++l){
                error_mat.col(l) = error[i][k][l];
            }*/
            // QR decomposition to find orthogonal directions of error (prolongation operator)
            /*Eigen::HouseholderQR<Eigen::MatrixXcd> qr(error_mat);
            Eigen::MatrixXcd QR_mat(Eigen::MatrixXcd::Identity(size, dir_count));
            QR_mat = qr.householderQ()*QR_mat;
            _P.emplace_back(QR_mat);
            _P.emplace_back(error_mat);
            M_h = (_P[i].transpose().conjugate() * M_h * _P[i]).sparseView();
            size = M_h.cols();*/
        }
    } while (_P[0][0].cols() < 20);
    // std::reverse(_P.begin(), _P.end());
    // make unitary
    for(int i=0; i<_level; i++)
    for (int k = 0; k<_P[i].size(); k++) {
        Eigen::HouseholderQR<Eigen::MatrixXcd> qr(_P[i][k]);
        Eigen::MatrixXcd tmp = qr.householderQ() * Eigen::MatrixXcd::Identity(_P[i][k].rows(), _P[i][k].cols());
        _P[i][k] = tmp;
    }
    _map = blocks;
    std::cout<< "1) number of sub-blocks at level 1 is " << _P[0].size() << std::endl;
    std::cout<< "2) dimension of a sub-block is " << _P[0][0].rows() << std::endl;
    std::cout<< "3) number of eigenbasis per sub-block is " << _P[0][55].cols() << std::endl;
}


Eigen::VectorXcd MG::recursive_solve_vcycle(Eigen::VectorXcd y, Eigen::VectorXcd x,
                                        Eigen::SparseMatrix<std::complex<double>> M, int level) {
    if (level==0) {
        // direct solve
        x = _CG.solve(M, y, x);
    }
    else {
        double omega = 0.01;
        // 1. pre-smoothing: forward solver (space V_h, fine)
        //Eigen::VectorXcd pre_corr_h = _K.solve(M.triangularView<Eigen::Lower>(), y - M * x, x,1e-10,1);
        Eigen::VectorXcd pre_corr_h = omega*(y-M * x);
        //GaussSeidel GS;
        // Eigen::VectorXcd pre_corr_h = GS.solve_forward(M, y - M * x, x, 1e-10, 1);
        //Eigen::VectorXcd pre_corr_h = M.triangularView<Eigen::Lower>().solve(y - M * x);

        x += pre_corr_h;

        // 2. corase grid correction (space V_H, coarse)
        // residual
        Eigen::VectorXcd r_h = y - M * x;
        // project residual onto coarse grid
        // Eigen::VectorXcd r_H = _P[level-1].transpose().conjugate() * r_h;
        std::vector<Eigen::VectorXcd> r_H;
        for (int block_ind =0; block_ind < _P[level-1].size(); block_ind++) {
            int const blocksize = _coarsening * _coarsening;
            // local residual vector
            Eigen::VectorXcd r_H_local(blocksize*2);

            // local to global map
            Eigen::VectorXi loc_to_glob(blocksize);
            for (int el = 0; el < blocksize; el ++) {
                loc_to_glob(el) = _map[level-1][block_ind][el];
            }

            //keep relevant r_h
            for (int el=0; el<blocksize; el++) {
                r_H_local(2*el) = r_h(2 *loc_to_glob(el));
                r_H_local(2*el+1) = r_h(2*loc_to_glob(el)+1);
            }

            // find block-local M (geometric coarsening)
            Eigen::MatrixXcd M_h_local(blocksize * 2, blocksize * 2);
            for (int row=0; row<blocksize; row++) {
                for (int col = 0; col < blocksize; col++) {
                    M_h_local(2 * row, 2 * col) = M.coeff(2 * loc_to_glob(row), 2 * loc_to_glob(col));
                    M_h_local(2 * row + 1, 2 * col) = M.coeff(2 * loc_to_glob(row) + 1, 2 * loc_to_glob(col));
                    M_h_local(2 * row, 2 * col + 1) = M.coeff(2 * loc_to_glob(row), 2 * loc_to_glob(col) + 1);
                    M_h_local(2 * row + 1, 2 * col + 1) = M.coeff(2 * loc_to_glob(row) + 1, 2 * loc_to_glob(col) + 1);
                }
            }

            // projection onto coarse grid (algebraic coarsening)
            Eigen::MatrixXcd M_H_local = _P[level-1][block_ind].transpose().conjugate() * M_h_local * _P[level-1][block_ind];

            // recursive call -> bloc-level correction
            Eigen::VectorXcd coarse_corr_H_local(_P[level-1][block_ind].cols());
            coarse_corr_H_local = recursive_solve_vcycle(_P[level-1][block_ind].transpose().conjugate() * r_H_local, coarse_corr_H_local, M_H_local.sparseView(), level-1);
            Eigen::VectorXcd coarse_corr_h = _P[level-1][block_ind] * coarse_corr_H_local;

            // update global x
            for (int el=0; el<blocksize; el++) {
                x(loc_to_glob(el) * 2) += coarse_corr_h(el * 2);
                x(loc_to_glob(el) * 2 + 1) += coarse_corr_h(el * 2 + 1);
            }

            //r_H.push_back();
        }
        /*
        // initialise correction
        Eigen::VectorXcd coarse_corr_H(r_H.size());
        coarse_corr_H.setRandom();
        Eigen::SparseMatrix<std::complex<double>> M_H = (_P[level-1].transpose().conjugate() * M * _P[level-1]).sparseView();
        coarse_corr_H = recursive_solve_vcycle(r_H, coarse_corr_H, M_H, level - 1);
        // update
        x += _P[level-1] * coarse_corr_H;
        */

        // 3. post-smoothing: backward solver (space V_h, fine)
        //Eigen::VectorXcd post_corr_h = _K.solve(M.triangularView<Eigen::Upper>(), y - M * x, x,1e-10,1);
        Eigen::VectorXcd post_corr_h = omega*(y-M * x);
        //Eigen::VectorXcd post_corr_h = GS.solve_backward(M, y - M * x, x, 1e-10, 1);
        //Eigen::VectorXcd post_corr_h = M.triangularView<Eigen::Lower>().solve(y - M * x);
        x += post_corr_h;
    }

    return x;
}
