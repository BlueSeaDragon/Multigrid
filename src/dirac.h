//
// Created by octavearevian on 12/3/23.
//

#ifndef MULTIGRID_DIRAC_H
#define MULTIGRID_DIRAC_H
#include "mesh.h"
#include "Field.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

class DiracOperator {
public:
    DiracOperator(Mesh* mesh, double const& mass, BosonField* b_field);

    FermionField operator()(FermionField f);

    Eigen::SparseMatrix<std::complex<double>> get_matrix() const { return _big_d;}


private:
    double _mass;
    Mesh* _mesh;
    BosonField* _b_field;
    Eigen::SparseMatrix<std::complex<double>> _big_d;

};

class DirectSolver{
public:
    DirectSolver(DiracOperator const& D);
    FermionField solve(FermionField target);
private:
    Eigen::SparseMatrix<std::complex<double>> _dirac_matrix;
    Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>, Eigen::COLAMDOrdering<int>> _solver;
};


#endif //MULTIGRID_DIRAC_H
