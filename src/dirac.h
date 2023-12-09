//
// Created by octavearevian on 12/3/23.
//

#ifndef MULTIGRID_DIRAC_H
#define MULTIGRID_DIRAC_H
#include "mesh.h"
#include "Field.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>


#include "ConjugateGradient.h"


class DiracOperator {
public:
    DiracOperator(Mesh* mesh, double const& mass, BosonField* b_field);

    FermionField operator()(FermionField f);

    typedef Eigen::SparseMatrix<std::complex<double>> matrix_type;

    matrix_type get_matrix() const { return _big_d;}



private:
    double _mass;
    Mesh* _mesh;
    BosonField* _b_field;
    matrix_type _big_d;

};

class DirectSolver{
public:
    DirectSolver(DiracOperator const& D);
    FermionField solve(FermionField target);
private:
    DiracOperator::matrix_type _dirac_matrix;
    Eigen::SparseLU<DiracOperator::matrix_type, Eigen::COLAMDOrdering<int>> _solver;
};


class CGSolver{
public:
    CGSolver(DiracOperator const& D);
    FermionField solve(FermionField target, double tol = 1e-8, int max_iter = 10000);
private:
    DiracOperator::matrix_type _dirac_matrix;
    ConjugateGradient<DiracOperator::matrix_type, FermionField::vector_type> _cgSolver;
};


#endif //MULTIGRID_DIRAC_H
