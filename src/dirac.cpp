//
// Created by octavearevian on 12/3/23.
//

#include "dirac.h"
#include <vector>

DiracOperator::DiracOperator(Mesh *mesh, const double &mass, BosonField* b_field):_mesh(mesh), _mass(mass),
                                            _b_field(b_field), _big_d(2*mesh->get_size(), 2*mesh->get_size()){

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    int size = _mesh->get_size();

    // fill D(x, y), dimension size*2 by size*2
    // loop over the second y index
    for (int y=0; y<size; y++) {
        // diagonal term x=y
        triplets.emplace_back(2*y, 2*y, 4 + _mass);
        triplets.emplace_back(2*y + 1, 2*y + 1, 4 + _mass);

        // off-diagonal x + e_t = y
        int x = _mesh->get_neighbour(y, -1, 0);
        Eigen::MatrixXcd factor(2, 2);
        factor << std::complex<double>(1,0), std::complex<double>(-1,0),
                std::complex<double>(-1,0), std::complex<double>(1,0);
        factor *= _b_field->U(x, 0);

        for (int i=0; i<factor.rows(); i++)
            for (int j=0; j<factor.cols(); j++)
                triplets.emplace_back(2*x + i, 2*y + j, -0.5 * factor(i, j));
        factor.setZero();


        // off-diagonal x + e_x = y
        x = _mesh->get_neighbour(y, 0, -1);
        factor << std::complex<double>(1,0), std::complex<double>(0,1),
                std::complex<double>(0,-1), std::complex<double>(1,0);
        factor *= _b_field->U(x, 1);
        for (int i=0; i<factor.rows(); i++)
            for (int j=0; j<factor.cols(); j++)
                triplets.emplace_back(2*x + i, 2*y + j, -0.5 * factor(i, j));
        factor.setZero();

        // off-diagonal x - e_t = y
        x = _mesh->get_neighbour(y, 1, 0);
        factor << std::complex<double>(1,0), std::complex<double>(1,0),
                std::complex<double>(1,0), std::complex<double>(1,0);
        factor *= conj(_b_field->U(y, 0));

        for (int i=0; i<factor.rows(); i++)
            for (int j=0; j<factor.cols(); j++)
                triplets.emplace_back(2*x + i, 2*y + j, -0.5 * factor(i, j));
        factor.setZero();


        // off-diagonal x - e_x = y
        x = _mesh->get_neighbour(y, 0, -1);
        factor << std::complex<double>(1,0), std::complex<double>(0,-1),
                std::complex<double>(0,1), std::complex<double>(1,0);
        factor *= conj(_b_field->U(y, 1));

        for (int i=0; i<factor.rows(); i++)
            for (int j=0; j<factor.cols(); j++)
                triplets.emplace_back(2*x + i, 2*y + j, -0.5 * factor(i, j));
    }

    /*
    for( int index = 0; index < _mesh->get_size(); ++index ){
        triplets.emplace_back(2*index, 2*index, 4 + _mass);
        triplets.emplace_back(2*index + 1, 2*index + 1, 4 + _mass);

        triplets.emplace_back(2* mesh->get_neighbour(index,1,0),
                              2* mesh->get_neighbour(index,1,0),
                              -(*_b_field)(index, BosonField::TDirection)/2.);
        triplets.emplace_back(2* mesh->get_neighbour(index,1,0)+1,
                              2* mesh->get_neighbour(index,1,0),
                              1. * (*_b_field)(index, BosonField::TDirection)/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,1,0),
                              2* mesh->get_neighbour(index,1,0) +1,
                              1. * (*_b_field)(index, BosonField::TDirection)/2.);
        triplets.emplace_back(2* mesh->get_neighbour(index,1,0)+1,
                              2* mesh->get_neighbour(index,1,0)+1,
                              -(*_b_field)(index, BosonField::TDirection)/2.);


        std::complex<double> i(0,1);
        triplets.emplace_back(2* mesh->get_neighbour(index,0,1),
                              2* mesh->get_neighbour(index,0,1),
                              -(*_b_field)(index, BosonField::XDirection)/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,0,1),
                              2* mesh->get_neighbour(index,0,1)+1,
                              -(*_b_field)(index, BosonField::XDirection) * i/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,0,1)+1,
                              2* mesh->get_neighbour(index,0,1),
                              -(*_b_field)(index, BosonField::XDirection)* (-i)/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,0,1) + 1,
                              2* mesh->get_neighbour(index,0,1) + 1,
                              -(*_b_field)(index, BosonField::XDirection)/2.);



        triplets.emplace_back(2* mesh->get_neighbour(index,-1,0),
                              2* mesh->get_neighbour(index,-1,0),
                              -conj((*_b_field)(mesh->get_neighbour(index, -1, 0), BosonField::TDirection))/2.);
        triplets.emplace_back(2* mesh->get_neighbour(index,-1,0)+1,
                              2* mesh->get_neighbour(index,-1,0),
                              -conj((*_b_field)(mesh->get_neighbour(index, -1, 0), BosonField::TDirection))/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,-1,0),
                              2* mesh->get_neighbour(index,-1,0) +1,
                              -conj((*_b_field)(mesh->get_neighbour(index, -1, 0), BosonField::TDirection))/2.);
        triplets.emplace_back(2* mesh->get_neighbour(index,-1,0)+1,
                              2* mesh->get_neighbour(index,-1,0)+1,
                              -conj((*_b_field)(mesh->get_neighbour(index, -1, 0), BosonField::TDirection))/2.);


        triplets.emplace_back(2* mesh->get_neighbour(index,0,-1),
                              2* mesh->get_neighbour(index,0,-1),
                              conj((*_b_field)(mesh->get_neighbour(index, 0, -1), BosonField::XDirection))/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,0,-1),
                              2* mesh->get_neighbour(index,0,-1)+1,
                              conj((*_b_field)(mesh->get_neighbour(index, 0, -1), BosonField::XDirection)) * (-i)/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,0,-1)+1,
                              2* mesh->get_neighbour(index,0,-1),
                              conj((*_b_field)(mesh->get_neighbour(index, 0, -1), BosonField::XDirection))* (i)/2.);

        triplets.emplace_back(2* mesh->get_neighbour(index,0,-1) + 1,
                              2* mesh->get_neighbour(index,0,-1) + 1,
                              conj((*_b_field)(mesh->get_neighbour(index, 0, -1), BosonField::XDirection))/2.);

    }
     */
    _big_d.setFromTriplets(triplets.begin(), triplets.end());
    _big_d.makeCompressed();
    _big_d.finalize();
}

FermionField DiracOperator::operator()(FermionField f) {
    auto fermion_values = f.get_data();

    return {_big_d * fermion_values, f.get_mesh()};
}


DirectSolver::DirectSolver(const DiracOperator &D): _dirac_matrix(D.get_matrix()) {
    _solver.analyzePattern(_dirac_matrix);
    _solver.factorize(_dirac_matrix);
}

FermionField DirectSolver::solve(FermionField target) {
    return {_solver.solve(target.get_data()), target.get_mesh()};
}



CGSolver::CGSolver(const DiracOperator &D):_dirac_matrix(D.get_matrix()),_cgSolver() {
}

FermionField CGSolver::solve(FermionField target, double tol, int max_iter ) {
    auto target_vec = target.get_data();
    FermionField::vector_type init_point(target_vec.size());
    init_point.setZero();

    auto solution = _cgSolver.solve(_dirac_matrix.transpose().conjugate()*_dirac_matrix, _dirac_matrix.transpose().conjugate()*target_vec, init_point, tol, max_iter);

    return {solution, target.get_mesh()};
}
