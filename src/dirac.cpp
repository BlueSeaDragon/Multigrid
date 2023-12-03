//
// Created by octavearevian on 12/3/23.
//

#include "dirac.h"
#include <vector>

DiracOperator::DiracOperator(Mesh *mesh, const double &mass, BosonField* b_field):_mesh(mesh), _mass(mass),
                                            _b_field(b_field), _big_d(2*mesh->get_size(), 2*mesh->get_size()){

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;

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
    _big_d.setFromTriplets(triplets.begin(), triplets.end());
    _big_d.finalize();
}

FermionField DiracOperator::operator()(FermionField f) {
    auto fermion_values = f.get_data();
    return FermionField(_big_d * fermion_values);
}


DirectSolver::DirectSolver(const DiracOperator &D): _dirac_matrix(D.get_matrix()) {
    _solver.analyzePattern(_dirac_matrix);
    _solver.factorize(_dirac_matrix);
}

FermionField DirectSolver::solve(FermionField target) {
    return FermionField(_solver.solve(target.get_data()));
}