//
// Created by octavearevian on 12/3/23.
//

#include "Field.h"
#include <iostream>

void BosonField::init_uniform(double const A) {
    int dim = _mesh.get_dim();
    _A = Eigen::VectorXd::Constant(dim * dim * 2, A);
}

void BosonField::init_staggered(double diff) {
    int dim = _mesh.get_dim();
    _A = Eigen::VectorXd::Constant(dim * dim * 2, 0.);

    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            int pos = i*dim + j;
            if ((i+j) % 2 == 0) {
                _A(2*pos) = diff/2.;
            }
            else {
                _A(2*pos+1) = diff/2.;
            }
        }
    }
}

std::complex<double> BosonField::U(Eigen::VectorXi loc, int mu) {
    int dim = _mesh.get_dim();
    // position index
    int pos = _mesh.loc_index(loc);

    // fetch value of A
    double A_loc = _A(2 * pos + mu);

    return std::exp(std::complex<double>(0, A_LAT * A_loc));
}

std::complex<double> BosonField::U(int loc, int mu) {
    // fetch value of A
    double A_loc = _A(2 * loc + mu);

    return std::exp(std::complex<double>(0, A_LAT * A_loc));
}

double BosonField::Q() {
    double q_sum = 0.;

    // only count top right plaquette per point
    for (int pos=0; pos<_mesh.get_size(); pos++) {
        int pos_up = _mesh.get_neighbour(pos, 1, 0);
        int pos_right = _mesh.get_neighbour(pos, 0, 1);

        double const A_sum = _A(2*pos) + _A(2*pos_up+1) - _A(2*pos_right) - _A(2*pos+1);
        q_sum += A_LAT*A_sum;
    }

    return q_sum;
}

void BosonField::printA() {
    int dim = _mesh.get_dim();
    for (int i=0; i<dim; i++) {
        std::cout<<" ";
        for (int j=0; j<dim; j++) {
            std::cout<<_A(2*(i*dim+j)+1)<<"  ";
        }
        std::cout<<std::endl;

        for (int j=0; j<dim; j++) {
            std::cout<<_A(2*(i*dim+j)) <<"  ";
        }
        std::cout<<std::endl;
    }
}

void FermionField::init_rand() {
    _values = Eigen::VectorXcd::Random(_mesh.get_size() * 2);
}

void FermionField::init_uniform(std::complex<double> val) {
    _values = Eigen::VectorXcd::Constant(_mesh.get_size() * 2, val);
}

std::complex<double> FermionField::Psi(Eigen::VectorXi loc, int i) {
    // position index
    int pos = _mesh.loc_index(loc);

    // fetch fermion value
    return _values(2 * pos + i);

}

std::complex<double> FermionField::Psi(int location, int spinor_idx) {
    return _values(2 * location + spinor_idx);
}