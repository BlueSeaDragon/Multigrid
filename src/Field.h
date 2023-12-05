//
// Created by octavearevian on 12/3/23.
//

#ifndef MULTIGRID_FIELD_H
#define MULTIGRID_FIELD_H
#include "mesh.h"
#include "Eigen/Dense"
#include <utility>
#include <vector>
#include "utils.h"

/*
template<typename FieldType>
class Field {
public:
    Field(FieldType const& field_values): _values(field_values){}

protected:
    FieldType _values;
};
 */

class BosonField {
public:
    //BosonField(Eigen::VectorXcd const& field_values):Field(field_values){};


    //construct boson field given A
    BosonField(Eigen::VectorXd  A, Mesh mesh): _A(std::move(A)), _mesh(mesh) {};
    BosonField(Mesh mesh): _mesh(mesh) { init_uniform(0.);};

    // gauge invariant initialisation where the magnitude per link = exp(i*a*A)
    void init_uniform (double A);
    void init_staggered(double diff);

    // return boson at location loc, direction mu, U(x)^(mu) = exp(i*a*A)
    std::complex<double> U(Eigen::VectorXi loc, int mu);
    std::complex<double> U(int loc, int mu);

    // invariant charge: sum over all plaquette
    double Q();

    void printA();

    static int const TDirection = 0;
    static int const XDirection = 1;

    //std::complex<double> operator()(int index, int direction){return _values[2*index + direction];}
    Mesh get_mesh() {return _mesh;};

protected:
    Eigen::VectorXd _A;
    Mesh _mesh;
};

//class FermionField: public Field<Eigen::VectorXcd>{
class FermionField {
public:
    //FermionField(Eigen::VectorXcd const& field_values): Field(field_values){};
    FermionField(Eigen::VectorXcd  field_values, Mesh mesh): _values(std::move(field_values)), _mesh(mesh){};
    FermionField(Mesh mesh): _mesh(mesh){ init_uniform(std::complex<double>(0,0));};

    // initialise to random value
    void init_rand();
    void init_uniform(std::complex<double> val);

    std::complex<double> operator()(int index, int component){ return _values[2* index + component]; }

    // return fermion at location, spinor_idx
    std::complex<double> Psi(Eigen::VectorXi location, int spinor_idx);
    std::complex<double> Psi(int location, int spinor_idx);

    Eigen::VectorXcd get_data(){return _values;}
    Mesh get_mesh() {return _mesh;};

private:
    Eigen::VectorXcd _values;
    Mesh _mesh;
};

#endif //MULTIGRID_FIELD_H
