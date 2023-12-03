//
// Created by octavearevian on 12/3/23.
//

#ifndef MULTIGRID_FIELD_H
#define MULTIGRID_FIELD_H
#include "mesh.h"
#include "Eigen/Dense"
#include <vector>
#include "utils.h"

template<typename FieldType>
class Field {
public:
    Field(FieldType const& field_values): _values(field_values){}

protected:
    FieldType _values;
};

class BosonField: public Field<Eigen::VectorXcd>{
public:
    BosonField(Eigen::VectorXcd const& field_values):Field(field_values){};

    static int const TDirection = 0;
    static int const XDirection = 1;

    std::complex<double> operator()(int index, int direction){return _values[2*index + direction];}
};

class FermionField: public Field<Eigen::VectorXcd>{
public:
    FermionField(Eigen::VectorXcd const& field_values): Field(field_values){};

    std::complex<double> operator()(int index, int component){ return _values[2* index + component]; }

    Eigen::VectorXcd get_data(){return _values;}


};

#endif //MULTIGRID_FIELD_H
