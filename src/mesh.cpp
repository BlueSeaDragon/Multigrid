//
// Created by octavearevian on 12/3/23.
//
#include "mesh.h"

Mesh::Mesh(int dim):_dim(dim) {}

int Mesh::get_neighbour(int const& index, int const& direction_t, int const& direction_x) {
    return (index + direction_t*_dim + direction_x + get_size())  % get_size();
}


Eigen::Vector4i Mesh::get_neighbours(int const& index) {
    Eigen::Vector4i output;
    output << get_neighbour(index, 1, 0),
                get_neighbour(index,0 , 1),
            get_neighbour(index, -1, 0),
            get_neighbour(index, 0, -1);
    return output;
}

int Mesh::get_size() const { return _dim*_dim;}

int Mesh::get_dim() const {return _dim;}

int Mesh::loc_index(Eigen::Vector2i x) const {return x(0)*_dim + x(1);}