//
// Created by octavearevian on 12/3/23.
//

#ifndef MULTIGRID_MESH_H
#define MULTIGRID_MESH_H

#include <Eigen/Dense>

class Mesh {
public:
    Mesh(int dim);

    Eigen::Vector4i get_neighbours(int const& index);
    // index of neighbour shifted in t and x
    int get_neighbour(int const& index, int const& direction_t, int const& direction_x);

    // number of grid points
    int get_size() const;
    // dimension of square grid
    int get_dim() const;
    // location index of 2-vector x
    int loc_index(Eigen::Vector2i x) const;

private:
    int _dim;
};


#endif //MULTIGRID_MESH_H
