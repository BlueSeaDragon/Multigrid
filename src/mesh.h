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
    int get_neighbour(int const& index, int const& direction_t, int const& direction_x);

    int get_size();



private:
    int _dim;
};


#endif //MULTIGRID_MESH_H
