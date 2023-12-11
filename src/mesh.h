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

    // #subblocks * #indices
    std::vector<std::vector<int>> block(int block_size){
        assert(_dim % block_size ==0);
        int n_blocks = _dim / block_size;
        std::vector<std::vector<int>> blocks(n_blocks*n_blocks);
        for(int i = 0; i <_dim;++i){
            for (int j = 0; j < _dim; ++j){
                int i_block = i/block_size;
                int j_block = j/block_size;
                blocks[i*n_blocks +j].emplace_back(loc_index({i,j}));

            }
        }
    }
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
