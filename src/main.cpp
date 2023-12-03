#include <iostream>
#include <Eigen/Sparse>




int main() {
    Eigen::SparseMatrix<double> m(4,4);
    for(int i = 0; i < 4; i++)
        m.insert(i,i) = i;

    m.finalize();

    std::cout << "Hello, World!" << std::endl;
    std::cout << "the matrix is " << m << std::endl;

    return 0;
}
