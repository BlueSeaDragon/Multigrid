//
// Created by octavearevian on 12/8/23.
//

#ifndef MULTIGRID_CONJUGATEGRADIENT_H
#define MULTIGRID_CONJUGATEGRADIENT_H

#include <algorithm>
#include <complex>

template<typename matrix_type, typename vector_type>
class ConjugateGradient {
public:
    ConjugateGradient(){}

    vector_type solve(matrix_type const& matrix, vector_type const& rhs ,
                      vector_type const& start_point,
                      double tolerance = 1e-10,
                      int max_iter = 10000){
        max_iter = std::min(max_iter, int(rhs.size()));

        vector_type x(start_point);// x

        vector_type r(rhs - matrix*x); //r
        
        vector_type r_temp(r);// temporary value for r_(k+1)
        vector_type p(r);//p
        vector_type q(matrix*p);// q = Ap (to avoid multiple matrix vector products)
        std::complex<double> alpha = 0;//alpha
        std::complex<double> beta = 0;//beta

        // < . , . > is for the scalar/ hermitian product
        for(int k = 0; k < max_iter; ++k ){
            alpha = r.dot(r)/ p.dot(q); // alpha_k = <r_k, r_k> / <p_k, A*p_k>
            x = x + alpha*p;// x_(k+1) = x_k + alpha_k*p_k
            r_temp = r - alpha*q; // r_(k+1) = r_k - alpha_k * A*p_k
            if(r_temp.squaredNorm() < tolerance) // if r_(k+1) is small enough return
                return x;
            beta = r_temp.dot(r_temp)/r.dot(r);// beta_k = <r_(k+1), r_(k+1)> / < r_k, r_k>
            std::swap(r, r_temp); // we don't need r_k anymore, change r = r_(k+1)
            p = r + beta*p;// p_(k+1) = r_(k+1) + beta_k*p_k
            q = matrix * p; // update q_(k+1) = A*p_(k+1)
        }

        return x;
    }
};


#endif //MULTIGRID_CONJUGATEGRADIENT_H
