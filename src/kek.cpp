#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <iostream>

int main() {
    arma::mat regressors({1.0, 2.0, 3.0});
    arma::rowvec responses({1.0, 4.0, 9.0});
    auto lr = mlpack::regression::LinearRegression(regressors, responses);
    arma::mat testX({2.0});
    arma::rowvec testY;
    lr.Predict(testX, testY);
    std::cout << testY << std::endl;
    
    bool status = mlpack::data::Save("zhopa.bin", "sraka", lr);
    std::cout << status << std::endl;
    
    return 0;
}
