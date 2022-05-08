#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <iostream>

int main() {
    arma::mat regressors({1.0, 2.0, 3.0});
    arma::Row<size_t> trainY = {0, 1, 5};

    auto lr = mlpack::regression::LogisticRegression<arma::mat>(regressors, trainY);
    std::cout << lr.Parameters() << std::endl;
    return 0;
}
