#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;
using namespace mlpack::neighbor; 
using namespace mlpack::metric; 

int main()
{
    arma::mat data;
    data::Load("../data.csv", data, true);
    NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(data);
    
    
    arma::Mat<size_t> neighbors;
    arma::mat distances; 
    
    nn.Search(1, neighbors, distances);
    
    // Print out each neighbor and its distance.
    for (size_t i = 0; i < neighbors.n_elem; ++i)
    {
        std::cout << "Nearest neighbor of point " << i << " is point "
        << neighbors[i] << " and the distance is " << distances[i] << ".\n";
    }
}