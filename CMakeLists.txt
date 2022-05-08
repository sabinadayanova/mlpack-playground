cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

project(play_ground)
add_executable(linreg src/kek.cpp)
add_executable(logreg src/lol.cpp)
add_executable(knn src/kuka.cpp)

target_link_libraries(linreg mlpack boost_serialization)
target_link_libraries(logreg mlpack boost_serialization armadillo)
target_link_libraries(knn mlpack boost_serialization)

# find_package(Boost COMPONENTS system REQUIRED)
# target_link_libraries(play ${Boost_LIBRARIES})

# SET(GCC_COVERAGE_LINK_FLAGS "-lboost_serialization")
# SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")