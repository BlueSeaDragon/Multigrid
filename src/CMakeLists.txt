cmake_minimum_required(VERSION 3.26)
project(Multigrid)

set(CMAKE_CXX_STANDARD 20)

# Set the path to the Eigen library
include_directories(${CMAKE_SOURCE_DIR}/lib/eigen-3.4.0)

add_executable(Multigrid main.cpp
        Field.h
        utils.h
        mesh.h
        dirac.h
        dirac.cpp
        Field.cpp
        mesh.cpp
        GCR.cpp
        GCR.h
        MG.cpp
        MG.h
        ConjugateGradient.h)

