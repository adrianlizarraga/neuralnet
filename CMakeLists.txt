cmake_minimum_required(VERSION 3.5)

project(neuralnet)

############################################################
# Create a library
############################################################

#Generate the shared library from the library sources
add_library(neuralnet SHARED
    src/MLPNetwork.cpp
)

target_include_directories(neuralnet
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external
)

############################################################
# Create excutables for examples
############################################################

# XOR example
add_executable(xor
    examples/xor.cpp
)

target_link_libraries(xor
    neuralnet
)

# Classification example
add_executable(classification
    examples/classification.cpp
)

target_link_libraries(classification
    neuralnet
)