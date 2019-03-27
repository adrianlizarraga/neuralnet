cmake_minimum_required(VERSION 3.5)

project(alai)

############################################################
# Create a library
############################################################

#Generate the shared library from the library sources
add_library(alai SHARED 
    src/MLPNetwork.cpp
)

target_include_directories(alai
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external
)

############################################################
# Create excutables for examples
############################################################

# XOR example
add_executable(xor-example
    examples/xor-example.cpp
)

target_link_libraries(xor-example
    PRIVATE 
        alai
)

# Classification example
add_executable(classification
    examples/classification.cpp
)

target_link_libraries(classification
    PRIVATE 
        alai
)