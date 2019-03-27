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
# Create an executable
############################################################

# Add an executable with the above sources
add_executable(xor-example
    examples/xor-example.cpp
)

# link the new hello_library target with the hello_binary target
target_link_libraries(xor-example
    PRIVATE 
        alai
)