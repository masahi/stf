cmake_minimum_required(VERSION 2.8)
include_directories("forest")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
add_executable(stf main.cpp)
