cmake_minimum_required(VERSION 2.8)
include_directories("forest")
add_definitions(-g  -gdwarf-2)
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})

set(BOOST_DIR /home/masa/boost_1_55_0)
include_directories(${BOOST_DIR})
set(BOOST_LIB_DIR ${BOOST_DIR}/stage/lib)
link_directories(${BOOST_LIB_DIR})
file(GLOB BOOST_LIBS ${BOOST_LIB_DIR}/*.so)

file(GLOB HEADERS forest/*.h)
set(LIBRARIES ${OpenCV_LIBS} ${TBB_LIBRARIES} ${BOOST_LIBS})

add_executable(test test.cpp ${HEADERS})
target_link_libraries(test ${LIBRARIES})

add_executable(segmentation segmentation.cpp ${HEADERS})
target_link_libraries(segmentation ${LIBRARIES})
