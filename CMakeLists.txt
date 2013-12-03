cmake_minimum_required(VERSION 2.8)
include_directories("forest")
add_definitions(-g  -gdwarf-2)
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

set(TBB_INCLUDE_DIR /usr/include/ CACHE FILEPATH "")
include_directories(${TBB_INCLUDE_DIR})
set(TBB_LIB_DIR /usr/lib/ CACHE FILEPATH "")
link_directories(${TBB_LIB_DIR})
set(TBB_LIBRARIES tbb tbbmalloc)

set(BOOST_DIR /home/masa/boost_1_55_0 CACHE FILEPATH "")
include_directories(${BOOST_DIR})
set(BOOST_LIB_DIR ${BOOST_DIR}/stage/lib CACHE FILEPATH "")
link_directories(${BOOST_LIB_DIR})
set(BOOST_LIBS boost_program_options boost_filesystem boost_system)

file(GLOB HEADERS forest/*.h)
set(LIBRARIES ${OpenCV_LIBS} ${TBB_LIBRARIES} ${BOOST_LIBS})

add_executable(test test.cpp ${HEADERS})
target_link_libraries(test ${LIBRARIES})

add_executable(segmentation segmentation.cpp ${HEADERS})
target_link_libraries(segmentation ${LIBRARIES})
