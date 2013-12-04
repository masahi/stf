cmake_minimum_required(VERSION 2.8)
include_directories("forest")
add_definitions(-g  -gdwarf-2)
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(TBB_INCLUDE_DIR /usr/include/ CACHE FILEPATH "")
include_directories(${TBB_INCLUDE_DIR})
set(TBB_LIB_DIR /usr/lib/ CACHE FILEPATH "")
link_directories(${TBB_LIB_DIR})

set(TBB_LIBRARIES tbb_debug tbbmalloc_debug)

set(GCO_DIR "" CACHE FILEPATH "")
include_directories(${GCO_DIR})
link_directories(${GCO_DIR}/build/Debug)
set(GCO_LIB gco)

set(BOOST_DIR /home/masa/boost_1_55_0 CACHE FILEPATH "")
include_directories(${BOOST_DIR})
set(BOOST_LIB_DIR ${BOOST_DIR}/stage/lib CACHE FILEPATH "")
link_directories(${BOOST_LIB_DIR})
if(WIN32)
 set(BOOST_LIBS libboost_program_options-vc120-mt-gd-1_55 libboost_filesystem-vc120-mt-gd-1_55 libboost_system-vc120-mt-gd-1_55)
else() 
 set(BOOST_LIBS boost_program_options boost_filesystem boost_system)
endif()

set(EIGEN_DIR /usr/local/include CACHE FILEPATH "")
include_directories(${EIGEN_DIR})

file(GLOB HEADERS forest/*.h)
set(LIBRARIES ${OpenCV_LIBS} ${TBB_LIBRARIES} ${BOOST_LIBS} ${GCO_LIB})

add_executable(test test.cpp ${HEADERS})
target_link_libraries(test ${LIBRARIES})

add_executable(segmentation segmentation.cpp ${HEADERS})
target_link_libraries(segmentation ${LIBRARIES})
