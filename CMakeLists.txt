cmake_minimum_required(VERSION 2.8)
include_directories("forest")
add_definitions(-g  -gdwarf-2)
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(TBB_INCLUDE_DIR /usr/include/ CACHE FILEPATH "")
include_directories(${TBB_INCLUDE_DIR})
set(TBB_LIB_DIR /usr/lib/ CACHE FILEPATH "")
link_directories(${TBB_LIB_DIR})

if(WIN32)
 set(TBB_LIBRARIES tbb_debug tbbmalloc_debug)
else()
 set(TBB_LIBRARIES tbb tbbmalloc)
endif()

set(BOOST_DIR /home/masa/project/lib/boost_1_55_0 CACHE FILEPATH "")
include_directories(${BOOST_DIR})
set(BOOST_LIB_DIR ${BOOST_DIR}/stage/lib CACHE FILEPATH "")
link_directories(${BOOST_LIB_DIR})
if(WIN32)
 set(BOOST_LIBS libboost_chrono-vc120-mt-gd-1_55 libboost_program_options-vc120-mt-gd-1_55 libboost_filesystem-vc120-mt-gd-1_55 libboost_system-vc120-mt-gd-1_55)
else() 
 set(BOOST_LIBS boost_program_options boost_filesystem boost_system boost_chrono)
endif()

set(EIGEN_DIR /usr/local/include CACHE FILEPATH "")
include_directories(${EIGEN_DIR})

file(GLOB FOREST_HEADERS forest/*.h util/*.h)
set(GENERAL_LIBRARIES ${TBB_LIBRARIES} ${BOOST_LIBS})

add_subdirectory(test)
add_subdirectory(msrc)
add_subdirectory(tumor)



