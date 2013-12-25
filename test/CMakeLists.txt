include_directories(../)

add_executable(test test.cpp util.h ${FOREST_HEADERS})
target_link_libraries(test ${GENERAL_LIBRARIES})

add_executable(test_gmm test_gmm.cpp)