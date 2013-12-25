find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(../)

add_executable(tumor tumor.cpp util.h ${FOREST_HEADERS} )
target_link_libraries(tumor ${GENERAL_LIBRARIES} ${ITK_LIBRARIES})
