find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(tumor tumor.cpp ${FOREST_HEADERS} )
target_link_libraries(tumor ${GENERAL_LIBRARIES} ${ITK_LIBRARIES})
