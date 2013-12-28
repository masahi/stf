find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(../)

add_executable(tumor tumor.cpp SpatialFeature.h util.h ../gmm/GMM.h ${FOREST_HEADERS} )
target_link_libraries(tumor ${GENERAL_LIBRARIES} ${ITK_LIBRARIES})
