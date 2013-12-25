find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(GCO_DIR /home/masa/project/lib/gco CACHE FILEPATH "")
include_directories(${GCO_DIR})
link_directories(${GCO_DIR}/build/Debug ${GCO_DIR}/build)
set(GCO_LIB gco)

include_directories(../)

add_executable(msrc msrc.cpp util.h ${FOREST_HEADERS} )
target_link_libraries(msrc ${GENERAL_LIBRARIES} ${OpenCV_LIBS} ${GCO_LIB})


