cmake_minimum_required(VERSION 3.7)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId})

add_executable(${ProjectId} main.cpp)

find_package(OpenCV REQUIRED)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectId} ${OpenCV_LIBS})

find_package(OpenMP REQUIRED)
target_link_libraries(${ProjectId} OpenMP::OpenMP_CXX)