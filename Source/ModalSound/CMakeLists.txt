cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(ModalSound LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)

file(GLOB HEADER_FILES Source/*.h)
include_directories(Source/)

file(GLOB SRC_FILES Source/*.cpp)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen igl::core)