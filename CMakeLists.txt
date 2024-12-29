cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

project(WaveBlender LANGUAGES C CXX)

option(FORCE_CPU "Force CPU-only compilation (even if GPU is available)" OFF)

find_package(Eigen3 3.4 REQUIRED)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)

find_package(CUDA)

add_subdirectory(Source/ModalSound)

add_subdirectory(Source/FluidSound)

file(GLOB H_FILES Source/*.h Source/FDTD/*.h)
include_directories(Source/ Source/FDTD/ Source/ThirdParty/ Source/ModalSound/Source/ Source/FluidSound/Source/)

if (CUDA_FOUND AND NOT FORCE_CPU)
	message(STATUS "Building WaveBlender for GPU")
	enable_language(CUDA)

	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

	file(GLOB SRC_FILES Source/*.cu Source/FDTD/*.cu Source/Shaders/*.cpp Source/Shaders/*.cu)
	add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES} ${H_FILES})

	target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
	target_link_libraries(${PROJECT_NAME} ModalSound FluidSound Eigen3::Eigen igl::core cublas)

else ()
	message(STATUS "Building for CPU (not implemented)")

endif ()
