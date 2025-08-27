cmake_minimum_required(VERSION 3.10)
project(distributed_mmio LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# add_library(distributed_mmio STATIC ${SRC_FILES})

add_library(distributed_mmio STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/mmio.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/mmio_utils.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/mmio_c_wrapper.cpp)
target_include_directories(distributed_mmio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(mtx_to_bmtx ${CMAKE_CURRENT_SOURCE_DIR}/src/mtx_to_bmtx.cpp)
target_include_directories(mtx_to_bmtx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mtx_to_bmtx PRIVATE distributed_mmio MPI::MPI_CXX)
