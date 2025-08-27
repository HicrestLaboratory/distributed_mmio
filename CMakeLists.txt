cmake_minimum_required(VERSION 3.10)
project(distributed_mmio LANGUAGES CXX)

option(DMMIO_ENABLE_MPI "Enable MPI for distributed reading" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/common)

### CCUTILS dependency ###
set(CCUTILS_ENABLE_MPI  ON  CACHE BOOL "")
include(FetchContent)
FetchContent_Declare(
    ccutils
    GIT_REPOSITORY https://github.com/ThomasPasquali/ccutils.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(ccutils)

### Distributed MMIO library ###
add_library(distributed_mmio STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/mmio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/mmio_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/mmio_c_wrapper.cpp)
target_link_libraries(distributed_mmio PRIVATE ccutils::ccutils)

### MTX to BMTX converter ###
option(DMMIO_TOOLS "Build mtx_to_bmtx tool" ON)
if(DMMIO_TOOLS)
    add_executable(mtx_to_bmtx ${CMAKE_CURRENT_SOURCE_DIR}/src/bmtx/mtx_to_bmtx.cpp)
    target_link_libraries(mtx_to_bmtx PRIVATE distributed_mmio ccutils::ccutils)
endif()

# ---------------- MPI (Optional) ----------------
if(DMMIO_ENABLE_MPI)
    message("DMMIO: Enabling MPI module")
    find_package(MPI REQUIRED COMPONENTS CXX)  # Provides MPI::MPI_C and MPI::MPI_CXX
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/common)
    target_sources(distributed_mmio PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/distributed/distributed_mmio.cpp
    )
    target_link_libraries(distributed_mmio PRIVATE MPI::MPI_C MPI::MPI_CXX ccutils::ccutils_mpi)
    target_link_libraries(mtx_to_bmtx PRIVATE MPI::MPI_C MPI::MPI_CXX ccutils::ccutils_mpi)
endif()