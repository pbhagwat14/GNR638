cmake_minimum_required(VERSION 3.18)
project(mytorch)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

find_package(OpenMP)

pybind11_add_module(mytorch
    cpp/mytorch.cpp
    cpp/bindings.cpp
)


if(OpenMP_CXX_FOUND)
    target_link_libraries(mytorch PRIVATE OpenMP::OpenMP_CXX)
endif()



if(MSVC)
    target_compile_options(mytorch PRIVATE /O2)
else()
    target_compile_options(mytorch PRIVATE -O3 -march=native)
endif()


# place compiled module inside python folder automatically
set_target_properties(mytorch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python

    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python
)

