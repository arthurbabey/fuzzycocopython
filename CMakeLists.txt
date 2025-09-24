cmake_minimum_required(VERSION 3.23)
project(fuzzycocopython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

# --- turn off tests/bench/examples in the subproject (cover common flags) ---
set(BUILD_TESTING OFF CACHE BOOL "Disable testing in subprojects" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

add_compile_options(-Wno-error)

# bring in fuzzycoco
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fuzzycoco)

find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(bindings)
