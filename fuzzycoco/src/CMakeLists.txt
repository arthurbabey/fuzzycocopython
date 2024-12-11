cmake_minimum_required(VERSION 3.16)
project(fuzzycoco VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(pybind11 REQUIRED)

# Include directories for headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/duktape
    # Add paths to python includes if not found automatically
    # /Users/arthurbabey/miniconda3/envs/fuzzypy-env/include/python3.11
    # /Users/arthurbabey/miniconda3/envs/fuzzypy-env/lib/python3.11/site-packages/pybind11/include
)

# Main source files
set(SOURCE_FILES
    bitarray.cpp
    coevolution_engine.cpp
    crossover_method.cpp
    dataframe.cpp
    evolution_engine.cpp
    file_utils.cpp
    fuzzy_coco.cpp
    fuzzy_coco_params.cpp
    fuzzy_coco_script_runner.cpp
    fuzzy_rule.cpp
    fuzzy_system.cpp
    fuzzy_system_fitness.cpp
    fuzzy_system_metrics_computer.cpp
    fuzzy_variable.cpp
    fuzzy_variables_db.cpp
    genome_codec.cpp
    mutation_method.cpp
    named_list.cpp
    selection_method.cpp
    duktape/duktape.c
    # fuzzy_coco_executable.cpp not necessarily added here if included in bindings.cpp
)

add_library(fuzzymodel MODULE bindings.cpp ${SOURCE_FILES})
target_link_libraries(fuzzymodel PRIVATE pybind11::module)

set_target_properties(fuzzymodel PROPERTIES PREFIX "" SUFFIX ".so")

message(STATUS "Building Python bindings (fuzzymodel.so)")
