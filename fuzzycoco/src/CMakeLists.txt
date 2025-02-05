cmake_minimum_required(VERSION 3.16)
project(fuzzycoco VERSION 1.0 LANGUAGES C CXX)

# we require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
# set(CMAKE_CXX_FLAGS "-g -fprofile-arcs -ftest-coverage")


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
)

# ==== fuzzy coco executable
add_executable(fuzzycoco.exe fuzzy_coco_executable.cpp)
target_link_libraries(fuzzycoco.exe fuzzycoco_static)

add_library(fuzzycoco SHARED ${SOURCE_FILES})
target_include_directories(fuzzycoco PRIVATE  duktape)

add_library(fuzzycoco_static STATIC ${SOURCE_FILES})
target_include_directories(fuzzycoco_static PRIVATE  duktape)
