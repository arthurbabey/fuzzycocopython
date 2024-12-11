file(GLOB SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/duktape/*.c
)

# Exclude fuzzy_coco_executable.cpp if needed
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*fuzzy_coco_executable\\.cpp$")

pybind11_add_module(fuzzycoco_core ${SOURCE_FILES})

target_include_directories(fuzzycoco_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/duktape
)

set_target_properties(fuzzycoco_core PROPERTIES PREFIX "" SUFFIX ".so")
