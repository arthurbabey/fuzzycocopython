cmake_minimum_required(VERSION 3.16)
project(FuzzyCocoPython LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(pybind11 REQUIRED)

# Add the subdirectory containing your C++ extension code
add_subdirectory(fuzzycoco)

# Install the built extension into the Python package directory
# Assuming the extension target is named "fuzzycocopython"
install(TARGETS _fuzzycoco_core LIBRARY DESTINATION fuzzycocopython)
