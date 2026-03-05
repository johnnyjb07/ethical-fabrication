cmake_minimum_required(VERSION 3.12)
project(VideoInferenceApp LANGUAGES CXX)

# 1. Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find pybind11
# This assumes pybind11 is installed via pip (pip install pybind11)
# or available in your system path.
find_package(pybind11 REQUIRED)

# 3. Define the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(ethicalfab main.cpp)

# 4. Link against pybind11 embedding library
# 'pybind11::embed' provides the necessary flags to embed the interpreter
target_link_libraries(ethicalfab PRIVATE pybind11::embed)

# 5. Copy Python scripts to the build directory (Optional but helpful)
# This ensures predictor.py is in the same folder as the executable
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/predictor.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})