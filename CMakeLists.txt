cmake_minimum_required(VERSION 3.16)
project(options_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP
find_package(OpenMP)

# pybind11 via FetchContent
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Sources
set(ENGINE_SOURCES
  src/black_scholes.cpp
  src/binomial.cpp
  src/monte_carlo.cpp
  src/implied_vol.cpp
)

# pybind11 module
pybind11_add_module(options_engine ${ENGINE_SOURCES} src/bindings.cpp)

target_include_directories(options_engine PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(OpenMP_CXX_FOUND)
  target_link_libraries(options_engine PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(options_engine PRIVATE USE_OPENMP)
endif()

# Install into python/ directory for convenience
install(TARGETS options_engine LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python)
