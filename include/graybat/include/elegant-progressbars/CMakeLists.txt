project(elegant-progressbars-dev)
cmake_minimum_required(VERSION 3.0.2)



###############################################################################
# Compiler flags
###############################################################################
option(CMAKE_BUILD_TYPE "build type" Slow) # to show the progress. Don't use in real code!
SET (CMAKE_CXX_FLAGS                "-Wall -Wextra -std=c++11")
SET (CMAKE_CXX_FLAGS_SLOW           "-O0")



###############################################################################
# Progressbar for a More Civilized Age Config
###############################################################################
find_package(elegant-progressbars 1.0.0 REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")



###############################################################################
# Installation for Unix systems 
###############################################################################
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
    DESTINATION "lib/elegant-progressbars"
    )

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    DESTINATION "lib/elegant-progressbars"
    )



###############################################################################
# Build executables
###############################################################################
add_executable(ProgressBarExample EXCLUDE_FROM_ALL examples/progressbar_example.cpp)



###############################################################################
# Custom targets
###############################################################################
# DOXYGEN
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)
