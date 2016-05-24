project(cracen)
cmake_minimum_required(VERSION 3.0.1)

###############################################################################
# PThreads
###############################################################################

find_package(Threads REQUIRED)
set(cracen_LIBARIES ${CMAKE_THREAD_LIBS_INIT})

###############################################################################
# Boost
###############################################################################

find_package(Boost 1.55.0 REQUIRED regex system)
set(cracen_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
set(cracen_LIBARIES ${cracen_LIBARIES} ${Boost_LIBRARIES})

################################################################################
# GRAYBAT
################################################################################
find_package(graybat REQUIRED CONFIG)
set(cracen_INCLUDE_DIRS ${cracen_INCLUDE_DIRS} ${graybat_INCLUDE_DIRS})
set(cracen_LIBARIES ${cracen_LIBARIES} ${graybat_LIBRARIES})