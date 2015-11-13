file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../include/elegant-progressbars/version.hpp" elegantProgressbars_VERSION_MAJOR_HPP REGEX "#define ELEGANTPROGRESSBARS_VERSION_MAJOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../include/elegant-progressbars/version.hpp"  elegantProgressbars_VERSION_MINOR_HPP REGEX "#define ELEGANTPROGRESSBARS_VERSION_MINOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../include/elegant-progressbars/version.hpp"  elegantProgressbars_VERSION_PATCH_HPP REGEX "#define ELEGANTPROGRESSBARS_VERSION_PATCH ")

string(REGEX MATCH "([0-9]+)" elegantProgressbars_VERSION_MAJOR  ${elegantProgressbars_VERSION_MAJOR_HPP})
string(REGEX MATCH "([0-9]+)" elegantProgressbars_VERSION_MINOR  ${elegantProgressbars_VERSION_MINOR_HPP})
string(REGEX MATCH "([0-9]+)" elegantProgressbars_VERSION_PATCH  ${elegantProgressbars_VERSION_PATCH_HPP})

set(PACKAGE_VERSION "${elegantProgressbars_VERSION_MAJOR}.${elegantProgressbars_VERSION_MINOR}.${elegantProgressbars_VERSION_PATCH}")

# Check whether the requested PACKAGE_FIND_VERSION is exactly the one requested
if("${PACKAGE_VERSION}" EQUAL "${PACKAGE_FIND_VERSION}")
    set(PACKAGE_VERSION_EXACT TRUE)
else()
    set(PACKAGE_VERSION_EXACT FALSE)
endif()

# Check whether the requested PACKAGE_FIND_VERSION is compatible
if("${PACKAGE_VERSION}" VERSION_LESS "${PACKAGE_FIND_VERSION}")
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if ("${PACKAGE_VERSION}" VERSION_EQUAL "${PACKAGE_FIND_VERSION}")
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
