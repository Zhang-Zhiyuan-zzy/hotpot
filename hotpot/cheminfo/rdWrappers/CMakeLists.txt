if(NOT HOTPOT_RDKIT_SOURCE_LOCK_VERIFIED)
    message(FATAL_ERROR
        "rdWrappers refuses an unverified RDKit source tree; "
        "use the offline locked-source driver"
    )
endif()
if(NOT RDKit_RELEASENAME STREQUAL "2026.09.1")
    message(FATAL_ERROR
        "rdWrappers requires RDKit 2026.09.1, got '${RDKit_RELEASENAME}'"
    )
endif()
if(NOT Boost_VERSION_STRING STREQUAL "1.86.0")
    message(FATAL_ERROR
        "rdWrappers requires locked Boost 1.86.0, got '${Boost_VERSION_STRING}'"
    )
endif()
if(NOT TARGET DistGeomHelpers_static)
    message(FATAL_ERROR "the pinned RDKit static ETKDG closure is unavailable")
endif()

set(HOTPOT_RDWRAPPERS_VERSION "0.5.4.0")
set(HOTPOT_RDWRAPPERS_SOVERSION "0")
set(HOTPOT_RDWRAPPERS_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(HOTPOT_RDWRAPPERS_BINARY_DIR "${CMAKE_BINARY_DIR}/hotpot-rdwrappers")
file(MAKE_DIRECTORY "${HOTPOT_RDWRAPPERS_BINARY_DIR}")
get_filename_component(
    HOTPOT_REPOSITORY_ROOT "${HOTPOT_RDWRAPPERS_ROOT}/../../.." ABSOLUTE
)

option(HOTPOT_RDWRAPPERS_BUILD_SDK "Build the standalone C/C++ SDK" ON)
option(HOTPOT_RDWRAPPERS_BUILD_PYTHON "Build the Python extension" ON)
set(
    HOTPOT_RDWRAPPERS_PYTHON_OUTPUT_DIR "" CACHE PATH
    "Optional build-tree output directory for _rd_native"
)

set(HOTPOT_RDWRAPPERS_CORE_SOURCES
    "${HOTPOT_RDWRAPPERS_ROOT}/_native/src/c_api.cpp"
    "${HOTPOT_RDWRAPPERS_ROOT}/_native/src/molecule_builder.cpp"
    "${HOTPOT_RDWRAPPERS_ROOT}/_native/src/sampler.cpp"
    "${HOTPOT_RDWRAPPERS_ROOT}/_native/src/validation.cpp"
)

add_library(hotpot_rdwrappers_core_objects OBJECT ${HOTPOT_RDWRAPPERS_CORE_SOURCES})
target_compile_features(hotpot_rdwrappers_core_objects PRIVATE cxx_std_20)
target_compile_definitions(
    hotpot_rdwrappers_core_objects PRIVATE HOTPOT_RDWRAPPERS_BUILDING_LIBRARY
)
target_include_directories(
    hotpot_rdwrappers_core_objects
    PRIVATE
        "${HOTPOT_RDWRAPPERS_ROOT}/_native/include"
        "${HOTPOT_REPOSITORY_ROOT}/hotpot/cheminfo/_native/include"
)
target_link_libraries(hotpot_rdwrappers_core_objects PRIVATE DistGeomHelpers_static)
set_target_properties(
    hotpot_rdwrappers_core_objects
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

if(HOTPOT_RDWRAPPERS_BUILD_SDK)
    include(CMakePackageConfigHelpers)

    add_library(
        hotpot_rdwrappers SHARED
        $<TARGET_OBJECTS:hotpot_rdwrappers_core_objects>
    )
    add_library(Hotpot::rdwrappers ALIAS hotpot_rdwrappers)
    set_target_properties(
        hotpot_rdwrappers
        PROPERTIES
            EXPORT_NAME rdwrappers
            OUTPUT_NAME hotpot_rdwrappers
            VERSION "${HOTPOT_RDWRAPPERS_VERSION}"
            SOVERSION "${HOTPOT_RDWRAPPERS_SOVERSION}"
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
            INSTALL_RPATH ""
            INSTALL_RPATH_USE_LINK_PATH OFF
            INSTALL_REMOVE_ENVIRONMENT_RPATH ON
    )
    target_link_libraries(hotpot_rdwrappers PRIVATE DistGeomHelpers_static)
    target_include_directories(
        hotpot_rdwrappers
        INTERFACE
            $<BUILD_INTERFACE:${HOTPOT_RDWRAPPERS_ROOT}/_native/include>
            $<BUILD_INTERFACE:${HOTPOT_REPOSITORY_ROOT}/hotpot/cheminfo/_native/include>
            $<INSTALL_INTERFACE:include>
    )
    if(UNIX AND NOT APPLE)
        target_link_options(
            hotpot_rdwrappers
            PRIVATE
                "LINKER:--version-script=${HOTPOT_RDWRAPPERS_ROOT}/_native/src/exports.map"
                "LINKER:--exclude-libs,ALL"
        )
        set_property(
            TARGET hotpot_rdwrappers APPEND PROPERTY LINK_DEPENDS
            "${HOTPOT_RDWRAPPERS_ROOT}/_native/src/exports.map"
        )
        add_custom_command(
            TARGET hotpot_rdwrappers POST_BUILD
            COMMAND "${CMAKE_COMMAND}"
                "-DHOTPOT_BINARY=$<TARGET_FILE:hotpot_rdwrappers>"
                "-DHOTPOT_READELF=${CMAKE_READELF}"
                -P "${HOTPOT_RDWRAPPERS_ROOT}/cmake/remove_build_rpath.cmake"
            VERBATIM
        )
    endif()

    configure_package_config_file(
        "${HOTPOT_RDWRAPPERS_ROOT}/cmake/HotpotRDWrappersConfig.cmake.in"
        "${HOTPOT_RDWRAPPERS_BINARY_DIR}/HotpotRDWrappersConfig.cmake"
        INSTALL_DESTINATION "lib/cmake/HotpotRDWrappers"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )
    write_basic_package_version_file(
        "${HOTPOT_RDWRAPPERS_BINARY_DIR}/HotpotRDWrappersConfigVersion.cmake"
        VERSION "${HOTPOT_RDWRAPPERS_VERSION}"
        COMPATIBILITY ExactVersion
    )

    install(
        TARGETS hotpot_rdwrappers
        EXPORT HotpotRDWrappersTargets
        LIBRARY DESTINATION lib COMPONENT rdwrappers_sdk
        RUNTIME DESTINATION bin COMPONENT rdwrappers_sdk
    )
    install(
        DIRECTORY "${HOTPOT_RDWRAPPERS_ROOT}/_native/include/hotpot/rdWrappers"
        DESTINATION include/hotpot
        COMPONENT rdwrappers_sdk
    )
    install(
        DIRECTORY "${HOTPOT_REPOSITORY_ROOT}/hotpot/cheminfo/_native/include/hotpot/native"
        DESTINATION include/hotpot
        COMPONENT rdwrappers_sdk
    )
    install(
        EXPORT HotpotRDWrappersTargets
        FILE HotpotRDWrappersTargets.cmake
        NAMESPACE Hotpot::
        DESTINATION lib/cmake/HotpotRDWrappers
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES
            "${HOTPOT_RDWRAPPERS_BINARY_DIR}/HotpotRDWrappersConfig.cmake"
            "${HOTPOT_RDWRAPPERS_BINARY_DIR}/HotpotRDWrappersConfigVersion.cmake"
        DESTINATION lib/cmake/HotpotRDWrappers
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES
            "${HOTPOT_REPOSITORY_ROOT}/LICENSE"
            "${HOTPOT_RDWRAPPERS_ROOT}/_native/THIRD_PARTY_NOTICES.md"
        DESTINATION share/licenses/hotpot-rdwrappers
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES "${CMAKE_SOURCE_DIR}/license.txt"
        DESTINATION share/licenses/hotpot-rdwrappers
        RENAME LICENSE.RDKit
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES "${HOTPOT_RDKIT_BOOST_SOURCE}/LICENSE_1_0.txt"
        DESTINATION share/licenses/hotpot-rdwrappers
        RENAME LICENSE.Boost
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES "${FETCHCONTENT_SOURCE_DIR_CATCH2}/LICENSE.txt"
        DESTINATION share/licenses/hotpot-rdwrappers
        RENAME LICENSE.Catch2
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES "${FETCHCONTENT_SOURCE_DIR_BETTER_ENUMS}/LICENSE.md"
        DESTINATION share/licenses/hotpot-rdwrappers
        RENAME LICENSE.better-enums
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES "${HOTPOT_RDKIT_EIGEN_SOURCE}/COPYING.MPL2"
        DESTINATION share/licenses/hotpot-rdwrappers
        RENAME LICENSE.Eigen
        COMPONENT rdwrappers_sdk
    )
    install(
        FILES
            "${CMAKE_SOURCE_DIR}/External/RingFamilies/RingDecomposerLib/LICENSE"
        DESTINATION share/licenses/hotpot-rdwrappers
        RENAME LICENSE.RingDecomposerLib
        COMPONENT rdwrappers_sdk
    )
endif()

if(HOTPOT_RDWRAPPERS_BUILD_PYTHON)
    set(CMAKE_DISABLE_FIND_PACKAGE_Python OFF)
    find_package(Python 3.10...<3.15 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 3 CONFIG REQUIRED)

    pybind11_add_module(
        hotpot_rd_native MODULE
        "${HOTPOT_RDWRAPPERS_ROOT}/_native/src/python_binding.cpp"
        $<TARGET_OBJECTS:hotpot_rdwrappers_core_objects>
    )
    target_compile_features(hotpot_rd_native PRIVATE cxx_std_20)
    target_include_directories(
        hotpot_rd_native
        PRIVATE "${HOTPOT_RDWRAPPERS_ROOT}/_native/include"
    )
    target_link_libraries(hotpot_rd_native PRIVATE DistGeomHelpers_static)
    set_target_properties(
        hotpot_rd_native
        PROPERTIES
            OUTPUT_NAME "_rd_native"
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
            INSTALL_RPATH ""
            INSTALL_RPATH_USE_LINK_PATH OFF
            INSTALL_REMOVE_ENVIRONMENT_RPATH ON
    )
    if(HOTPOT_RDWRAPPERS_PYTHON_OUTPUT_DIR)
        set_target_properties(
            hotpot_rd_native
            PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${HOTPOT_RDWRAPPERS_PYTHON_OUTPUT_DIR}"
        )
    endif()
    if(UNIX AND NOT APPLE)
        target_link_options(
            hotpot_rd_native
            PRIVATE
                "LINKER:--version-script=${HOTPOT_RDWRAPPERS_ROOT}/_native/src/python_exports.map"
                "LINKER:--exclude-libs,ALL"
        )
        set_property(
            TARGET hotpot_rd_native APPEND PROPERTY LINK_DEPENDS
            "${HOTPOT_RDWRAPPERS_ROOT}/_native/src/python_exports.map"
        )
        add_custom_command(
            TARGET hotpot_rd_native POST_BUILD
            COMMAND "${CMAKE_COMMAND}"
                "-DHOTPOT_BINARY=$<TARGET_FILE:hotpot_rd_native>"
                "-DHOTPOT_READELF=${CMAKE_READELF}"
                -P "${HOTPOT_RDWRAPPERS_ROOT}/cmake/remove_build_rpath.cmake"
            VERBATIM
        )
    endif()

    install(
        TARGETS hotpot_rd_native
        LIBRARY DESTINATION hotpot/cheminfo/rdWrappers
            COMPONENT rdwrappers_python
        RUNTIME DESTINATION hotpot/cheminfo/rdWrappers
            COMPONENT rdwrappers_python
    )
    install(
        FILES "${HOTPOT_RDWRAPPERS_ROOT}/_rd_native.pyi"
        DESTINATION hotpot/cheminfo/rdWrappers
        COMPONENT rdwrappers_python
    )
    install(
        FILES
            "${HOTPOT_RDWRAPPERS_ROOT}/_native/THIRD_PARTY_NOTICES.md"
            "${CMAKE_SOURCE_DIR}/license.txt"
            "${HOTPOT_RDKIT_BOOST_SOURCE}/LICENSE_1_0.txt"
            "${FETCHCONTENT_SOURCE_DIR_CATCH2}/LICENSE.txt"
            "${FETCHCONTENT_SOURCE_DIR_BETTER_ENUMS}/LICENSE.md"
            "${HOTPOT_RDKIT_EIGEN_SOURCE}/COPYING.MPL2"
            "${CMAKE_SOURCE_DIR}/External/RingFamilies/RingDecomposerLib/LICENSE"
        DESTINATION hotpot/cheminfo/rdWrappers/_native/licenses
        COMPONENT rdwrappers_python
    )
endif()
