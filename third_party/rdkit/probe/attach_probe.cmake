get_property(probe_already_attached GLOBAL PROPERTY HOTPOT_RDKIT_PROBE_ATTACHED)
if(probe_already_attached)
    return()
endif()
set_property(GLOBAL PROPERTY HOTPOT_RDKIT_PROBE_ATTACHED TRUE)

set(HOTPOT_RDKIT_PROBE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/distgeom_probe.cpp")

function(hotpot_record_target target output)
    get_property(seen GLOBAL PROPERTY HOTPOT_RDKIT_PROBE_SEEN)
    if(target IN_LIST seen)
        return()
    endif()
    list(APPEND seen "${target}")
    set_property(GLOBAL PROPERTY HOTPOT_RDKIT_PROBE_SEEN "${seen}")

    get_target_property(target_type "${target}" TYPE)
    get_target_property(direct_links "${target}" LINK_LIBRARIES)
    get_target_property(interface_links "${target}" INTERFACE_LINK_LIBRARIES)
    if(direct_links STREQUAL "direct_links-NOTFOUND")
        set(direct_links "")
    endif()
    if(interface_links STREQUAL "interface_links-NOTFOUND")
        set(interface_links "")
    endif()
    set(all_links ${direct_links} ${interface_links})
    list(REMOVE_DUPLICATES all_links)
    string(REPLACE ";" "," rendered_links "${all_links}")
    file(APPEND "${output}" "${target} [${target_type}] -> ${rendered_links}\n")

    foreach(link IN LISTS all_links)
        if(TARGET "${link}")
            hotpot_record_target("${link}" "${output}")
        endif()
    endforeach()
endfunction()

function(hotpot_add_distgeom_probe)
    if(NOT RDKit_RELEASENAME STREQUAL "2026.09.1")
        message(FATAL_ERROR "expected RDKit 2026.09.1, got ${RDKit_RELEASENAME}")
    endif()
    if(NOT TARGET DistGeomHelpers_static)
        message(FATAL_ERROR "RDKit static DistGeomHelpers target is missing")
    endif()

    add_executable(hotpot_distgeom_probe "${HOTPOT_RDKIT_PROBE_SOURCE}")
    target_compile_features(hotpot_distgeom_probe PRIVATE cxx_std_20)
    target_link_libraries(hotpot_distgeom_probe PRIVATE DistGeomHelpers_static)

    set(link_map "${CMAKE_BINARY_DIR}/distgeom-target-link-map.txt")
    file(WRITE "${link_map}" "")
    hotpot_record_target(hotpot_distgeom_probe "${link_map}")
endfunction()

cmake_language(
    DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL hotpot_add_distgeom_probe
)

