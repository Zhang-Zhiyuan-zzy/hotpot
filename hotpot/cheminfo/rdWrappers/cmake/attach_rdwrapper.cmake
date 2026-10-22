get_property(
    hotpot_rdwrappers_attached GLOBAL PROPERTY HOTPOT_RDWRAPPERS_ATTACHED
)
if(hotpot_rdwrappers_attached)
    return()
endif()
set_property(GLOBAL PROPERTY HOTPOT_RDWRAPPERS_ATTACHED TRUE)

get_filename_component(
    HOTPOT_RDWRAPPERS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE
)

function(hotpot_attach_rdwrappers)
    include("${HOTPOT_RDWRAPPERS_SOURCE_DIR}/CMakeLists.txt")
endfunction()

cmake_language(
    DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL hotpot_attach_rdwrappers
)
