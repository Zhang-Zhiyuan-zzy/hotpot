if(NOT DEFINED HOTPOT_RDKIT_BOOST_SOURCE)
    message(FATAL_ERROR "HOTPOT_RDKIT_BOOST_SOURCE is required")
endif()
if(NOT EXISTS "${HOTPOT_RDKIT_BOOST_SOURCE}/boost/version.hpp")
    message(FATAL_ERROR "locked Boost headers are missing")
endif()

set(Boost_FOUND TRUE)
set(Boost_VERSION 108600)
set(Boost_VERSION_STRING "1.86.0")
set(Boost_LIB_VERSION "1_86")
set(Boost_INCLUDE_DIR "${HOTPOT_RDKIT_BOOST_SOURCE}")
set(Boost_INCLUDE_DIRS "${HOTPOT_RDKIT_BOOST_SOURCE}")
set(Boost_LIBRARIES "")

if(NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED GLOBAL)
    set_target_properties(
        Boost::boost
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HOTPOT_RDKIT_BOOST_SOURCE}"
    )
endif()

