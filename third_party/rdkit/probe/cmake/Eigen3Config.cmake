if(NOT DEFINED HOTPOT_RDKIT_EIGEN_SOURCE)
    message(FATAL_ERROR "HOTPOT_RDKIT_EIGEN_SOURCE is required")
endif()
if(NOT EXISTS "${HOTPOT_RDKIT_EIGEN_SOURCE}/Eigen/Core")
    message(FATAL_ERROR "locked Eigen headers are missing")
endif()

set(Eigen3_FOUND TRUE)
set(Eigen3_VERSION "3.4.0")
set(EIGEN3_INCLUDE_DIR "${HOTPOT_RDKIT_EIGEN_SOURCE}")
set(EIGEN3_INCLUDE_DIRS "${HOTPOT_RDKIT_EIGEN_SOURCE}")

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(
        Eigen3::Eigen
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HOTPOT_RDKIT_EIGEN_SOURCE}"
    )
endif()

