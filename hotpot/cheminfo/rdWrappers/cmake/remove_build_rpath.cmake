if(NOT DEFINED HOTPOT_BINARY OR NOT EXISTS "${HOTPOT_BINARY}")
    message(FATAL_ERROR "rdWrappers RPATH audit received no binary")
endif()

file(RPATH_REMOVE FILE "${HOTPOT_BINARY}")

if(DEFINED HOTPOT_READELF AND EXISTS "${HOTPOT_READELF}")
    execute_process(
        COMMAND "${HOTPOT_READELF}" -d "${HOTPOT_BINARY}"
        RESULT_VARIABLE audit_status
        OUTPUT_VARIABLE dynamic_section
        ERROR_VARIABLE audit_error
    )
    if(NOT audit_status EQUAL 0)
        message(FATAL_ERROR
            "readelf failed while auditing ${HOTPOT_BINARY}: ${audit_error}"
        )
    endif()
    if(dynamic_section MATCHES "\\((RPATH|RUNPATH)\\).*[[]/")
        message(FATAL_ERROR
            "rdWrappers binary retains an absolute runtime path: "
            "${HOTPOT_BINARY}"
        )
    endif()
    if(dynamic_section MATCHES "Shared library: [[]lib(RDKit|boost)")
        message(FATAL_ERROR
            "rdWrappers binary dynamically links a private RDKit/Boost library: "
            "${HOTPOT_BINARY}"
        )
    endif()
endif()
