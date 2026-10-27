cmake_minimum_required(VERSION 3.16)
project(hotpot_rdwrappers_sdk_consumer LANGUAGES CXX)

find_package(HotpotRDWrappers 0.5.4.5 EXACT CONFIG REQUIRED)

add_executable(hotpot_rdwrappers_sdk_consumer main.cpp)
target_compile_features(hotpot_rdwrappers_sdk_consumer PRIVATE cxx_std_17)
if(DEFINED HOTPOT_CONSUMER_GLIBCXX_ABI)
    target_compile_definitions(
        hotpot_rdwrappers_sdk_consumer
        PRIVATE _GLIBCXX_USE_CXX11_ABI=${HOTPOT_CONSUMER_GLIBCXX_ABI}
    )
endif()
target_link_libraries(
    hotpot_rdwrappers_sdk_consumer PRIVATE Hotpot::rdwrappers
)
