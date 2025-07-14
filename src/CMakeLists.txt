# Core library
add_library(darkpool-core STATIC
    config.cpp
)

target_link_libraries(darkpool-core
    PUBLIC
        Threads::Threads
        yaml-cpp
        TBB::tbb
        Boost::system
        Boost::thread
        Boost::filesystem
        Boost::program_options
        CURL::libcurl
        ZLIB::ZLIB
)

if(prometheus-cpp_FOUND OR TARGET PkgConfig::prometheus-cpp)
    target_link_libraries(darkpool-core PUBLIC
        $<IF:$<TARGET_EXISTS:prometheus-cpp::pull>,prometheus-cpp::pull,PkgConfig::prometheus-cpp>
    )
endif()

# Add subdirectories
add_subdirectory(core)
add_subdirectory(ml)
add_subdirectory(protocols)
add_subdirectory(strategies)
add_subdirectory(utils)
add_subdirectory(visualization)

# Main executable
add_executable(darkpool-detector main.cpp)
target_link_libraries(darkpool-detector
    darkpool-core
    darkpool-protocols
    darkpool-ml
    darkpool-strategies
    darkpool-utils
    darkpool-visualization
)
