cmake_minimum_required(VERSION 3.16)
project(darkpool-detector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_LTO "Enable Link Time Optimization" OFF)
option(ENABLE_MARCH_NATIVE "Enable -march=native" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
        if(ENABLE_MARCH_NATIVE)
            add_compile_options(-march=native)
        else()
            add_compile_options(-march=x86-64-v3)
        endif()
        add_compile_options(-mtune=generic)
        add_compile_options(-ffast-math -fno-finite-math-only)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g3 -ggdb)
    endif()
    
    if(ENABLE_LTO)
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS system thread filesystem program_options)
find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)

find_package(prometheus-cpp CONFIG QUIET)
if(NOT prometheus-cpp_FOUND)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(prometheus-cpp IMPORTED_TARGET prometheus-cpp)
    endif()
endif()

find_package(GTest QUIET)
find_package(benchmark QUIET)
find_package(PkgConfig)

if(PkgConfig_FOUND)
    pkg_check_modules(WebSocketPP IMPORTED_TARGET websocketpp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options(-Wno-unused-parameter -Wno-unused-variable)
    
    if(DEFINED SANITIZER)
        if(SANITIZER STREQUAL "address")
            add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
            add_link_options(-fsanitize=address)
        elseif(SANITIZER STREQUAL "thread")
            add_compile_options(-fsanitize=thread -g)
            add_link_options(-fsanitize=thread)
        elseif(SANITIZER STREQUAL "undefined")
            add_compile_options(-fsanitize=undefined -g)
            add_link_options(-fsanitize=undefined)
        endif()
    endif()
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX2)
    add_compile_definitions(HAS_AVX2)
endif()
if(COMPILER_SUPPORTS_AVX512)
    add_compile_definitions(HAS_AVX512)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

install(DIRECTORY include/darkpool DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/darkpool-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/darkpool-config-version.cmake"
    DESTINATION lib/cmake/darkpool
)
