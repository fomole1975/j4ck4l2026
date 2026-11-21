cmake_minimum_required(VERSION 3.20)
project(shopping_sessions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

# PyArrow's binary wheels ship the same native Parquet headers and shared
# libraries as the Apache C++ distribution. Python is used only at configure time.
find_program(Python3_EXECUTABLE NAMES python3 REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_library_dirs()[0])"
    OUTPUT_VARIABLE arrow_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
foreach(lib IN ITEMS parquet arrow)
    file(GLOB libraries "${arrow_dir}/lib${lib}.so.*" "${arrow_dir}/lib${lib}.[0-9]*.dylib")
    if(NOT libraries)
        message(FATAL_ERROR "PyArrow shared library ${lib} is missing in ${arrow_dir}")
    endif()
    list(GET libraries 0 path)
    add_library(${lib}_native SHARED IMPORTED)
    set_target_properties(${lib}_native PROPERTIES IMPORTED_LOCATION "${path}")
endforeach()

set(pipeline_sources src/main.cpp src/input.cpp src/key.cpp src/engine.cpp src/output.cpp src/run_parquet.cpp)

add_library(shopping-writer SHARED src/output_arrow.cpp)
target_include_directories(shopping-writer SYSTEM PRIVATE "${arrow_dir}/include")
target_link_libraries(shopping-writer PRIVATE parquet_native arrow_native)
set_target_properties(shopping-writer PROPERTIES BUILD_RPATH "${arrow_dir}" INSTALL_RPATH "$ORIGIN")
install(TARGETS shopping-writer LIBRARY DESTINATION lib COMPONENT writer)

function(add_pipeline name)
    add_executable(${name} ${pipeline_sources})
    add_dependencies(${name} shopping-writer)
    target_compile_definitions(${name} PRIVATE SHOPPING_WRITER_NAME="$<TARGET_FILE_NAME:shopping-writer>")
    target_link_libraries(${name} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${arrow_dir}" INSTALL_RPATH "$ORIGIN/../lib")
    install(TARGETS ${name} RUNTIME DESTINATION bin COMPONENT ${name})
endfunction()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    # Dispatch once, outside the timed inner loops. Separate executables let the
    # compiler inline the key and parsing operations consistently at each ISA.
    add_executable(shopping-sessions src/dispatch.cpp)
    install(TARGETS shopping-sessions RUNTIME DESTINATION bin COMPONENT shopping-sessions)
    foreach(variant IN ITEMS generic avx2 avx512 vbmi2)
        add_pipeline(shopping-sessions-${variant})
        target_compile_options(shopping-sessions-${variant} PRIVATE -mtune=skylake-avx512 -Wa,-mbranches-within-32B-boundaries -fno-semantic-interposition -fno-plt -fomit-frame-pointer)
    endforeach()
    target_compile_options(shopping-sessions-avx2 PRIVATE -mavx2 -mbmi -mbmi2 -mpopcnt -mpclmul)
    target_compile_options(shopping-sessions-avx512 PRIVATE -mavx2 -mbmi -mbmi2 -mpopcnt -mpclmul -mavx512f -mavx512bw)
    target_compile_options(shopping-sessions-vbmi2 PRIVATE -mavx2 -mbmi -mbmi2 -mpopcnt -mpclmul -mavx512f -mavx512bw -mavx512vbmi2)
else()
    add_pipeline(shopping-sessions)
endif()
