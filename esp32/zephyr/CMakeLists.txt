cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(pocketadcs_ads)
target_sources(app PRIVATE src/main.cpp)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Identity of the complete application inputs (not a timestamp or git HEAD:
# uncommitted shared headers must change it too). Paths and digests are sorted
# and concatenated; build_manifest records the same inputs and final binaries.
file(GLOB_RECURSE identity_inputs CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/*")
list(SORT identity_inputs)
set(identity_material "")
foreach(input IN LISTS identity_inputs)
    if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${input}")
        file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${input}" input_sha)
        string(APPEND identity_material "${input}:${input_sha}\n")
    endif()
endforeach()
string(SHA256 source_digest "${identity_material}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/adcs_identity_inputs.txt" "${identity_material}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/adcs_build_id.txt" "${source_digest}\n")
target_compile_definitions(app PRIVATE ADCS_BUILD_ID="${source_digest}")
# Target, not host, stack-usage evidence retained alongside the ELF.
target_compile_options(app PRIVATE -fstack-usage)
