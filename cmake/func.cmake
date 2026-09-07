# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# define functions

# usage: recursive_add_subdirectory()
# If OP_LIST is set (comma-separated), only include those ops; otherwise include all.
macro(recursive_add_subdirectory)
    file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUB_DIR ${CURRENT_DIRS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/${ARCH_DIR}/CMakeLists.txt")
            set(_INCLUDE_OP 1)
            if(NOT "${OP_LIST}" STREQUAL "")
                set(_INCLUDE_OP 0)
                foreach(_OP ${OP_LIST})
                    if("${_OP}" STREQUAL "${SUB_DIR}")
                        set(_INCLUDE_OP 1)
                        break()
                    endif()
                endforeach()
            endif()
            if(_INCLUDE_OP)
                add_subdirectory(${SUB_DIR}/${ARCH_DIR})
            endif()
        endif()
        # ops/ 容器: 类别下 ops/{算子名}/{ARCH_DIR} 两层嵌套 (kernel 算子集中管理)
        # ops/ 容器: 类别下 ops/{算子名}/{ARCH_DIR} 两层嵌套 (kernel 算子集中管理)
        if("${SUB_DIR}" STREQUAL "ops")
            file(GLOB _OPS_SUB_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/ops/*)
            foreach(_OPS_SUB ${_OPS_SUB_DIRS})
                if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_OPS_SUB}/${ARCH_DIR}/CMakeLists.txt")
                    get_filename_component(_OPS_NAME "${_OPS_SUB}" NAME)
                    set(_INCLUDE_OP 1)
                    if(NOT "${OP_LIST}" STREQUAL "")
                        set(_INCLUDE_OP 0)
                        foreach(_OP ${OP_LIST})
                            if("${_OP}" STREQUAL "${_OPS_NAME}")
                                set(_INCLUDE_OP 1)
                                break()
                            endif()
                        endforeach()
                    endif()
                    if(_INCLUDE_OP)
                        add_subdirectory(${_OPS_SUB}/${ARCH_DIR})
                    endif()
                endif()
            endforeach()
        endif()
    endforeach()
endmacro()

# usage: add_sources()
# Collect source files from current op directory and accumulate into global OP_ALL_SOURCES.
macro(add_sources)
    # clear CMAKE_CXX_FLAGS to avoid affecting bisheng compile
    unset(CMAKE_CXX_FLAGS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_C_COMPILER ${BISHENG})
    set(CMAKE_CXX_COMPILER ${BISHENG})
    set(CMAKE_LINKER ${BISHENG})

    message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

    # get parent dir name as OP_NAME
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${PARENT_DIR} NAME)
    message(STATUS "OP_NAME: ${OP_NAME}")

    # get compile flags for current op
    set(COMPILE_FLAGS "--npu-arch=${NPU_ARCH} -xasc --cce-long-scbz=true --gcc-toolchain=/usr -mno-outline-atomics -I${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/tiling -I${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/utils ")
    message(STATUS "COMPILE FLAGS: ${COMPILE_FLAGS}")

    # recursively get source files
    file(GLOB_RECURSE SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.asc)
    message(STATUS "SOURCE FILES: ${SOURCE_FILES}")
    if(SOURCE_FILES STREQUAL "")
        message(FATAL_ERROR "No source files found in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    # accumulate sources into global property
    foreach(SRC ${SOURCE_FILES})
        set(FULL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
        set_property(SOURCE ${FULL_PATH} PROPERTY COMPILE_FLAGS "${COMPILE_FLAGS}")
        set_property(SOURCE ${FULL_PATH} PROPERTY LANGUAGE CXX)
        set_property(GLOBAL APPEND PROPERTY OP_ALL_SOURCES ${FULL_PATH})
    endforeach()
endmacro()

# usage: add_op_library()
# Create a single shared library from all collected sources.
function(add_op_library)
    get_property(ALL_SOURCES GLOBAL PROPERTY OP_ALL_SOURCES)

    set(COMPILE_FLAGS "--npu-arch=${NPU_ARCH} -xasc --cce-long-scbz=true --gcc-toolchain=/usr -mno-outline-atomics -I${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/tiling -I${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/utils ")
    foreach(SRC ${ALL_SOURCES})
        set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "${COMPILE_FLAGS}")
    endforeach()

    set(TARGET_NAME ops_multimodal_fusion)
    add_library(${TARGET_NAME} SHARED ${ALL_SOURCES})
    set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PREFIX "lib"
        SUFFIX ".so"
        LINKER_LANGUAGE CXX
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})
    target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_DIRECTORIES})
    target_link_directories(${TARGET_NAME} PRIVATE ${LINK_DIRECTORIES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LINK_LIBRARIES})

    get_property(COMPILE_DEFS GLOBAL PROPERTY OP_COMPILE_DEFINITIONS)
    if(COMPILE_DEFS)
        target_compile_definitions(${TARGET_NAME} PRIVATE ${COMPILE_DEFS})
    endif()
endfunction()
