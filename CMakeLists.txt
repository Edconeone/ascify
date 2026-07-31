cmake_minimum_required(VERSION 3.16.8)

project(ascify-clang)

include(GNUInstallDirs)

option(ASCIFY_CLANG_TESTS
  "Build ascify-clang and register the release checks" OFF)
option(ASCIFY_CLANG_TESTS_ONLY
  "Register release checks without building ascify-clang" OFF)
option(ASCIFY_INCLUDE_IN_ASCIFY_SDK "Include ASCIFY in ASCIFY SDK" OFF)
option(ASCIFY_INSTALL_CLANG_HEADERS "Include clang headers in ASCIFY installation" ON)

if(ASCIFY_CLANG_TESTS OR ASCIFY_CLANG_TESTS_ONLY)
  set(ASCIFY_TEST "ON")
else()
  set(ASCIFY_TEST "OFF")
endif()

if(ASCIFY_CLANG_TESTS_ONLY)
  set(ASCIFY_BUILD "OFF")
else()
  set(ASCIFY_BUILD "ON")
endif()

message(STATUS "ASCIFY config:")
message(STATUS "   - Build ascify-clang    : ${ASCIFY_BUILD}")
message(STATUS "   - Test ascify-clang     : ${ASCIFY_TEST}")
message(STATUS "   - Is part of ASCIFY SDK    : ${ASCIFY_INCLUDE_IN_ASCIFY_SDK}")
message(STATUS "   - Install clang headers : ${ASCIFY_INSTALL_CLANG_HEADERS}")

if(NOT ASCIFY_CLANG_TESTS_ONLY)
  if(ASCIFY_INCLUDE_IN_ASCIFY_SDK)
    if(NOT WIN32)
      message(FATAL_ERROR "ASCIFY_INCLUDE_IN_ASCIFY_SDK is only supported on Windows")
    elseif(CMAKE_GENERATOR MATCHES "Visual Studio")
      message(FATAL_ERROR "ASCIFY_INCLUDE_IN_ASCIFY_SDK is not targeting Visual Studio")
    endif()
  else()
    find_package(Clang REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})
    find_package(LLVM REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})

    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}:")
    message(STATUS "   - CMake module path     : ${LLVM_CMAKE_DIR}")
    message(STATUS "   - Clang include path    : ${CLANG_INCLUDE_DIRS}")
    message(STATUS "   - LLVM Include path     : ${LLVM_INCLUDE_DIRS}")
    message(STATUS "   - Binary path           : ${LLVM_TOOLS_BINARY_DIR}")
  endif()

  list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
  include(AddLLVM)
endif()

if (NOT ASCIFY_CLANG_TESTS_ONLY)
  if(MSVC AND MSVC_VERSION VERSION_LESS "1900")
    message(SEND_ERROR "ascify-clang could be built by Visual Studio 14 2015 or higher.")
    return()
  endif()

  include_directories(${CLANG_INCLUDE_DIRS})
  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  file(GLOB_RECURSE ASCIFY_SOURCES src/*.cpp)
  file(GLOB_RECURSE ASCIFY_HEADERS src/*.h)

  # In order to be compatible with an LLVM that uses LLVM_LINK_LLVM_DYLIB=ON,
  # ascify-clang must link core LLVM libraries via LLVM_LINK_COMPONENTS vs directly adding
  # them.
  set(ASCIFY_ADDL_LINK_LIBS)
  set(LLVM_LINK_COMPONENTS
    ProfileData
    Support
    MCParser
    MC
    BitReader
    Option
    Core
  )
  
  if(LLVM_PACKAGE_VERSION VERSION_GREATER "6.0.1")
    list(APPEND ASCIFY_ADDL_LINK_LIBS clangToolingInclusions)
  endif()

  if(LLVM_PACKAGE_VERSION VERSION_GREATER "9.0.1")
    list(APPEND LLVM_LINK_COMPONENTS FrontendOpenMP)
  endif()

  if(LLVM_PACKAGE_VERSION VERSION_EQUAL "15.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "15.0.0")
    list(APPEND LLVM_LINK_COMPONENTS WindowsDriver)
    list(APPEND ASCIFY_ADDL_LINK_LIBS clangSupport)
  endif()


  add_llvm_executable(ascify-clang
    ${ASCIFY_SOURCES} ${ASCIFY_HEADERS}
  )
  target_link_directories(ascify-clang PRIVATE ${LLVM_LIBRARY_DIRS})

  if(ASCIFY_INCLUDE_IN_ASCIFY_SDK)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
      message(FATAL_ERROR "In order to include ASCIFY in ASCIFY SDK, ASCIFY needs to be built with LLVM_EXTERNAL_PROJECTS")
    endif()

    # Need to add clang include directories explicitly if building as part of llvm.
    if(LLVM_EXTERNAL_CLANG_SOURCE_DIR)
      target_include_directories(ascify-clang
        PRIVATE
          ${LLVM_BINARY_DIR}/tools/clang/include
          ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include)
    endif()

    # Need to add lld include directories explicitly if building as part of llvm.
    if(LLVM_EXTERNAL_LLD_SOURCE_DIR)
      target_include_directories(ascify-clang
        PRIVATE
          ${LLVM_BINARY_DIR}/tools/lld/include
          ${LLVM_EXTERNAL_LLD_SOURCE_DIR}/include)
    endif()

  # Keep the existing compiler since MSVC-specific flags will be set later.
  elseif(MSVC)
  else()
    # Overriding CMAKE_CXX_COMPILER/CMAKE_C_COMPILER below is sketchy - could
    # that be removed or reworked instead?
    set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
    set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
  endif()

  # Link against LLVM and CLANG libraries.
  target_link_libraries(ascify-clang PRIVATE
    clangASTMatchers
    clangFrontend
    clangTooling
    clangParse
    clangSerialization
    clangSema
    clangEdit
    clangFormat
    clangLex
    clangAnalysis
    clangDriver
    clangAST
    clangToolingCore
    clangRewrite
    clangBasic
    ${ASCIFY_ADDL_LINK_LIBS}
  )

  if(LLVM_PACKAGE_VERSION VERSION_EQUAL "5.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "5.0.0")
    if(MSVC)
      set(STD "/std:c++17")
    else()
      set(STD "-std=c++17")
    endif()
  elseif(LLVM_PACKAGE_VERSION VERSION_GREATER "3.4.0")
    if(MSVC)
      set(STD "/std:c++14")
    else()
      set(STD "-std=c++14")
    endif()
  endif()

  if(MSVC)
    target_link_libraries(ascify-clang PRIVATE version)
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    target_compile_options(ascify-clang PRIVATE ${STD} /Od /GR- /EHs- /EHc- /MP)
    if((LLVM_PACKAGE_VERSION VERSION_EQUAL "18.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "18.0.0") AND MSVC_VERSION VERSION_GREATER "1919")
      target_compile_options(ascify-clang PRIVATE /Zc:preprocessor)
    endif()
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /SUBSYSTEM:WINDOWS")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STD} -pthread -fno-rtti -fvisibility-inlines-hidden")
  endif()

  # Address Sanitize Flag.
  if(ADDRESS_SANITIZER)
    set(addr_var -fsanitize=address)
  else()
    set(addr_var )
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS} ${addr_var}")

  if(LLVM_PACKAGE_VERSION VERSION_EQUAL "16.0.0" OR LLVM_PACKAGE_VERSION VERSION_GREATER "16.0.0")
    set(LIB_CLANG_RES ${LLVM_VERSION_MAJOR})
  else()
    set(LIB_CLANG_RES ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH})
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS} -DLIB_CLANG_RES=${LIB_CLANG_RES} ${addr_var}")

  set(INSTALL_PATH_DOC_STRING "ascify-clang Installation Path")
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/dist" CACHE PATH ${INSTALL_PATH_DOC_STRING} FORCE)
  endif()

  set(ASCIFY_BIN_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/bin")

  set(ASCIFY_LIBEXEC_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBEXECDIR}/ascify")

  install(TARGETS ascify-clang DESTINATION bin)

  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ascify-compat-headers)

  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/acl_cub/aclcub.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/acl_cub
    COMPONENT ascify-compat-headers)

  # Headers are already included in HIP SDK, so skip those if including ASCIFY in HIP SDK.
  if( ASCIFY_INSTALL_CLANG_HEADERS)
    if(NOT ASCIFY_INCLUDE_IN_HIP_SDK)
      # Install all folders under clang/version/ in CMAKE_INSTALL_PREFIX path.
      install(
        DIRECTORY ${LLVM_LIBRARY_DIRS}/clang/${LIB_CLANG_RES}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ascify/${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT clang-resource-headers
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.modulemap"
        PATTERN "algorithm"
        PATTERN "complex"
        PATTERN "new"
        PATTERN "ppc_wrappers" EXCLUDE
        PATTERN "omp-tools.h" EXCLUDE
        PATTERN "omp.h" EXCLUDE
        PATTERN "ompt.h" EXCLUDE
        PATTERN "openmp_wrappers" EXCLUDE)
    endif()
  endif()

  if(UNIX)
    # Get rid of any RPATH definitions already.
    set_target_properties(ascify-clang PROPERTIES INSTALL_RPATH "")
    if(APPLE)
        set_target_properties(ascify-clang PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/../lib")
    else()
        set_target_properties(ascify-clang PROPERTIES LINK_FLAGS "-Wl,--enable-new-dtags -Wl,--rpath,\\$ORIGIN/../lib")
    endif()
  endif()

endif() # if (NOT ASCIFY_CLANG_TESTS_ONLY)

if(ASCIFY_CLANG_TESTS OR ASCIFY_CLANG_TESTS_ONLY)

  message(STATUS "---- The below configuring for ascify-clang testing only ----")

  include(CTest)
  enable_testing()

  find_package(Python3 3.0 REQUIRED COMPONENTS Interpreter)
  find_program(ASCIFY_TEST_SHELL sh)
  if(NOT ASCIFY_TEST_SHELL)
    message(FATAL_ERROR
      "Can't find a POSIX sh executable required by tests/run_release_checks.sh")
  endif()

  set(ASCIFY_TEST_BINARY "" CACHE FILEPATH
    "Existing ascify-clang binary used with ASCIFY_CLANG_TESTS_ONLY")
  set(ASCIFY_TEST_CUDA_PATH "" CACHE PATH
    "CUDA parsing root used by translated-fixture release checks")
  set(ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY "" CACHE PATH
    "Clang resource directory used by translated-fixture release checks")

  if(NOT ASCIFY_CLANG_TESTS_ONLY AND
     ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY STREQUAL "")
    set(ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY
      "${LLVM_LIBRARY_DIRS}/clang/${LIB_CLANG_RES}")
  endif()

  set(ASCIFY_RELEASE_CHECK_ENV
    "PYTHON=${Python3_EXECUTABLE}")
  if(ASCIFY_CLANG_TESTS_ONLY AND NOT ASCIFY_TEST_BINARY STREQUAL "")
    if(ASCIFY_TEST_CUDA_PATH STREQUAL "" OR
       ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY STREQUAL "")
      message(FATAL_ERROR
        "ASCIFY_TEST_BINARY requires ASCIFY_TEST_CUDA_PATH and "
        "ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY")
    endif()
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_BINARY=${ASCIFY_TEST_BINARY}")
  elseif(NOT ASCIFY_CLANG_TESTS_ONLY AND
         NOT ASCIFY_TEST_CUDA_PATH STREQUAL "")
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_BINARY=$<TARGET_FILE:ascify-clang>")
  endif()
  if(NOT ASCIFY_TEST_CUDA_PATH STREQUAL "")
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_CUDA_PATH=${ASCIFY_TEST_CUDA_PATH}")
  endif()
  if(NOT ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY STREQUAL "")
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_CLANG_RESOURCE_DIRECTORY=${ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY}")
  endif()

  add_test(
    NAME ascify-release-checks
    COMMAND
      ${CMAKE_COMMAND} -E env ${ASCIFY_RELEASE_CHECK_ENV}
      ${ASCIFY_TEST_SHELL}
      ${CMAKE_CURRENT_LIST_DIR}/tests/run_release_checks.sh)
  set_tests_properties(ascify-release-checks PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

  add_custom_target(test-ascify
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      -R "^ascify-release-checks$"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL)
  if(NOT ASCIFY_CLANG_TESTS_ONLY)
    add_dependencies(test-ascify ascify-clang)
  endif()

  add_custom_target(test-ascify-clang)
  add_dependencies(test-ascify-clang test-ascify)
  set_target_properties(test-ascify test-ascify-clang PROPERTIES
    FOLDER "Tests")

endif() # if(ASCIFY_CLANG_TESTS OR ASCIFY_CLANG_TESTS_ONLY)
