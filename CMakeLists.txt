cmake_minimum_required(VERSION 3.5)
project(needle C CXX)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# find correct version of Python
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()


###########################
### MLIR COMPILER BACKEND #
###########################
# Option to enable/disable MLIR compiler build
option(NEEDLE_BUILD_MLIR "Build the MLIR compiler backend" OFF)

if(NEEDLE_BUILD_MLIR)
  message(STATUS "Building MLIR compiler backend")

  # Find MLIR
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "MLIR found at: ${MLIR_DIR}")

  # Include MLIR headers
  include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

  # Create compiler shared library
  add_library(needle_compiler MODULE
    src/compiler/needle_compiler.cc
    src/compiler/mlir_gen.cc
    src/compiler/passes.cc
    src/compiler/jit_engine.cc
  )

  target_link_libraries(needle_compiler PUBLIC
    ${LINKER_LIBS}
    MLIRLinalgDialect
    MLIRAffineDialect
    MLIRSCFDialect
    MLIRVectorDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMDialect
    LLVMOrcJIT
  )

  # C++17 required for MLIR
  target_compile_features(needle_compiler PUBLIC cxx_std_17)

  pybind11_extension(needle_compiler)
  pybind11_strip(needle_compiler)

  # Output to backend_mlir directory
  set_target_properties(needle_compiler
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_mlir
    CXX_VISIBILITY_PRESET "hidden"
  )

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET needle_compiler PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
  endif()

  message(STATUS "MLIR compiler backend will be built")
else()
  message(STATUS "MLIR compiler backend disabled (set -DNEEDLE_BUILD_MLIR=ON to enable)")
endif()

####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()

