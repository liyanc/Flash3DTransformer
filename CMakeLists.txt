cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(Flash3DXFMR VERSION 1.0 LANGUAGES CXX CUDA )
project(Flash3DXFMR VERSION 1.0 LANGUAGES CXX CUDA )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH $ORIGIN)

find_package(CUDAToolkit  REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(glog REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(src)

message(WARNING "CUDA_ARCH = ${CMAKE_CUDA_ARCHITECTURES}")
add_definitions(${TK_FLAGS})

pybind11_add_module(pshattn MODULE
    src/f3d_frontend.cu
    src/common/runtime.cpp
    src/common/device_manager.cu
    src/common/kernel_dispatcher.cu
    src/psh/batch_psh_bucket_dispatcher.cu
    src/psh/psh_label.cu
    src/psh/rand_gen.cu
    src/psh/bucket_sanitizer_cpu.cpp
    src/psh/gemm_dispatcher.cu
    src/psh/additive_unpool_dispatcher.cu
    src/psh/batch_subbuck_pooling_dispatcher.cu
    src/psh/batch_subbuck_pooling_bwd_dispatcher.cu
    )

target_compile_options(pshattn PRIVATE
    -fuse-ld=lld
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --use_fast_math --restrict --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress=186>
)
target_link_options(pshattn PRIVATE
    -Wl,--disable-new-dtags)

target_include_directories(pshattn PUBLIC
    ${TORCH_INCLUDE_DIRS}
    third_party/ThunderKittens/include
    )


target_link_libraries(pshattn PUBLIC
    torch
    ${TORCH_PYTHON_LIBRARY}
    pybind11::module pybind11::lto
    CUDA::cublas
    glog::glog
    )

install(IMPORTED_RUNTIME_ARTIFACTS
    pshattn
    DESTINATION flash3dxfmr/lib)

install(TARGETS pshattn
    RUNTIME DESTINATION flash3dxfmr/lib
    LIBRARY DESTINATION flash3dxfmr/lib
    )
