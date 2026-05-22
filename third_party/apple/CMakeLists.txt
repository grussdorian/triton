include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

if(APPLE)
  add_subdirectory(include)
  add_subdirectory(lib)
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonApple ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_apple.cc
    LINK_LIBS TritonAppleGPUToLLVM)
  target_link_libraries(TritonApple PRIVATE Python3::Module pybind11::headers)
endif()
