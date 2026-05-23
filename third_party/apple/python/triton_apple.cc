#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Pass/PassManager.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
void add_pass_wrapper(mlir::PassManager &pm, int32_t warpSize) {
  pm.addPass(mlir::triton::createConvertTritonAppleGPUToLLVMPass(warpSize));
}
}

void init_triton_apple(py::module &&m) {
  m.doc() = "Triton Apple GPU backend bindings";

  auto passes = m.def_submodule("passes");
  auto ttgpuir = passes.def_submodule("ttgpuir");
  ttgpuir.def("add_to_llvmir", &add_pass_wrapper);
}
