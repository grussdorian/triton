#ifndef TRITON_APPLE_GPU_TO_LLVM_PASSES_H
#define TRITON_APPLE_GPU_TO_LLVM_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
namespace triton {

std::unique_ptr<Pass> createConvertTritonAppleGPUToLLVMPass(int32_t warpSize);

#define GEN_PASS_DECL
#include "TritonAppleGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_APPLE_GPU_TO_LLVM_PASSES_H
