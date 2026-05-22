#ifndef TRITON_APPLE_GPU_TO_LLVM_PASSES_H
#define TRITON_APPLE_GPU_TO_LLVM_PASSES_H

#include <memory>

namespace mlir {
class Pass;
namespace triton {

std::unique_ptr<Pass> createConvertTritonAppleGPUToLLVMPass(int32_t warpSize);

} // namespace triton
} // namespace mlir

#endif // TRITON_APPLE_GPU_TO_LLVM_PASSES_H
