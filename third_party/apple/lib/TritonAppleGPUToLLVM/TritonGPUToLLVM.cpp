#include "TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton {

std::unique_ptr<Pass> createConvertTritonAppleGPUToLLVMPass(int32_t warpSize) {
  return nullptr;
}

} // namespace mlir::triton
