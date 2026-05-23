//===- Passes.td - TritonAppleGPUToLLVM pass definitions ----*- tablegen -*-===//
//
// Part of the MLIR project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_APPLE_GPU_TO_LLVM_PASSES_TD
#define TRITON_APPLE_GPU_TO_LLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTritonAppleGPUToLLVM : Pass<"convert-triton-apple-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Triton Apple GPU to LLVM";
  let constructor = "mlir::triton::createConvertTritonAppleGPUToLLVMPass(32)";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::cf::ControlFlowDialect"
  ];

  let options = [
    Option<"warpSize", "warp-size", "int32_t", /*default=*/"32",
           "warp size">,
  ];
}

#endif // TRITON_APPLE_GPU_TO_LLVM_PASSES_TD
