#include "TritonAppleGPUToLLVM/Passes.h"
#include "TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONAPPLEGPUTOLLVM
#include "TritonAppleGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonAppleGPUToLLVM
    : public triton::impl::ConvertTritonAppleGPUToLLVMBase<
          ConvertTritonAppleGPUToLLVM> {
  explicit ConvertTritonAppleGPUToLLVM(int32_t warpSize) {
    this->warpSize = warpSize;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Initialize type converter
    auto targetInfo = TargetInfoApple(warpSize);
    TritonGPUToLLVMTypeConverter typeConverter(context, targetInfo);

    // Convert function signatures
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // Convert TritonGPU ops to LLVM
    TritonLLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(
        typeConverter, patterns, benefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::populateAssertOpToLLVMPattern(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(
        typeConverter, patterns, targetInfo, benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(
        typeConverter, patterns, targetInfo);
    mlir::triton::populateFpSanToLLVMPatterns(typeConverter, patterns);

    // Additional MLIR conversions
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // Lower remaining control flow
    {
      TritonLLVMFunctionConversionTarget cfTarget(*context);
      cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
        return op->getDialect() !=
               context->getLoadedDialect<cf::ControlFlowDialect>();
      });
      RewritePatternSet cfPatterns(context);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                             cfPatterns);
      if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<Pass> createConvertTritonAppleGPUToLLVMPass(int32_t warpSize) {
  return std::make_unique<ConvertTritonAppleGPUToLLVM>(warpSize);
}

} // namespace mlir::triton
