#include "TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton {

bool TargetInfoApple::supportMaximumMinimum() const { return true; }

Value TargetInfoApple::getClusterCTAId(RewriterBase &rewriter,
                                        Location loc) const {
  llvm_unreachable("cluster barrier not supported on Apple GPU");
}

Value TargetInfoApple::ballot(RewriterBase &rewriter, Location loc, Type type,
                               Value cmp) const {
  llvm_unreachable("ballot not yet implemented for Apple GPU");
}

void TargetInfoApple::barrier(Location loc, RewriterBase &rewriter,
                               triton::gpu::AddrSpace targets) const {
  auto ctx = rewriter.getContext();
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i32ty = IntegerType::get(ctx, 32);

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  const char *funcName = "__spirv_ControlBarrier";
  auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32ty, i32ty, i32ty});
    func = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcTy);
    func.setPrivate();
  }

  // Workgroup scope (2), Workgroup memory scope (2),
  // SequentiallyConsistent | WorkgroupMemory (0x100 | 0x2)
  auto execScope = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                       IntegerAttr::get(i32ty, 2));
  auto memScope = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                       IntegerAttr::get(i32ty, 2));
  auto memSem = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                       IntegerAttr::get(i32ty, 0x102));
  LLVM::createLLVMCallOp(rewriter, loc, func,
                         ValueRange{execScope, memScope, memSem});
}

void TargetInfoApple::clusterBarrier(Location loc,
                                      RewriterBase &rewriter) const {
  barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

void TargetInfoApple::warpSync(Location loc, RewriterBase &rewriter) const {
  auto ctx = rewriter.getContext();
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i32ty = IntegerType::get(ctx, 32);

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  const char *funcName = "__spirv_ControlBarrier";
  auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32ty, i32ty, i32ty});
    func = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcTy);
    func.setPrivate();
  }

  // Subgroup scope (3), Subgroup memory scope (3),
  // AcquireRelease | WorkgroupMemory (0x8 | 0x100 | 0x2)
  auto execScope = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                       IntegerAttr::get(i32ty, 3));
  auto memScope = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                       IntegerAttr::get(i32ty, 3));
  auto memSem = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                       IntegerAttr::get(i32ty, 0x10A));
  LLVM::createLLVMCallOp(rewriter, loc, func,
                         ValueRange{execScope, memScope, memSem});
}

void TargetInfoApple::storeDShared(RewriterBase &rewriter, Location loc,
                                    Value ptr, std::optional<Value> ctaId,
                                    Value val, Value pred) const {
  llvm_unreachable("storeDShared not yet implemented for Apple GPU");
}

Value TargetInfoApple::loadDShared(RewriterBase &rewriter, Location loc,
                                    Value ptr, std::optional<Value> ctaId,
                                    Type elemTy, Value pred,
                                    Operation *localLoadOp) const {
  llvm_unreachable("loadDShared not yet implemented for Apple GPU");
}

Value TargetInfoApple::shuffleXor(RewriterBase &rewriter, Location loc,
                                   Value val, int i) const {
  auto ctx = rewriter.getContext();
  auto ty = val.getType();
  auto i32ty = IntegerType::get(ctx, 32);

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  const char *funcName = "__spirv_SubgroupShuffleXor";
  auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funcTy = LLVM::LLVMFunctionType::get(ty, {ty, i32ty});
    func = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcTy);
    func.setPrivate();
  }
  auto maskVal = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                     IntegerAttr::get(i32ty, i));
  return LLVM::createLLVMCallOp(rewriter, loc, func,
                                ValueRange{val, maskVal}).getResult();
}

Value TargetInfoApple::shuffleUp(RewriterBase &rewriter, Location loc,
                                  Value val, int i) const {
  auto ctx = rewriter.getContext();
  auto ty = val.getType();
  auto i32ty = IntegerType::get(ctx, 32);

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  const char *funcName = "__spirv_SubgroupShuffleUp";
  auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funcTy = LLVM::LLVMFunctionType::get(ty, {ty, i32ty});
    func = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcTy);
    func.setPrivate();
  }
  auto deltaVal = LLVM::ConstantOp::create(rewriter, loc, i32ty,
                       IntegerAttr::get(i32ty, i));
  return LLVM::createLLVMCallOp(rewriter, loc, func,
                                ValueRange{val, deltaVal}).getResult();
}

Value TargetInfoApple::shuffleIdx(RewriterBase &rewriter, Location loc,
                                   Value val, int i) const {
  auto idxVal = LLVM::ConstantOp::create(rewriter, loc,
                     IntegerType::get(rewriter.getContext(), 32),
                     IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32), i));
  return shuffleIdx(rewriter, loc, val, idxVal);
}

Value TargetInfoApple::shuffleIdx(RewriterBase &rewriter, Location loc,
                                   Value val, Value i) const {
  auto ctx = rewriter.getContext();
  auto ty = val.getType();
  auto i32ty = IntegerType::get(ctx, 32);

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  const char *funcName = "__spirv_SubgroupShuffleIdx";
  auto func = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funcTy = LLVM::LLVMFunctionType::get(ty, {ty, i32ty});
    func = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcTy);
    func.setPrivate();
  }
  return LLVM::createLLVMCallOp(rewriter, loc, func,
                                ValueRange{val, i}).getResult();
}

Value TargetInfoApple::permute(RewriterBase &rewriter, Location loc,
                                Value a, Value b, Value selector) const {
  llvm_unreachable("permute not yet implemented for Apple GPU");
}

Value TargetInfoApple::programId(RewriterBase &rewriter, Location loc,
                                  ModuleOp moduleOp,
                                  ProgramIDDim axis) const {
  auto ctx = rewriter.getContext();
  auto i32ty = IntegerType::get(ctx, 32);

  const char *funcName = nullptr;
  switch (axis) {
  case ProgramIDDim::X:
    funcName = "__spirv_WorkgroupId_x";
    break;
  case ProgramIDDim::Y:
    funcName = "__spirv_WorkgroupId_y";
    break;
  case ProgramIDDim::Z:
    funcName = "__spirv_WorkgroupId_z";
    break;
  }

  auto funcTy = LLVM::LLVMFunctionType::get(i32ty, {});
  auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    func = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcTy);
    func.setPrivate();
  }
  ValueRange emptyArgs;
  return LLVM::createLLVMCallOp(rewriter, loc, func, emptyArgs).getResult();
}

bool TargetInfoApple::warpReduce(RewriterBase &rewriter, Location loc,
                                  SmallVector<Value> &acc,
                                  triton::ReduceOp op,
                                  unsigned reduceLaneIdMask) const {
  llvm_unreachable("warpReduce not yet implemented for Apple GPU");
}

std::string TargetInfoApple::getMulhiFuncName(Type resultElementTy) const {
  llvm_unreachable("getMulhiFuncName not yet implemented for Apple GPU");
}

void TargetInfoApple::printf(RewriterBase &rewriter, Value formatStrStart,
                              int formatStrByteCount, ValueRange args,
                              ArrayRef<bool> isSigned) const {
  llvm_unreachable("printf not supported on Apple GPU");
}

void TargetInfoApple::printf(RewriterBase &rewriter, StringRef msg,
                              ValueRange args, ArrayRef<bool> isSigned) const {
  llvm_unreachable("printf not supported on Apple GPU");
}

void TargetInfoApple::assertFail(RewriterBase &rewriter, Location loc,
                                  StringRef message, StringRef file,
                                  StringRef func, int line) const {
  llvm_unreachable("assertFail not yet implemented for Apple GPU");
}

int TargetInfoApple::getSharedAddressSpace() const {
  return 3;
}

int TargetInfoApple::getAddressSpace(Attribute addressSpace) const {
  return 0;
}

bool TargetInfoApple::supportVectorizedAtomics() const { return false; }

} // namespace mlir::triton
