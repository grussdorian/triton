#include "TargetInfo.h"
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
  llvm_unreachable("barrier not yet implemented for Apple GPU");
}

void TargetInfoApple::clusterBarrier(Location loc,
                                      RewriterBase &rewriter) const {
  llvm_unreachable("cluster barrier not supported on Apple GPU");
}

void TargetInfoApple::warpSync(Location loc, RewriterBase &rewriter) const {
  llvm_unreachable("warpSync not yet implemented for Apple GPU");
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
  llvm_unreachable("shuffleXor not yet implemented for Apple GPU");
}

Value TargetInfoApple::shuffleUp(RewriterBase &rewriter, Location loc,
                                  Value val, int i) const {
  llvm_unreachable("shuffleUp not yet implemented for Apple GPU");
}

Value TargetInfoApple::shuffleIdx(RewriterBase &rewriter, Location loc,
                                   Value val, int i) const {
  llvm_unreachable("shuffleIdx not yet implemented for Apple GPU");
}

Value TargetInfoApple::shuffleIdx(RewriterBase &rewriter, Location loc,
                                   Value val, Value i) const {
  llvm_unreachable("shuffleIdx not yet implemented for Apple GPU");
}

Value TargetInfoApple::permute(RewriterBase &rewriter, Location loc,
                                Value a, Value b, Value selector) const {
  llvm_unreachable("permute not yet implemented for Apple GPU");
}

Value TargetInfoApple::programId(RewriterBase &rewriter, Location loc,
                                  ModuleOp moduleOp,
                                  ProgramIDDim axis) const {
  llvm_unreachable("programId not yet implemented for Apple GPU");
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
