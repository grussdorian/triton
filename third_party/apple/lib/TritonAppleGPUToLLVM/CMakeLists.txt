add_triton_library(TritonAppleGPUToLLVM
  TargetInfo.cpp
  TritonGPUToLLVM.cpp

  DEPENDS
  TritonAppleGPUConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonGPUToLLVM
  TritonInstrumentToLLVM
  TritonAnalysis
  MLIRReconcileUnrealizedCasts
  MLIRUBToLLVM
)
