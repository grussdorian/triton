set(HEADERS
  Passes.h
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonAppleGPUToLLVM)
add_public_tablegen_target(TritonAppleGPUConversionPassIncGen)
