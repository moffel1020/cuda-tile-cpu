add_mlir_conversion_library(CudaTileCPUToLLVM
  CudaTileCPUToLLVM.cpp

  DEPENDS
  CudaTileCPUToLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialectUtils
  MLIRTransforms
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  CudaTileDialect
  CudaTileCPUDialect
)
