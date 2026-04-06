add_mlir_conversion_library(CudaTileToStandard
  CudaTileToStandard.cpp

  DEPENDS
  CudaTileToStandardPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialectUtils
  MLIRTransforms
  CudaTileDialect
  CudaTileCPUDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRPtrDialect
  MLIRFuncDialect
  MLIRMathDialect
)
