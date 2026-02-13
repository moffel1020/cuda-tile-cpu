add_mlir_conversion_library(CudaTileToStd
  CudaTileToStd.cpp

  DEPENDS
  CudaTileToStdPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRDialectUtils
  MLIRTransforms
  CudaTileDialect
  MLIRVectorDialect
  MLIRSCFDialect
  MLIRFuncDialect
)