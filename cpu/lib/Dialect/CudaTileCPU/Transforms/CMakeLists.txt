add_mlir_dialect_library(CudaTileCPUTransforms
  InsertParallelLoops.cpp
  TileAndFuseIntoStore.cpp
  TilingInterfaceImpl.cpp

  DEPENDS
  CudaTileCPUTransformsIncGen

  LINK_LIBS PUBLIC
  CudaTileCPUDialect
  MLIRIR
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms

  LINK_COMPONENTS
  Support
)
