add_mlir_dialect_library(CudaTileCPUTransforms
  InsertParallelLoops.cpp
  TileAndFuseIntoStore.cpp
  TilingInterfaceImpl.cpp
  VectorizeLinalg.cpp

  DEPENDS
  CudaTileCPUTransformsIncGen

  LINK_LIBS PUBLIC
  CudaTileCPUDialect
  MLIRIR
  MLIRLinalgTransforms
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms

  LINK_COMPONENTS
  Support
)
