add_mlir_dialect_library(CudaTileCPUTransforms
  InsertParallelLoops.cpp
  TilingInterfaceImpl.cpp

  DEPENDS
  CudaTileCPUTransformsIncGen

  LINK_LIBS PUBLIC
  CudaTileCPUDialect
  MLIRIR
  MLIRTransforms

  LINK_COMPONENTS
  Support
)
