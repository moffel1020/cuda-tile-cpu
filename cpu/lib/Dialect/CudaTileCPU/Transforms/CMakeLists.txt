add_mlir_dialect_library(CudaTileCPUTilingInterface
  TilingInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  CudaTileCPUDialect
)
