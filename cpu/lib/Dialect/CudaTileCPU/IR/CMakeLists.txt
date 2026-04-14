add_mlir_library(CudaTileCPUDialect
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  CudaTileCPUIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
