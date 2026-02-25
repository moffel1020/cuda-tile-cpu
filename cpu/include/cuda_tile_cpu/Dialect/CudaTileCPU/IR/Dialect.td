#ifndef CUDATILE_CPU_DIALECT_TD
#define CUDATILE_CPU_DIALECT_TD

include "mlir/IR/OpBase.td"

def CudaTileCPU_Dialect: Dialect {
  let name = "cuda_tile_cpu";
  let summary = "Cuda Tile intermediate CPU dialect.";
  let cppNamespace = "::mlir::cuda_tile::cpu";

  let description = [{
    Intermediate dialect used to convert Cuda Tile ops to standard ops.
  }];

  let dependentDialects = [
    "cuda_tile::CudaTileDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif
