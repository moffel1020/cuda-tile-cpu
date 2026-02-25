#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.cpp.inc"

using namespace mlir::cuda_tile;
using namespace mlir::cuda_tile::cpu;

void CudaTileCPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Ops.cpp.inc"
      >();
}
