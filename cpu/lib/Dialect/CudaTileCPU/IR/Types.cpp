#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.cpp.inc"

using namespace mlir;
using namespace mlir::cuda_tile::cpu;

void mlir::cuda_tile::cpu::CudaTileCPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.cpp.inc"
      >();
}
