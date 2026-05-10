#ifndef CUDATILE_CPU_DIALECT_H
#define CUDATILE_CPU_DIALECT_H

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"

#define GET_OP_CLASSES
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h.inc"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Ops.h.inc"

#endif
