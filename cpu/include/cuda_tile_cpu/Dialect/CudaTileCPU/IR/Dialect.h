#ifndef CUDATILE_CPU_DIALECT_H
#define CUDATILE_CPU_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#define GET_OP_CLASSES
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Ops.h.inc"

#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h.inc"

#endif
