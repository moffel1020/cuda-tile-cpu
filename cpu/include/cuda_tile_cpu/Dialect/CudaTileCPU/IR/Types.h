#ifndef CUDATILE_CPU_TYPES_H
#define CUDATILE_CPU_TYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.h.inc"

#endif // CUDATILE_CPU_TYPES_H
