#ifndef CUDA_TILE_CPU_TRANSFORM_PASSES_H
#define CUDA_TILE_CPU_TRANSFORM_PASSES_H

#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cuda_tile {
namespace cpu {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

#endif
