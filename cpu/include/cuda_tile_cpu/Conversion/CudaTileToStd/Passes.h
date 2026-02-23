#ifndef CUDA_TILE_CONVERSTION_TO_STD_PASSES_H
#define CUDA_TILE_CONVERSTION_TO_STD_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cuda_tile {
namespace cpu {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createCudaTileConvertToStd();

#define GEN_PASS_DECL
#include "cuda_tile_cpu/Conversion/CudaTileToStd/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "cuda_tile_cpu/Conversion/CudaTileToStd/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

#endif