#ifndef CUDA_TILE_CPU_CONVERSION_TO_LLVM_PASSES_H
#define CUDA_TILE_CPU_CONVERSION_TO_LLVM_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cuda_tile {
namespace cpu {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertCudaTileCPUToLLVM();

#define GEN_PASS_DECL
#include "cuda_tile_cpu/Conversion/CudaTileCPUToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "cuda_tile_cpu/Conversion/CudaTileCPUToLLVM/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

#endif
