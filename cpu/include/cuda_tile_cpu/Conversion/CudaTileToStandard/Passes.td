#ifndef CUDATILETOSTD_CONVERSION_PASSES
#define CUDATILETOSTD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCudaTileToStandard : Pass<"convert-cuda-tile-to-standard", "mlir::ModuleOp"> {
  let summary = "Convert cuda tile to standard dialects";

  let constructor = "mlir::cuda_tile::cpu::createConvertCudaTileToStandard()";
  let dependentDialects = ["mlir::cuda_tile::CudaTileDialect",
                           "mlir::cuda_tile::cpu::CudaTileCPUDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::func::FuncDialect"];
}

def LowerCudaTileCPUMemOps : Pass<"lower-cuda-tile-cpu-mem-ops", "mlir::ModuleOp"> {
  let summary = "Convert cuda tile CPU load/store ops to affine and memref patterns";

  let constructor = "mlir::cuda_tile::cpu::createLowerCudaTileCPUMemOps()";
  let dependentDialects = ["mlir::cuda_tile::CudaTileDialect",
                           "mlir::cuda_tile::cpu::CudaTileCPUDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::func::FuncDialect"];
}

#endif
