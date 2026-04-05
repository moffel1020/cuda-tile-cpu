#ifndef CUDATILETOSTD_CONVERSION_PASSES
#define CUDATILETOSTD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCudaTileToStandard : Pass<"convert-cuda-tile-to-standard", "mlir::ModuleOp"> {
  let summary = "Convert cuda tile to standard dialects";

  let constructor = "mlir::cuda_tile::cpu::createConvertCudaTileToStandard()";
  let dependentDialects = ["mlir::cuda_tile::CudaTileDialect",
                           "mlir::cuda_tile::cpu::CudaTileCPUDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::vector::VectorDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::ptr::PtrDialect",
                           "mlir::func::FuncDialect"];
}

#endif
