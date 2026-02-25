#ifndef CUDATILETOSTD_CONVERSION_PASSES
#define CUDATILETOSTD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCudaTileToStandard : Pass<"convert-cuda-tile-to-standard", "mlir::ModuleOp"> {
  let summary = "Convert cuda tile to standard dialects";
  let description = [{
  }];

  let constructor = "mlir::cuda_tile::cpu::createConvertCudaTileToStandard()";
  let dependentDialects = ["mlir::cuda_tile::CudaTileDialect",
                           "mlir::cuda_tile::cpu::CudaTileCPUDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::vector::VectorDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::func::FuncDialect"];
}

#endif
