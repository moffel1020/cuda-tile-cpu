#ifndef CUDATILETOSTD_CONVERSION_PASSES
#define CUDATILETOSTD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CudaTileConvertToStd : Pass<"cuda-tile-convert-to-std", "mlir::ModuleOp"> {
  let summary = "Convert cuda tile to standard dialects";
  let description = [{
  }];

  let constructor = "mlir::cuda_tile::createCudaTileConvertToStd()";
  let dependentDialects = ["mlir::cuda_tile::CudaTileDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::vector::VectorDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::func::FuncDialect"];
}

#endif
