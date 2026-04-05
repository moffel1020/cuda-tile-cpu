#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry
      .insert<cuda_tile::CudaTileDialect, cuda_tile::cpu::CudaTileCPUDialect,
              mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
              mlir::arith::ArithDialect, mlir::vector::VectorDialect,
              mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
              mlir::func::FuncDialect, mlir::affine::AffineDialect>();

  mlir::MLIRContext context(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
