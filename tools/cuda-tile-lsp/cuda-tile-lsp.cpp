#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry
      .insert<cuda_tile::CudaTileDialect, cuda_tile::cpu::CudaTileCPUDialect>();

  mlir::MLIRContext context(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
