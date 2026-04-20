//===- cuda-tile-opt.cpp - CUDA Tile Dialect Test Driver --------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"
#include "cuda_tile_cpu/Conversion/CudaTileCPUToLLVM/Passes.h"
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/TilingInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::cuda_tile::CudaTileDialect,
                  mlir::cuda_tile::cpu::CudaTileCPUDialect>();

  mlir::cuda_tile::cpu::registerTilingInterfaceExternalModels(registry);

  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::cuda_tile::registerCudaTilePasses();
  mlir::cuda_tile::cpu::registerCudaTileToStandardPasses();
  mlir::cuda_tile::cpu::registerCudaTileCPUToLLVMPasses();
  mlir::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CudaTile test driver\n", registry));
}
