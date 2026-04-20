#ifndef CUDA_TILE_CPU_TILING_INTERFACE_H
#define CUDA_TILE_CPU_TILING_INTERFACE_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace cuda_tile {
namespace cpu {

void registerTilingInterfaceExternalModels(DialectRegistry &registry);

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_CPU_TILING_INTERFACE_H
