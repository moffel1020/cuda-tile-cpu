#ifndef CUDATILE_CPU_TRANSFORMS_TD
#define CUDATILE_CPU_TRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def InsertParallelLoopsPass : Pass<
  "cuda-tile-cpu-insert-parallel-loops", "::mlir::ModuleOp"
> {
  let summary = "Insert scf.parallel loop and replace get_block_num and get_block_id to access the loop";

  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

def TileAndFuseIntoStorePass : Pass<
  "cuda-tile-cpu-tile-and-fuse-into-store", "::mlir::ModuleOp"
> {
  let summary = "Fuse tileable ops into the store_ptr_tile op";

  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for store_ptr_tile iteration dimensions. If empty, "
               "all dimensions use tile size 32.">
  ];

  let dependentDialects = [
    "mlir::scf::SCFDialect",
  ];
}

#endif
