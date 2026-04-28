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

#endif
