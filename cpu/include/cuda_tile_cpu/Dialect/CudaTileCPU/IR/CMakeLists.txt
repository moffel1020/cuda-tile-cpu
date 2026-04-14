set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=cuda_tile_cpu)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=cuda_tile_cpu)
add_mlir_doc(CudaTileCPUDialect CudaTileCPUDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(CudaTileCPUOps CudaTileCPUOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=cuda_tile_cpu)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=cuda_tile_cpu)

add_public_tablegen_target(CudaTileCPUIncGen)
