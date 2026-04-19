add_llvm_executable(cuda-tile-lsp cuda-tile-lsp.cpp PARTIAL_SOURCES_INTENDED)
mlir_check_all_link_libraries(cuda-tile-lsp)

llvm_update_compile_flags(cuda-tile-lsp)
target_link_libraries(cuda-tile-lsp PRIVATE
  CudaTileDialect
  CudaTileCPUDialect
  CudaTileToStandard

  MLIRLspServerLib

  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
)

mlir_check_all_link_libraries(cuda-tile-lsp)
