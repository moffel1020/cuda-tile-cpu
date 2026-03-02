add_llvm_library(ctcpu-runtime runtime.cpp PARTIAL_SOURCES_INTENDED)

target_link_libraries(ctcpu-runtime PRIVATE
  MLIRExecutionEngine
  MLIRExecutionEngineUtils
)

mlir_check_all_link_libraries(ctcpu-runtime)
