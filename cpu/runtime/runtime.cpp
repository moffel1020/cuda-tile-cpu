#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <iostream>

// print functions copied from "mlir/ExecutionEngine/RunnerUtils.h"
static inline void printSpace(std::ostream &os, int count) {
  for (int i = 0; i < count; ++i) {
    os << ' ';
  }
}

template <typename T> struct MemRefDataPrinter {
  static void print(std::ostream &os, T *base, int64_t dim, int64_t rank,
                    int64_t offset, const int64_t *sizes,
                    const int64_t *strides);
  static void printFirst(std::ostream &os, T *base, int64_t dim, int64_t rank,
                         int64_t offset, const int64_t *sizes,
                         const int64_t *strides);
  static void printLast(std::ostream &os, T *base, int64_t dim, int64_t rank,
                        int64_t offset, const int64_t *sizes,
                        const int64_t *strides);
};

template <typename T>
void MemRefDataPrinter<T>::printFirst(std::ostream &os, T *base, int64_t dim,
                                      int64_t rank, int64_t offset,
                                      const int64_t *sizes,
                                      const int64_t *strides) {
  os << "[";
  print(os, base, dim - 1, rank, offset, sizes + 1, strides + 1);
  // If single element, close square bracket and return early.
  if (sizes[0] <= 1) {
    os << "]";
    return;
  }
  os << ", ";
  if (dim > 1)
    os << "\n";
}

template <typename T>
void MemRefDataPrinter<T>::print(std::ostream &os, T *base, int64_t dim,
                                 int64_t rank, int64_t offset,
                                 const int64_t *sizes, const int64_t *strides) {
  if (dim == 0) {
    os << base[offset];
    return;
  }
  printFirst(os, base, dim, rank, offset, sizes, strides);
  for (unsigned i = 1; i + 1 < sizes[0]; ++i) {
    printSpace(os, rank - dim + 1);
    print(os, base, dim - 1, rank, offset + i * strides[0], sizes + 1,
          strides + 1);
    os << ", ";
    if (dim > 1)
      os << "\n";
  }
  if (sizes[0] <= 1)
    return;
  printLast(os, base, dim, rank, offset, sizes, strides);
}

template <typename T>
void MemRefDataPrinter<T>::printLast(std::ostream &os, T *base, int64_t dim,
                                     int64_t rank, int64_t offset,
                                     const int64_t *sizes,
                                     const int64_t *strides) {
  printSpace(os, rank - dim + 1);
  print(os, base, dim - 1, rank, offset + (sizes[0] - 1) * (*strides),
        sizes + 1, strides + 1);
  os << "]";
}

template <typename T> void printMemRef(const DynamicMemRefType<T> &m) {
  if (m.rank == 0)
    std::cout << "[";
  MemRefDataPrinter<T>::print(std::cout, m.data, m.rank, m.rank, m.offset,
                              m.sizes, m.strides);
  if (m.rank == 0)
    std::cout << "]";
}

// runtime exports
extern "C" void ct_cpu_print_str(const char *str) { std::cout << str; }

#define CT_CPU_PRINT_MEMREF(ID, TYPE)                                          \
  extern "C" void ct_cpu_print_memref_##ID(const char *str, int64_t rank,      \
                                           void *descriptor) {                 \
    std::cout << str;                                                          \
    printMemRef(                                                               \
        DynamicMemRefType<TYPE>(UnrankedMemRefType<TYPE>{rank, descriptor}));  \
  }                                                                            \
  static_assert(true, "")
// force semicolon ^

CT_CPU_PRINT_MEMREF(i32, int32_t);
CT_CPU_PRINT_MEMREF(u32, uint32_t);
CT_CPU_PRINT_MEMREF(i64, int64_t);
CT_CPU_PRINT_MEMREF(u64, uint64_t);
CT_CPU_PRINT_MEMREF(f32, float);
CT_CPU_PRINT_MEMREF(f64, double);
