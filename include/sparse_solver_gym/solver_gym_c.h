#ifndef SPARSE_SOLVER_GYM_SOLVER_GYM_C_H
#define SPARSE_SOLVER_GYM_SOLVER_GYM_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SSG_SOLVER_ABI_VERSION_1 1u
#define SSG_CREATE_SOLVER_V1_SYMBOL "ssg_create_solver_v1"

typedef uint32_t ssg_status_t;
enum {
  SSG_STATUS_OK = 0u,
  SSG_STATUS_FAIL = 1u
};

typedef uint32_t ssg_dtype_t;
enum {
  SSG_DTYPE_F32 = 0u,
  SSG_DTYPE_F64 = 1u,
  SSG_DTYPE_C64 = 2u,
  SSG_DTYPE_C128 = 3u
};

typedef uint32_t ssg_itype_t;
enum {
  SSG_ITYPE_I32 = 0u,
  SSG_ITYPE_I64 = 1u
};

typedef uint32_t ssg_matrix_order_t;
enum {
  SSG_MATRIX_ORDER_ROW_MAJOR = 0u,
  SSG_MATRIX_ORDER_COL_MAJOR = 1u
};

typedef uint32_t ssg_sparse_storage_t;
enum {
  SSG_SPARSE_STORAGE_COO = 0u,
  SSG_SPARSE_STORAGE_CSC = 1u,
  SSG_SPARSE_STORAGE_CSR = 2u
};

typedef struct ssg_complex64 {
  float real;
  float imag;
} ssg_complex64;

typedef struct ssg_complex128 {
  double real;
  double imag;
} ssg_complex128;

typedef struct ssg_const_matrix_view {
  uint64_t struct_size;
  ssg_dtype_t dtype;
  ssg_matrix_order_t order;
  int64_t nrows;
  int64_t ncols;
  int64_t ld;
  const void* data;
} ssg_const_matrix_view;

typedef struct ssg_matrix_view {
  uint64_t struct_size;
  ssg_dtype_t dtype;
  ssg_matrix_order_t order;
  int64_t nrows;
  int64_t ncols;
  int64_t ld;
  void* data;
} ssg_matrix_view;

typedef struct ssg_sparse_graph {
  uint64_t struct_size;
  ssg_itype_t itype;
  int64_t nrows;
  int64_t ncols;
  int64_t nnz;
  ssg_sparse_storage_t storage;
  const void* rids;
  const void* cids;
  const void* offs;
} ssg_sparse_graph;

typedef struct ssg_numeric_values {
  uint64_t struct_size;
  ssg_dtype_t dtype;
  int64_t count;
  const void* data;
} ssg_numeric_values;

typedef struct ssg_solver_v1 {
  uint64_t struct_size;
  uint32_t abi_version;
  uint32_t reserved_flags;
  void* instance;
  const char* (*name)(void* instance);
  const char* (*last_error)(void* instance);
  ssg_status_t (*setup)(void* instance);
  ssg_status_t (*symbolic)(void* instance, const ssg_sparse_graph* graph);
  ssg_status_t (*numeric)(void* instance, const ssg_numeric_values* values);
  ssg_status_t (*solve)(void* instance, const ssg_const_matrix_view* in, ssg_matrix_view* out);
  void (*destroy)(void* instance);
} ssg_solver_v1;

typedef ssg_status_t (*ssg_create_solver_v1_fn)(ssg_solver_v1* out_solver);

/*
 * Contract notes for plugin authors:
 * - Export `ssg_create_solver_v1` with C linkage and fill `out_solver`.
 * - All callbacks must be safe to call through a C ABI and must not throw.
 * - Returned strings must remain valid until `destroy` is called.
 * - Input views borrow caller-owned memory and must not be retained after the call returns.
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SPARSE_SOLVER_GYM_SOLVER_GYM_C_H */
