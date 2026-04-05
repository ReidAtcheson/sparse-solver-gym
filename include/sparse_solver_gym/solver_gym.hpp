#pragma once
#include <cstdint>
#include <string>
#include <complex>

namespace sparse_solver_gym {

// Field value type
enum class DType{
  f32,
  f64,
  c64,
  c128
};

// Index value type
enum class IType{
  // Signed integers
  i32,
  i64
};

enum class MatrixOrder{
  RowMajor,
  ColMajor
};

struct MatrixView{
  DType dtype;
  MatrixOrder order;
  int64_t nrows;
  int64_t ncols;
  int64_t ld;
  union {
    float* f32;
    double* f64;
    std::complex<float>* c64;
    std::complex<double>* c128;
  } data;
};

enum class SparseStorage{
  Coo,
  Csc,
  Csr
};

struct SparseGraph{
  IType itype;
  int64_t nrows;
  int64_t ncols;
  int64_t nnz;
  SparseStorage storage;
  union {
    int32_t* i32;
    int64_t* i64;
  } rids; // Null if CSR
  union {
    int32_t* i32;
    int64_t* i64;
  } cids; // Null if CSC
  union {
    int32_t* i32;
    int64_t* i64;
  } offs; // Null if COO
};

// Intended calling sequence:
// solver.setup() -- Once per process, init code.
// solver.symbolic(graph) -- Once per graph
// solver.numeric(dtype,data) -- must follow symbolic(), but can call multiple times
// solver.solve(in,out) -- must follow numeric() but can call as many times as necessary
//
// Possible improvements: Enable low rank up/down dates, possibly sparse.
class ISolver{
  public:
    virtual std::string name() = 0;
    virtual void setup() = 0;
    virtual void symbolic(SparseGraph& g) = 0;
    virtual void numeric(DType dtype,void* data) = 0;
    virtual void solve(const MatrixView& in,MatrixView& out) = 0;
    virtual ~ISolver() = default;
};

class IBenchmarkLogger{
  public:
    enum class Phase{
      setup,
      symbolic,
      numeric,
      solve
    };
    // This contains all possible information we will log as official
    // outputs from the benchmarking suite. error/residual is 
    // only populated for solves, duration is always populated.
    struct Log{
      Phase phase;
      int64_t duration_ns;
      double relative_error;
      double relative_residual;
    };
    virtual void record_arbitrary(const std::string&) = 0;
    virtual void record(const Log& log) = 0;
    virtual ~IBenchmarkLogger() = default;
};

class IBenchmark{
  public:
    virtual void run(ISolver* solver,IBenchmarkLogger* logger) = 0;
    virtual ~IBenchmark() = default;
};


}  // namespace sparse_solver_gym
