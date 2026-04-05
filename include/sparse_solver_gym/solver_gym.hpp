#pragma once
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>

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
  void* data;
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
  SparseStorage storage;
  void* rids; // Null if CSR
  void* cids; // Null if CSC
  void* offs; // Null if COO
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
};

class IBenchmarkLogger{
  public:
    enum class Phase{
      setup,
      symbolic,
      numeric,
      solve
    };
    struct Log{
      Phase phase;
      int64_t duration_ns;
    };
    virtual void record(const Log& log) = 0;
};

class IBenchmark{
  public:
    virtual void run(ISolver* solver,IBenchmarkLogger* logger) = 0;
};


}  // namespace sparse_solver_gym
