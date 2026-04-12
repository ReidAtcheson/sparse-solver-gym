#include "sparse_solver_gym/solver_gym_cxx.hpp"

#include <stdexcept>
#include <string>

namespace sparse_solver_gym {
namespace {

static_assert(sizeof(ssg_complex64) == sizeof(std::complex<float>));
static_assert(alignof(ssg_complex64) == alignof(std::complex<float>));
static_assert(sizeof(ssg_complex128) == sizeof(std::complex<double>));
static_assert(alignof(ssg_complex128) == alignof(std::complex<double>));

ISolver::Status to_cpp_status(const ssg_status_t status) noexcept {
  return status == SSG_STATUS_OK ? ISolver::Status::Ok : ISolver::Status::Fail;
}

ssg_dtype_t to_c_dtype(const DType dtype) noexcept {
  switch (dtype) {
    case DType::f32:
      return SSG_DTYPE_F32;
    case DType::f64:
      return SSG_DTYPE_F64;
    case DType::c64:
      return SSG_DTYPE_C64;
    case DType::c128:
      return SSG_DTYPE_C128;
  }
  return SSG_DTYPE_F32;
}

ssg_itype_t to_c_itype(const IType itype) noexcept {
  switch (itype) {
    case IType::i32:
      return SSG_ITYPE_I32;
    case IType::i64:
      return SSG_ITYPE_I64;
  }
  return SSG_ITYPE_I32;
}

ssg_matrix_order_t to_c_order(const MatrixOrder order) noexcept {
  switch (order) {
    case MatrixOrder::RowMajor:
      return SSG_MATRIX_ORDER_ROW_MAJOR;
    case MatrixOrder::ColMajor:
      return SSG_MATRIX_ORDER_COL_MAJOR;
  }
  return SSG_MATRIX_ORDER_ROW_MAJOR;
}

ssg_sparse_storage_t to_c_storage(const SparseStorage storage) noexcept {
  switch (storage) {
    case SparseStorage::Coo:
      return SSG_SPARSE_STORAGE_COO;
    case SparseStorage::Csc:
      return SSG_SPARSE_STORAGE_CSC;
    case SparseStorage::Csr:
      return SSG_SPARSE_STORAGE_CSR;
  }
  return SSG_SPARSE_STORAGE_COO;
}

const void* dense_data_ptr(const MatrixView& view) noexcept {
  switch (view.dtype) {
    case DType::f32:
      return view.data.f32;
    case DType::f64:
      return view.data.f64;
    case DType::c64:
      return view.data.c64;
    case DType::c128:
      return view.data.c128;
  }
  return nullptr;
}

void* dense_data_ptr(MatrixView& view) noexcept {
  switch (view.dtype) {
    case DType::f32:
      return view.data.f32;
    case DType::f64:
      return view.data.f64;
    case DType::c64:
      return view.data.c64;
    case DType::c128:
      return view.data.c128;
  }
  return nullptr;
}

const void* graph_index_ptr(const SparseGraph& graph, const int which) noexcept {
  switch (graph.itype) {
    case IType::i32:
      if (which == 0) {
        return graph.rids.i32;
      }
      if (which == 1) {
        return graph.cids.i32;
      }
      return graph.offs.i32;
    case IType::i64:
      if (which == 0) {
        return graph.rids.i64;
      }
      if (which == 1) {
        return graph.cids.i64;
      }
      return graph.offs.i64;
  }
  return nullptr;
}

ssg_numeric_values to_c_numeric_values(const NumericValues& values) noexcept {
  ssg_numeric_values c_values{};
  c_values.struct_size = sizeof(c_values);
  c_values.dtype = to_c_dtype(dtype_of(values));
  c_values.count = static_cast<int64_t>(std::visit([](const auto& typed_values) {
    return typed_values.size();
  }, values));
  c_values.data = std::visit([](const auto& typed_values) -> const void* {
    return typed_values.data();
  }, values);
  return c_values;
}

ssg_const_matrix_view to_c_const_matrix_view(const MatrixView& view) noexcept {
  ssg_const_matrix_view c_view{};
  c_view.struct_size = sizeof(c_view);
  c_view.dtype = to_c_dtype(view.dtype);
  c_view.order = to_c_order(view.order);
  c_view.nrows = view.nrows;
  c_view.ncols = view.ncols;
  c_view.ld = view.ld;
  c_view.data = dense_data_ptr(view);
  return c_view;
}

ssg_matrix_view to_c_matrix_view(MatrixView& view) noexcept {
  ssg_matrix_view c_view{};
  c_view.struct_size = sizeof(c_view);
  c_view.dtype = to_c_dtype(view.dtype);
  c_view.order = to_c_order(view.order);
  c_view.nrows = view.nrows;
  c_view.ncols = view.ncols;
  c_view.ld = view.ld;
  c_view.data = dense_data_ptr(view);
  return c_view;
}

ssg_sparse_graph to_c_sparse_graph(const SparseGraph& graph) noexcept {
  ssg_sparse_graph c_graph{};
  c_graph.struct_size = sizeof(c_graph);
  c_graph.itype = to_c_itype(graph.itype);
  c_graph.nrows = graph.nrows;
  c_graph.ncols = graph.ncols;
  c_graph.nnz = graph.nnz;
  c_graph.storage = to_c_storage(graph.storage);
  c_graph.rids = graph_index_ptr(graph, 0);
  c_graph.cids = graph_index_ptr(graph, 1);
  c_graph.offs = graph_index_ptr(graph, 2);
  return c_graph;
}

void validate_solver_api(const ssg_solver_v1& solver) {
  if (solver.struct_size < sizeof(ssg_solver_v1)) {
    throw std::invalid_argument("C solver API struct_size is too small");
  }
  if (solver.abi_version != SSG_SOLVER_ABI_VERSION_1) {
    throw std::invalid_argument("C solver ABI version is not supported");
  }
  if (solver.name == nullptr || solver.setup == nullptr || solver.symbolic == nullptr ||
      solver.numeric == nullptr || solver.solve == nullptr) {
    throw std::invalid_argument("C solver API is missing required callbacks");
  }
}

class CSolverAdapter final : public ISolver {
  public:
    explicit CSolverAdapter(ssg_solver_v1 solver)
        : solver_(solver) {
      validate_solver_api(solver_);
      if (const char* raw_name = solver_.name(solver_.instance); raw_name != nullptr) {
        name_ = raw_name;
      }
    }

    ~CSolverAdapter() override {
      if (solver_.destroy != nullptr) {
        solver_.destroy(solver_.instance);
      }
    }

    std::string name() override {
      return name_;
    }

    Status setup() override {
      return to_cpp_status(solver_.setup(solver_.instance));
    }

    Status symbolic(SparseGraph& graph) override {
      const ssg_sparse_graph c_graph = to_c_sparse_graph(graph);
      return to_cpp_status(solver_.symbolic(solver_.instance, &c_graph));
    }

    Status numeric(const NumericValues& values) override {
      const ssg_numeric_values c_values = to_c_numeric_values(values);
      return to_cpp_status(solver_.numeric(solver_.instance, &c_values));
    }

    Status solve(const MatrixView& in, MatrixView& out) override {
      const ssg_const_matrix_view c_in = to_c_const_matrix_view(in);
      ssg_matrix_view c_out = to_c_matrix_view(out);
      return to_cpp_status(solver_.solve(solver_.instance, &c_in, &c_out));
    }

  private:
    ssg_solver_v1 solver_;
    std::string name_;
};

}  // namespace

std::unique_ptr<ISolver> make_solver_from_c_api(ssg_solver_v1 solver) {
  return std::make_unique<CSolverAdapter>(solver);
}

}  // namespace sparse_solver_gym
