#include "sparse_solver_gym/solver_gym_c.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct FakeSolver {
  bool setup_done = false;
  bool symbolic_done = false;
  bool numeric_done = false;
  int64_t n = 0;
  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  std::vector<double> values;
};

const char* fake_name(void*) {
  return "fake-plugin-solver";
}

const char* fake_last_error(void*) {
  return "";
}

ssg_status_t fake_setup(void* instance) {
  auto* solver = static_cast<FakeSolver*>(instance);
  solver->setup_done = true;
  return SSG_STATUS_OK;
}

ssg_status_t fake_symbolic(void* instance, const ssg_sparse_graph* graph) {
  auto* solver = static_cast<FakeSolver*>(instance);
  if (!solver->setup_done || graph == nullptr || graph->nrows <= 0 ||
      graph->nrows != graph->ncols || graph->nnz <= 0 || graph->itype != SSG_ITYPE_I32 ||
      graph->storage != SSG_SPARSE_STORAGE_COO || graph->rids == nullptr ||
      graph->cids == nullptr) {
    return SSG_STATUS_FAIL;
  }

  solver->n = graph->nrows;
  const auto* rows = static_cast<const int32_t*>(graph->rids);
  const auto* cols = static_cast<const int32_t*>(graph->cids);
  solver->rows.assign(rows, rows + graph->nnz);
  solver->cols.assign(cols, cols + graph->nnz);
  solver->values.clear();
  solver->symbolic_done = true;
  solver->numeric_done = false;
  return SSG_STATUS_OK;
}

ssg_status_t fake_numeric(void* instance, const ssg_numeric_values* values) {
  auto* solver = static_cast<FakeSolver*>(instance);
  if (!solver->symbolic_done || values == nullptr || values->dtype != SSG_DTYPE_F64 ||
      values->count != static_cast<int64_t>(solver->rows.size()) || values->data == nullptr) {
    return SSG_STATUS_FAIL;
  }
  const auto* numeric_values = static_cast<const double*>(values->data);
  solver->values.assign(numeric_values, numeric_values + values->count);
  solver->numeric_done = true;
  return SSG_STATUS_OK;
}

bool dense_solve(std::vector<double> a, std::vector<double> b, std::vector<double>& x) {
  const auto n = static_cast<int64_t>(b.size());
  for (int64_t col = 0; col < n; ++col) {
    int64_t pivot = col;
    double pivot_abs = std::abs(a[static_cast<std::size_t>(col * n + col)]);
    for (int64_t row = col + 1; row < n; ++row) {
      const double candidate_abs = std::abs(a[static_cast<std::size_t>(row * n + col)]);
      if (candidate_abs > pivot_abs) {
        pivot = row;
        pivot_abs = candidate_abs;
      }
    }
    if (pivot_abs == 0.0) {
      return false;
    }
    if (pivot != col) {
      for (int64_t j = col; j < n; ++j) {
        std::swap(a[static_cast<std::size_t>(col * n + j)],
                  a[static_cast<std::size_t>(pivot * n + j)]);
      }
      std::swap(b[static_cast<std::size_t>(col)], b[static_cast<std::size_t>(pivot)]);
    }

    for (int64_t row = col + 1; row < n; ++row) {
      const double factor =
          a[static_cast<std::size_t>(row * n + col)] /
          a[static_cast<std::size_t>(col * n + col)];
      a[static_cast<std::size_t>(row * n + col)] = 0.0;
      for (int64_t j = col + 1; j < n; ++j) {
        a[static_cast<std::size_t>(row * n + j)] -=
            factor * a[static_cast<std::size_t>(col * n + j)];
      }
      b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(col)];
    }
  }

  x.assign(static_cast<std::size_t>(n), 0.0);
  for (int64_t row = n - 1; row >= 0; --row) {
    double sum = b[static_cast<std::size_t>(row)];
    for (int64_t col = row + 1; col < n; ++col) {
      sum -= a[static_cast<std::size_t>(row * n + col)] * x[static_cast<std::size_t>(col)];
    }
    x[static_cast<std::size_t>(row)] = sum / a[static_cast<std::size_t>(row * n + row)];
  }
  return true;
}

ssg_status_t fake_solve(void* instance, const ssg_const_matrix_view* in, ssg_matrix_view* out) {
  auto* solver = static_cast<FakeSolver*>(instance);
  if (!solver->numeric_done || in == nullptr || out == nullptr || in->dtype != SSG_DTYPE_F64 ||
      out->dtype != SSG_DTYPE_F64 || in->data == nullptr || out->data == nullptr ||
      in->nrows != solver->n || out->nrows != solver->n || in->ncols != 1 || out->ncols != 1) {
    return SSG_STATUS_FAIL;
  }

  const auto* input = static_cast<const double*>(in->data);
  auto* output = static_cast<double*>(out->data);

  std::vector<double> dense(static_cast<std::size_t>(solver->n * solver->n), 0.0);
  for (std::size_t i = 0; i < solver->values.size(); ++i) {
    dense[static_cast<std::size_t>(solver->rows[i] * solver->n + solver->cols[i])] +=
        solver->values[i];
  }

  std::vector<double> rhs(input, input + solver->n);
  std::vector<double> solution;
  if (!dense_solve(std::move(dense), std::move(rhs), solution)) {
    return SSG_STATUS_FAIL;
  }

  std::copy(solution.begin(), solution.end(), output);
  return SSG_STATUS_OK;
}

void fake_destroy(void* instance) {
  delete static_cast<FakeSolver*>(instance);
}

}  // namespace

extern "C" ssg_status_t ssg_create_solver_v1(ssg_solver_v1* out_solver) {
  if (out_solver == nullptr) {
    return SSG_STATUS_FAIL;
  }

  out_solver->struct_size = sizeof(ssg_solver_v1);
  out_solver->abi_version = SSG_SOLVER_ABI_VERSION_1;
  out_solver->reserved_flags = 0;
  out_solver->instance = new FakeSolver();
  out_solver->name = &fake_name;
  out_solver->last_error = &fake_last_error;
  out_solver->setup = &fake_setup;
  out_solver->symbolic = &fake_symbolic;
  out_solver->numeric = &fake_numeric;
  out_solver->solve = &fake_solve;
  out_solver->destroy = &fake_destroy;
  return SSG_STATUS_OK;
}
