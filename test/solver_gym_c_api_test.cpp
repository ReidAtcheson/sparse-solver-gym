#include <array>
#include <memory>

#include <gtest/gtest.h>

#include "sparse_solver_gym/solver_gym_c.h"
#include "sparse_solver_gym/solver_gym_cxx.hpp"

namespace {

struct FakeCSolver {
  bool* destroyed;
  int setup_calls = 0;
  int symbolic_calls = 0;
  int numeric_calls = 0;
  int solve_calls = 0;
  ssg_dtype_t last_numeric_dtype = SSG_DTYPE_F32;
  int64_t last_numeric_count = 0;
  int64_t last_graph_nrows = 0;
  int64_t last_graph_nnz = 0;
};

const char* fake_name(void*) {
  return "fake-c-solver";
}

ssg_status_t fake_setup(void* instance) {
  auto* solver = static_cast<FakeCSolver*>(instance);
  ++solver->setup_calls;
  return SSG_STATUS_OK;
}

ssg_status_t fake_symbolic(void* instance, const ssg_sparse_graph* graph) {
  auto* solver = static_cast<FakeCSolver*>(instance);
  ++solver->symbolic_calls;
  solver->last_graph_nrows = graph->nrows;
  solver->last_graph_nnz = graph->nnz;
  return SSG_STATUS_OK;
}

ssg_status_t fake_numeric(void* instance, const ssg_numeric_values* values) {
  auto* solver = static_cast<FakeCSolver*>(instance);
  ++solver->numeric_calls;
  solver->last_numeric_dtype = values->dtype;
  solver->last_numeric_count = values->count;
  return SSG_STATUS_OK;
}

ssg_status_t fake_solve(void* instance, const ssg_const_matrix_view* in, ssg_matrix_view* out) {
  auto* solver = static_cast<FakeCSolver*>(instance);
  ++solver->solve_calls;

  auto* in_values = static_cast<const double*>(in->data);
  auto* out_values = static_cast<double*>(out->data);
  out_values[0] = in_values[0] + 1.0;
  out_values[1] = in_values[1] + 1.0;
  return SSG_STATUS_OK;
}

void fake_destroy(void* instance) {
  auto* solver = static_cast<FakeCSolver*>(instance);
  *solver->destroyed = true;
  delete solver;
}

}  // namespace

TEST(SolverGymCTest, WrapsCAbiSolverAsCppSolver) {
    bool destroyed = false;
    ssg_solver_v1 c_solver{};
    c_solver.struct_size = sizeof(c_solver);
    c_solver.abi_version = SSG_SOLVER_ABI_VERSION_1;
    c_solver.instance = new FakeCSolver{.destroyed = &destroyed};
    c_solver.name = &fake_name;
    c_solver.setup = &fake_setup;
    c_solver.symbolic = &fake_symbolic;
    c_solver.numeric = &fake_numeric;
    c_solver.solve = &fake_solve;
    c_solver.destroy = &fake_destroy;

    {
        auto solver = sparse_solver_gym::make_solver_from_c_api(c_solver);
        EXPECT_EQ(solver->name(), "fake-c-solver");
        EXPECT_EQ(solver->setup(), sparse_solver_gym::ISolver::Status::Ok);

        std::array<int32_t, 2> rids{0, 1};
        std::array<int32_t, 2> cids{0, 1};
        sparse_solver_gym::SparseGraph graph{};
        graph.itype = sparse_solver_gym::IType::i32;
        graph.nrows = 2;
        graph.ncols = 2;
        graph.nnz = 2;
        graph.storage = sparse_solver_gym::SparseStorage::Coo;
        graph.rids.i32 = rids.data();
        graph.cids.i32 = cids.data();
        graph.offs.i32 = nullptr;
        EXPECT_EQ(solver->symbolic(graph), sparse_solver_gym::ISolver::Status::Ok);

        const std::array<double, 2> numeric_values{2.0, 3.0};
        EXPECT_EQ(solver->numeric(sparse_solver_gym::NumericValues{std::span(numeric_values)}),
                  sparse_solver_gym::ISolver::Status::Ok);

        std::array<double, 2> in_values{4.0, 5.0};
        std::array<double, 2> out_values{0.0, 0.0};
        sparse_solver_gym::MatrixView in{};
        in.dtype = sparse_solver_gym::DType::f64;
        in.order = sparse_solver_gym::MatrixOrder::ColMajor;
        in.nrows = 2;
        in.ncols = 1;
        in.ld = 2;
        in.data.f64 = in_values.data();

        sparse_solver_gym::MatrixView out{};
        out.dtype = sparse_solver_gym::DType::f64;
        out.order = sparse_solver_gym::MatrixOrder::ColMajor;
        out.nrows = 2;
        out.ncols = 1;
        out.ld = 2;
        out.data.f64 = out_values.data();

        EXPECT_EQ(solver->solve(in, out), sparse_solver_gym::ISolver::Status::Ok);
        EXPECT_DOUBLE_EQ(out_values[0], 5.0);
        EXPECT_DOUBLE_EQ(out_values[1], 6.0);
    }

    EXPECT_TRUE(destroyed);
}
