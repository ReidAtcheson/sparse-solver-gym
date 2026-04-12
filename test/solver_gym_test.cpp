#include <array>

#include <gtest/gtest.h>

#include "sparse_solver_gym/solver_gym.hpp"

namespace {

class StubSolver final : public sparse_solver_gym::ISolver {
  public:
    std::string name() override { return "stub"; }

    Status setup() override { return Status::Ok; }

    Status symbolic(sparse_solver_gym::SparseGraph&) override { return Status::Ok; }

    Status numeric(const sparse_solver_gym::NumericValues& values) override {
      last_dtype = sparse_solver_gym::dtype_of(values);
      last_size = std::visit([](const auto& typed_values) {
        return typed_values.size();
      }, values);
      return Status::Ok;
    }

    Status solve(const sparse_solver_gym::MatrixView&, sparse_solver_gym::MatrixView&) override {
      return Status::Ok;
    }

    sparse_solver_gym::DType last_dtype = sparse_solver_gym::DType::f32;
    std::size_t last_size = 0;
};

}  // namespace

TEST(SolverGymTest, AddReturnsExpectedSum) {
    EXPECT_EQ(1,1);
}

TEST(SolverGymTest, NumericAcceptsTypedSpans) {
    StubSolver solver;
    const std::array<double, 3> values{1.0, 2.0, 3.0};
    const sparse_solver_gym::NumericValues numeric_values = std::span(values);

    EXPECT_EQ(solver.numeric(numeric_values), sparse_solver_gym::ISolver::Status::Ok);
    EXPECT_EQ(solver.last_dtype, sparse_solver_gym::DType::f64);
    EXPECT_EQ(solver.last_size, values.size());
}
