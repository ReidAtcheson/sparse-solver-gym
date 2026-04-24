#include "sparse_solver_gym/solver_gym_cxx.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace sparse_solver_gym {
namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

class StdoutBenchmarkLogger final : public IBenchmarkLogger {
  public:
    FrameId get_frame(const std::string& label) override {
      if (auto existing = frame_ids_.find(label); existing != frame_ids_.end()) {
        return existing->second;
      }

      const auto next_id = static_cast<FrameId>(frame_labels_.size());
      frame_labels_.push_back(label);
      frame_ids_.emplace(frame_labels_.back(), next_id);
      return next_id;
    }

    std::string_view get_frame_label(const FrameId id) override {
      if (id >= frame_labels_.size()) {
        return "<unknown>";
      }
      return frame_labels_[id];
    }

    void on_event(const Event& event) override {
      std::visit([this](const auto& payload) {
        using T = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<T, Frame>) {
          const char* tag = payload.tag == FrameTag::beg ? "begin" : "end";
          std::cout << "frame." << tag << " " << get_frame_label(payload.id) << '\n';
        } else {
          for (const Value& value : payload) {
            std::cout << "value " << value.label << '=';
            std::visit([](const auto& typed_value) {
              std::cout << typed_value;
            }, value.d);
            std::cout << '\n';
          }
        }
      }, event);
    }

  private:
    std::vector<std::string> frame_labels_;
    std::map<std::string, FrameId, std::less<>> frame_ids_;
};

void emit_frame(IBenchmarkLogger& logger,
                const IBenchmarkLogger::FrameId frame,
                const IBenchmarkLogger::FrameTag tag) {
  logger.on_event(IBenchmarkLogger::Frame{.id = frame, .tag = tag});
}

void emit_values(IBenchmarkLogger& logger, std::span<IBenchmarkLogger::Value> values) {
  logger.on_event(values);
}

double infinity_norm(std::span<const double> values) {
  double norm = 0.0;
  for (const double value : values) {
    norm = std::max(norm, std::abs(value));
  }
  return norm;
}

double safe_relative_norm(const double numerator, const double denominator) {
  if (denominator == 0.0) {
    return numerator == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return numerator / denominator;
}

struct TimedStatus {
  ISolver::Status status = ISolver::Status::Fail;
  double seconds = 0.0;
};

template <class Fn>
TimedStatus time_solver_operation(Fn&& fn) {
  const auto start = Clock::now();
  const ISolver::Status status = fn();
  const auto stop = Clock::now();
  return {
      .status = status,
      .seconds = std::chrono::duration<double>(stop - start).count(),
  };
}

std::string required_string(const json& source, const char* key) {
  if (!source.contains(key)) {
    throw std::invalid_argument(std::string("missing required string field: ") + key);
  }
  return source.at(key).get<std::string>();
}

int64_t required_i64(const json& source, const char* key) {
  if (!source.contains(key)) {
    throw std::invalid_argument(std::string("missing required integer field: ") + key);
  }
  return source.at(key).get<int64_t>();
}

double json_double_or(const json& source, const char* key, const double fallback) {
  return source.contains(key) ? source.at(key).get<double>() : fallback;
}

uint64_t json_u64_or(const json& source, const char* key, const uint64_t fallback) {
  return source.contains(key) ? source.at(key).get<uint64_t>() : fallback;
}

std::string json_string_or(const json& source, const char* key, std::string fallback) {
  return source.contains(key) ? source.at(key).get<std::string>() : std::move(fallback);
}

class RandomBandedSparseBenchmark final : public IBenchmark {
  public:
    struct Config {
      std::string name = "random_banded_sparse";
      int64_t n = 0;
      int64_t matrix_count = 10;
      int64_t nnz_per_row = 0;
      int64_t bandwidth = 0;
      double value_min = -1.0;
      double value_max = 1.0;
      double solution_min = -1.0;
      double solution_max = 1.0;
      uint64_t seed = 1;
    };

    explicit RandomBandedSparseBenchmark(Config config)
        : config_(std::move(config)) {
      validate_config();
    }

    static std::unique_ptr<IBenchmark> from_json(const json& config) {
      Config parsed{};
      parsed.name = json_string_or(config, "name", parsed.name);
      parsed.n = required_i64(config, "n");
      parsed.matrix_count = config.value("matrix_count", parsed.matrix_count);
      parsed.nnz_per_row = required_i64(config, "nnz_per_row");
      parsed.bandwidth = required_i64(config, "bandwidth");
      parsed.value_min = json_double_or(config, "value_min", parsed.value_min);
      parsed.value_max = json_double_or(config, "value_max", parsed.value_max);
      parsed.solution_min = json_double_or(config, "solution_min", parsed.solution_min);
      parsed.solution_max = json_double_or(config, "solution_max", parsed.solution_max);
      parsed.seed = json_u64_or(config, "seed", parsed.seed);
      return std::make_unique<RandomBandedSparseBenchmark>(std::move(parsed));
    }

    void run(ISolver* solver, IBenchmarkLogger* logger) override {
      if (solver == nullptr || logger == nullptr) {
        return;
      }

      std::mt19937_64 rng(config_.seed);
      std::uniform_real_distribution<double> value_dist(config_.value_min, config_.value_max);
      std::uniform_real_distribution<double> solution_dist(config_.solution_min,
                                                           config_.solution_max);

      const auto benchmark_frame = logger->get_frame("benchmark." + config_.name);
      emit_frame(*logger, benchmark_frame, IBenchmarkLogger::FrameTag::beg);

      const TimedStatus setup = run_solver_operation(*logger, "setup", [&]() {
        return solver->setup();
      });
      log_operation(*logger, "setup", -1, setup);
      if (setup.status != ISolver::Status::Ok) {
        emit_frame(*logger, benchmark_frame, IBenchmarkLogger::FrameTag::end);
        return;
      }

      for (int64_t matrix_index = 0; matrix_index < config_.matrix_count; ++matrix_index) {
        const auto matrix_frame = logger->get_frame("matrix." + std::to_string(matrix_index));
        emit_frame(*logger, matrix_frame, IBenchmarkLogger::FrameTag::beg);

        GeneratedSystem system = generate_system(rng, value_dist, solution_dist);
        SparseGraph graph{};
        graph.itype = IType::i32;
        graph.nrows = config_.n;
        graph.ncols = config_.n;
        graph.nnz = static_cast<int64_t>(system.values.size());
        graph.storage = SparseStorage::Coo;
        graph.rids.i32 = system.rows.data();
        graph.cids.i32 = system.cols.data();
        graph.offs.i32 = nullptr;

        const TimedStatus symbolic = run_solver_operation(*logger, "symbolic", [&]() {
          return solver->symbolic(graph);
        });
        log_operation(*logger, "symbolic", matrix_index, symbolic);
        if (symbolic.status != ISolver::Status::Ok) {
          emit_frame(*logger, matrix_frame, IBenchmarkLogger::FrameTag::end);
          continue;
        }

        const TimedStatus numeric = run_solver_operation(*logger, "numeric", [&]() {
          return solver->numeric(NumericValues{std::span<const double>(system.values)});
        });
        log_operation(*logger, "numeric", matrix_index, numeric);
        if (numeric.status != ISolver::Status::Ok) {
          emit_frame(*logger, matrix_frame, IBenchmarkLogger::FrameTag::end);
          continue;
        }

        std::vector<double> computed_solution(static_cast<std::size_t>(config_.n), 0.0);
        MatrixView rhs{};
        rhs.dtype = DType::f64;
        rhs.order = MatrixOrder::ColMajor;
        rhs.nrows = config_.n;
        rhs.ncols = 1;
        rhs.ld = config_.n;
        rhs.data.f64 = system.rhs.data();

        MatrixView solution{};
        solution.dtype = DType::f64;
        solution.order = MatrixOrder::ColMajor;
        solution.nrows = config_.n;
        solution.ncols = 1;
        solution.ld = config_.n;
        solution.data.f64 = computed_solution.data();

        const TimedStatus solve = run_solver_operation(*logger, "solve", [&]() {
          return solver->solve(rhs, solution);
        });
        log_operation(*logger, "solve", matrix_index, solve);

        if (solve.status == ISolver::Status::Ok) {
          log_accuracy(*logger, matrix_index, system, computed_solution);
        }

        emit_frame(*logger, matrix_frame, IBenchmarkLogger::FrameTag::end);
      }

      emit_frame(*logger, benchmark_frame, IBenchmarkLogger::FrameTag::end);
    }

  private:
    struct GeneratedSystem {
      std::vector<int32_t> rows;
      std::vector<int32_t> cols;
      std::vector<double> values;
      std::vector<double> exact_solution;
      std::vector<double> rhs;
    };

    void validate_config() const {
      if (config_.n <= 0) {
        throw std::invalid_argument("random_banded_sparse requires n > 0");
      }
      if (config_.matrix_count <= 0) {
        throw std::invalid_argument("random_banded_sparse requires matrix_count > 0");
      }
      if (config_.nnz_per_row <= 0) {
        throw std::invalid_argument("random_banded_sparse requires nnz_per_row > 0");
      }
      if (config_.bandwidth < 0) {
        throw std::invalid_argument("random_banded_sparse requires bandwidth >= 0");
      }
      if (config_.value_min > config_.value_max) {
        throw std::invalid_argument("random_banded_sparse requires value_min <= value_max");
      }
      if (config_.solution_min > config_.solution_max) {
        throw std::invalid_argument("random_banded_sparse requires solution_min <= solution_max");
      }

      for (int64_t row = 0; row < config_.n; ++row) {
        if (band_size(row) < config_.nnz_per_row) {
          std::ostringstream message;
          message << "random_banded_sparse row " << row << " has only " << band_size(row)
                  << " unique band positions, but nnz_per_row is " << config_.nnz_per_row;
          throw std::invalid_argument(message.str());
        }
      }
    }

    int64_t band_begin(const int64_t row) const {
      return std::max<int64_t>(0, row - config_.bandwidth);
    }

    int64_t band_end(const int64_t row) const {
      return std::min<int64_t>(config_.n - 1, row + config_.bandwidth);
    }

    int64_t band_size(const int64_t row) const {
      return band_end(row) - band_begin(row) + 1;
    }

    template <class Rng, class ValueDistribution, class SolutionDistribution>
    GeneratedSystem generate_system(Rng& rng,
                                    ValueDistribution& value_dist,
                                    SolutionDistribution& solution_dist) const {
      GeneratedSystem system{};
      const auto n = static_cast<std::size_t>(config_.n);
      const auto nnz = static_cast<std::size_t>(config_.n * config_.nnz_per_row);
      system.rows.reserve(nnz);
      system.cols.reserve(nnz);
      system.values.reserve(nnz);
      system.exact_solution.resize(n);
      system.rhs.assign(n, 0.0);

      for (double& value : system.exact_solution) {
        value = solution_dist(rng);
      }

      for (int64_t row = 0; row < config_.n; ++row) {
        std::set<int64_t> columns;
        columns.insert(row);

        std::uniform_int_distribution<int64_t> col_dist(band_begin(row), band_end(row));
        while (static_cast<int64_t>(columns.size()) < config_.nnz_per_row) {
          columns.insert(col_dist(rng));
        }

        for (const int64_t col : columns) {
          const double value = value_dist(rng);
          system.rows.push_back(static_cast<int32_t>(row));
          system.cols.push_back(static_cast<int32_t>(col));
          system.values.push_back(value);
          system.rhs[static_cast<std::size_t>(row)] +=
              value * system.exact_solution[static_cast<std::size_t>(col)];
        }
      }

      return system;
    }

    template <class Fn>
    static TimedStatus run_solver_operation(IBenchmarkLogger& logger,
                                            const std::string& operation,
                                            Fn&& fn) {
      const auto frame = logger.get_frame("solver." + operation);
      emit_frame(logger, frame, IBenchmarkLogger::FrameTag::beg);
      const TimedStatus result = time_solver_operation(std::forward<Fn>(fn));
      emit_frame(logger, frame, IBenchmarkLogger::FrameTag::end);
      return result;
    }

    void log_operation(IBenchmarkLogger& logger,
                       const std::string& operation,
                       const int64_t matrix_index,
                       const TimedStatus& result) const {
      const std::string ok_label = operation + ".ok";
      const std::string seconds_label = operation + ".seconds";
      std::array<IBenchmarkLogger::Value, 3> values{
          IBenchmarkLogger::Value{.label = "matrix_index", .d = matrix_index},
          IBenchmarkLogger::Value{.label = ok_label, .d = result.status == ISolver::Status::Ok},
          IBenchmarkLogger::Value{.label = seconds_label, .d = result.seconds},
      };
      emit_values(logger, values);
    }

    void log_accuracy(IBenchmarkLogger& logger,
                      const int64_t matrix_index,
                      const GeneratedSystem& system,
                      std::span<const double> computed_solution) const {
      std::vector<double> residual(system.rhs);
      std::vector<double> error(system.exact_solution);

      for (std::size_t entry = 0; entry < system.values.size(); ++entry) {
        const auto row = static_cast<std::size_t>(system.rows[entry]);
        const auto col = static_cast<std::size_t>(system.cols[entry]);
        residual[row] -= system.values[entry] * computed_solution[col];
      }
      for (std::size_t i = 0; i < error.size(); ++i) {
        error[i] -= computed_solution[i];
      }

      const double relative_residual =
          safe_relative_norm(infinity_norm(residual), infinity_norm(system.rhs));
      const double relative_error =
          safe_relative_norm(infinity_norm(error), infinity_norm(system.exact_solution));

      std::array<IBenchmarkLogger::Value, 3> values{
          IBenchmarkLogger::Value{.label = "matrix_index", .d = matrix_index},
          IBenchmarkLogger::Value{.label = "relative_residual_inf", .d = relative_residual},
          IBenchmarkLogger::Value{.label = "relative_error_inf", .d = relative_error},
      };
      emit_values(logger, values);
    }

    Config config_;
};

std::unique_ptr<IBenchmark> make_benchmark_from_json(const json& config) {
  const std::string type = required_string(config, "type");
  if (type == "random_banded_sparse") {
    return RandomBandedSparseBenchmark::from_json(config);
  }
  throw std::invalid_argument("unknown benchmark type: " + type);
}

std::vector<std::unique_ptr<IBenchmark>> make_benchmarks_from_json(const json& config) {
  const json* benchmark_configs = nullptr;
  if (config.is_array()) {
    benchmark_configs = &config;
  } else if (config.is_object() && config.contains("benchmarks") &&
             config.at("benchmarks").is_array()) {
    benchmark_configs = &config.at("benchmarks");
  } else {
    throw std::invalid_argument(
        "benchmark config must be an array of benchmark specs or an object with a benchmarks array");
  }

  std::vector<std::unique_ptr<IBenchmark>> benchmarks;
  for (const json& benchmark_config : *benchmark_configs) {
    benchmarks.push_back(make_benchmark_from_json(benchmark_config));
  }
  if (benchmarks.empty()) {
    throw std::invalid_argument("benchmark config must contain at least one benchmark");
  }
  return benchmarks;
}

json default_config() {
  return json{
      {"benchmarks",
       json::array({
           {
               {"type", "random_banded_sparse"},
               {"name", "default_random_banded_sparse"},
               {"n", 8},
               {"matrix_count", 10},
               {"nnz_per_row", 3},
               {"bandwidth", 2},
               {"value_min", -1.0},
               {"value_max", 1.0},
               {"solution_min", -1.0},
               {"solution_max", 1.0},
               {"seed", 1},
           },
       })},
  };
}

json load_config(const char* path) {
  if (path == nullptr) {
    return default_config();
  }

  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error(std::string("failed to open benchmark config: ") + path);
  }

  json config;
  input >> config;
  return config;
}

class SharedObjectHandle final {
  public:
    explicit SharedObjectHandle(const char* path)
        : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {}

    SharedObjectHandle(const SharedObjectHandle&) = delete;
    SharedObjectHandle& operator=(const SharedObjectHandle&) = delete;

    ~SharedObjectHandle() {
      if (handle_ != nullptr) {
        dlclose(handle_);
      }
    }

    [[nodiscard]] explicit operator bool() const noexcept {
      return handle_ != nullptr;
    }

    [[nodiscard]] void* symbol(const char* name) const {
      dlerror();
      void* result = dlsym(handle_, name);
      if (const char* error = dlerror(); error != nullptr) {
        throw std::runtime_error(error);
      }
      return result;
    }

  private:
    void* handle_ = nullptr;
};

int run_child(const char* solver_shared_object, const char* config_path) {
  const json config = load_config(config_path);
  std::vector<std::unique_ptr<IBenchmark>> benchmarks = make_benchmarks_from_json(config);

  SharedObjectHandle shared_object(solver_shared_object);
  if (!shared_object) {
    std::cerr << "failed to load solver shared object: " << dlerror() << '\n';
    return 2;
  }

  const auto create_solver = reinterpret_cast<ssg_create_solver_v1_fn>(
      shared_object.symbol(SSG_CREATE_SOLVER_V1_SYMBOL));

  ssg_solver_v1 c_solver{};
  const ssg_status_t create_status = create_solver(&c_solver);
  if (create_status != SSG_STATUS_OK) {
    std::cerr << "solver factory returned failure\n";
    return 3;
  }

  auto solver = make_solver_from_c_api(c_solver);
  StdoutBenchmarkLogger logger;
  std::cout << "solver.name=" << solver->name() << '\n';

  for (const auto& benchmark : benchmarks) {
    benchmark->run(solver.get(), &logger);
  }

  solver.reset();
  return 0;
}

std::string current_executable_path(const char* argv0) {
  std::array<char, 4096> buffer{};
  const ssize_t count = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (count > 0) {
    buffer[static_cast<std::size_t>(count)] = '\0';
    return buffer.data();
  }
  return argv0;
}

int run_parent(const char* executable_path,
               const char* solver_shared_object,
               const char* config_path) {
  const pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "failed to fork benchmark subprocess: " << std::strerror(errno) << '\n';
    return 1;
  }

  if (pid == 0) {
    if (config_path == nullptr) {
      execl(executable_path, executable_path, "--child", solver_shared_object, nullptr);
    } else {
      execl(executable_path, executable_path, "--child", solver_shared_object, config_path, nullptr);
    }
    std::cerr << "failed to exec benchmark subprocess: " << std::strerror(errno) << '\n';
    _exit(127);
  }

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    std::cerr << "failed to wait for benchmark subprocess: " << std::strerror(errno) << '\n';
    return 1;
  }

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    std::cerr << "benchmark subprocess terminated by signal " << WTERMSIG(status) << '\n';
    return 128 + WTERMSIG(status);
  }
  return 1;
}

void print_usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " <solver-shared-object> [benchmark-config.json]\n";
}

}  // namespace
}  // namespace sparse_solver_gym

int main(int argc, char** argv) {
  try {
    if ((argc == 3 || argc == 4) && std::string_view(argv[1]) == "--child") {
      return sparse_solver_gym::run_child(argv[2], argc == 4 ? argv[3] : nullptr);
    }
    if (argc != 2 && argc != 3) {
      sparse_solver_gym::print_usage(argv[0]);
      return 64;
    }

    const std::string executable_path = sparse_solver_gym::current_executable_path(argv[0]);
    return sparse_solver_gym::run_parent(executable_path.c_str(),
                                         argv[1],
                                         argc == 3 ? argv[2] : nullptr);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
