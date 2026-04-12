#pragma once

#include <memory>

#include "sparse_solver_gym/solver_gym.hpp"
#include "sparse_solver_gym/solver_gym_c.h"

namespace sparse_solver_gym {

[[nodiscard]] std::unique_ptr<ISolver> make_solver_from_c_api(ssg_solver_v1 solver);

}  // namespace sparse_solver_gym
