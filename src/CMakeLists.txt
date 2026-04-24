add_library(sparse_solver_gym STATIC
    solver_gym.cpp
    solver_gym_cxx.cpp
)

target_link_libraries(sparse_solver_gym
    PUBLIC
        nlohmann_json::nlohmann_json
)

target_include_directories(sparse_solver_gym
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(sparse_solver_gym PUBLIC cxx_std_23)

add_executable(sparse_solver_gym_runner
    solver_gym_main.cpp
)

set_target_properties(sparse_solver_gym_runner
    PROPERTIES
        OUTPUT_NAME sparse-solver-gym
)

target_link_libraries(sparse_solver_gym_runner
    PRIVATE
        sparse_solver_gym
        ${CMAKE_DL_LIBS}
)

target_compile_features(sparse_solver_gym_runner PRIVATE cxx_std_23)
