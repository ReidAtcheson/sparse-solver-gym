add_library(sparse_solver_gym STATIC
    solver_gym.cpp
)

target_link_libraries(sparse_solver_gym
    PUBLIC
        nlohmann_json::nlohmann_json
)

target_include_directories(sparse_solver_gym
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(sparse_solver_gym PUBLIC cxx_std_17)
