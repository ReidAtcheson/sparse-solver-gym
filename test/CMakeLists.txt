find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(sparse_solver_gym_tests
    json_test.cpp
    solver_gym_c_api_test.cpp
    solver_gym_test.cpp
)

target_link_libraries(sparse_solver_gym_tests
    PRIVATE
        sparse_solver_gym
        GTest::gtest_main
)

gtest_discover_tests(sparse_solver_gym_tests)

add_library(fake_solver_plugin MODULE
    fake_solver_plugin.cpp
)

target_include_directories(fake_solver_plugin
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(fake_solver_plugin PRIVATE cxx_std_23)

add_test(
    NAME sparse_solver_gym_runner_smoke
    COMMAND
        $<TARGET_FILE:sparse_solver_gym_runner>
        $<TARGET_FILE:fake_solver_plugin>
        ${CMAKE_CURRENT_SOURCE_DIR}/random_banded_sparse_config.json
)
