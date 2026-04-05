find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(sparse_solver_gym_tests
    json_test.cpp
    solver_gym_test.cpp
)

target_link_libraries(sparse_solver_gym_tests
    PRIVATE
        sparse_solver_gym
        GTest::gtest_main
)

gtest_discover_tests(sparse_solver_gym_tests)
