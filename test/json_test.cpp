#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

TEST(JsonIntegrationTest, CanConstructSerializeAndParseJson) {
    nlohmann::json payload = {
        {"solver", "demo"},
        {"iterations", 3},
        {"success", true},
    };

    const auto serialized = payload.dump();
    const auto parsed = nlohmann::json::parse(serialized);

    EXPECT_EQ(parsed.at("solver").get<std::string>(), "demo");
    EXPECT_EQ(parsed.at("iterations").get<int>(), 3);
    EXPECT_TRUE(parsed.at("success").get<bool>());
}
