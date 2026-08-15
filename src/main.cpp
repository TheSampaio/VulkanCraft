#include <exception>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "vulkancraft/core/Application.hpp"

int main(const int argument_count, char** arguments) {
    try {
        const bool smoke_test = argument_count == 2 && std::string_view{arguments[1]} == "--smoke-test";
        const bool wireframe_smoke_test =
            argument_count == 2 && std::string_view{arguments[1]} == "--wireframe-smoke-test";
        const bool streaming_smoke_test =
            argument_count == 2 && std::string_view{arguments[1]} == "--streaming-smoke-test";
        const bool edit_smoke_test =
            argument_count == 2 && std::string_view{arguments[1]} == "--edit-smoke-test";
        const bool night_smoke_test =
            argument_count == 2 && std::string_view{arguments[1]} == "--night-smoke-test";
        const bool benchmark =
            argument_count == 2 && std::string_view{arguments[1]} == "--benchmark";
        const bool visual_smoke_test =
            argument_count == 2 && std::string_view{arguments[1]} == "--visual-smoke-test";
        const bool console_smoke_test =
            argument_count == 2 && std::string_view{arguments[1]} == "--console-smoke-test";
        if (argument_count > 1 && !smoke_test && !wireframe_smoke_test &&
            !streaming_smoke_test && !edit_smoke_test && !night_smoke_test && !benchmark &&
            !visual_smoke_test && !console_smoke_test) {
            std::cerr <<
                "Usage: vulkancraft [--smoke-test|--wireframe-smoke-test|"
                "--streaming-smoke-test|--edit-smoke-test|--night-smoke-test|"
                "--visual-smoke-test|--console-smoke-test|--benchmark]\n";
            return 2;
        }
        const bool automated_test = smoke_test || wireframe_smoke_test || streaming_smoke_test ||
                                    edit_smoke_test || night_smoke_test || benchmark ||
                                    visual_smoke_test;
        const bool any_automated_test = automated_test || console_smoke_test;
        vulkancraft::core::Application application{
            !any_automated_test || visual_smoke_test || console_smoke_test,
            wireframe_smoke_test,
            console_smoke_test,
            visual_smoke_test};
        constexpr std::uint32_t benchmark_frame_count = 600U;
        const auto benchmark_started_at = std::chrono::steady_clock::now();
        application.run(
            any_automated_test
                ? std::optional<std::uint32_t>{
                      benchmark ? benchmark_frame_count
                      : (visual_smoke_test || console_smoke_test) ? 10000U
                      : (streaming_smoke_test ? 10000U : 3U)}
                : std::nullopt,
            streaming_smoke_test,
            edit_smoke_test,
            night_smoke_test ? 885.0F : 0.0F);
        if (benchmark) {
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - benchmark_started_at)
                                       .count();
            std::cout << std::fixed << std::setprecision(1)
                      << "Benchmark: " << static_cast<double>(benchmark_frame_count) / seconds
                      << " FPS across " << benchmark_frame_count << " frames.\n";
        } else if (any_automated_test) {
            std::cout << "Vulkan smoke test passed\n";
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "VulkanCraft stopped: " << exception.what() << '\n';
        return 1;
    }
}
