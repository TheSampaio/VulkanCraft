#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace vulkancraft::core {

inline constexpr std::size_t world_worker_count = 4U;

/** Owns the four persistent workers shared by generation and meshing. */
class WorldThreadPool final {
public:
    /** Returns the process-wide world worker pool. */
    [[nodiscard]] static WorldThreadPool& instance();

    /**
     * Executes every indexed task and waits until all four workers finish.
     *
     * @param task_count Number of independent tasks.
     * @param function Work executed exactly once for each task index.
     */
    void execute(std::size_t task_count, std::function<void(std::size_t)> function);

    WorldThreadPool(const WorldThreadPool&) = delete;
    WorldThreadPool& operator=(const WorldThreadPool&) = delete;
    WorldThreadPool(WorldThreadPool&&) = delete;
    WorldThreadPool& operator=(WorldThreadPool&&) = delete;

private:
    WorldThreadPool();
    ~WorldThreadPool();

    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Executes independent indexed work across at most four worker threads.
 *
 * The caller remains the coordinator, so a streaming rebuild uses four workers
 * without consuming the render thread.
 *
 * @tparam Function Callable accepting one std::size_t task index.
 * @param task_count Number of independent tasks.
 * @param function Work executed exactly once for each task index.
 */
template <typename Function>
void parallel_for(const std::size_t task_count, Function&& function) {
    WorldThreadPool::instance().execute(
        task_count,
        [callable = std::forward<Function>(function)](const std::size_t task) mutable {
            callable(task);
        });
}

}  // namespace vulkancraft::core
