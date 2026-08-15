#include "vulkancraft/core/Parallel.hpp"

#define NOMINMAX
#include <Windows.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

namespace vulkancraft::core {

class WorldThreadPool::Impl final {
public:
    Impl() {
        for (std::jthread& worker : workers_) {
            worker = std::jthread([this] { worker_loop(); });
        }
    }

    ~Impl() {
        {
            const std::scoped_lock lock{state_mutex_};
            stopping_ = true;
            ++generation_;
        }
        work_available_.notify_all();
        for (std::jthread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void execute(
        const std::size_t task_count,
        std::function<void(std::size_t)> function) {
        if (task_count == 0U) {
            return;
        }
        const std::scoped_lock submission_lock{submission_mutex_};
        std::unique_lock state_lock{state_mutex_};
        function_ = std::move(function);
        task_count_ = task_count;
        next_task_.store(0U, std::memory_order_relaxed);
        completed_workers_ = 0U;
        exception_ = nullptr;
        ++generation_;
        work_available_.notify_all();
        work_completed_.wait(state_lock, [this] {
            return completed_workers_ == world_worker_count;
        });
        function_ = {};
        if (exception_ != nullptr) {
            std::rethrow_exception(exception_);
        }
    }

private:
    void worker_loop() {
        static_cast<void>(SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL));
        std::uint64_t observed_generation = 0U;
        while (true) {
            std::unique_lock state_lock{state_mutex_};
            work_available_.wait(state_lock, [this, observed_generation] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) {
                return;
            }
            observed_generation = generation_;
            state_lock.unlock();

            while (true) {
                const std::size_t task = next_task_.fetch_add(1U, std::memory_order_relaxed);
                if (task >= task_count_) {
                    break;
                }
                try {
                    function_(task);
                } catch (...) {
                    const std::scoped_lock exception_lock{exception_mutex_};
                    if (exception_ == nullptr) {
                        exception_ = std::current_exception();
                    }
                }
            }

            state_lock.lock();
            ++completed_workers_;
            if (completed_workers_ == world_worker_count) {
                work_completed_.notify_one();
            }
        }
    }

    std::array<std::jthread, world_worker_count> workers_;
    std::mutex submission_mutex_;
    std::mutex state_mutex_;
    std::mutex exception_mutex_;
    std::condition_variable work_available_;
    std::condition_variable work_completed_;
    std::function<void(std::size_t)> function_;
    std::atomic_size_t next_task_{};
    std::size_t task_count_{};
    std::size_t completed_workers_{};
    std::uint64_t generation_{};
    std::exception_ptr exception_;
    bool stopping_{false};
};

WorldThreadPool& WorldThreadPool::instance() {
    static WorldThreadPool pool;
    return pool;
}

WorldThreadPool::WorldThreadPool() : impl_(std::make_unique<Impl>()) {}

WorldThreadPool::~WorldThreadPool() = default;

void WorldThreadPool::execute(
    const std::size_t task_count,
    std::function<void(std::size_t)> function) {
    impl_->execute(task_count, std::move(function));
}

}  // namespace vulkancraft::core
