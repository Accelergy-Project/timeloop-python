#pragma once

#include <benchmark/benchmark.h>

#include "pytimeloop/model/accelerator-pool.h"

class FakeAcceleratorPool {
 public:
  FakeAcceleratorPool(int nthreads) : terminate_(false) {
    for (int i = 0; i < nthreads; i++) {
      workers_.push_back(std::thread(&FakeAcceleratorPool::worker_loop, this));
    }
  }

  void Evaluate(int x) { task_q_.Push(x); }

  int GetResult() {
    int res;
    res_q_.Pop(res);
    return res;
  }

  void Terminate() {
    terminate_ = true;

    for (auto& worker : workers_) {
      worker.join();
    }

    workers_.clear();
  }

 private:
  std::atomic_bool terminate_;
  std::vector<std::thread> workers_;

  ConcurrentQueue<int> task_q_;
  ConcurrentQueue<int> res_q_;

  void worker_loop() {
    while (!terminate_) {
      int task;
      task_q_.Pop(task);

      res_q_.Push(task);
    }
  }
};

class FakeAcceleratorBenchmarks_1000Iter : public benchmark::Fixture {
 public:
  std::unique_ptr<FakeAcceleratorPool> pool;
  const int N = 1000;

  void SetUp(const ::benchmark::State& state) {
    pool = std::make_unique<FakeAcceleratorPool>(state.range(0));
  }

  void TearDown(const ::benchmark::State&) { pool->Terminate(); }
};

BENCHMARK_DEFINE_F(FakeAcceleratorBenchmarks_1000Iter, FakeAcceleratorPool)
(benchmark::State& state) {
  for (auto _ : state) {
    for (int i = 0; i < N; i++) {
      pool->Evaluate(i);
      [[maybe_unused]] auto res = pool->GetResult();
    }
  }
}

BENCHMARK_REGISTER_F(FakeAcceleratorBenchmarks_1000Iter, FakeAcceleratorPool)
    ->RangeMultiplier(2)
    ->Range(1, 8)
    ->UseRealTime();
