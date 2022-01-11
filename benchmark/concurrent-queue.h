#pragma once

#include <benchmark/benchmark.h>

#include <queue>

#include "pytimeloop/model/accelerator-pool.h"

class ConcurrentQueueFixture : public benchmark::Fixture {
 public:
  std::unique_ptr<SpscQueue<int>> spsc_queue_;
  ConcurrentQueue<int> queue_;
  std::queue<int> seq_queue_;

  void SetUp(const ::benchmark::State& state) {
    spsc_queue_ = std::make_unique<SpscQueue<int>>(16);
  }
};

BENCHMARK_F(ConcurrentQueueFixture, SpscSequentialTest)
(benchmark::State& state) {
  for (auto _ : state) {
    spsc_queue_->push(0);
    int res;
    spsc_queue_->pop(res);
  }
}

BENCHMARK_F(ConcurrentQueueFixture, SequentialTest)(benchmark::State& state) {
  for (auto _ : state) {
    queue_.push(0);

    int res;
    queue_.pop(res);
  }
}

BENCHMARK_F(ConcurrentQueueFixture, ReferenceTest)(benchmark::State& state) {
  for (auto _ : state) {
    seq_queue_.push(0);

    auto res = seq_queue_.front();
    seq_queue_.pop();
  }
}
