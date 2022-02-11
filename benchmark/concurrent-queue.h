#pragma once

#include <benchmark/benchmark.h>

#include <queue>

#include "pytimeloop/model/accelerator-pool.h"

class ConcurrentQueueFixture : public benchmark::Fixture {
 public:
  ConcurrentQueue<int> queue_;
  std::queue<int> seq_queue_;
};

BENCHMARK_F(ConcurrentQueueFixture, SequentialTest)(benchmark::State& state) {
  for (auto _ : state) {
    queue_.Push(0);

    int res;
    queue_.Pop(res);
  }
}

BENCHMARK_F(ConcurrentQueueFixture, ReferenceTest)(benchmark::State& state) {
  for (auto _ : state) {
    seq_queue_.push(0);

    [[maybe_unused]] auto res = seq_queue_.front();
    seq_queue_.pop();
  }
}
