#pragma once

#include <atomic>
#include <boost/test/unit_test.hpp>
#include <thread>

#include "pytimeloop/utils/concurrent-queue.h"

using namespace boost::unit_test;

BOOST_AUTO_TEST_CASE(test_concurrent_queue_int) {
  const int N = 100;

  ConcurrentQueue<int> queue;

  for (unsigned i = 0; i < N; ++i) {
    queue.Push(i);
  }
  for (unsigned i = 0; i < N; ++i) {
    int val;
    queue.Pop(val);
    BOOST_TEST_REQUIRE(val == i);
  }
}

BOOST_AUTO_TEST_CASE(test_concurrent_queue_terminate_pop) {
  ConcurrentQueue<int> queue;

  std::atomic_bool terminate;

  std::thread t(
      [](std::atomic_bool& terminate, ConcurrentQueue<int>& queue) {
        int val;
        queue.Pop(val, [&terminate]() { return terminate.load(); });
      },
      std::ref(terminate), std::ref(queue));

  terminate.store(true);
  t.join();
}
