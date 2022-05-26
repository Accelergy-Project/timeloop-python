#pragma once

#include <boost/test/unit_test.hpp>

#include "pytimeloop/utils/worker-pool.h"

using namespace boost::unit_test;

struct AddOneWorker {
  typedef int Task;
  typedef int Result;

  Result operator()(Task& task) { return task + 1; }
};

AddOneWorker AddOneFactory() { return AddOneWorker(); }

BOOST_AUTO_TEST_CASE(test_worker_pool_int) {
  const int N = 100;
  const int N_WORKERS = 4;
  WorkerPool<AddOneWorker> pool(N_WORKERS, AddOneFactory);

  std::map<uint64_t, int> id_to_task;
  for (int i = 0; i < N; ++i) {
    auto task_id = pool.PushTask(i);
    BOOST_TEST_REQUIRE((id_to_task.find(task_id) == id_to_task.end()));
    id_to_task.insert({task_id, i});
  }
  for (int i = 0; i < N; ++i) {
    auto result = pool.PopResult();
    BOOST_TEST_REQUIRE(id_to_task.at(result.id) + 1 == result.val);
  }

  pool.Terminate();
}
