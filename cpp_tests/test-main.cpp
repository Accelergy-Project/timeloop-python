#define BOOST_TEST_MODULE Test Accelerator
#include <boost/test/included/unit_test.hpp>

#include "fixture-configs.h"
// Running Accelerator test breaks mapper test because of Timeloop issue:
// https://github.com/NVlabs/timeloop/issues/138
// #include "test-accelerator.h"
#include "test-concurrent-queue.h"
#include "test-mapper.h"
#include "test-worker-pool.h"
