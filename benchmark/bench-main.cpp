#include <benchmark/benchmark.h>

#include <queue>

#include "pytimeloop/model/accelerator-pool.h"
#include "pytimeloop/model/accelerator.h"

// Timeloop
#include <compound-config/compound-config.hpp>
#include <mapping/parser.hpp>
#include <model/sparse-optimization-parser.hpp>
#include <workload/workload.hpp>

bool gTerminate = false;
bool gTerminateEval = false;

#include "accelerator-pool.h"
#include "accelerator.h"
#include "concurrent-queue.h"

BENCHMARK_MAIN();
