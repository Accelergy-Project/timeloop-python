#pragma once

#include <boost/test/unit_test.hpp>
#include <compound-config/compound-config.hpp>
#include <filesystem>

using namespace boost::unit_test;

namespace fs = std::filesystem;

struct TimeloopExerciseConfigGenerator {
  TimeloopExerciseConfigGenerator(const std::string& timeloop_exercise_path)
      : TIMELOOP_EXERCISE_PATH(timeloop_exercise_path) {}

  config::CompoundConfig GetModelOneLevelConfig() {
    const fs::path MODEL_DIR = TIMELOOP_EXERCISE_PATH / ISPASS_EXERCISES_DIR /
                               "timeloop/00-model-conv1d-1level";

    const fs::path ARCH_PATH = MODEL_DIR / "arch/1level.arch.yaml";
    const fs::path MAP_PATH = MODEL_DIR / "map/conv1d-1level.map.yaml";
    const fs::path PROB_PATH = MODEL_DIR / "prob/conv1d.prob.yaml";

    return config::CompoundConfig(
        {ARCH_PATH.string(), MAP_PATH.string(), PROB_PATH.string()});
  }

  config::CompoundConfig GetModelThreeLevelSpatialConfig() {
    const fs::path MODEL_DIR = TIMELOOP_EXERCISE_PATH / ISPASS_EXERCISES_DIR /
                               "timeloop/04-model-conv1d+oc-3levelspatial";

    const fs::path ARCH_PATH = MODEL_DIR / "arch/3levelspatial.arch.yaml";
    const fs::path MAP_PATH =
        MODEL_DIR / "map/conv1d+oc+ic-3levelspatial-cp-ws.map.yaml";
    const fs::path PROB_PATH = MODEL_DIR / "prob/conv1d+oc+ic.prob.yaml";

    return config::CompoundConfig(
        {ARCH_PATH.string(), MAP_PATH.string(), PROB_PATH.string()});
  }

  config::CompoundConfig GetMapperThreeLevelFreeBypassConfig() {
    const fs::path MAPPER_DIR = TIMELOOP_EXERCISE_PATH / ISPASS_EXERCISES_DIR /
                                "timeloop/05-mapper-conv1d+oc-3level";

    const fs::path ARCH_PATH = MAPPER_DIR / "arch/3level.arch.yaml";
    const fs::path CONSTRAINTS_PATH =
        MAPPER_DIR / "constraints/conv1d+oc-3level-freebypass.constraints.yaml";
    const fs::path PROB_PATH = MAPPER_DIR / "prob/conv1d+oc.prob.yaml";
    const fs::path MAPPER_PATH = MAPPER_DIR / "mapper/exhaustive.mapper.yaml";
    const fs::path ERT_PATH =
        MAPPER_DIR / "ref-output/freebypass/timeloop-mapper.ERT.yaml";

    return config::CompoundConfig(
        {ARCH_PATH.string(), CONSTRAINTS_PATH.string(), PROB_PATH.string(),
         MAPPER_PATH.string(), ERT_PATH.string()});
  }

  config::CompoundConfig GetMapperEyerissConfig() {
    const fs::path MAPPER_DIR = TIMELOOP_EXERCISE_PATH / ISPASS_EXERCISES_DIR /
                                "timeloop/06-mapper-convlayer-eyeriss";

    const fs::path ARCH_PATH = MAPPER_DIR / "arch/eyeriss_like.yaml";
    const fs::path ARCH_RF_PATH =
        MAPPER_DIR / "arch/components/smartbuffer_RF.yaml";
    const fs::path ARCH_SRAM_PATH =
        MAPPER_DIR / "arch/components/smartbuffer_SRAM.yaml";
    const fs::path ARCH_CONSTR_PATH =
        MAPPER_DIR / "constraints/eyeriss_like_arch_constraints.yaml";
    const fs::path MAP_CONSTR_PATH =
        MAPPER_DIR / "constraints/eyeriss_like_map_constraints.yaml";
    const fs::path PROB_PATH = "prob/VGG02_layer5.yaml";
    const fs::path MAPPER_PATH = "mapper/mapper.yaml";

    return config::CompoundConfig(
        {ARCH_PATH.string(), ARCH_RF_PATH.string(), ARCH_SRAM_PATH.string(),
         ARCH_CONSTR_PATH.string(), MAP_CONSTR_PATH.string(),
         PROB_PATH.string(), MAPPER_PATH.string()});
  }

  static TimeloopExerciseConfigGenerator CreateFromPath(
      const std::string& timeloop_exercise_path) {
    return TimeloopExerciseConfigGenerator(timeloop_exercise_path);
  }

  const fs::path TIMELOOP_EXERCISE_PATH;
  const fs::path ISPASS_EXERCISES_DIR = "workspace/exercises/2020.ispass";
};

struct TestConfig {
  TestConfig() {
    BOOST_TEST_REQUIRE(boost::unit_test::framework::master_test_suite().argc ==
                       2);
  }

  void setup() {
    config_gen = std::make_unique<TimeloopExerciseConfigGenerator>(
        TimeloopExerciseConfigGenerator::CreateFromPath(
            framework::master_test_suite().argv[1]));
  }

  static inline std::unique_ptr<TimeloopExerciseConfigGenerator> config_gen;
};

BOOST_TEST_GLOBAL_FIXTURE(TestConfig);
