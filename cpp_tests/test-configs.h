#pragma once

#include <compound-config/compound-config.hpp>
#include <filesystem>

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

  static TimeloopExerciseConfigGenerator CreateFromPath(
      const std::string& timeloop_exercise_path) {
    return TimeloopExerciseConfigGenerator(timeloop_exercise_path);
  }

  const fs::path TIMELOOP_EXERCISE_PATH;
  const fs::path ISPASS_EXERCISES_DIR = "workspace/exercises/2020.ispass";
};
