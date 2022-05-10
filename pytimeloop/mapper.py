from enum import Enum


class Betterness(Enum):
    BETTER = 1
    SLIGHTLY_BETTER = 2
    WORSE = -1
    SLIGHTLY_WORSE = -2


class SearchTask:
    def __init__(self, task_id, mapping, only_bypass_changed):
        self.task_id = task_id
        self.mapping = mapping
        self.only_bypass_changed = only_bypass_changed
