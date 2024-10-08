from collections import defaultdict
from collections.abc import Mapping, Sequence

from combinatorics.integer import integer_factorizations_to_n_parts


class ShapeSubspace:
    def __init__(self,
                 rank_shapes: Mapping[str, int],
                 rank_order: Sequence[str]):
        self.rank_shapes = rank_shapes

        rank_idxs = defaultdict(list)
        for i, r in enumerate(rank_order):
            rank_idxs[r].append(i)

        self.permutation_map = []
        self.choices = []
        for rank, idxs in rank_idxs.items():
            self.permutation_map += idxs
            self.choices.append([
                (shapes, rank, shapes[-1])
                for shapes in get_shapes_of_int(rank_shapes[rank], len(idxs))
            ])

    def __iter__(self) -> 'ShapeSubspaceIterator':
        return ShapeSubspaceIterator(self)


class ShapeSubspaceIterator:
    def __init__(self, subspace: ShapeSubspace):
        self.subspace = subspace
        self.permutation_map = subspace.permutation_map
        self.choices = subspace.choices

        self.choice_iterators = [iter(choice) for choice in self.choices]
        self.is_started = False
        self.is_done = False
        self.choice = None
        self.is_first_choice = [False]*len(self.choices)
        self.just_skipped = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_started:
            self.is_started = True
            try:
                self.choice = [next(it) for it in self.choice_iterators]
                self.is_first_choice = [True]*len(self.choice_iterators)
            except StopIteration as e:
                self.is_done = True
        elif not self.just_skipped:
            self.is_done = True
            for i in range(len(self.choice_iterators)):
                idx = len(self.choice_iterators)-i-1
                try:
                    self.move_iterator(idx)
                    break
                except StopIteration as e:
                    self.restart_iterator(idx)
        else:
            self.just_skipped = False

        if self.is_done:
            raise StopIteration()

        return self.permute_choices_to_original()

    def skip_current_rank_iteration(self, chain_skip_if_first=True):
        if self.is_done:
            return

        skip_limit = 1
        if chain_skip_if_first:
            for i in range(len(self.choice_iterators)):
                idx = len(self.choice_iterators)-i-1
                if not self.is_first_choice[idx]:
                    break
            skip_limit = i+1

        if skip_limit == len(self.choice_iterators):
            self.is_done = True
            self.just_skipped = True
            return

        for i in range(skip_limit):
            idx = len(self.choice_iterators)-i-1
            self.restart_iterator(idx)

        self.is_done = True
        for i in range(skip_limit, len(self.choice_iterators)):
            idx = len(self.choice_iterators)-i-1
            try:
                self.move_iterator(idx)
                break
            except StopIteration as e:
                self.restart_iterator(idx)

        self.just_skipped = True

    def permute_choices_to_original(self):
        reordered_choice = [None]*len(self.permutation_map)

        leftover_rank_shapes = self.subspace.rank_shapes.copy()

        appended_choice = []
        for chosen_shape, rank, leftover_shape in self.choice:
            appended_choice += chosen_shape
            leftover_rank_shapes[rank] = leftover_shape

        for i_prime, val in zip(self.permutation_map, appended_choice):
            reordered_choice[i_prime] = val

        return reordered_choice, leftover_rank_shapes

    def restart_iterator(self, idx):
        self.choice_iterators[idx] = iter(self.choices[idx])
        self.choice[idx] = next(self.choice_iterators[idx])
        self.is_first_choice[idx] = True

    def move_iterator(self, idx):
        val = next(self.choice_iterators[idx])
        self.choice[idx] = val
        self.is_first_choice[idx] = False
        self.is_done = False


def get_shapes_of_int(n, levels):
    factors = [tup[0] for tup in integer_factorizations_to_n_parts(n, 2)]
    if levels == 1:
        for f in factors:
            yield [f]
        return
    for f in factors:
        for shape in get_shapes_of_int(f, levels-1):
            yield [f] + shape
