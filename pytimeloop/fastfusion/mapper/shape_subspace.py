from combinatorics.integer import integer_factorizations_to_n_parts


class ShapeSubspace:
    def __init__(self, rank_shapes: dict[int, int], ranks: list[int]):
        self.rank_shapes = rank_shapes
        self.ranks = ranks

        self.position_to_last = {}
        self.fill_position_to_next()

    def fill_position_to_next(self):
        self.rank_to_last = {}
        for i, r in enumerate(self.ranks):
            if r in self.rank_to_last:
                self.position_to_last[i] = self.rank_to_last[r]
            else:
                self.position_to_last[i] = None
            self.rank_to_last[r] = i

    def __iter__(self) -> 'ShapeSubspaceIterator':
        return ShapeSubspaceIterator(self)


class ShapeSubspaceIterator:
    def __init__(self, shape_subspace: ShapeSubspace):
        self.shapes = shape_subspace.rank_shapes
        self.ranks = shape_subspace.ranks
        self.choice_generators = self.make_choice_generators(shape_subspace)
        self.pos_to_last = shape_subspace.position_to_last

        self.is_started = False
        self.is_done = False
        self.just_skipped = False

        self.choice_iterators = None
        self.choice = None
        self.is_first_choice = None

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_started:
            self.is_started = True
            self.initialize_choice_iterators()
        elif not self.just_skipped:
            self.is_done = True
            for i in range(len(self.choice_iterators)):
                idx = len(self.choice_iterators)-i-1
                try:
                    self.move_iterator(idx)
                    break
                except StopIteration as e:
                    pass
            if len(self.choice_iterators) == 0:
                idx = 0
            for j in range(idx+1, len(self.choice_iterators)):
                self.restart_iterator(j)
        else:
            self.just_skipped = False

        if self.is_done:
            raise StopIteration()

        return self.choice.copy()

    def skip_current_rank_iteration(self, chain_skip_if_first=True):
        if self.is_done:
            return

        skip_limit = 1
        if chain_skip_if_first:
            for i in range(len(self.choice_iterators)):
                idx = len(self.choice_iterators)-i-1
                if not self.is_first_choice[idx]:
                    break
            if len(self.choice_iterators) == 0:
                skip_limit = 0
            else:
                skip_limit = i+1

        if skip_limit == len(self.choice_iterators):
            self.is_done = True
            self.just_skipped = True
            return

        self.is_done = True
        for i in range(skip_limit, len(self.choice_iterators)):
            idx = len(self.choice_iterators)-i-1
            try:
                self.move_iterator(idx)
                break
            except StopIteration as e:
                self.restart_iterator(idx)
        for j in range(idx+1, len(self.choice_iterators)):
            self.restart_iterator(j)

        self.just_skipped = True

    def make_choice_generators(self, shape_subspace: ShapeSubspace):
        choice_generators = []

        def gen(shape):
            if shape == 1:
                return [1]
            else:
                return [
                    s[0] for s in
                    integer_factorizations_to_n_parts(shape, 2)
                ][:-1]

        for _ in shape_subspace.ranks:
            choice_generators.append(gen)
        return choice_generators

    def initialize_choice_iterators(self):
        self.choice_iterators = [None]*len(self.choice_generators)
        self.choice = [None]*len(self.choice_generators)
        self.is_first_choice = [None]*len(self.choice_generators)
        for i in range(len(self.choice_generators)):
            try:
                self.restart_iterator(i)
            except StopIteration as e:
                self.is_done = True
                return

            self.is_first_choice[i] = True

    def restart_iterator(self, idx):
        last = self.pos_to_last[idx]
        if last is None:
            shape = self.shapes[self.ranks[idx]]
        else:
            shape = self.choice[last]
        self.choice_iterators[idx] = \
            iter(self.choice_generators[idx](shape))
        self.choice[idx] = next(self.choice_iterators[idx])
        self.is_first_choice[idx] = True

    def move_iterator(self, idx):
        val = next(self.choice_iterators[idx])
        self.choice[idx] = val
        self.is_first_choice[idx] = False
        self.is_done = False
