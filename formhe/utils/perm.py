import itertools
import logging
import math
import random
from typing import Any

import z3

from formhe.trinity.ng_enumerator import NextGenEnumerator

logger = logging.getLogger("formhe.permutations")


class PermutationGenerator:

    def __init__(self, seed: Any, n_holes: int, n_mutations: int, do_init=True):
        self.n_holes = n_holes
        self.n_mutations = n_mutations
        self.random = random.Random(seed)
        self.closed = []
        if do_init:
            self._open = list(itertools.combinations(range(self.n_holes), self.n_mutations))
        else:
            self._open = []

    def open(self, perm: list[bool]):
        hole_positions = tuple(i for i in range(self.n_holes) if perm[i])
        self._open.append(hole_positions)

    def pick(self) -> list[bool]:
        hole_positions = self.random.choice(self._open)
        return [True if i in hole_positions else False for i in range(self.n_holes)]

    def close(self, perm: list[bool]):
        hole_positions = tuple(i for i in range(self.n_holes) if perm[i])
        self._open.remove(hole_positions)
        self.closed.append(hole_positions)

    def is_done(self):
        return not self._open


class PermutationGeneratorHelper:

    def __init__(self, seed: Any, n_holes: int, n_mutations: int, enumerator: NextGenEnumerator, return_perms=False):
        self.permutation_generator = PermutationGenerator(seed, n_holes, n_mutations, do_init=False)
        self.enumerator = enumerator
        self.in_startup_phase = True
        self.enumerator.solver.push()
        self.enumerator.set_mutation_count(n_mutations)
        self.blocked_models = []
        self.return_perms = return_perms

    def next(self):
        if self.in_startup_phase:
            prog = self.enumerator.next()

            if prog is None:
                self.in_startup_phase = False
                self.enumerator.solver.pop()
                for blocked_model in self.blocked_models:
                    self.enumerator.block_model(blocked_model)
                logger.info("Startup phase done. Opened %d/%d permutations (%.2f%%).",
                            len(self.permutation_generator._open),
                            math.comb(self.permutation_generator.n_holes, self.permutation_generator.n_mutations),
                            len(self.permutation_generator._open) / math.comb(self.permutation_generator.n_holes, self.permutation_generator.n_mutations) * 100)
                return self.next()

            self.blocked_models.append(self.enumerator.model)

            perm = self.enumerator.model_relaxation_values(self.enumerator.model)
            # logger.debug("Opening %s", str(perm))

            self.permutation_generator.open(perm)

            self.enumerator.block_relaxation(perm)

            if self.return_perms:
                return prog, perm
            else:
                return prog

        else:

            if self.permutation_generator.is_done():
                # logger.debug("Permutations done")
                if self.return_perms:
                    return None, None
                else:
                    return None

            perm = self.permutation_generator.pick()
            prog = self.enumerator.next(z3.substitute_vars(self.enumerator.relaxation_template, *[z3.BoolVal(p) for p in perm]))

            if prog is None:
                # logger.debug("Closing %s", str(perm))
                self.permutation_generator.close(perm)
                return self.next()

            if self.return_perms:
                return prog, perm
            else:
                return prog
