import heapq
import random
from collections import Counter
from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence

from ordered_set import OrderedSet

from formhe.utils import config

T = TypeVar('T')


@dataclass
class HeapElement(Generic[T]):
    priority: float
    insertion_id: int
    bandit: T

    def __lt__(self, other):
        return (self.priority, self.insertion_id) < (other.priority, other.insertion_id)


REMOVED = '<removed>'


class MultiArmedBandit(Generic[T]):

    def __init__(self):
        self.bandits = Counter()
        self.random = random.Random(config.get().seed)
        self.epsilon = config.get().bandit_starting_epsilon
        self.epsilon_update = config.get().bandit_epsilon_multiplier
        self.first_n = config.get().bandit_exploration_count

    def __iter__(self):
        return iter(self.bandits.keys())

    def add_bandit(self, bandit: T, score: int = 0):
        self.bandits[bandit] = score

    def get_bandits(self, count: int, start: int = 0) -> Sequence[T]:
        if self.first_n != 0:
            self.first_n -= 1
        else:
            self.epsilon *= self.epsilon_update
        real_count = min(count, len(self.bandits))
        bandits = OrderedSet()
        bandit_i = start
        for _ in range(real_count):
            if self.random.random() > self.epsilon and self.first_n == 0:
                while True:
                    bandit = self.bandits.most_common(bandit_i + 1)[bandit_i][0]
                    bandit_i += 1
                    if bandit not in bandits:
                        bandits.add(bandit)
                        break
            else:
                while True:
                    bandit = self.random.choice(list(self.bandits.keys()))
                    if bandit not in bandits:
                        bandits.add(bandit)
                        break
        return bandits

    def update_bandit(self, bandit: T, score: int):
        self.bandits[bandit] += score

    def __repr__(self) -> str:
        return f'MultiArmedBandit(epsilon={self.epsilon}, values={repr(self.bandits.values())})'

# class MultiArmedBandit(Generic[T]):
#
#     def __init__(self):
#         self.bandits: list[HeapElement[T]] = []
#         self.bandit_mapper: dict[T, HeapElement[T]] = {}
#         self.insertion_counter = 0
#         self.random = random.Random(config.get().seed)
#         self.epsilon = 0.01
#
#     def __iter__(self):
#         return iter([heap_element.bandit for heap_element in sorted(self.bandits, key=lambda h: h.priority) if heap_element.bandit is not REMOVED])
#
#     def add_bandit(self, bandit: T, score: float = 0):
#         heap_element = HeapElement(score, self.insertion_counter, bandit)
#         heapq.heappush(self.bandits, heap_element)
#         self.bandit_mapper[bandit] = heap_element
#
#     def get_bandit(self) -> T:
#         self.epsilon *= 0.999
#         if self.random.random() > self.epsilon:
#             while True:
#                 bandit = self.bandits[0].bandit
#                 if bandit is REMOVED:
#                     heapq.heappop(self.bandits)
#                 else:
#                     return bandit
#         else:
#             while True:
#                 bandit = self.random.choice(self.bandits).bandit
#                 if bandit is REMOVED:
#                     if bandit == self.bandits[0].bandit:
#                         heapq.heappop(self.bandits)
#                 else:
#                     return bandit
#
#     def update_bandit(self, bandit, score):
#         previous_score = self.bandit_mapper[bandit].priority
#         self.remove_bandit(bandit)
#         self.add_bandit(bandit, previous_score + score)
#
#     def remove_bandit(self, bandit):
#         heap_element = self.bandit_mapper.pop(bandit)
#         heap_element.bandit = REMOVED
#
#     def __repr__(self) -> str:
#         return "\n".join(map(lambda h: f'Bandit(p={h.priority}, b={repr(h.bandit)})', sorted(self.bandits, key=lambda h: h.priority)))
