from itertools import chain, combinations, cycle, islice, product

from ordered_set import OrderedSet


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s), -1, -1))


def toggleset(iterable):
    return sorted(product(*[(elem, None) if elem is not None else [elem] for elem in iterable]), key=lambda combo: sum(x is None for x in combo))


def toggleset_add_one_to_base(iterable):
    return OrderedSet(
        [combo if sum(x is None for x in combo) != 0 or len(combo) == 0 else combo + (None,) for combo in
         sorted(product(*[(elem, None) for elem in iterable]), key=lambda combo: sum(x is None for x in combo))])


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def drop_nones(l: list) -> list:
    return [a for a in l if a is not None]
