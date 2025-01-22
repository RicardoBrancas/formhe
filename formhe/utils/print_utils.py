from collections import defaultdict

from ordered_set import OrderedSet


def simplify(obj):
    if isinstance(obj, defaultdict) or isinstance(obj, dict):
        return {k: simplify(v) for (k, v) in obj.items()}

    elif isinstance(obj, OrderedSet) or isinstance(obj, frozenset) or isinstance(obj, list):
        return [simplify(o) for o in obj]

    return obj
