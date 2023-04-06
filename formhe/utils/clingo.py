import clingo.ast


def parse_string(*args, **kwargs):
    x = []
    clingo.ast.parse_string(*args, **kwargs, callback=x.append)
    return x
