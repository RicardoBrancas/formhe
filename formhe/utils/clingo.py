import clingo.ast


def parse_string(*args, **kwargs):
    x = []
    clingo.ast.parse_string(*args, **kwargs, callback=x.append, logger=lambda x, y: None)
    return x
