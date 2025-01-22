import clingo.ast


def parse_string(*args, **kwargs):
    x = []

    def append_custom(elem):
        if elem.ast_type == clingo.ast.ASTType.Program:
            return
        x.append(elem)

    clingo.ast.parse_string(*args, **kwargs, callback=append_custom, logger=lambda x, y: None)
    return x
