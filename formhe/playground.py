import itertools

from utils.iterutils import toggleset_add_one_to_base

preset_statements = [('head1', ['body1a', 'body1b']), (None, ['body2a', 'body2b'])]

stmt_combos = []
for head, body in preset_statements:
    if head is None:
        stmt_combos.append([(False, None, body_combo) for body_combo in toggleset_add_one_to_base(body)])
    else:
        stmt_combos.append([(True, head_combo, body_combo) for body_combo in toggleset_add_one_to_base(body) for head_combo in [head, None]])

for combo in itertools.product(*stmt_combos):
    print(combo)
