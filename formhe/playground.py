from contextlib import contextmanager, ExitStack, redirect_stdout, redirect_stderr

import os

from formhe.asp.instance import Instance


@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


instance = Instance('buggy_instances/nqueens/0.lp')

for f in instance.ast:
    print(f)

print()

for f in instance.instrumented_ast:
    print(f)

unsats = instance.find_mcs("queen(7,1). queen(6, 2).", minimum=False)

print(unsats)

#

#
# for f in instance.instrumenter.assumption_combos():
#     print(f)
