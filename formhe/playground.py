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

instance.find_wrong_models(1000)

print(unsats)

#

#
# for f in instance.instrumenter.assumption_combos():
#     print(f)
