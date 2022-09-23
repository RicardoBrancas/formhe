import logging
from functools import cached_property

import cvc5.pythonic

from formhe.trinity.DSL import TyrellSpec
from formhe.trinity.smt_enumerator import SmtEnumerator
from formhe.utils import config

logger = logging.getLogger('formhe.asp.enumerator')


class Cvc5Enumerator(SmtEnumerator):

    def __init__(self, spec: TyrellSpec, depth=None, n_roots=1, predicates_names=None):
        super().__init__(spec, depth, n_roots, predicates_names)

    def init_solver(self):
        self.solver = cvc5.pythonic.SolverFor('QF_LIA')

        try:
            self.solver.setOption('seed', config.get().seed)
            self.solver.setOption('stats', True)
        except:
            pass

    @cached_property
    def smt_namespace(self):
        original_Or = cvc5.pythonic.Or

        def Or(*args):
            if isinstance(args[0], list):
                return Or(*args[0])
            if len(args) == 1:
                return args[0]
            else:
                return original_Or(*args)

        cvc5.pythonic.Or = Or

        return cvc5.pythonic
