from itertools import chain
from typing import List

import clingo.ast
import runhelper
from ordered_set import OrderedSet

from fl.fault_localizer import FaultLocalizer
from formhe.asp.highlithing_visitor import AspHighlithingVisitor, MaxScorer, RuleScoreCalculator
from formhe.utils import config


class SBFL(FaultLocalizer):

    def fault_localize(self) -> List:
        try:
            highlighter = AspHighlithingVisitor()

            highlighter.highlight(config.get().input_file, list(chain.from_iterable(self.instance.missing_models)))

            scorer = MaxScorer()
            rule_score_calculator = RuleScoreCalculator(highlighter.graph.weights, scorer)

            clingo.ast.parse_files([config.get().input_file], lambda x: rule_score_calculator.visit(x))

            selected_lines = OrderedSet([rule for rule, score in sorted(rule_score_calculator.scorer.get_scores(), key=lambda x: x[1], reverse=True)[:1]])

            runhelper.log_any('fl.sbfl', sorted(rule_score_calculator.scorer.get_scores(), key=lambda x: x[1], reverse=True))

            return [selected_lines]
        except:
            return []
