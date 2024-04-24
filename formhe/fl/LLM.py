import logging
from typing import List

from _bentoml_impl.client import AbstractClient

import runhelper
from asp.instance import Instance
from asp.synthesis.AspInterpreter import AspInterpreter
from fl.FaultLocalizer import FaultLocalizer

logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger('formhe.asp.fault_localization')


class ListInterpreter(AspInterpreter):

    def eval_stmt_and(self, node, args):
        return [arg for arg in args]


class LLMFaultLocalizer(FaultLocalizer):

    def __init__(self, instance: Instance, bentoml_client: AbstractClient):
        super().__init__(instance)
        self.bentoml_client = bentoml_client
        self.scores = None

    def create_prompt(self) -> str:
        if self.instance.config.fl_prompt_version == 1:
            return self.instance.get_program_str()

        elif self.instance.config.fl_prompt_version == 2:
            correct = self.instance.ground_truth.get_program_str()
            incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(self.instance.get_program_lines())))
            prompt = f"<correct>{correct}\n<incorrect>{incorrect}"
            return prompt

        else:
            raise NotImplementedError()

    def parse_response(self, response: str) -> list:
        return [frozenset([i for (i, e) in enumerate(response) if e >= 0.5])]

    def fault_localize(self) -> List:
        runhelper.timer_start("fl.llm.time")

        prompt = self.create_prompt()

        logger.info("FL Prompt:\n%s", prompt)

        response = self.bentoml_client.fl(prompt)

        logger.info("FL Response:\n%s", response)

        self.scores = response[1:]
        lines = self.parse_response(response[1:])

        runhelper.timer_stop("fl.llm.time")

        runhelper.log_any('fl.llm', [set(s) for s in lines])
        runhelper.log_any('fl.llm.missing.lines.probability', response[0])

        return lines
