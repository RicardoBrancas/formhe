import logging

from _bentoml_impl.client import AbstractClient

import runhelper
from asp.instance import Instance
from asp.synthesis.AspInterpreter import AspInterpreter
from repair.repair import RepairModule
from utils import config
from utils.llm import repair_prompt

logger = logging.getLogger('formhe.repair')


class LLMRepair(RepairModule):

    def __init__(self, instance: Instance, bentoml_client: AbstractClient):
        super().__init__(instance)
        self.bentoml_client = bentoml_client
        self.candidate = None

    def create_prompt(self, fl, missing_lines: bool = False) -> str:
        return repair_prompt(version=self.instance.config.repair_prompt_version,
                             incorrect_program=self.instance.get_program_str(),
                             correct_program=self.instance.reference.get_program_str(),
                             reference_program=self.instance.reference.get_program_str(),
                             title=self.instance.problem.title,
                             fl=fl,
                             missing_lines=missing_lines)

    def parse_response(self, response: str) -> str:
        return response

    def repair(self, fls, missing_lines: bool = False) -> bool:
        runhelper.timer_start("repair.llm.time")
        runhelper.timer_start('enum.fail.time')

        modified_instance = Instance(config.get().input_file, skips=fls, reference_instance=self.instance.reference, canon_instance=self.instance.canon, suppress_override_message=True)
        self.interpreter = AspInterpreter(modified_instance)

        prompt = self.create_prompt(fls, missing_lines)

        logger.info("Repair Prompt:\n%s", prompt)

        response = self.bentoml_client.repair(prompt)

        logger.info("Repair Response:\n%s", response)

        repair_candidate = self.parse_response(response)
        self.candidate = repair_candidate

        result = self.test_candidate(repair_candidate, fls)

        runhelper.timer_stop("repair.llm.time")

        return result
