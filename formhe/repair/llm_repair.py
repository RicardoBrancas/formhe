import logging

from _bentoml_impl.client import AbstractClient

import runhelper
from asp.instance import Instance
from asp.synthesis.AspInterpreter import AspInterpreter
from repair.repair import RepairModule
from utils import config

logger = logging.getLogger('formhe.repair')


class LLMRepair(RepairModule):

    def __init__(self, instance: Instance, bentoml_client: AbstractClient):
        super().__init__(instance)
        self.bentoml_client = bentoml_client

    def create_prompt(self, fl) -> str:
        if self.instance.config.repair_prompt_version == 1:
            correct = self.instance.ground_truth.get_program_str()
            incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(self.instance.get_program_lines())))
            fl = " ".join(map(str, fl))
            prompt = f"Reference implementation:\n{correct}\nStudent submission:\n{incorrect}\nIncorrect lines:\n{fl}\nCorrection:\n"
            return prompt
        else:
            raise NotImplementedError()

    def parse_response(self, response: str) -> str:
        return response

    def repair(self, fls, predicates) -> bool:
        runhelper.timer_start("repair.llm.time")
        runhelper.timer_start('enum.fail.time')

        fl = fls[0]  # todo change me?

        modified_instance = Instance(config.get().input_file, skips=fl, ground_truth_instance=self.instance.ground_truth, suppress_override_message=True)
        self.interpreter = AspInterpreter(modified_instance, predicates.keys())

        prompt = self.create_prompt(fl)

        logger.info("Repair Prompt:\n%s", prompt)

        response = self.bentoml_client.repair(prompt)

        logger.info("Repair Response:\n%s", response)

        repair_candidate = self.parse_response(response)

        result = self.test_candidate(repair_candidate)

        runhelper.timer_stop("repair.llm.time")

        return result
