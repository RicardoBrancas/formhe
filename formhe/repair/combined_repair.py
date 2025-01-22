import logging

import bentoml

import runhelper
from asp.instance import Instance
from repair.llm_repair import LLMRepair
from repair.mutation_repair import NextGenRepair
from repair.repair import RepairModule
from utils import config

logger = logging.getLogger('formhe.repair')


class CombinedRepair(RepairModule):

    def repair(self, fls, missing_lines: bool = False, iteration=0):
        res = False

        if not config.get().skip_llm_repair:
            with bentoml.SyncHTTPClient(self.instance.config.llm_url, timeout=600) as client:
                llm_repair = LLMRepair(self.instance, client)

            res = llm_repair.repair(fls, missing_lines)

            runhelper.log_any("llm.repair.success", res)

            if res:
                return True

            logger.info("LLM-based repair was unsuccessful. Trying to proceed using attempt as mutation starting point...")

            llm_repair_candidate = llm_repair.candidate

            if config.get().mutate_llm_attempt or (config.get().iterative_llm and iteration < config.get().max_llm_iterations):
                try:
                    modified_instance = Instance(config.get().input_file, skips=fls, reference_instance=self.instance.reference, canon_instance=self.instance.canon, suppress_override_message=True)
                    program_minus_fl = modified_instance.get_program_lines()
                    llm_repair_candidate_lines = llm_repair_candidate.split("\n")
                    new_program = program_minus_fl + llm_repair_candidate_lines
                    logger.info("New program:\n%s", "\n".join(new_program))

                    new_instance = Instance(ast=new_program, parent_config=modified_instance.config, suppress_override_message=True)
                    object.__setattr__(new_instance.config, "selfeval_lines", None)
                    new_fl = list(range(len(program_minus_fl), len(new_program)))
                    logger.info("New FL: %s", str(new_fl))
                    runhelper.log_any("llm.repair.candidate.usable", True)

                    if config.get().iterative_llm and iteration < config.get().max_llm_iterations:
                        import asp_integrated
                        res = asp_integrated.main(new_instance, iteration + 1)

                        if res:
                            return True

                except Exception as e:
                    runhelper.log_any("llm.repair.candidate.usable", False)
                    logger.warning("Could not use LLM attempt because: %s", str(e))
                    new_instance = self.instance
                    new_fl = fls

            if not config.get().mutate_llm_attempt:
                new_instance = self.instance
                new_fl = fls
        else:
            new_instance = self.instance
            new_fl = fls

        if config.get().mutation_based_repair and (config.get().mutate_llm_attempt or iteration == 0):
            logger.info("Starting mutation repair for:\n%s", new_instance.get_program_str())
            ng_repair = NextGenRepair(new_instance)
            res = ng_repair.repair(new_fl)

        return res
