# import atexit
# import logging
# import signal
# import time
# from collections import Counter
# from typing import Any
#
# logger = logging.getLogger('formhe.perf')
#
# counters = []
# counters_secondary = []
# start_time = time.perf_counter_ns()
#
#
# def new_counter():
#     counters.append(0)
#     counters_secondary.append(0)
#     return len(counters) - 1
#
#
# def new_counter_counter():
#     counters.append(Counter())
#     counters_secondary.append(0)
#     return len(counters) - 1
# #
# #
# # BLOCK_TIME = new_counter()
# # EVAL_TIME = new_counter()
# # START_TIME = new_counter()
# # FAULT_LOCALIZATION_TIME = new_counter()
# # ANSWER_SET_ENUM_TIME = new_counter()
# # SMT_ENUM_CONSTRUCTION_TIME = new_counter()
# # Z3_ENUM_TIME = new_counter()
# # BUILD_PROG_TIME = new_counter()
# # GROUNDING_TIME = new_counter()
# # ENUM_PROGRAMS = new_counter()
# # BLOCK_PROGRAMS = new_counter()
# # EVAL_FAIL_PROGRAMS = new_counter()
# # SAT = new_counter()
# # UNIQUE_SAT = new_counter()
# # UNSAT = new_counter()
# # UNIQUE_UNSAT = new_counter()
# # CANDIDATE_CHECKS_COUNTER = new_counter_counter()
# # GT_CHECKS_COUNTER = new_counter_counter()
# # EVAL_FAIL_TIME = new_counter()
# # TMP_TIME = new_counter()
# # TMP_TIME2 = new_counter()
#
#
# def timer_start(timer: int):
#     if __debug__:
#         counters_secondary[timer] = time.perf_counter_ns()
#
#
# def timer_stop(timer: int):
#     if __debug__:
#         counters[timer] += time.perf_counter_ns() - counters_secondary[timer]
#
#
# def counter_inc(counter: int, inc: int = 1):
#     if __debug__:
#         counters[counter] += inc
#
#
# def update_avg(counter: int, value: float):
#     if __debug__:
#         counters[counter] += value
#         counters_secondary[counter] += 1
#
#
# def update_counter(counter: int, value: Any):
#     if __debug__:
#         counters[counter][value] += 1
#
#
# def log_custom(tag_name, value):
#     logger.info(f'perf.{tag_name}=%s', str(value))
#
#
# @atexit.register
# def log():
#     # logger.info('perf.fault.localization.time=%f', float(counters[FAULT_LOCALIZATION_TIME]) / 1e9)
#     # logger.info('perf.answer.set.enum.time=%f', float(counters[ANSWER_SET_ENUM_TIME]) / 1e9)
#     # logger.info('perf.smt.enum.construction.time=%f', float(counters[SMT_ENUM_CONSTRUCTION_TIME]) / 1e9)
#     # logger.info('perf.smt.enum.time=%f', float(counters[Z3_ENUM_TIME]) / 1e9)
#     # logger.info('perf.build_program.time=%f', float(counters[BUILD_PROG_TIME]) / 1e9)
#     # logger.info('perf.block.time=%f', float(counters[BLOCK_TIME]) / 1e9)
#     # logger.info('perf.eval.time=%f', float(counters[EVAL_TIME]) / 1e9)
#     # logger.info('perf.grounding.time=%f', float(counters[GROUNDING_TIME]) / 1e9)
#     # logger.info('perf.tmp.1.time=%f', float(counters[TMP_TIME]) / 1e9)
#     # logger.info('perf.tmp.2.time=%f', float(counters[TMP_TIME2]) / 1e9)
#     # logger.info('perf.enumerated.count=%d', counters[ENUM_PROGRAMS])
#     # logger.info('perf.blocked.count=%d', counters[BLOCK_PROGRAMS])
#     # logger.info('perf.eval.failed.count=%d', counters[EVAL_FAIL_PROGRAMS])
#     # logger.info('perf.eval.failed.time=%f', float(counters[EVAL_FAIL_TIME]) / 1e9)
#     # logger.info('perf.sat.calls.count=%d', counters[SAT])
#     # logger.info('perf.sat.calls.unique.count=%d', counters[UNIQUE_SAT])
#     # logger.info('perf.unsat.calls.count=%d', counters[UNSAT])
#     # logger.info('perf.unsat.calls.unique.count=%d', counters[UNIQUE_UNSAT])
#     # logger.info('perf.candidate.checks.multicount=%s', str(counters[CANDIDATE_CHECKS_COUNTER]))
#     # logger.info('perf.ground_truth.checks.multicount=%s', str(counters[GT_CHECKS_COUNTER]))
#     logger.info('perf.total.time=%f', float(time.perf_counter_ns() - start_time) / 1e9)
#
#
# def termination_handler(signum=None, frame=None):
#     logger.warning('Termination signal received. Exiting...')
#     log()
#     exit()
#
#
# signal.signal(signal.SIGTERM, termination_handler)
