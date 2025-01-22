PROBLEM_TK = "<|problem|>"
REFERENCE_PROGRAM_TK = "<|reference_program|>"
INCORRECT_PROGRAM_TK = "<|incorrect_program|>"
FAULT_LOCALIZATION_TK = "<|fl|>"
MISSING_LINES_TK = "<|missing_lines|>"
CORRECTION_TK = "<|correction|>"
SPECIAL_TOKENS = [PROBLEM_TK, REFERENCE_PROGRAM_TK, INCORRECT_PROGRAM_TK, FAULT_LOCALIZATION_TK, MISSING_LINES_TK, CORRECTION_TK]


def requires(d: dict, key, extra_info=""):
    if key not in d.keys():
        raise TypeError(f"Argument {key} is required{' for ' + extra_info if extra_info else ''}")


def fl_prompt(version: int, **kwargs):
    if version == 1:
        requires(kwargs, "incorrect_program", "fault localization prompt version 1")
        return kwargs["incorrect_program"]

    elif version == 2:
        requires(kwargs, "correct_program", "fault localization prompt version 2")
        requires(kwargs, "incorrect_program", "fault localization prompt version 2")
        correct = kwargs["correct_program"]
        incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(kwargs["incorrect_program"].splitlines())))
        prompt = f"<correct>{correct}\n<incorrect>{incorrect}"
        return prompt

    elif version == 3:
        requires(kwargs, "title", "fault localization prompt version 3")
        requires(kwargs, "reference_program", "fault localization prompt version 3")
        requires(kwargs, "incorrect_program", "fault localization prompt version 3")

        incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(kwargs["incorrect_program"].splitlines())))
        return (f"{PROBLEM_TK}{kwargs['title']}\n"
                f"{REFERENCE_PROGRAM_TK}{kwargs['reference_program']}\n"
                f"{INCORRECT_PROGRAM_TK}{incorrect}")

    else:
        raise ValueError(f"Invalid fault localization prompt version {version}")


def fl_prompt_raw(version: int, **kwargs):
    if version == 1:
        requires(kwargs, "title", "fault localization prompt version 3")
        requires(kwargs, "reference_program", "fault localization prompt version 3")
        requires(kwargs, "incorrect_program", "fault localization prompt version 3")

        incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(kwargs["incorrect_program"].splitlines())))
        return (f"Your goal is to identify the lines with bugs in a Answer Set Programming submission aiming to solve the {kwargs['title']} problem.\n"
                f"\n"
                f"Here is the incorrect submission:\n"
                f"```\n"
                f"{incorrect}\n"
                f"```\n"
                f"\n"
                f"And here is an example of a correct reference implementation:\n"
                f"```\n"
                f"{kwargs['reference_program']}\n"
                f"```")

    else:
        raise ValueError(f"Invalid fault localization prompt version {version}")


def repair_prompt(version: int, include_response=False, **kwargs):
    if version == 1:
        requires(kwargs, "correct_program", "repair prompt version 1")
        requires(kwargs, "incorrect_program", "repair prompt version 1")
        requires(kwargs, "fl", "repair prompt version 1")

        incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(kwargs["incorrect_program"].splitlines())))
        correction = "\n".join(kwargs["correction"])
        fl = " ".join(map(str, kwargs["fl"]))
        prompt = f"Reference implementation:\n{kwargs['correct_program']}\nStudent submission:\n{incorrect}\nIncorrect lines:\n{fl}\nCorrection:\n"

        if include_response:
            requires(kwargs, "correction", "repair prompt version 1")
            requires(kwargs, "eos_token", "repair prompt version 1")
            prompt += correction + kwargs["eos_token"]

        return prompt

    elif version == 2:
        requires(kwargs, "correct_program", "repair prompt version 2")
        requires(kwargs, "incorrect_program", "repair prompt version 2")
        requires(kwargs, "fl", "repair prompt version 2")
        requires(kwargs, "missing_lines", "repair prompt version 2")

        incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(kwargs["incorrect_program"].splitlines())))
        fl = " ".join(map(str, kwargs["fl"]))
        prompt = f"Reference implementation:\n{kwargs['correct_program']}\nStudent submission:\n{incorrect}\nIncorrect lines:\n{fl}\nMissing lines: {'yes' if kwargs['missing_lines'] else 'no'}\nCorrection:\n"

        if include_response:
            requires(kwargs, "correction", "repair prompt version 2")
            requires(kwargs, "eos_token", "repair prompt version 2")
            correction = "\n".join(kwargs["correction"])
            prompt += correction + kwargs["eos_token"]

        return prompt

    elif version == 3:
        requires(kwargs, "title", "repair prompt version 3")
        requires(kwargs, "reference_program", "repair prompt version 3")
        requires(kwargs, "incorrect_program", "repair prompt version 3")
        requires(kwargs, "fl", "repair prompt version 3")
        requires(kwargs, "missing_lines", "repair prompt version 3")

        incorrect = "\n".join(map(lambda x: f"<{x[0]}>{x[1]}", enumerate(kwargs["incorrect_program"].splitlines())))
        fl = " ".join(map(str, kwargs["fl"]))
        prompt = (f"{PROBLEM_TK}{kwargs['title']}\n"
                  f"{REFERENCE_PROGRAM_TK}{kwargs['reference_program']}\n"
                  f"{INCORRECT_PROGRAM_TK}{incorrect}\n"
                  f"{FAULT_LOCALIZATION_TK}{fl}\n"
                  f"{MISSING_LINES_TK}{'yes' if kwargs['missing_lines'] else 'no'}\n"
                  f"{CORRECTION_TK}")

        if include_response:
            requires(kwargs, "correction", "repair prompt version 3")
            requires(kwargs, "eos_token", "repair prompt version 3")
            correction = "\n".join(kwargs["correction"])
            prompt += correction + kwargs["eos_token"]

        return prompt

    else:
        raise ValueError(f"Invalid repair prompt version {version}")


def get_repair_response(output: str, repair_prompt_version: int, eos_token: str) -> str:
    if repair_prompt_version == 1 or repair_prompt_version == 2:
        return output.split("Correction:\n", maxsplit=1)[1].removesuffix(eos_token)

    elif repair_prompt_version == 3:
        return output.split(CORRECTION_TK, maxsplit=1)[1].removesuffix(eos_token)

    else:
        raise ValueError(f"Invalid repair prompt version {repair_prompt_version}")
