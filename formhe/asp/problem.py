import glob
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

from dataclass_wizard import YAMLWizard


@dataclass
class Problem(YAMLWizard):
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    optimization: bool = False

    inputs: list[str] = field(default_factory=list)
    input_predicates: list[str] = field(default_factory=list)
    input_constants: list[str] = field(default_factory=list)

    output_predicates: list[str] = field(default_factory=list)

    correct_implementations: list[str] = field(default_factory=list)
    canon_implementation: Optional[str] = None


    @property
    def input_predicates_tuple(self) -> list[tuple[str, int]]:
        return [(a, int(b)) for a, b in [s.split("/") for s in self.input_predicates]]

    @property
    def input_predicates_zero_arity(self) -> list[str]:
        return [pred for pred, arity in self.input_predicates_tuple if arity == 0]

    @property
    def output_predicates_tuple(self) -> list[tuple[str, int]]:
        return [(a, int(b)) for a, b in [s.split("/") for s in self.output_predicates]]

    @property
    def all_correct_implementations(self) -> list[str]:
        return list(chain.from_iterable(map(lambda s: glob.glob(s, recursive=True), self.correct_implementations)))