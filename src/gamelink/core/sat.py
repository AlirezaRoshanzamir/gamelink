from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import product
from typing import ClassVar, cast, override


class ExpressionResult(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def from_optional_bool(cls, value: bool | None) -> ExpressionResult:  # noqa: FBT001
        if value is None:
            return ExpressionResult.UNKNOWN
        if value:
            return ExpressionResult.TRUE
        return ExpressionResult.FALSE

    def __bool__(self) -> bool:
        if self == ExpressionResult.TRUE:
            return True
        if self == ExpressionResult.FALSE:
            return False
        msg = f"The {self} cannot be converted to bool."
        raise ValueError(msg)

    def __invert__(self) -> ExpressionResult:
        if self == ExpressionResult.TRUE:
            return ExpressionResult.FALSE
        if self == ExpressionResult.FALSE:
            return ExpressionResult.TRUE
        return ExpressionResult.UNKNOWN

    def __or__(self, other: ExpressionResult) -> ExpressionResult:
        if ExpressionResult.TRUE in (self, other):
            return ExpressionResult.TRUE
        if ExpressionResult.UNKNOWN in (self, other):
            return ExpressionResult.UNKNOWN
        return ExpressionResult.FALSE

    def __and__(self, other: ExpressionResult) -> ExpressionResult:
        if ExpressionResult.FALSE in (self, other):
            return ExpressionResult.FALSE
        if ExpressionResult.UNKNOWN in (self, other):
            return ExpressionResult.UNKNOWN
        return ExpressionResult.TRUE


class Expression(ABC):
    @abstractmethod
    def evaluate(
        self,
        terminals: Mapping[TerminalIdentifier, bool],
    ) -> ExpressionResult:
        pass

    @abstractmethod
    def extract_terminals(self) -> set[Terminal]:
        pass


class Constant(Expression):
    value: ExpressionResult

    @override
    def evaluate(
        self,
        terminals: Mapping[TerminalIdentifier, bool],
    ) -> ExpressionResult:
        return self.value

    @override
    def extract_terminals(self) -> set[Terminal]:
        return set()

    def __str__(self) -> str:
        return self.value.value


@dataclass(frozen=True)
class Terminal(Expression):
    symbol: str

    @override
    def evaluate(
        self,
        terminals: Mapping[TerminalIdentifier, bool],
    ) -> ExpressionResult:
        return ExpressionResult.from_optional_bool(terminals.get(self.symbol))

    @override
    def extract_terminals(self) -> set[Terminal]:
        return {self}

    def __str__(self) -> str:
        return self.symbol


TerminalIdentifier = str | Terminal


@dataclass(frozen=True)
class Not(Expression):
    operand: Expression

    @override
    def evaluate(
        self,
        terminals: Mapping[TerminalIdentifier, bool],
    ) -> ExpressionResult:
        return ~self.operand.evaluate(terminals)

    @override
    def extract_terminals(self) -> set[Terminal]:
        return self.operand.extract_terminals()

    def __str__(self) -> str:
        operand_string = str(self.operand)

        if " " not in operand_string or operand_string.startswith("("):
            return f"~{operand_string}"

        return f"~({operand_string})"


@dataclass
class Operator(Expression):
    operands: Sequence[Expression]
    operator_symbol: ClassVar[str]

    @override
    def extract_terminals(self) -> set[Terminal]:
        return set.union(*(operand.extract_terminals() for operand in self.operands))

    def __str__(self) -> str:
        return f" {self.operator_symbol} ".join(
            f"({operand!s})" if isinstance(operand, Operator) else str(operand)
            for operand in self.operands
        ).strip()


@dataclass
class Union(Operator):
    operator_symbol = "v"

    @override
    def evaluate(
        self,
        terminals: Mapping[TerminalIdentifier, bool],
    ) -> ExpressionResult:
        return reduce(
            ExpressionResult.__or__,
            (operand.evaluate(terminals) for operand in self.operands),
            ExpressionResult.FALSE,
        )


@dataclass
class Intersection(Operator):
    operator_symbol = "^"

    @override
    def evaluate(
        self,
        terminals: Mapping[TerminalIdentifier, bool],
    ) -> ExpressionResult:
        return reduce(
            ExpressionResult.__and__,
            (operand.evaluate(terminals) for operand in self.operands),
            ExpressionResult.TRUE,
        )


class Solver(ABC):
    @abstractmethod
    def solve(self, expression: Expression) -> Iterable[Mapping[Terminal, bool]]:
        pass


class BruteForceSolver(Solver):
    def __init__(
        self,
        precondition: Callable[[Mapping[Terminal, bool]], bool] | None = None,
    ) -> None:
        self._precondition: Callable[[Mapping[Terminal, bool]], bool] = (
            precondition or (lambda _: True)
        )

    @override
    def solve(self, expression: Expression) -> Iterable[Mapping[Terminal, bool]]:
        terminals = list(expression.extract_terminals())
        index_to_terminal = dict(enumerate(terminals))

        for record in product(*((True, False) for _ in range(len(terminals)))):
            terminal_values = {
                index_to_terminal[i]: value for i, value in enumerate(record)
            }

            if not self._precondition(terminal_values):
                continue

            evaluation = expression.evaluate(
                cast("dict[TerminalIdentifier, bool]", terminal_values),
            )
            if evaluation:
                yield terminal_values


if __name__ == "__main__":
    expression = Intersection(
        operands=[
            Union(operands=[Terminal(symbol="x"), Terminal(symbol="y")]),
            Union(operands=[Not(operand=Terminal(symbol="z")), Terminal(symbol="x")]),
        ],
    )

    solver = BruteForceSolver(precondition=lambda x: list(x.values()).count(True) == 1)
