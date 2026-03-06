from __future__ import annotations

import contextlib
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Any, override


class DecisionSelector(ABC):
    def __init__(self) -> None:
        self._last_number_of_decisions = 0
        self._last_selected_decision_index = -1

    @property
    def last_number_of_decisions(self) -> int:
        return self._last_number_of_decisions

    @property
    def last_selected_decision_index(self) -> int:
        return self._last_selected_decision_index

    def select[TDecision](
        self,
        decisions: Sequence[TDecision],
        weights: Sequence[float],
    ) -> TDecision:
        return decisions[self.select_index(decisions, weights)]

    def select_index[TDecision](
        self,
        decisions: Sequence[TDecision],
        weights: Sequence[float],
    ) -> int:
        self._last_number_of_decisions = len(decisions)
        selected_index = self.select_index_hook(decisions, weights)
        self._last_selected_decision_index = selected_index
        return selected_index

    @abstractmethod
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[TDecision],
        weights: Sequence[float],
    ) -> int:
        pass


class DelegatedDecisionSelector(DecisionSelector):
    def __init__(self, delegation_function: Callable[[], DecisionSelector]) -> None:
        self._delegation_function = delegation_function

    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[TDecision],
        weights: Sequence[float],
    ) -> int:
        return self._delegation_function().select_index_hook(decisions, weights)


class SamplingDecisionSelector(DecisionSelector):
    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[TDecision],
        weights: Sequence[float],
    ) -> int:
        return random.choices(list(range(len(decisions))), weights)[0]


class DecisionProducer:
    def __init__(self, decision_selector: DecisionSelector | None = None) -> None:
        self._decision_selector = decision_selector or SamplingDecisionSelector()

    @contextlib.contextmanager
    def with_decision_selector(
        self,
        decision_selector: DecisionSelector,
    ) -> Iterator[None]:
        pre_decision_selector = self._decision_selector
        self._decision_selector = decision_selector
        yield
        self._decision_selector = pre_decision_selector

    def select_decision[TDecision](
        self,
        decisions: Sequence[TDecision],
        weights: Sequence[float],
    ) -> TDecision:
        return self._decision_selector.select(decisions, weights)


class Game[TState: State, TPlayer: Player[Any]](DecisionProducer):
    def __init__(self) -> None:
        super().__init__()
        self._players: set[TPlayer] = set()
        self._decision_producers: list[DecisionProducer] = []
        self._state = self.create_initial_state()

    @classmethod
    @abstractmethod
    def create_initial_state(cls) -> TState:
        pass

    def join_player(self, player: TPlayer) -> None:
        self._players.add(player)
        self.join_decision_producer(player)

    def join_decision_producer(self, producer: DecisionProducer) -> None:
        self._decision_producers.append(producer)

    @override
    @contextlib.contextmanager
    def with_decision_selector(
        self,
        decision_selector: DecisionSelector,
    ) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for decision_producer in self._decision_producers:
                stack.enter_context(
                    decision_producer.with_decision_selector(decision_selector),
                )
            stack.enter_context(super().with_decision_selector(decision_selector))
            yield

    @property
    def players(self) -> set[TPlayer]:
        return self._players

    @property
    def finished(self) -> bool:
        return self.state.finished

    @property
    def state(self) -> TState:
        return self._state

    def in_state(self, state: TState) -> bool:
        return hash(self.state) == hash(state)

    def step_all_forward(self) -> None:
        while not self.finished:
            self.step_forward()

    @abstractmethod
    def step_forward(self) -> None:
        pass

    @abstractmethod
    def step_backward(self) -> None:
        pass


class Player[TState](DecisionProducer, ABC):
    @abstractmethod
    def act(self, state: TState) -> Action:
        pass

    @abstractmethod
    def score(self) -> float:
        pass


class State:
    @property
    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


class Action:
    @abstractmethod
    def is_feasible(self) -> bool:
        pass

    @abstractmethod
    def do(self) -> None:
        pass

    @abstractmethod
    def revert(self) -> None:
        pass
