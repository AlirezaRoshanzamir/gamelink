from __future__ import annotations

import contextlib
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence, Iterable, Mapping
from typing import Any, cast, override
from dataclasses import dataclass
import sys

type Readonly[T] = T


@dataclass
class Probabilistic[TEvent]:
    event: TEvent
    probability: float

    def __post_init__(self) -> None:
        if self.probability < 0 or self.probability > 1:
            raise ValueError(f"The {self.probability=} is not valid.")

    @classmethod
    def many_from_mapping(
        cls, each_event_probability: Mapping[TEvent, float]
    ) -> Sequence[Probabilistic[TEvent]]:
        return [
            Probabilistic(event=event, probability=probability)
            for event, probability in each_event_probability.items()
        ]

    @classmethod
    def many_uniform(cls, events: Sequence[TEvent]) -> Sequence[Probabilistic[TEvent]]:
        return [Probabilistic(event, 1.0 / len(events)) for event in events]

    @classmethod
    def deterministic(cls, event: TEvent) -> Probabilistic[TEvent]:
        return cls(event=event, probability=1.0)


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
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> TDecision:
        return decisions[self.select_index(decisions, title)].event

    def select_index[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        self._last_number_of_decisions = len(decisions)
        selected_index = self.select_index_hook(decisions, title)
        self._last_selected_decision_index = selected_index
        return selected_index

    @abstractmethod
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        pass


class DelegatedDecisionSelector(DecisionSelector):
    def __init__(self, delegation_function: Callable[[], DecisionSelector]) -> None:
        self._delegation_function = delegation_function

    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        return self._delegation_function().select_index_hook(decisions, title)


class SamplingDecisionSelector(DecisionSelector):
    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        return random.choices(
            population=list(range(len(decisions))),
            weights=[decision.probability for decision in decisions],
        )[0]


class CliDecisionSelector(DecisionSelector):
    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        decisions_part = ", ".join(
            f"{i + 1}: {decision.event}" for i, decision in enumerate(decisions)
        )
        title_part = f'"{title}" ' if title is not None else ""

        while True:
            try:
                n = int(input(f"Select {title_part}between ({decisions_part}): "))
            except ValueError:
                continue
            if not 1 <= n <= len(decisions):
                continue
            break

        return n - 1


class DecisionProducer:
    def __init__(
        self,
        decision_selector: DecisionSelector | None = None,
    ) -> None:  # For cooperative multiple inheritance
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
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> TDecision:
        return self._decision_selector.select(decisions, title)


class Game[TObservation: Observation, TAction: Action[Any], TPlayer: Player[Any, Any]](
    DecisionProducer
):
    def __init__(self) -> None:
        super().__init__()
        self._players: set[TPlayer] = set()
        self._decision_producers: list[DecisionProducer] = []

    def join_player(self, player: TPlayer) -> None:
        self._players.add(player)
        self.join_decision_producer(player)

    def join_decision_producer(self, producer: DecisionProducer) -> None:
        self._decision_producers.append(producer)

    def replace_player(self, player: TPlayer, replacement: TPlayer) -> None:
        self._players.remove(player)
        self._players.add(replacement)
        self.replace_decision_producer(player, replacement)

    def replace_players(self, replacements: Mapping[TPlayer, TPlayer]) -> None:
        for player, replacement in replacements.items():
            self.replace_player(player, replacement)

    def replace_decision_producer(
        self, producer: DecisionProducer, replacement: DecisionProducer
    ) -> None:
        self._decision_producers.remove(producer)
        self._decision_producers.append(replacement)

    @contextlib.contextmanager
    def with_player_replacement(
        self, player: TPlayer, replacement: TPlayer
    ) -> Iterator[None]:
        self.replace_player(player, replacement)
        yield
        self.replace_player(replacement, player)

    @contextlib.contextmanager
    def with_players_replacement(
        self, replacements: Mapping[TPlayer, TPlayer]
    ) -> Iterator[None]:
        self.replace_players(replacements)
        yield
        self.replace_players(
            {replacement: player for player, replacement in replacements.items()}
        )

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

    def others(self, player: TPlayer) -> set[TPlayer]:
        return self._players - {player}

    @property
    def players(self) -> set[TPlayer]:
        return self._players

    @property
    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def observation_for_player(self, player: TPlayer) -> TObservation:
        pass

    @abstractmethod
    def possible_actions_for_player(self, player: TPlayer) -> Iterable[TAction]:
        pass

    @abstractmethod
    def final_score_for_player(self, player: TPlayer) -> float:
        pass

    def step_all_forward(self) -> None:
        while not self.finished:
            self.step_forward()

    @abstractmethod
    def step_forward(self) -> None:
        pass

    @abstractmethod
    def step_backward(self) -> None:
        pass


class Observation:
    pass


class Player[TObservation: Observation, TAction: Action[Any]](DecisionProducer, ABC):
    @abstractmethod
    def act(self, observation: TObservation) -> TAction:
        pass

    def select_integer(
        self,
        low: int | None = None,
        high: int | None = None,
        exclude: set[int] | None = None,
        title: str | None = None,
    ) -> int:
        low = low if low is not None else (-sys.maxsize + 1)
        high = high if high is not None else sys.maxsize
        exclude = exclude or set()
        return self.select_decision(
            decisions=Probabilistic.many_uniform(
                [i for i in range(low, high + 1) if i not in exclude]
            ),
            title=title,
        )


class GenericPlayer[
    TGame: Game[Any, Any, Any],
    TObservation: Observation,
    TAction: Action[Any],
](Player[Any, Any]):
    def __init__(
        self,
        readonly_game: Readonly[TGame] | None = None,
        decision_selector: DecisionSelector | None = None,
    ) -> None:
        super().__init__(decision_selector)
        self._readonly_game = readonly_game

    @override
    def act(self, observation: TObservation) -> TAction:
        final_readonly_game: Readonly[Game[Any, Any, Any]]

        if self._readonly_game is not None:
            final_readonly_game = self._readonly_game
        elif isinstance(observation, Game):
            final_readonly_game = observation
        else:
            raise RuntimeError("Cannot find a game to get possible actions.")

        return (
            cast(  # Due to Python limitation using type variable as a type parameter.
                TAction,
                self.select_decision(
                    Probabilistic.many_uniform(
                        list(final_readonly_game.possible_actions_for_player(self))
                    )
                ),
            )
        )


class Action[TGame]:
    @abstractmethod
    def is_feasible(self, game: Readonly[TGame]) -> bool:
        pass

    @abstractmethod
    def do(self, game: TGame) -> None:
        pass

    @abstractmethod
    def revert(self, game: TGame) -> None:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass
