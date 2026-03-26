from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, cast, override

from gamelink.core.game import (
    Action,
    DecisionSelector,
    Game,
    Observation,
    Player,
    Probabilistic,
)
from gamelink.utils import Timeline


class BruteForceBot[
    TGame: Game[Any, Any, Any],
    TObservation: Observation,
    TAction: Action[Any],
    TPlayer: Player[Any, Any],
](Player[TObservation, TAction]):
    def __init__(
        self,
        possible_games_generator: Callable[
            [TObservation], Sequence[Probabilistic[TGame]],
        ],
        actions_generator: Callable[[TObservation], Sequence[TAction]],
        player_scorer: PlayerScorer[TGame, TPlayer],
        decision_selector: DecisionSelector | None = None,
        score_cutoff: float | None = None,
        **kwargs: Any,  # noqa: ANN401  # For cooperative multiple inheritance
    ) -> None:
        super().__init__(decision_selector=decision_selector, **kwargs)
        self._possible_games_generator = possible_games_generator
        self._actions_generator = actions_generator
        self._player_scorer = player_scorer
        self._score_cutoff = score_cutoff if score_cutoff is not None else math.inf

    @override
    def act(self, observation: TObservation) -> TAction:
        each_action_score: dict[TAction, float] = defaultdict(lambda: 0)
        for probabilistic_game in self._possible_games_generator(observation):
            game, probability = probabilistic_game.event, probabilistic_game.probability
            for action in game.possible_actions_for_player(self):
                # TODO: Use the Game class interfaces for simulating the game.
                action.do(game)
                current_score = self._player_scorer.score(
                    game,
                    cast(
                        "TPlayer", self,
                    ),  # Because of Python typing limitation on CRTP.
                )

                each_action_score[action] += probability * current_score
                action.revert(game)

                if current_score >= self._score_cutoff:
                    break

        best_action = max(
            each_action_score.keys(), key=lambda key: each_action_score[key],
        )
        each_action_probability = {
            action: 1.0 if action is best_action else 0.0
            for action, score in each_action_score.items()
        }

        return self.select_decision(
            decisions=Probabilistic.many_from_mapping(each_action_probability),
        )


class PlayerScorer[TGame: Game[Any, Any, Any], TPlayer: Player[Any, Any]](ABC):
    @abstractmethod
    def score(self, game: TGame, player: TPlayer) -> float:
        pass


class BruteForcePlayerScorer[TPlayer: Player[Any, Any]](
    PlayerScorer[Game[Any, Any, Any], TPlayer],
):
    def __init__(self, players_replacer: Callable[[TPlayer, TPlayer], TPlayer]) -> None:
        super().__init__()
        self._players_replacer = players_replacer

    @override
    def score(self, game: Game[Any, Any, Any], player: TPlayer) -> float:
        if game.finished:
            return game.final_score_for_player(player)

        decision_selector = BacktrackingDecisionSelector()

        scores_sum = 0
        paths_count = 0

        with game.simulate(
            player_replacements={
                replacing_player: self._players_replacer(player, replacing_player)
                for replacing_player in game.players
            },
            decision_selector=decision_selector,
        ):
            while True:
                while not game.finished:
                    decision_selector.checkpoint_current()
                    game.step_forward()

                scores_sum += game.final_score_for_player(player)
                paths_count += 1

                while decision_selector.any_checkpoint():
                    game.step_backward()
                    decision_selector.pop_then_seek_checkpoint()

                    if decision_selector.any_remaining_decision_in_future_selectors():
                        break

                if not decision_selector.any_remaining_decision_in_future_selectors():
                    break

        return scores_sum / paths_count


class BacktrackingDecisionSelector(DecisionSelector):
    def __init__(self) -> None:
        super().__init__()
        self._dummy_root_selector = MovingDecisionSelector()
        self._path_timeline: Timeline[MovingDecisionSelector] = Timeline(
            root=self._dummy_root_selector,
        )

    def any_checkpoint(self) -> bool:
        return self._path_timeline.any_checkpoint()

    def checkpoint_current(self) -> None:
        self._path_timeline.checkpoint_current()

    def pop_then_seek_checkpoint(self) -> None:
        self._path_timeline.seek(self._path_timeline.pop_checkpoint())

    def any_remaining_decision_in_future_selectors(self) -> bool:
        return any(selector.has_next() for selector in self._path_timeline.future)

    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        if not self._path_timeline.future:
            self._path_timeline.append(MovingDecisionSelector())

        self._path_timeline.step_forward()

        if (
            not self.any_remaining_decision_in_future_selectors()
            and self._path_timeline.current.has_next()
        ):
            self._path_timeline.prune_future()
            self._path_timeline.current.to_next()

        return self._path_timeline.current.select_index(decisions, title)


class MovingDecisionSelector(DecisionSelector):
    def __init__(self) -> None:
        super().__init__()
        self._current_selecting_decision_index: int = 0

    def to_next(self) -> None:
        if not self.has_next():
            raise RuntimeError("No remaining decision to move to.")

        self._current_selecting_decision_index += 1

    def has_next(self) -> bool:
        return (
            self._current_selecting_decision_index < self.last_number_of_decisions - 1
        )

    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        return self._current_selecting_decision_index
