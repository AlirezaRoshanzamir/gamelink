from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from collections import defaultdict
from typing import Any, cast, override

from gamelink.core.game import (
    Action,
    DecisionSelector,
    DelegatedDecisionSelector,
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
            [TObservation], Sequence[Probabilistic[TGame]]
        ],
        actions_generator: Callable[[TObservation], Sequence[TAction]],
        player_scorer: PlayerScorer[TGame, TPlayer],
        decision_selector: DecisionSelector | None = None,
        **kwargs: Any,  # noqa: ANN401  # For cooperative multiple inheritance
    ) -> None:
        super().__init__(decision_selector=decision_selector, **kwargs)
        self._possible_games_generator = possible_games_generator
        self._actions_generator = actions_generator
        self._player_scorer = player_scorer

    @override
    def act(self, observation: TObservation) -> TAction:
        each_action_score: dict[TAction, float] = defaultdict(lambda: 0)
        for probabilistic_game in self._possible_games_generator(observation):
            game, probability = probabilistic_game.event, probabilistic_game.probability
            for action in game.possible_actions_for_player(self):
                # TODO: Use the Game class interfaces for simulating the game.
                action.do(game)
                current_evaluation = self._player_scorer.score(
                    game,
                    cast(
                        "TPlayer", self
                    ),  # Because of Python typing limitation on CRTP.
                )
                each_action_score[action] += probability * current_evaluation
                action.revert(game)

        return self.select_decision(
            decisions=Probabilistic.many_from_mapping(each_action_score),
        )


class PlayerScorer[TGame: Game[Any, Any, Any], TPlayer: Player[Any, Any]](ABC):
    @abstractmethod
    def score(self, game: TGame, player: TPlayer) -> float:
        pass


class GameSimulatorPlayerScorer[TPlayer: Player[Any, Any]](
    PlayerScorer[Game[Any, Any, Any], TPlayer]
):
    def __init__(self, other_players_replacer: Callable[[TPlayer], TPlayer]) -> None:
        super().__init__()
        self._other_players_replacer = other_players_replacer

    @override
    def score(self, game: Game[Any, Any, Any], player: TPlayer) -> float:
        root_selector = BacktrackingDecisionSelector()
        root_selector.path_timeline.checkpoint()

        scores_sum = 0
        paths_count = 0

        with (
            game.with_players_replacement(
                {
                    other_player: self._other_players_replacer(other_player)
                    for other_player in game.others(player)
                }
            ),
            game.with_decision_selector(
                DelegatedDecisionSelector(lambda: root_selector.path_timeline.current),
            ),
        ):
            while not game.finished:
                while not game.finished:
                    game.step_forward()
                    root_selector.path_timeline.checkpoint()

                scores_sum += game.final_score_for_player(player)
                paths_count += 1

                while not root_selector.path_timeline.current.any_remaining_decision():
                    game.step_backward()
                    root_selector.path_timeline.seek(
                        root_selector.path_timeline.pop_checkpoint(),
                    )

        return scores_sum / paths_count


class BacktrackingDecisionSelector(DecisionSelector):
    def __init__(
        self,
        path_timeline: Timeline[BacktrackingDecisionSelector] | None = None,
    ) -> None:
        super().__init__()
        self._path_timeline = path_timeline or Timeline(root=self)

    @property
    def path_timeline(self) -> Timeline[BacktrackingDecisionSelector]:
        return self._path_timeline

    def any_remaining_decision(self) -> bool:
        return (
            self.any_remaining_decision_in_this_selector()
            or self.any_remaining_decision_in_future_selectors()
        )

    def any_remaining_decision_in_this_selector(self) -> bool:
        return self.last_selected_decision_index <= self.last_number_of_decisions - 1

    def any_remaining_decision_in_future_selectors(self) -> bool:
        return any(
            selector.any_remaining_decision_in_this_selector()
            for selector in self._path_timeline.future
        )

    @override
    def select_index_hook[TDecision](
        self,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        if self.any_remaining_decision_in_future_selectors():
            selecting_decision_index = self.last_selected_decision_index
        elif self.any_remaining_decision_in_this_selector():
            selecting_decision_index = self.last_selected_decision_index + 1
            self._path_timeline.prune_future()
        else:
            msg = "There is no decision left to select."
            raise RuntimeError(msg)

        if selecting_decision_index == 0:
            self._path_timeline.append(
                BacktrackingDecisionSelector(path_timeline=self._path_timeline),
            )
        self._path_timeline.step_forward()

        return selecting_decision_index
