from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, override

from gamelink.core.game import (
    Action,
    DecisionSelector,
    DelegatedDecisionSelector,
    Game,
    Player,
    State,
)
from gamelink.utils import Timeline


class BruteForceBot[TState: State](Player[TState]):
    def __init__(
        self,
        state_to_actions: Callable[[TState], Sequence[Action]],
        state_evaluator: PlayerStateEvaluator[Player[TState]],
    ) -> None:
        super().__init__()
        self._state_to_actions = state_to_actions
        self._state_evaluator = state_evaluator

    @override
    def act(self, state: TState) -> Action:
        possible_actions = self._state_to_actions(state)

        best_action_index = -1
        best_evaluation = 0.0
        for i, action in enumerate(possible_actions):
            action.do()
            current_evaluation = self._state_evaluator.evaluate(self)
            if best_action_index == -1 or current_evaluation > best_evaluation:
                best_action_index = i
                best_evaluation = current_evaluation
            action.revert()

        return self.select_decision(
            decisions=possible_actions,
            weights=[
                1.0 if i == best_action_index else 0.0
                for i in range(len(possible_actions))
            ],
        )


class PlayerStateEvaluator[TPlayer: Player[Any]](ABC):
    @abstractmethod
    def evaluate(self, player: TPlayer) -> float:
        pass


class GameSimulatorPlayerStateEvaluator(PlayerStateEvaluator[Player[Any]]):
    def __init__(self, game: Game[Any, Any]) -> None:
        self._game = game

    @override
    def evaluate(self, player: Player[Any]) -> float:
        root_selector = BacktrackingDecisionSelector()
        root_selector.path_timeline.checkpoint()

        scores_sum = 0
        paths_count = 0

        with self._game.with_decision_selector(
            DelegatedDecisionSelector(lambda: root_selector.path_timeline.current),
        ):
            while not self._game.finished:
                while not self._game.finished:
                    self._game.step_forward()
                    root_selector.path_timeline.checkpoint()

                scores_sum += player.score()
                paths_count += 1

                while not root_selector.path_timeline.current.any_remaining_decision():
                    self._game.step_backward()
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
        decisions: Sequence[TDecision],
        weights: Sequence[float],
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
