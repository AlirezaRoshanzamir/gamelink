from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import Any, override

from gamelink.core.game import (
    Action,
    CliDecisionSelector,
    DecisionSelector,
    Game,
    GenericPlayer,
    Probabilistic,
    Readonly,
    State,
)
from gamelink.core.learning import (
    BruteForceFOVFunction,
    FOQFunctionAsQFunction,
    FOVFunctionAsFOQFunction,
    MinimaxBacktrackingDecisionSelectorNodeFactory,
    OptimalPlayer,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class PigGame(Game[Readonly["PigGame"], "PigAction", "PigPlayer"], State):
    TARGET_SCORE = 10
    DICE_SIDES = 3

    def __init__(self) -> None:
        super().__init__()
        self._players_list: list[PigPlayer] = []
        self._total_scores: dict[PigPlayer, int] = {}
        self._current_player_index: int = 0
        self._turn_score: int = 0
        self._history: list[PigAction] = []

    @property
    def current_player(self) -> PigPlayer:
        return self._players_list[self._current_player_index]

    @property
    def turn_score(self) -> int:
        return self._turn_score

    @turn_score.setter
    def turn_score(self, value: int) -> None:
        self._turn_score = value

    @property
    def total_scores(self) -> dict[PigPlayer, int]:
        return dict(self._total_scores)

    def add_to_player_score(self, player: PigPlayer, amount: int) -> None:
        self._total_scores[player] += amount

    def advance_turn(self) -> None:
        self._current_player_index = (self._current_player_index + 1) % len(
            self._players_list,
        )

    def retreat_turn(self) -> None:
        self._current_player_index = (self._current_player_index - 1) % len(
            self._players_list,
        )

    @override
    @property
    def finished(self) -> bool:
        return any(score >= self.TARGET_SCORE for score in self._total_scores.values())

    @override
    def possible_actions_for_player(
        self, player: PigPlayer, state: Readonly[PigGame]
    ) -> Sequence[PigAction]:
        if player is not self.current_player:
            return []
        return [Roll(), Hold()]

    @override
    def final_score_for_player(self, player: PigPlayer) -> float:
        winner = self.winner()
        if winner is player:
            return 1.0
        if winner is not None:
            return -1.0
        return 0.0

    def winner(self) -> PigPlayer | None:
        for player in self._players_list:
            if self._total_scores.get(player, 0) >= self.TARGET_SCORE:
                return player
        return None

    @override
    def join_player(self, player: PigPlayer) -> None:
        self._players_list.append(player)
        self._total_scores[player] = 0
        super().join_player(player)

    @override
    def replace_player(self, player: PigPlayer, replacement: PigPlayer) -> None:
        self._players_list[self._players_list.index(player)] = replacement
        self._total_scores[replacement] = self._total_scores.pop(player, 0)
        return super().replace_player(player, replacement)

    @override
    def step_forward(self) -> None:
        player = self.current_player
        action = player.act(self)
        action.do(self)
        self._history.append(action)
        self.log(
            "Player %s: %s | Turn score: %d | Totals: %s",
            player.name,
            action,
            self._turn_score,
            {p.name: s for p, s in self._total_scores.items()},
        )

    @override
    def step_backward(self) -> None:
        if self._history:
            action = self._history.pop()
            action.revert(self)

    def __str__(self) -> str:
        lines = []
        for player in self._players_list:
            score = self._total_scores.get(player, 0)
            marker = " <-- current" if player is self.current_player else ""
            lines.append(f"  {player.name}: {score}{marker}")
        lines.append(f"  Turn score: {self._turn_score}")
        return "\n".join(lines)


class PigPlayer(GenericPlayer["PigGame", Readonly["PigGame"], "PigAction"]):
    def __init__(
        self,
        name: str,
        decision_selector: DecisionSelector | None = None,
        **kwargs: Any,  # noqa: ANN401  # For cooperative multiple inheritance
    ) -> None:
        super().__init__(decision_selector=decision_selector, **kwargs)
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class PigAction(Action["PigGame"]):
    pass


class Roll(PigAction):
    """Roll the die.

    A roll of 1 (pig) loses the turn score and passes the turn.
    Any other value is added to the turn score.
    """

    def __init__(self) -> None:
        self._rolled: int | None = None
        self._prev_turn_score: int = 0

    @override
    def is_feasible(self, game: Readonly[PigGame]) -> bool:
        return True

    @override
    def do(self, game: PigGame) -> None:
        self._prev_turn_score = game.turn_score
        self._rolled = game.select_decision(
            Probabilistic.many_uniform(list(range(1, game.DICE_SIDES))),
            title="die",
        )
        if self._rolled == 1:
            game.turn_score = 0
            game.advance_turn()
        else:
            game.turn_score += self._rolled

    @override
    def revert(self, game: PigGame) -> None:
        if self._rolled == 1:
            game.retreat_turn()
        game.turn_score = self._prev_turn_score
        self._rolled = None

    @override
    def __hash__(self) -> int:
        return hash("Roll")

    def __str__(self) -> str:
        return f"Roll({self._rolled})" if self._rolled is not None else "Roll"


class Hold(PigAction):
    """Bank the current turn score into the total and pass the turn."""

    def __init__(self) -> None:
        self._held_score: int = 0

    @override
    def is_feasible(self, game: Readonly[PigGame]) -> bool:
        return True

    @override
    def do(self, game: PigGame) -> None:
        self._held_score = game.turn_score
        game.add_to_player_score(game.current_player, self._held_score)
        game.turn_score = 0
        game.advance_turn()

    @override
    def revert(self, game: PigGame) -> None:
        game.retreat_turn()
        game.add_to_player_score(game.current_player, -self._held_score)
        game.turn_score = self._held_score
        self._held_score = 0

    @override
    def __hash__(self) -> int:
        return hash("Hold")

    def __str__(self) -> str:
        return f"Hold({self._held_score})"


class PigBruteForceBot(
    OptimalPlayer[PigGame, Readonly[PigGame], PigAction, PigPlayer],
    PigPlayer,
):
    def __init__(
        self,
        name: str,
        decision_selector: DecisionSelector | None = None,
    ) -> None:
        super().__init__(
            name=name,
            decision_selector=decision_selector,
            possible_actions_generator=lambda player, readonly_game: (
                readonly_game.possible_actions_for_player(
                    player,
                    readonly_game,
                )
            ),
            q_function=FOQFunctionAsQFunction(
                foq_function=FOVFunctionAsFOQFunction(
                    fov_function=BruteForceFOVFunction(
                        node_factory=MinimaxBacktrackingDecisionSelectorNodeFactory(),
                        players_replacer=lambda current_player, replacing_player: (
                            PigPlayer(name=name)
                        ),
                        max_depth=10,
                    )
                ),
                possible_games_generator=lambda readonly_game: [
                    Probabilistic.deterministic(readonly_game),
                ],
            ),
        )


if __name__ == "__main__":
    game = PigGame()
    game.join_player(PigBruteForceBot("Bob"))
    game.join_player(PigPlayer("Alice", decision_selector=CliDecisionSelector()))
    game.step_all_forward()
    winner = game.winner()
    print(f"\nGame over! Winner: {winner.name if winner else 'nobody'}")  # noqa: T201
    print(game)  # noqa: T201
