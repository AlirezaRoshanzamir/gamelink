from __future__ import annotations

import logging
import sys
from enum import StrEnum
from typing import Any, override
from collections.abc import Sequence, Callable, Iterable
from gamelink.core.game import (
    Action,
    DecisionSelector,
    Game,
    Player,
    Readonly,
    Observation,
    Probabilistic,
    SamplingDecisionSelector,
    CliDecisionSelector,
    GenericPlayer,
)
from gamelink.core.bot import (
    BruteForceBot,
    GameSimulatorPlayerScorer,
    PlayerScorer,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class XOGame(Game[Readonly["XOGame"], "Select", "XOPlayer"], Observation):
    ROWS_COUNT = 3
    COLS_COUNT = 3

    def __init__(self) -> None:
        super().__init__()
        self._history: list[Select] = []
        self._role_to_player: dict[XOPlayerRole, XOPlayer] = {}
        self._table: list[list[XOPlayerRole | None]] = [
            [None for _ in range(XOGame.COLS_COUNT)] for _ in range(XOGame.ROWS_COUNT)
        ]
        self._turn = XOPlayerRole.X

    @property
    def table(self) -> list[list[XOPlayerRole | None]]:
        return self._table

    @property
    def turn(self) -> XOPlayerRole:
        return self._turn

    @turn.setter
    def turn(self, value: XOPlayerRole) -> None:
        self._turn = value

    @override
    @property
    def finished(self) -> bool:
        if self.winner() is not None:
            return True
        return all(cell is not None for row in self._table for cell in row)

    @override
    def observation_for_player(self, player: XOPlayer) -> Readonly[XOGame]:
        return self

    @override
    def possible_actions_for_player(self, player: XOPlayer) -> Sequence[Select]:
        return [
            Select(row, col, player.role)
            for row in range(XOGame.ROWS_COUNT)
            for col in range(XOGame.COLS_COUNT)
            if self.table[row][col] is None
        ]

    @override
    def final_score_for_player(self, player: XOPlayer) -> float:
        winner = self.winner()
        if winner is None:
            return 0
        elif winner == player.role:
            return 1.0
        else:
            return -1.0

    def winner(self) -> XOPlayerRole | None:
        lines: list[list[XOPlayerRole | None]] = []
        lines.extend(self._table)
        lines.extend([[self._table[r][c] for r in range(3)] for c in range(3)])
        lines.append([self._table[i][i] for i in range(3)])
        lines.append([self._table[i][2 - i] for i in range(3)])

        for line in lines:
            a, b, c = line
            if a is not None and a == b == c:
                return a
        return None

    def __str__(self) -> str:
        rows = [
            " | ".join(cell.value if cell else "." for cell in row)
            for row in self._table
        ]
        return "\n---------\n".join(rows)

    @override
    def join_player(self, player: XOPlayer) -> None:
        if player.role in self._role_to_player:
            raise ValueError(f"A player with role {player.role} has already joined.")
        self._role_to_player[player.role] = player
        super().join_player(player)

    @override
    def replace_player(self, player: XOPlayer, replacement: XOPlayer) -> None:
        if replacement.role != player.role:
            raise ValueError(
                "Replacement player must have the same role as the original player."
            )
        self._role_to_player[player.role] = replacement
        return super().replace_player(player, replacement)

    @override
    def step_forward(self) -> None:
        player = self._role_to_player[self._turn]
        action = player.act(self)
        action.do(self)
        self._history.append(action)
        winner = self.winner()
        logger.info(
            "Action: %s, Winner: %s, Game Board:\n%s",
            action,
            winner.value if winner else "none",
            self,
        )

    @override
    def step_backward(self) -> None:
        if self._history:
            action = self._history.pop()
            action.revert(self)
            logger.info(
                "Action reverted: %s, Game Board:\n%s",
                action,
                self,
            )


class XOPlayerRole(StrEnum):
    X = "X"
    O = "O"

    def opposite(self) -> XOPlayerRole:
        return XOPlayerRole.X if self == XOPlayerRole.O else XOPlayerRole.O


class XOPlayer(GenericPlayer[XOGame, Readonly[XOGame], "Select"]):
    def __init__(
        self,
        role: XOPlayerRole,
        decision_selector: DecisionSelector | None = None,
        **kwargs: Any,  # noqa: ANN401  # For cooperative multiple inheritance
    ) -> None:
        super().__init__(decision_selector=decision_selector, **kwargs)
        self._role = role

    @property
    def role(self) -> XOPlayerRole:
        return self._role


class Select(Action[XOGame]):
    def __init__(
        self,
        row: int,
        col: int,
        role: XOPlayerRole,
    ) -> None:
        self._row = row
        self._col = col
        self._role = role

    @property
    def row(self) -> int:
        return self._row

    @property
    def col(self) -> int:
        return self._col

    @override
    def is_feasible(self, game: Readonly[XOGame]) -> bool:
        return game.table[self._row][self._col] is None

    @override
    def do(self, game: XOGame) -> None:
        game.table[self._row][self._col] = self._role
        game.turn = game.turn.opposite()

    @override
    def revert(self, game: XOGame) -> None:
        game.table[self._row][self._col] = None
        game.turn = game.turn.opposite()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._row}, {self._col}, {self._role.value})"
        )

    @override
    def __hash__(self) -> int:
        return 2 * (self._row * XOGame.ROWS_COUNT + self._col) + (
            0 if self._role == XOPlayerRole.X else 1
        )


class XOBruteForceBot(
    BruteForceBot[XOGame, Readonly[XOGame], Select, XOPlayer], XOPlayer
):
    def __init__(
        self,
        role: XOPlayerRole,
        decision_selector: DecisionSelector | None = None,
    ) -> None:
        super().__init__(
            role=role,
            decision_selector=decision_selector,
            possible_games_generator=lambda readonly_game: [
                Probabilistic.deterministic(readonly_game)
            ],
            actions_generator=lambda readonly_game: (
                readonly_game.possible_actions_for_player(self)
            ),
            player_scorer=GameSimulatorPlayerScorer(
                other_players_replacer=lambda other_player: XOPlayer(other_player.role),
            ),
        )


if __name__ == "__main__":
    game = XOGame()
    game.join_player(XOBruteForceBot(XOPlayerRole.X))
    game.join_player(XOPlayer(XOPlayerRole.O, decision_selector=CliDecisionSelector()))
    game.step_forward()
    # logger.info("Winner: %s", game.winner())
