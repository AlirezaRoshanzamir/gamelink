from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from enum import StrEnum
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


class XOGame(Game[Readonly["XOGame"], "Select", "XOPlayer"], State):
    BOARD_SIZE = 3

    def __init__(self) -> None:
        super().__init__()
        self._history: list[Select] = []
        self._role_to_player: dict[XOPlayerRole, XOPlayer] = {}
        self._board: list[list[XOPlayerRole | None]] = [
            [None for _ in range(XOGame.BOARD_SIZE)] for _ in range(XOGame.BOARD_SIZE)
        ]
        self._turn = XOPlayerRole.X

    @property
    def board(self) -> list[list[XOPlayerRole | None]]:
        return self._board

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
        return all(cell is not None for row in self._board for cell in row)

    @override
    def possible_actions_for_player(
        self, player: XOPlayer, state: Readonly[XOGame]
    ) -> Sequence[Select]:
        return [
            Select(row, col, player.role)
            for row in range(XOGame.BOARD_SIZE)
            for col in range(XOGame.BOARD_SIZE)
            if self.board[row][col] is None
        ]

    @override
    def final_score_for_player(self, player: XOPlayer) -> float:
        winner = self.winner()
        if winner is None:
            return 0
        if winner == player.role:
            return 1.0
        return -1.0

    def winner(self) -> XOPlayerRole | None:
        lines: list[list[XOPlayerRole | None]] = []

        # Row-wise
        lines.extend(self._board)

        # Column-wise
        lines.extend(
            [
                [self._board[r][c] for r in range(XOGame.BOARD_SIZE)]
                for c in range(XOGame.BOARD_SIZE)
            ],
        )

        # Main diagonal
        lines.append([self._board[i][i] for i in range(XOGame.BOARD_SIZE)])

        # Second diagonal
        lines.append(
            [
                self._board[i][XOGame.BOARD_SIZE - i - 1]
                for i in range(XOGame.BOARD_SIZE)
            ],
        )

        for line in lines:
            if line[0] is not None and all(cell == line[0] for cell in line):
                return line[0]
        return None

    def number_of_filled_cells(self) -> int:
        return sum(cell is not None for row in self._board for cell in row)

    def __str__(self) -> str:
        rows = [
            " | ".join(cell.value if cell else "." for cell in row)
            for row in self._board
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
                "Replacement player must have the same role as the original player.",
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
        self.log(
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
            self.log(
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
        return game.board[self._row][self._col] is None

    @override
    def do(self, game: XOGame) -> None:
        game.board[self._row][self._col] = self._role
        game.turn = game.turn.opposite()

    @override
    def revert(self, game: XOGame) -> None:
        game.board[self._row][self._col] = None
        game.turn = game.turn.opposite()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._row}, {self._col}, {self._role.value})"
        )

    @override
    def __hash__(self) -> int:
        return 2 * (self._row * XOGame.BOARD_SIZE + self._col) + (
            0 if self._role == XOPlayerRole.X else 1
        )


class XOBruteForceBot(
    OptimalPlayer[XOGame, Readonly[XOGame], Select, XOPlayer],
    XOPlayer,
):
    def __init__(
        self,
        role: XOPlayerRole,
        decision_selector: DecisionSelector | None = None,
    ) -> None:
        super().__init__(
            role=role,
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
                            XOPlayer(
                                replacing_player.role,
                            )
                        ),
                    )
                ),
                possible_games_generator=lambda readonly_game: [
                    Probabilistic.deterministic(readonly_game),
                ],
            ),
        )


if __name__ == "__main__":
    game = XOGame()
    game.join_player(XOBruteForceBot(XOPlayerRole.X))
    game.join_player(XOPlayer(XOPlayerRole.O, decision_selector=CliDecisionSelector()))
    game.step_all_forward()
