from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import override

from gamelink.core.game import Action, DecisionSelector, Game, Player, State
from gamelink.core.minimax import (
    BruteForceBot,
    GameSimulatorPlayerStateEvaluator,
    PlayerStateEvaluator,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class XOPlayerRole(StrEnum):
    X = "X"
    Y = "Y"

    def opposite(self) -> XOPlayerRole:
        return XOPlayerRole.X if self == XOPlayerRole.Y else XOPlayerRole.Y


class XOState(State):
    def __init__(
        self,
        table: list[list[XOPlayerRole | str | None]] | None = None,
        turn: XOPlayerRole = XOPlayerRole.X,
    ) -> None:
        rows_count = 3
        cols_count = 3

        self._table: list[list[XOPlayerRole | None]]
        if table is None:
            self._table = [[None for _ in range(cols_count)] for _ in range(rows_count)]
        else:
            if len(table) != rows_count or any(len(row) != cols_count for row in table):
                msg = "table must be 3x3"
                raise ValueError(msg)
            self._table = [
                [self._normalize_cell(cell) for cell in row] for row in table
            ]
        self._turn = turn

    @property
    def table(self) -> list[list[XOPlayerRole | None]]:
        return self._table

    @property
    def turn(self) -> XOPlayerRole:
        return self._turn

    def _normalize_cell(self, cell: XOPlayerRole | str | None) -> XOPlayerRole | None:
        if cell is None:
            return None
        if isinstance(cell, XOPlayerRole):
            return cell
        if cell in (XOPlayerRole.X.value, XOPlayerRole.Y.value):
            return XOPlayerRole(cell)
        msg = f"Invalid cell value: {cell!r}"
        raise ValueError(msg)

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

    @property
    @override
    def finished(self) -> bool:
        if self.winner() is not None:
            return True
        return all(cell is not None for row in self._table for cell in row)

    def __str__(self) -> str:
        rows = [
            " | ".join(cell.value if cell else "." for cell in row)
            for row in self._table
        ]
        return "\n---------\n".join(rows)

    @override
    def __hash__(self) -> int:
        return hash(
            tuple(tuple(cell for cell in row) for row in self._table)
        ) * 31 + hash(self._turn)


class XOPlayer(Player[XOState]):
    def __init__(
        self,
        role: XOPlayerRole,
        decision_selector: DecisionSelector | None = None,
    ) -> None:
        super().__init__(decision_selector)
        self._role = role

    @property
    def role(self) -> XOPlayerRole:
        return self._role

    @override
    def score(self) -> float:
        return 0.0


class XOGame(Game[XOState, XOPlayer]):
    def __init__(self) -> None:
        super().__init__()
        self._history: list[Action] = []
        self._role_to_player: dict[XOPlayerRole, XOPlayer] = {}

    @override
    @classmethod
    def create_initial_state(cls) -> XOState:
        return XOState()

    @override
    def join_player(self, player: XOPlayer) -> None:
        if player.role in self._role_to_player:
            raise ValueError(f"A player with role {player.role} has already joined.")
        self._role_to_player[player.role] = player
        super().join_player(player)

    @override
    def step_forward(self) -> None:
        player = self._role_to_player[self._state._turn]
        action = player.act(self._state)
        action.do()
        self._history.append(action)
        self._state._turn = self._state._turn.opposite()
        assert isinstance(action, Select)
        winner = self._state.winner()
        logger.info(
            "Player %s played (%d, %d)\n%s\nWinner: %s",
            action._role.value,
            action._row,
            action._col,
            self._state,
            winner.value if winner else "none",
        )

    @override
    def step_backward(self) -> None:
        if self._history:
            action = self._history.pop()
            action.revert()
            self._state._turn = self._state._turn.opposite()
            assert isinstance(action, Select)
            logger.info(
                "Reverted Player %s's move at (%d, %d)\n%s",
                action._role.value,
                action._row,
                action._col,
                self._state,
            )


class Select(Action):
    def __init__(
        self,
        state: XOState,
        row: int,
        col: int,
        role: XOPlayerRole,
    ) -> None:
        self._state = state
        self._row = row
        self._col = col
        self._role = role

    @override
    def is_feasible(self) -> bool:
        return self._state._table[self._row][self._col] is None

    @override
    def do(self) -> None:
        self._state._table[self._row][self._col] = self._role

    @override
    def revert(self) -> None:
        self._state._table[self._row][self._col] = None


class BaseXOPlayer(XOPlayer):
    @override
    def act(self, state: XOState) -> Action:
        empty = [
            (r, c) for r in range(3) for c in range(3) if state._table[r][c] is None
        ]
        decisions = [Select(state, r, c, self._role) for r, c in empty]
        return self.select_decision(decisions, [1.0] * len(decisions))


class CliXOPlayer(XOPlayer):
    @override
    def act(self, state: XOState) -> Action:
        minimum_cell_number = 1
        maximum_cell_number = 9
        while True:
            try:
                n = int(
                    input(
                        f"Player {self._role.value} - "
                        f"enter a number ({minimum_cell_number}-"
                        f"{maximum_cell_number}): ",
                    ),
                )
            except ValueError:
                continue
            if not minimum_cell_number <= n <= maximum_cell_number:
                continue
            row, col = divmod(n - 1, 3)
            action = Select(state, row, col, self._role)
            if action.is_feasible():
                break
        return action


class _XOBruteForceInternalBot(BruteForceBot[XOState]):
    def __init__(
        self,
        state_to_actions: Callable[[XOState], Sequence[Action]],
        state_evaluator: PlayerStateEvaluator[Player[XOState]],
        role: XOPlayerRole,
        game_provider: Callable[[], XOGame],
    ) -> None:
        super().__init__(state_to_actions, state_evaluator)
        self._role = role
        self._game_provider = game_provider

    @override
    def score(self) -> float:
        game = self._game_provider()
        winner = game.state.winner()
        if winner == self._role:
            return 1.0
        if winner is None:
            return 0.0
        return -1.0


class XOBruteForceBot(XOPlayer):
    def __init__(self, role: XOPlayerRole) -> None:
        super().__init__(role)
        self._sim_game = XOGame()
        self._sim_game.join_player(BaseXOPlayer(XOPlayerRole.X))
        self._sim_game.join_player(BaseXOPlayer(XOPlayerRole.Y))

        self._internal_bot = _XOBruteForceInternalBot(
            self._state_to_actions,
            GameSimulatorPlayerStateEvaluator(self._sim_game),
            role,
            lambda: self._sim_game,
        )

    def _state_to_actions(self, state: State) -> Sequence[Action]:
        assert isinstance(state, XOState)
        empty = [
            (r, c) for r in range(3) for c in range(3) if state._table[r][c] is None
        ]
        return [Select(state, r, c, self.role) for r, c in empty]

    @override
    def act(self, state: XOState) -> Action:
        self._sim_game._state = state
        flat = [c for r in state.table for c in r]
        x_count = flat.count(XOPlayerRole.X)
        o_count = flat.count(XOPlayerRole.Y)
        self._sim_game.state._turn = (
            XOPlayerRole.X if (x_count + o_count) % 2 else XOPlayerRole.Y
        )

        return self._internal_bot.act(state)


if __name__ == "__main__":
    game = XOGame()
    game.join_player(XOBruteForceBot(XOPlayerRole.X))
    game.join_player(BaseXOPlayer(XOPlayerRole.Y))
    game.step_all_forward()

    state = game.state
    assert isinstance(state, XOState)
    winner = state.winner()
