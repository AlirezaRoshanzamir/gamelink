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


class XOGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self._state = XOGameState()
        self._turn = 0
        self._history: list[Action] = []
        self._role_to_player: dict[XOPlayerRole, BaseXOPlayer] = {}

    @override
    def join_player(self, player: Player) -> None:
        if not isinstance(player, BaseXOPlayer):
            msg = f"player must be a BaseXOPlayer, got {type(player).__name__}"
            raise TypeError(
                msg,
            )
        if player.role in self._role_to_player:
            msg = f"A player with role {player.role} has already joined"
            raise ValueError(msg)
        self._role_to_player[player.role] = player
        super().join_player(player)

    @property
    @override
    def players(self) -> set[Player]:
        return self._players

    @property
    @override
    def finished(self) -> bool:
        return self._state.finished

    @property
    @override
    def state(self) -> State:
        return self._state

    @override
    def step_forward(self) -> None:
        role = XOPlayerRole.X if self._turn % 2 == 0 else XOPlayerRole.Y
        player = self._role_to_player[role]
        action = player.act(self._state)
        action.do()
        self._history.append(action)
        self._turn += 1
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
            self._turn -= 1
            assert isinstance(action, Select)
            logger.info(
                "Reverted Player %s's move at (%d, %d)\n%s",
                action._role.value,
                action._row,
                action._col,
                self._state,
            )


class XOGameState(State):
    def __init__(
        self,
        table: list[list[XOPlayerRole | str | None]] | None = None,
    ) -> None:
        if table is None:
            self._table = [[None for _ in range(3)] for _ in range(3)]
        else:
            if len(table) != 3 or any(len(row) != 3 for row in table):
                msg = "table must be 3x3"
                raise ValueError(msg)
            self._table = [
                [self._normalize_cell(cell) for cell in row] for row in table
            ]

    def _normalize_cell(self, cell: XOPlayerRole | str | None) -> XOPlayerRole | None:
        if cell is None:
            return None
        if isinstance(cell, XOPlayerRole):
            return cell
        if cell in (XOPlayerRole.X.value, XOPlayerRole.Y.value):
            return XOPlayerRole(cell)
        msg = f"Invalid cell value: {cell!r}"
        raise ValueError(msg)

    @property
    def table(self) -> list[list[XOPlayerRole | None]]:
        return [list(row) for row in self._table]

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
        acc = 0
        for row in self._table:
            for cell in row:
                acc = acc * 3 + (
                    0 if cell is None else (1 if cell == XOPlayerRole.X else 2)
                )
        return acc


class Select(Action):
    def __init__(
        self,
        state: XOGameState,
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


class XOPlayerRole(StrEnum):
    X = "X"
    Y = "Y"


class BaseXOPlayer(Player):
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


class XOPlayer(BaseXOPlayer):
    @override
    def act(self, state: State) -> Action:
        assert isinstance(state, XOGameState)
        empty = [
            (r, c) for r in range(3) for c in range(3) if state._table[r][c] is None
        ]
        decisions = [Select(state, r, c, self._role) for r, c in empty]
        return self.select_decision(decisions, [1.0] * len(decisions))


class CliXOPlayer(BaseXOPlayer):
    @override
    def act(self, state: State) -> Action:
        assert isinstance(state, XOGameState)
        while True:
            try:
                n = int(input(f"Player {self._role.value} — enter a number (1-9): "))
                if not 1 <= n <= 9:
                    raise ValueError
                row, col = divmod(n - 1, 3)
                action = Select(state, row, col, self._role)
                if action.is_feasible():
                    return action
            except ValueError:
                pass


class _XOBruteForceInternalBot(BruteForceBot):
    def __init__(
        self,
        state_to_actions: Callable[[State], Sequence[Action]],
        state_evaluator: PlayerStateEvaluator,
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


class XOBruteForceBot(BaseXOPlayer):
    def __init__(self, role: XOPlayerRole) -> None:
        super().__init__(role)
        self._sim_game = XOGame()
        self._sim_game.join_player(XOPlayer(XOPlayerRole.X))
        self._sim_game.join_player(XOPlayer(XOPlayerRole.Y))

        self._internal_bot = _XOBruteForceInternalBot(
            self._state_to_actions,
            GameSimulatorPlayerStateEvaluator(self._sim_game),
            role,
            lambda: self._sim_game,
        )

    def _state_to_actions(self, state: State) -> Sequence[Action]:
        assert isinstance(state, XOGameState)
        empty = [
            (r, c) for r in range(3) for c in range(3) if state._table[r][c] is None
        ]
        return [Select(state, r, c, self.role) for r, c in empty]

    @override
    def act(self, state: State) -> Action:
        self._sim_game._state = state
        flat = [c for r in state.table for c in r]
        x_count = flat.count(XOPlayerRole.X)
        o_count = flat.count(XOPlayerRole.Y)
        self._sim_game._turn = x_count + o_count

        return self._internal_bot.act(state)


if __name__ == "__main__":
    game = XOGame()
    game.join_player(XOBruteForceBot(XOPlayerRole.X))
    game.join_player(XOPlayer(XOPlayerRole.Y))
    game.step_all_forward()

    state = game.state
    assert isinstance(state, XOGameState)
    winner = state.winner()
