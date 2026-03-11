from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import override
from dataclasses import dataclass, field

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


@dataclass
class XOState(State):
    ROWS_COUNT = 3
    COLS_COUNT = 3

    turn: XOPlayerRole = XOPlayerRole.X
    table: list[list[XOPlayerRole | None]] = field(
        default_factory=lambda: [
            [None for _ in range(XOState.COLS_COUNT)] for _ in range(XOState.ROWS_COUNT)
        ]
    )

    def __post_init__(self) -> None:
        if len(self.table) != self.ROWS_COUNT or any(
            len(row) != self.COLS_COUNT for row in self.table
        ):
            raise ValueError(f"table must be {self.ROWS_COUNT}x{self.COLS_COUNT}")

    def winner(self) -> XOPlayerRole | None:
        lines: list[list[XOPlayerRole | None]] = []
        lines.extend(self.table)
        lines.extend([[self.table[r][c] for r in range(3)] for c in range(3)])
        lines.append([self.table[i][i] for i in range(3)])
        lines.append([self.table[i][2 - i] for i in range(3)])

        for line in lines:
            a, b, c = line
            if a is not None and a == b == c:
                return a
        return None

    @property
    def finished(self) -> bool:
        if self.winner() is not None:
            return True
        return all(cell is not None for row in self.table for cell in row)

    def __str__(self) -> str:
        rows = [
            " | ".join(cell.value if cell else "." for cell in row)
            for row in self.table
        ]
        return "\n---------\n".join(rows)

    @override
    def __hash__(self) -> int:
        return hash(
            tuple(tuple(cell for cell in row) for row in self.table)
        ) * 31 + hash(self.turn)


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
    @property
    def finished(self) -> bool:
        return self._state.finished

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
        player = self._role_to_player[self._state.turn]
        action = player.act(self._state)
        action.do()
        self._history.append(action)
        self._state.turn = self._state.turn.opposite()
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
            self._state.turn = self._state.turn.opposite()
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
        return self._state.table[self._row][self._col] is None

    @override
    def do(self) -> None:
        self._state.table[self._row][self._col] = self._role

    @override
    def revert(self) -> None:
        self._state.table[self._row][self._col] = None


class BaseXOPlayer(XOPlayer):
    @override
    def act(self, state: XOState) -> Action:
        empty = [
            (r, c) for r in range(3) for c in range(3) if state.table[r][c] is None
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
            (r, c) for r in range(3) for c in range(3) if state.table[r][c] is None
        ]
        return [Select(state, row, col, self.role) for row, col in empty]

    @override
    def act(self, state: XOState) -> Action:
        self._sim_game._state = state
        flat = [col for row in state.table for col in row]
        x_count = flat.count(XOPlayerRole.X)
        o_count = flat.count(XOPlayerRole.Y)
        self._sim_game._state.turn = (
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
