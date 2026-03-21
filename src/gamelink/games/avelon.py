# type: ignore

from __future__ import annotations

import contextlib
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import override

from gamelink.core import sat


class Role(Enum):
    MERLIN_LOYAL = "merlin"
    SIMPLE_LOYAL = "loyal"
    TRAITOR = "traitor"

    @property
    def loyal(self) -> bool:
        return self in (Role.MERLIN_LOYAL, Role.SIMPLE_LOYAL)


class QuestVote(Enum):
    VICTORY = "victory"
    DEFEAT = "defeat"


class SelectionVote(Enum):
    YES = "yes"
    NO = "no"

    def __bool__(self) -> bool:
        return self == SelectionVote.YES


class Player(ABC):
    def __init__(self, name: str) -> None:
        self._name = name
        self._private_knowledge_base = ...
        self._public_knowledge_base = ...

    @property
    def public_knowledge_base(self) -> KnowledgeBase:
        return self._public_knowledge_base

    @property
    @abstractmethod
    def role(self) -> Role:
        pass

    def as_terminal(self) -> sat.Terminal:
        return sat.Terminal(symbol=self._name)

    @abstractmethod
    def offer_selection(self) -> set[Player]:
        pass

    @abstractmethod
    def vote_in_quest(
        self,
        leader: Player,
        other_players: set[Player],
    ) -> Generator[QuestVote, set[QuestVote], None]:
        pass

    @abstractmethod
    def vote_in_selection(
        self,
        leader: Player,
        selection: set[Player],
    ) -> Generator[SelectionVote, Mapping[SelectionVote, int], None]:
        pass


class TraitorPlayer(Player):
    pass


class BaseLoyalPlayer(Player):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._private_knowledge_base = KnowledgeBase(
            traitor_expression=sat.Not(operand=self.as_terminal()),
            final_traitors={},
        )
        self._public_knowledge_base = KnowledgeBase(
            traitor_expression=sat.Not(operand=self.as_terminal()),
            final_traitors={},
        )

    @override
    def vote_in_quest(
        self,
        leader: Player,
        other_players: set[Player],
    ) -> Generator[QuestVote, Mapping[QuestVote, int], None]:
        yield QuestVote.VICTORY
        # TODO: Should be completed

    @override
    def vote_in_selection(
        self,
        leader: Player,
        quest_players: set[Player],
    ) -> Generator[SelectionVote, Mapping[SelectionVote, int], None]:
        pass


class SimpleLoyalPlayer(BaseLoyalPlayer):
    pass


class MerlinLoyalPlayer(Player):
    def __init__(self, name: str, traitors: set[Player]) -> None:
        super().__init__(name)
        self._private_knowledge_base = KnowledgeBase(
            traitor_expression=sat.Not(operand=self.as_terminal()),
            final_traitors=traitors,
        )
        self._public_knowledge_base = KnowledgeBase(
            traitor_expression=sat.Not(operand=self.as_terminal()),
            final_traitors={},
        )


@dataclass
class KnowledgeBase:
    traitor_expression: sat.Expression
    final_traitors: set[Player]


class Game:
    def __init__(self, number_of_players: int, number_of_traitors: int) -> None:
        traitors = [TraitorPlayer(name=f"t{i}") for i in range(number_of_traitors)]
        self._players: list[Player] = [
            *[
                SimpleLoyalPlayer(name=f"l{i}")
                for i in range(number_of_players - number_of_traitors - 1)
            ],
            *traitors,
            MerlinLoyalPlayer(name="m", traitors=set(traitors)),
        ]
        self._leader_index = random.randint(0, number_of_players)

    def step(self) -> None:
        leader = self._players[self._leader_index]

        selection = leader.offer_selection()

        selection_votes: dict[SelectionVote, int] = defaultdict(lambda: 0)
        selection_generators: list[Generator] = []

        for player in self._players:
            generator = player.vote_in_selection(leader, selection)
            selection_votes[generator.send(None)] += 1
            selection_generators.append(generator)

        for generator in selection_generators:
            with contextlib.suppress(StopIteration):
                generator.send(selection_votes)

        if (
            len(list(selection_votes).count(SelectionVote.YES))
            >= len(self._players) // 2
        ):
            quest_votes: dict[SelectionVote, int] = defaultdict(lambda: 0)
            quest_generators: list[Generator] = []

            for player in selection:
                generator = player.vote_in_quest(leader, selection - player)
                quest_votes[generator.send(None)] += 1
                quest_generators.append(generator)

            for generator in quest_generators:
                with contextlib.suppress(StopIteration):
                    generator.send(quest_votes)

        self._leader_index = (self._leader_index + 1) % len(self._players)
