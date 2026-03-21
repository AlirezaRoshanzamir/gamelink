# type: ignore

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import override


class Token(StrEnum):
    BLUE = "blue"
    RED = "red"
    GREEN = "green"
    BROWN = "brown"
    WHITE = "white"
    GOLD = "gold"


@dataclass
class FamousCard:
    needing_rants: set[Token]
    score: int


@dataclass
class RantCard:
    needing_tokens: Mapping[Token, int]
    score: int
    resulting_token: Token
    level: int


@dataclass
class Player:
    reserved_rant_cards: set[RantCard] = field(default_factory=set)
    bought_rant_cards: set[RantCard] = field(default_factory=set)
    achieved_famous_cards: set[FamousCard] = field(default_factory=set)
    coin_tokens_count: Mapping[Token, int] = field(default_factory=dict)

    def calculate_total_tokens_count(self, token: Token) -> int:
        return self.coin_tokens_count.get(token, 0) + self.calculate_rant_tokens_count(
            token,
        )

    def calculate_rant_tokens_count(self, token: Token) -> int:
        return sum(
            1 for card in self.bought_rant_cards if card.resulting_token == token
        )

    def fund_coin_tokens(self, coin_tokens: Mapping[Token, int]) -> None:
        for token, count in coin_tokens.items():
            self.coin_tokens_count[token] += count

    def return_coin_tokens(self, coin_tokens: Mapping[Token, int]) -> None:
        for token, count in coin_tokens.items():
            self.coin_tokens_count[token] -= count

    def pay_and_return_coin_tokens_count(self, token: Token, count: int) -> int:
        from_rant_tokens_count = min(count, self.calculate_rant_tokens_count(token))
        from_coin_tokens_count = count - from_rant_tokens_count

        if from_coin_tokens_count < 0:
            msg = f"Cannot give #{count} {token}."
            raise AssertionError(msg)

        self.coin_tokens_count[token] = (
            self.coin_tokens_count.get(token, 0) - from_coin_tokens_count
        )

        return from_coin_tokens_count

    @property
    def total_score(self) -> int:
        return self.total_achieved_famous_cards_score + self.total_rant_cards_score

    @property
    def total_achieved_famous_cards_score(self) -> int:
        return sum(card.score for card in self.achieved_famous_cards)

    @property
    def total_rant_cards_score(self) -> int:
        return sum(card.score for card in self.bought_rant_cards)

    def choose_action(self) -> Action:
        pass


class Action(ABC):
    def __init__(self, player: Player, game: Game) -> None:
        self._player = player
        self._game = game
        self._returned_coin_tokens: Mapping[Token, int] = {}
        self._funded_coin_tokens: Mapping[Token, int] = {}

    @abstractmethod
    def is_feasible(self) -> bool:
        pass

    @abstractmethod
    def do(self) -> None:
        pass

    def revert(self) -> None:
        self.custom_revert()
        self._game.take_coin_tokens(self._returned_coin_tokens)
        self._player.fund_coin_tokens(self._returned_coin_tokens)

    @abstractmethod
    def custom_revert(self) -> None:
        pass

    def return_coin_tokens(self, coin_tokens: Mapping[Token, int]) -> None:
        for token, count in coin_tokens.items():
            self._returned_coin_tokens[token] = (
                self._returned_coin_tokens.get(token, 0) + count
            )
        self._game.return_coin_tokens({token: count})
        self._player.return_coin_tokens({token: count})

    def fund_player_coin_tokens(self, coin_tokens: Mapping[Token, int]) -> None:
        for token, count in coin_tokens.items():
            self._funded_coin_tokens[token] = (
                self._funded_coin_tokens.get(token, 0) + count
            )
        self._player.fund_coin_tokens({token: count})
        self._game.take_coin_tokens({token: count})


class BuyRantCard(Action):
    def __init__(self, player: Player, buying_rant_card: RantCard, game: Game) -> None:
        super().__init__(player, game)
        self._buying_rant_card = buying_rant_card
        self._reserved = buying_rant_card in player.reserved_rant_cards

    @override
    def is_feasible(self) -> bool:
        return self._apply(enable_returning_coin_tokens=False)

    @override
    def do(self) -> None:
        self._apply(enable_returning_coin_tokens=True)
        self._player.bought_rant_cards.add(self._buying_rant_card)
        if self._reserved:
            self._player.reserved_rant_cards.remove(self._buying_rant_card)

    @override
    def custom_revert(self) -> None:
        self._player.bought_rant_cards.remove(self._buying_rant_card)
        if self._reserved:
            self._player.reserved_rant_cards.add(self._buying_rant_card)

    def _apply(self, *, enable_returning_coin_tokens: bool) -> bool:
        remaining_tokens_count = 0

        for token, count in self._buying_rant_card.needing_tokens.items():
            giving_tokens_count = min(
                count,
                self._player.calculate_total_tokens_count(token),
            )
            remaining_tokens_count = count - giving_tokens_count
            if enable_returning_coin_tokens:
                self.return_coin_tokens(
                    {
                        token: self._player.pay_and_return_coin_tokens_count(
                            token,
                            giving_tokens_count,
                        ),
                    },
                )

        new_golds_count = (
            self._player.calculate_total_tokens_count(Token.GOLD)
            - remaining_tokens_count
        )
        if enable_returning_coin_tokens:
            self.return_coin_tokens(
                {
                    Token.GOLD: self._player.pay_and_return_coin_tokens_count(
                        Token.GOLD,
                        remaining_tokens_count,
                    ),
                },
            )

        return new_golds_count >= 0


class ReserveRantCard(Action):
    def __init__(
        self,
        player: Player,
        reserving_rant_card: RantCard,
        returning_token: Token | None,
        game: Game,
    ) -> None:
        super().__init__(player, game)
        self._reserving_rant_card = reserving_rant_card
        self._returning_token = returning_token

    @override
    def is_feasible(self) -> bool:
        raise NotImplementedError

    @override
    def do(self) -> None:
        self._player.reserved_rant_cards.add(self._reserving_rant_card)
        self.fund_player_coin_tokens({Token.GOLD: 1})
        if self._returning_token is not None:
            self.return_coin_tokens({self._returning_token: 1})

    @override
    def custom_revert(self) -> None:
        self._player.reserved_rant_cards.remove(self._reserving_rant_card)


class GetCoinTokens(Action):
    def __init__(
        self,
        player: Player,
        getting_coin_tokens: Mapping[Token, int],
        returning_coin_tokens: Mapping[Token, int],
        game: Game,
    ) -> None:
        super().__init__(player, game)
        self._getting_coin_tokens = getting_coin_tokens
        self._returned_coin_tokens = returning_coin_tokens

    @override
    def is_feasible(self) -> bool:
        raise NotImplementedError

    @override
    def do(self) -> None:
        self._game.take_coin_tokens(self._getting_coin_tokens)
        self._player.fund_coin_tokens(self._getting_coin_tokens)
        self._game.return_coin_tokens(self._returned_coin_tokens)
        self._player.return_coin_tokens(self._returned_coin_tokens)

    @override
    def custom_revert(self) -> None:
        pass


@dataclass
class Game:
    players: Sequence[Player]
    coin_tokens: Mapping[Token, int]
    each_row_rants: Sequence[Sequence[RantCard]]
    famous_cards: set[FamousCard]
    turn: int = 0

    def play(self) -> None:
        current_player = self.players[self.turn]
        action = current_player.choose_action()

        action.do()

        self.turn += 1
        self.turn %= len(self.players)

    def take_coin_tokens(self, coin_tokens: Mapping[Token, int]) -> None:
        for coin_token, count in coin_tokens.items():
            self.coin_tokens[coin_token] -= count

    def return_coin_tokens(self, coin_tokens: Mapping[Token, int]) -> None:
        for coin_token, count in coin_tokens.items():
            self.coin_tokens[coin_token] += count


class NextRantCardGenerator(ABC):
    @abstractmethod
    def generate(self, used_rant_cards: set[RantCard]) -> RantCard:
        pass


if __name__ == "__main__":
    first_player = Player()
    second_player = Player()
    game = Game(
        players=[first_player, second_player],
        coin_tokens={token.value: 6 for token in list(Token)},
        each_row_rants=[],
        famous_cards=set(),
    )
